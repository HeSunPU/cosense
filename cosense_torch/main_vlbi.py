import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional

torch.set_default_dtype(torch.float32)
import torch.optim as optim
import pickle
import math

from torchkbnufft import KbNufft, AdjKbNufft
from torchkbnufft.mri.dcomp_calc import calculate_radial_dcomp_pytorch
from torchkbnufft.math import absolute

# from loupe import models_vlbi, layers_vlbi # loupe package
import ehtim as eh # eht imaging package

from ehtim.observing.obs_helpers import *
import ehtim.const_def as ehc
from scipy.ndimage import gaussian_filter
import skimage.transform
import csv
import sys
import datetime
import warnings
import copy

import gc
import cv2

from astropy.io import fits
from pynfft.nfft import NFFT


from interferometry_helpers import *
import models

import argparse
import time

from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description="Deep Probabilistic Imaging Trainer for Interferometry")
parser.add_argument("--cuda", default=0, type=int, help="cuda index in use")
parser.add_argument("--array", default='/home/groot/BoumanLab/eht-imaging/arrays/EHT2019.txt', type=str, help="groud-truth EHT image file path")
# parser.add_argument("--array", default='/home/groot/BoumanLab/eht-imaging/arrays/EHT2025.txt', type=str, help="groud-truth EHT image file path")
# parser.add_argument("--array", default='/home/groot/BoumanLab/eht-imaging/arrays/Location_sites_SEFD_April.txt', type=str, help="groud-truth EHT image file path")
parser.add_argument("--target", default='m87', type=str, help="EHT target")
parser.add_argument("--fov", default=120, type=int, help="field of view of target")
parser.add_argument("--npix", default=32, type=int, help="# of pixel of reconstructed image")

# parser.add_argument("--save_path", default='./checkpoint/eht2019vis', type=str, help="file save path")
# parser.add_argument("--save_path", default='./checkpoint/eht2025vis', type=str, help="file save path")
# parser.add_argument("--save_path", default='./checkpoint/ehtmorevis', type=str, help="file save path")
parser.add_argument("--save_path", default='./checkpoint/eht2019cphaseamp', type=str, help="file save path")





parser.add_argument("--lr", default=1e-2, type=float, help="learning rate")
parser.add_argument("--n_batch", default=512, type=int, help="batch size")
# parser.add_argument("--n_batch", default=64, type=int, help="batch size")
parser.add_argument("--n_epoch", default=100, type=int, help="number of training epochs")
parser.add_argument("--sparsity_weight", default=1e-2, type=float, help="sparisty weight")
parser.add_argument("--diversity_weight", default=1e-2, type=float, help="diversity weight")

parser.add_argument("--dropout_rate", default=0.0, type=float, help="diversity weight")


parser.add_argument("--ttype", default='nfft', type=str, help="fourier transform computation method")
parser.add_argument("--dset", default='fashion_pad', type=str, help="dataset used for training the sampler")
# parser.add_argument("--data_product", default='vis', type=str, help="data product used for reconstruction")
# parser.add_argument("--data_product", default='cphase_amp', type=str, help="data product used for reconstruction")
parser.add_argument("--data_product", default='cphase_logcamp', type=str, help="data product used for reconstruction")
# parser.add_argument("--loss_func", default='l1', type=str, help="loss function used for reconstruction")
parser.add_argument("--loss_func", default='cross_correlation', type=str, help="loss function used for reconstruction")







if __name__ == "__main__":
	args = parser.parse_args()

	save_path = args.save_path
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	add_th_noise = False # False if you *don't* want to add thermal error. If there are no sefds in obs_orig it will use the sigma for each data point
	phasecal = True # True if you don't want to add atmospheric phase error. if False then it adds random phases to simulate atmosphere
	ampcal = True # True if you don't want to add atmospheric amplitude error. if False then add random gain errors 
	stabilize_scan_phase = False # if true then add a single phase error for each scan to act similar to adhoc phasing
	stabilize_scan_amp = False # if true then add a single gain error at each scan
	jones = False # apply jones matrix for including noise in the measurements (including leakage)
	inv_jones = False # no not invert the jones matrix
	frcal = True # True if you do not include effects of field rotation
	dcal = True # True if you do not include the effects of leakage
	dterm_offset = 0.05 # a random offset of the D terms is given at each site with this standard deviation away from 1
	dtermp = 0

	tint_sec = 10
	tadv_sec = 600
	tstart_hr = 0
	tstop_hr = 24
	bw_hz = 4e9

	rf = 230e9
	mjd = 57853 # day of observation


	ttype = args.ttype #'nfft'
	fov = args.fov * eh.RADPERUAS
	npix = args.npix
	target = args.target
	data_product = args.data_product#'vis'
	loss_func = args.loss_func


	if target == 'm87':
		ra = 12.513728717168174
		dec = 12.39112323919932
	elif target == 'sgrA':
		ra = 19.414182210498385
		dec = -29.24170032236311


	array = args.array #'/home/groot/BoumanLab/eht-imaging/arrays/EHT2019.txt'
	# array = '/home/groot/BoumanLab/eht-imaging/arrays/EHT2025.txt'
	eht = eh.array.load_txt(array)


	sites = np.sort(eht.tarr['site'])
	n_sites = len(sites)
	sites_dic = {}
	for k in range(n_sites):
		sites_dic[sites[k]] = k


	zbl = 1.0
	prior_fwhm = 60*eh.RADPERUAS#
	simim = eh.image.make_empty(npix, fov, ra, dec, rf=rf, source='random', mjd=mjd).add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))
	# simim.imvec = np.zeros((npix, npix, 1)).reshape((-1, 1))#xdata[0, :, :, :].reshape((-1, 1))
	obs = simim.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, add_th_noise=add_th_noise, ampcal=ampcal, phasecal=phasecal, 
				    stabilize_scan_phase=stabilize_scan_phase, stabilize_scan_amp=stabilize_scan_amp,
				    jones=jones,inv_jones=inv_jones,dcal=dcal, frcal=frcal, dterm_offset=dterm_offset)


	# define the eht observation function using non-uniform fft
	if torch.cuda.is_available():
		device = torch.device('cuda:{}'.format(0))
	else:
		device = torch.device('cpu')
	nufft_ob = KbNufft(im_size=(npix, npix), numpoints=3)
	dft_mat, ktraj_vis, pulsefac_vis_torch, cphase_ind_list, cphase_sign_list, camp_ind_list = Obs_params_torch(obs, simim, snrcut=0.0, ttype=ttype, data_product=data_product)

	obs_sigma = torch.tensor(255.0 * obs.data['sigma'], dtype=torch.float32)
	eht_obs_torch = eht_observation_pytorch(npix, nufft_ob, dft_mat, ktraj_vis, pulsefac_vis_torch, cphase_ind_list, cphase_sign_list, camp_ind_list, device, ttype=ttype, sigma=obs_sigma, data_product=data_product)


	vis_t1 = []
	vis_t2 = []
	for k in range(len(obs.data['vis'])):
		vis_t1.append(sites_dic[obs.data['t1'][k]])
		vis_t2.append(sites_dic[obs.data['t2'][k]])

	vis_t1 = np.array(vis_t1)
	vis_t2 = np.array(vis_t2)

	n_vis = len(obs.data['vis'])

	if data_product == 'cphase_amp' or data_product == 'cphase_logcamp':
		cphase_t1 = []
		cphase_t2 = []
		cphase_t3 = []
		for k in range(len(obs.cphase['cphase'])):
			cphase_t1.append(sites_dic[obs.cphase['t1'][k]])
			cphase_t2.append(sites_dic[obs.cphase['t2'][k]])
			cphase_t3.append(sites_dic[obs.cphase['t3'][k]])

		cphase_t1 = np.array(cphase_t1)
		cphase_t2 = np.array(cphase_t2)
		cphase_t3 = np.array(cphase_t3)

		n_cphase = len(obs.cphase['cphase'])

	if data_product == 'cphase_logcamp':
		camp_t1 = []
		camp_t2 = []
		camp_t3 = []
		camp_t4 = []
		for k in range(len(obs.camp['camp'])):
			camp_t1.append(sites_dic[obs.camp['t1'][k]])
			camp_t2.append(sites_dic[obs.camp['t2'][k]])
			camp_t3.append(sites_dic[obs.camp['t3'][k]])
			camp_t4.append(sites_dic[obs.camp['t4'][k]])

		camp_t1 = np.array(camp_t1)
		camp_t2 = np.array(camp_t2)
		camp_t3 = np.array(camp_t3)
		camp_t4 = np.array(camp_t4)

		n_camp = len(obs.camp['camp'])







	#####################################################################################
	## Load the fashion-mnist images for training the sampler and the reconstructor
	#####################################################################################
	list_of_transforms = transforms.Compose([transforms.ToTensor()])

	dset = args.dset#'fashion_pad'

	if dset == "fashion_pad":
		train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
		                            transform=list_of_transforms)
		test_dataset = datasets.FashionMNIST('./data', train=False, download=True,
		                           transform=list_of_transforms)
		# train_data = 30.0 * train_dataset.data / torch.sum(train_dataset.data, (1, 2)).unsqueeze(-1).unsqueeze(-1)
		# test_data = 30.0 * test_dataset.data / torch.sum(test_dataset.data, (1, 2)).unsqueeze(-1).unsqueeze(-1)
		train_data = train_dataset.data / 256.0
		test_data = test_dataset.data / 256.0
		train_data = functional.pad(train_data, (2, 2, 2, 2, 0, 0), "constant", 0)
		test_data = functional.pad(test_data, (2, 2, 2, 2, 0, 0), "constant", 0)

		n_train = train_data.shape[0]
		n_test = test_data.shape[0]



	#####################################################################################
	## Define the Ising sampler and the reconstructor (dirty image encoder + unnet)
	#####################################################################################

	# Ising smapler
	sampler = models.IsingNet(nsensors=n_sites, nsteps=3).to(device=device)

	# fully connected nn encoder - convert data products to a dirty image 
	if data_product == 'vis':
		dirty_im_encoder = nn.Sequential(
		            nn.Flatten(),
		            nn.Linear(2*n_vis, 2*npix*npix),
		        ).to(device)
		nn.init.normal_(dirty_im_encoder[1].weight, mean=0, std=1e-3) 
		dirty_im_encoder[1].bias.data.fill_(0)

	elif args.data_product == 'cphase_amp':
		dirty_im_encoder = nn.Sequential(
		            nn.Linear(n_vis+n_cphase, 1000),
		            nn.LeakyReLU(negative_slope=0.01, inplace=True),
		            nn.BatchNorm1d(1000, eps=1e-2, affine=True),
		            nn.Linear(1000, 1000),
		            nn.LeakyReLU(negative_slope=0.01, inplace=True),
		            nn.BatchNorm1d(1000, eps=1e-2, affine=True),
		            nn.Linear(1000, 2*npix*npix),
		        ).to(device)
		nn.init.normal_(dirty_im_encoder[0].weight, mean=0, std=1e-3) 
		dirty_im_encoder[0].bias.data.fill_(0)
		nn.init.normal_(dirty_im_encoder[3].weight, mean=0, std=1e-3) 
		dirty_im_encoder[3].bias.data.fill_(0)
		nn.init.normal_(dirty_im_encoder[6].weight, mean=0, std=1e-3) 
		dirty_im_encoder[6].bias.data.fill_(0)

	elif args.data_product == 'cphase_logcamp':
		dirty_im_encoder = nn.Sequential(
		            nn.Linear(n_camp+n_cphase, 1000),
		            nn.LeakyReLU(negative_slope=0.01, inplace=True),
		            nn.BatchNorm1d(1000, eps=1e-2, affine=True),
		            nn.Linear(1000, 1000),
		            nn.LeakyReLU(negative_slope=0.01, inplace=True),
		            nn.BatchNorm1d(1000, eps=1e-2, affine=True),
		            nn.Linear(1000, 2*npix*npix),
		        ).to(device)
		nn.init.normal_(dirty_im_encoder[0].weight, mean=0, std=1e-3) 
		dirty_im_encoder[0].bias.data.fill_(0)
		nn.init.normal_(dirty_im_encoder[3].weight, mean=0, std=1e-3) 
		dirty_im_encoder[3].bias.data.fill_(0)
		nn.init.normal_(dirty_im_encoder[6].weight, mean=0, std=1e-3) 
		dirty_im_encoder[6].bias.data.fill_(0)

	# unit reconstructor
	reconstructor = models.LOUPEUNet(in_chans=2, out_chans=1, chans=64, num_pool_layers=4, drop_prob=args.dropout_rate).to(device)


	#####################################################################################
	## Train the sampler and reconstructor
	#####################################################################################
	n_epoch = args.n_epoch#100
	n_batch = args.n_batch#512#1024#128
	sparisty_weight = args.sparsity_weight#1e-2#1e-3#
	diversity_weight = args.diversity_weight#1e-2#1e-3#
	lr = args.lr#1e-2#1e-3#

	params_list = list(sampler.parameters()) + list(dirty_im_encoder.parameters()) + list(reconstructor.parameters())
	optimizer = optim.Adam(params_list, lr = lr)

	loss_list = []
	loss_recon_list = []
	loss_sparsity_list = []
	loss_diversity_list = []

	loss_val_list = []
	loss_val_recon_list = []
	loss_val_sparsity_list = []
	loss_val_diversity_list = []

	t_start = time.time()
	for epoch in range(n_epoch):
		reconstructor.train()

		loss_train = 0
		loss_recon_train = 0
		loss_sparsity_train = 0
		loss_diversity_train = 0

		for k in range(n_train//n_batch+1):
			true_im = train_data[k*n_batch:min((k+1)*n_batch, n_train)].unsqueeze(1).to(device)
			vis, visamp, cphase, logcamp = eht_obs_torch(true_im)

			z_samples = torch.randn(true_im.shape[0], n_sites).to(device=device)
			site_masks, energy = sampler.forward(z_samples)

			site_masks = 0.5 * (site_masks + 1)


			if data_product == 'vis':
				vis_mask = site_masks[:, vis_t1] * site_masks[:, vis_t2]

				vis_selected = vis * vis_mask.unsqueeze(1)
				input_data = vis_selected

			elif args.data_product == 'cphase_amp':
				vis_mask = site_masks[:, vis_t1] * site_masks[:, vis_t2]
				cphase_mask = site_masks[:, cphase_t1] * site_masks[:, cphase_t2] * site_masks[:, cphase_t3]

				visamp_selected = visamp * vis_mask
				cphase_selected = cphase * cphase_mask
				input_data = torch.cat([visamp_selected, cphase_selected], -1)

			elif args.data_product == 'cphase_logcamp':
				cphase_mask = site_masks[:, cphase_t1] * site_masks[:, cphase_t2] * site_masks[:, cphase_t3]
				camp_mask = site_masks[:, camp_t1] * site_masks[:, camp_t2] * site_masks[:, camp_t3] * site_masks[:, camp_t4]

				logcamp_selected = logcamp * camp_mask
				cphase_selected = cphase * cphase_mask
				input_data = torch.cat([logcamp_selected, cphase_selected], -1)


			dirty_im = dirty_im_encoder.forward(input_data.to(torch.float32))
			dirty_im = dirty_im.reshape((-1, 2, npix, npix))

			recon_im = reconstructor.forward(dirty_im)

			if loss_func == 'l1':
				loss_recon = torch.sum(torch.abs(recon_im - true_im), (1, 2, 3)) / (npix*npix)
			elif loss_func == 'cross_correlation':
				normal_scalar = torch.sqrt(torch.sum(recon_im**2, (1, 2, 3)) + 1e-5) * torch.sqrt(torch.sum(true_im**2, (1, 2, 3)) + 1e-5)
				loss_recon = 1 - (functional.max_pool2d(functional.conv2d(recon_im.transpose(0, 1), true_im, groups=true_im.shape[0], padding=16), 32).squeeze() / normal_scalar)

			loss_sparsity = torch.sum(site_masks, 1)

			loss = torch.mean(loss_recon  + sparisty_weight * loss_sparsity + diversity_weight * energy)

			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			# nn.utils.clip_grad_norm_(params_list, args.clip)
			optimizer.step()

			loss_recon_train = loss_recon_train + torch.sum(loss_recon).detach().cpu().numpy()
			loss_sparsity_train = loss_sparsity_train + torch.sum(loss_sparsity).detach().cpu().numpy()
			loss_diversity_train = loss_diversity_train + torch.sum(energy).detach().cpu().numpy()


		loss_train = loss_recon_train + sparisty_weight * loss_sparsity_train + diversity_weight * loss_diversity_train

		loss_list.append(loss_train/n_train)
		loss_recon_list.append(loss_recon_train/n_train)
		loss_sparsity_list.append(loss_sparsity_train/n_train)
		loss_diversity_list.append(loss_diversity_train/n_train)


		print(f"epoch: {epoch:}, loss: {loss_list[-1]:.5f}, loss recon: {loss_recon_list[-1]:.5f}, loss sparisty: {loss_sparsity_list[-1]:.5f}, energy: {loss_diversity_list[-1]:.5f}")

		
		reconstructor.eval()

		loss_val = 0
		loss_recon_val = 0
		loss_sparsity_val = 0
		loss_diversity_val = 0

		for k in range(n_test//n_batch+1):
			true_im = train_data[k*n_batch:min((k+1)*n_batch, n_test)].unsqueeze(1).to(device)
			vis, visamp, cphase, logcamp = eht_obs_torch(true_im)

			z_samples = torch.randn(true_im.shape[0], n_sites).to(device=device)
			site_masks, energy = sampler.forward(z_samples)

			site_masks = 0.5 * (site_masks + 1)

			if data_product == 'vis':
				vis_mask = site_masks[:, vis_t1] * site_masks[:, vis_t2]

				vis_selected = vis * vis_mask.unsqueeze(1)
				input_data = vis_selected

			elif args.data_product == 'cphase_amp':
				vis_mask = site_masks[:, vis_t1] * site_masks[:, vis_t2]
				cphase_mask = site_masks[:, cphase_t1] * site_masks[:, cphase_t2] * site_masks[:, cphase_t3]

				visamp_selected = visamp * vis_mask
				cphase_selected = cphase * cphase_mask
				input_data = torch.cat([visamp_selected, cphase_selected], -1)

			elif args.data_product == 'cphase_logcamp':
				cphase_mask = site_masks[:, cphase_t1] * site_masks[:, cphase_t2] * site_masks[:, cphase_t3]
				camp_mask = site_masks[:, camp_t1] * site_masks[:, camp_t2] * site_masks[:, camp_t3] * site_masks[:, camp_t4]

				logcamp_selected = logcamp * camp_mask
				cphase_selected = cphase * cphase_mask
				input_data = torch.cat([logcamp_selected, cphase_selected], -1)


			dirty_im = dirty_im_encoder.forward(input_data.to(torch.float32))
			dirty_im = dirty_im.reshape((-1, 2, npix, npix))

			recon_im = reconstructor.forward(dirty_im)

			if loss_func == 'l1':
				loss_recon = torch.sum(torch.abs(recon_im - true_im), (1, 2, 3)) / (npix*npix)
			elif loss_func == 'cross_correlation':
				normal_scalar = torch.sqrt(torch.sum(recon_im**2, (1, 2, 3)) + 1e-5) * torch.sqrt(torch.sum(true_im**2, (1, 2, 3)) + 1e-5)
				loss_recon = 1 - (functional.max_pool2d(functional.conv2d(recon_im.transpose(0, 1), true_im, groups=true_im.shape[0], padding=16), 32).squeeze() / normal_scalar)


			loss_sparsity = torch.sum(site_masks, 1)


			loss_recon_val = loss_recon_val + torch.sum(loss_recon).detach().cpu().numpy()
			loss_sparsity_val = loss_sparsity_val + torch.sum(loss_sparsity).detach().cpu().numpy()
			loss_diversity_val = loss_diversity_val + torch.sum(energy).detach().cpu().numpy()


		loss_val = loss_recon_val + sparisty_weight * loss_sparsity_val + diversity_weight * loss_diversity_val

		loss_val_list.append(loss_val/n_test)
		loss_val_recon_list.append(loss_recon_val/n_test)
		loss_val_sparsity_list.append(loss_sparsity_val/n_test)
		loss_val_diversity_list.append(loss_diversity_val/n_test)

		print(f"val loss: {loss_val_list[-1]:.5f}, val loss recon: {loss_val_recon_list[-1]:.5f}, val loss sparisty: {loss_val_sparsity_list[-1]:.5f}, val energy: {loss_val_diversity_list[-1]:.5f}")


	t_end = time.time()

	list(sampler.parameters()) + list(dirty_im_encoder.parameters()) + list(reconstructor.parameters())

	torch.save(sampler.state_dict(), save_path+'/ising_sampler')
	torch.save(dirty_im_encoder.state_dict(), save_path+'/dirtyim_encoder')
	torch.save(reconstructor.state_dict(), save_path+'/unet_reconstructor')

	loss_all = {}
	loss_all['time'] = t_end - t_start
	loss_all['total'] = np.array(loss_list)
	loss_all['recon'] = np.array(loss_recon_list)
	loss_all['sparsity'] = np.array(loss_sparsity_list)
	loss_all['diversity'] = np.array(loss_diversity_list)
	loss_all['val_total'] = np.array(loss_val_list)
	loss_all['val_recon'] = np.array(loss_val_recon_list)
	loss_all['val_sparsity'] = np.array(loss_val_sparsity_list)
	loss_all['val_diversity'] = np.array(loss_val_diversity_list)


	np.save(save_path+'/loss.npy', loss_all)


	#####################################################################################
	## visualize the ising model parameters
	#####################################################################################
	delta = sampler.delta.detach().cpu().numpy()
	M = sampler.M.detach().cpu().numpy()
	M = M + M.T

	plt.figure(), 
	plt.figure(), plt.plot(delta, 'ro')
	plt.ylim([-0.5, 1.5])
	plt.title(r'Ising model parameter ($\theta_{jj}$)', fontsize=18)
	plt.xticks(range(len(list(sites_dic.keys()))), list(sites_dic.keys()), size='small', rotation=35, fontsize=12)
	plt.yticks(fontsize=14)
	plt.savefig(save_path+'/ising_delta.pdf')


	plt.figure(), plt.imshow(M), plt.set_cmap('seismic')
	cb = plt.colorbar()
	cb.ax.tick_params(labelsize=14)
	plt.title(r'Ising model parameter ($\theta_{jk}$)', fontsize=18)
	plt.clim(-0.6, 0.6)
	plt.xticks(range(len(list(sites_dic.keys()))), list(sites_dic.keys()), size='small', rotation=35, fontsize=12)
	plt.yticks(range(len(list(sites_dic.keys()))), list(sites_dic.keys()), size='small', fontsize=12)
	plt.savefig(save_path+'/ising_M.pdf')



