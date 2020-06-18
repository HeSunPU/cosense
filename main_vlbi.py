import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomUniform, Constant
import keras.models
import keras.layers
import keras.initializers
import keras.regularizers
import keras.callbacks
from keras import backend as K
from keras import losses
from keras.datasets import fashion_mnist
from keras.datasets import mnist


# from loupe import models_vlbi, layers_vlbi # loupe package
import ehtim as eh # eht imaging package

from ehtim.observing.obs_helpers import *
from scipy.ndimage import gaussian_filter
import skimage.transform
# import helpers as hp
import csv
import sys
import datetime
import warnings


from models_posci import IsingVisNet, IsingCpAmpNet, IsingMutipleVisNet, IsingMutipleCpAmpNet, IsingVisFeatureNet, IsingCpAmpFeatureNet
from losses_posci import site_sparsity, energy, Lambda_similarity, Lambda_angle_diff
from data_augmentation import elastic_transform

import gc

# mute the verbose warnings
tf.logging.set_verbosity(tf.logging.ERROR)
warnings.filterwarnings("ignore")

# initialize GPU
K.clear_session()
gpu_id = 0
gpu = '/gpu:' + str(gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
set_session(tf.Session(config=config))


plt.ion()


# import numpy as np

def observe_prob(params,limit):
	return(1./(1.+np.exp(-params[0]*(limit-params[1]))))

cutoff = 3.0
with open('cdf_params.txt','r') as fi:
	lines = fi.readlines()

cdf_params = {}
obs_prob = {}
for line in lines:
	cdf_params[line.split('\t')[0]] = np.array([float(line.split('\t')[1]),float(line.split('\t')[2].rstrip())])
	obs_prob[line.split('\t')[0]] = observe_prob(cdf_params[line.split('\t')[0]],cutoff)

print(obs_prob)

nsamp = 10000 #100000
def Prepare_EHT_Data(fov_param, flux_label, blur_param, sefd_param, eht_array='eht2019', target='m87', data_augmentation=False, npix=32):
	"""
    Prepare the EHT training data for learning a probabilistic sensing for computational imaging!
    
	fov_param: field of view
	flux_label: 0 represents varying flux, 1 represents constant flux
	blur_param: fraction of nominal resolution

    """

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

	###############################################################################
	# define EHT array
	###############################################################################
	if eht_array == 'eht2019':
		# please change this to the folder of ehtim
		# array = '/home/groot/BoumanLab/eht-imaging/arrays/EHT2019.txt'
		array = '/home/groot/BoumanLab/eht-imaging/arrays/EHT2019_no_carma.txt'
	elif eht_array == 'eht2025':
		# please change this to the folder of ehtim
		array = '/home/groot/BoumanLab/eht-imaging/arrays/EHT2025.txt'

	eht = eh.array.load_txt(array)

	###############################################################################
	# define observation field of view
	###############################################################################
	fov = fov_param * eh.RADPERUAS

	###############################################################################
	# define scientific target
	###############################################################################
	if target == 'm87':
		ra = 12.513728717168174
		dec = 12.39112323919932
	elif target == 'sgrA':
		ra = 19.414182210498385
		dec = -29.24170032236311

	###############################################################################
	# generate the discrete Fourier transform matrix for complex visibilities
	###############################################################################
	rf = 230e9
	# npix = 32
	mjd = 57853 # day of observation
	simim = eh.image.make_empty(npix, fov, ra, dec, rf=rf, source='random', mjd=mjd)
	simim.imvec = np.zeros((npix, npix, 1)).reshape((-1, 1))#xdata[0, :, :, :].reshape((-1, 1))
	obs = simim.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, add_th_noise=add_th_noise, ampcal=ampcal, phasecal=phasecal, 
				    stabilize_scan_phase=stabilize_scan_phase, stabilize_scan_amp=stabilize_scan_amp,
				    jones=jones,inv_jones=inv_jones,dcal=dcal, frcal=frcal, dterm_offset=dterm_offset)

	obs_data = obs.unpack(['u', 'v', 'vis', 'sigma'])
	uv = np.hstack((obs_data['u'].reshape(-1,1), obs_data['v'].reshape(-1,1)))
	F = ftmatrix(simim.psize, simim.xdim, simim.ydim, uv, pulse=simim.pulse)

	t1 = obs.data['t1']
	t2 = obs.data['t2']
	n_sites = np.unique(np.concatenate([t1, t2])).shape[0] + 1

	###############################################################################
	# generate the discrete Fourier transform matrices for closure phases
	###############################################################################

	# obs.add_cphase(count='min')
	obs.add_cphase(count='max')
	tc1 = obs.cphase['t1']
	tc2 = obs.cphase['t2']
	tc3 = obs.cphase['t3']

	cphase_map = np.zeros((len(obs.cphase['time']), 3))

	zero_symbol = 10000
	for k1 in range(cphase_map.shape[0]):
		for k2 in list(np.where(obs.data['time']==obs.cphase['time'][k1])[0]):
			if obs.data['t1'][k2] == obs.cphase['t1'][k1] and obs.data['t2'][k2] == obs.cphase['t2'][k1]:
				cphase_map[k1, 0] = k2
				if k2 == 0:
					cphase_map[k1, 0] = zero_symbol
			elif obs.data['t2'][k2] == obs.cphase['t1'][k1] and obs.data['t1'][k2] == obs.cphase['t2'][k1]:
				cphase_map[k1, 0] = -k2
				if k2 == 0:
					cphase_map[k1, 0] = -zero_symbol
			elif obs.data['t1'][k2] == obs.cphase['t2'][k1] and obs.data['t2'][k2] == obs.cphase['t3'][k1]:
				cphase_map[k1, 1] = k2
				if k2 == 0:
					cphase_map[k1, 1] = zero_symbol
			elif obs.data['t2'][k2] == obs.cphase['t2'][k1] and obs.data['t1'][k2] == obs.cphase['t3'][k1]:
				cphase_map[k1, 1] = -k2
				if k2 == 0:
					cphase_map[k1, 1] = -zero_symbol
			elif obs.data['t1'][k2] == obs.cphase['t3'][k1] and obs.data['t2'][k2] == obs.cphase['t1'][k1]:
				cphase_map[k1, 2] = k2
				if k2 == 0:
					cphase_map[k1, 2] = zero_symbol
			elif obs.data['t2'][k2] == obs.cphase['t3'][k1] and obs.data['t1'][k2] == obs.cphase['t1'][k1]:
				cphase_map[k1, 2] = -k2
				if k2 == 0:
					cphase_map[k1, 2] = -zero_symbol


	F_cphase = np.zeros((cphase_map.shape[0], npix*npix, 3), dtype=np.complex64)
	cphase_proj = np.zeros((cphase_map.shape[0], F.shape[0]), dtype=np.float32)
	for k in range(cphase_map.shape[0]):
		for j in range(cphase_map.shape[1]):
			if cphase_map[k][j] > 0:
				if int(cphase_map[k][j]) == zero_symbol:
					cphase_map[k][j] = 0
				F_cphase[k, :, j] = F[int(cphase_map[k][j]), :]
				cphase_proj[k, int(cphase_map[k][j])] = 1
			else:
				if np.abs(int(cphase_map[k][j])) == zero_symbol:
					cphase_map[k][j] = 0
				F_cphase[k, :, j] = np.conj(F[int(-cphase_map[k][j]), :])
				cphase_proj[k, int(-cphase_map[k][j])] = -1

	###############################################################################
	# load the data
	###############################################################################

	# nsamp = 10000

	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
	(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

	# xdata_train = 1.0*x_train[0:int(nsamp*0.7)]
	xdata_train = 1.0*x_train[[k%60000 for k in range(int(nsamp*0.7))]]
	if data_augmentation:
		rot_random = np.random.rand(int(nsamp*0.7)) * 360
		for k in range(int(nsamp*0.7)):
			im = np.expand_dims(xdata_train[k], -1)
			im = np.concatenate([im, np.zeros(im.shape)], -1)
			# im_deform = elastic_transform(im, 15, 2, 0.5)
			im_deform = elastic_transform(im, 20, 2, 0.5)
			xdata_train[k] = im_deform[:, :, 0]
			xdata_train[k] = skimage.transform.rotate(xdata_train[k], rot_random[k])
	xdata_train = np.pad(xdata_train, ((0,0), (2,2), (2,2)), 'constant')  # get to 32x32

	xdata_train = xdata_train[..., np.newaxis]/255


	# xdata_test = 1.0*x_train_mnist[0:int(nsamp*0.3)]
	xdata_test = 1.0*x_train_mnist[[k%60000 for k in range(int(nsamp*0.3))]]
	xdata_test = np.pad(xdata_test, ((0,0), (2,2), (2,2)), 'constant')  # get to 32x32
	xdata_test = xdata_test[..., np.newaxis]/255
	for k in range(int(0.3*nsamp)):
		xdata_test[k] = 2.2 * gaussian_filter(xdata_test[k], 2)

	# res = obs.res()
	# for k in range(int(0.3*nsamp)):
	# 	simim.imvec = xdata_test[k, :, :, :].reshape((-1, 1))
	# 	im_out = simim.blur_circ(0.3*res)
	# 	xdata_test[k, :, :, 0] = im_out.imvec.reshape((32, 32))

	xdata = np.concatenate([xdata_train, xdata_test], 0)

	###############################################################################
	# define the flux label
	###############################################################################
	if flux_label != 0:
		for k in range(nsamp):
			xdata[k] = 224.46*xdata[k] / np.sum(xdata[k])

	###############################################################################
	# define blurry effect: the fraction of the nominal resolution
	###############################################################################
	xdata_blur = np.zeros(xdata.shape)
	res = obs.res()
	for k in range(xdata.shape[0]):
		simim.imvec = xdata[k, :, :, :].reshape((-1, 1))
		im_out = simim.blur_circ(blur_param*res)
		xdata_blur[k, :, :, 0] = im_out.imvec.reshape((32, 32))

	###############################################################################
	# thermal noises: 0 represents no thermal noises, 1 represents site-varying thermal noises, 2 represents site-equivalent thermal noises
	###############################################################################
	if sefd_param == 1:
		sigma = 224.46 * np.concatenate([np.expand_dims(obs.data['sigma'], -1), np.expand_dims(obs.data['sigma'], -1)], -1)
	elif sefd_param == 2:
		sigma = 224.46 * np.concatenate([np.expand_dims(obs.data['sigma'], -1), np.expand_dims(obs.data['sigma'], -1)], -1)
		sigma = np.mean(sigma.reshape((-1, ))) * np.ones(sigma.shape)
	else:
		sigma = None

	return xdata, xdata_blur, t1, t2, F, tc1, tc2, tc3, F_cphase, cphase_proj, sigma


def Train_IsingVisNet(eht_array, target, fov_param, flux_label, blur_param, sefd_param, lr, nb_epochs_train, sample_weight, ising_weight, batch_size = 32, n_ising_layers = 5, models_dir='../joint_opt/models/anti-aliasing/12302019/', savefile_name='nn_params', weather=False):
	
	###############################################################################
	# prepare the training data
	###############################################################################
	xdata, xdata_blur, t1, t2, F, tc1, tc2, tc3, F_cphase, cphase_proj, sigma = Prepare_EHT_Data(fov_param, flux_label, blur_param, sefd_param, eht_array, target)
	n_sites = np.unique(np.concatenate([t1, t2])).shape[0] + 1
	###############################################################################
	# define the model
	###############################################################################
	# nsamp = 10000

	# model = IsingVisNet(t1, t2, F, n_ising_layers=n_ising_layers, slope_const=3, sigma=sigma)

	if weather:
		model = IsingVisNet(t1, t2, F, n_ising_layers=n_ising_layers, slope_const=3, obs_prob=obs_prob)
		# model = IsingCpAmpNet(t1, t2, tc1, tc2, tc3, F, F_cphase, cphase_proj, n_ising_layers=n_ising_layers, slope_const=3, sigma=sigma, obs_prob=obs_prob)
	else:
		model = IsingVisNet(t1, t2, F, n_ising_layers=n_ising_layers, slope_const=3)
		# model = IsingCpAmpNet(t1, t2, tc1, tc2, tc3, F, F_cphase, cphase_proj, n_ising_layers=n_ising_layers, slope_const=3, sigma=sigma)

	
	adam_opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

	recon_weight = 1.0
	model.compile(optimizer=adam_opt, loss={'recon': 'mean_absolute_error', 'sampling': site_sparsity, 'ising': energy}, 
				loss_weights={'recon': recon_weight, 'sampling': sample_weight, 'ising': ising_weight})

	# modelname = os.path.join(models_dir, savefile_name+'best.h5')

	checkpoint = ModelCheckpoint(models_dir+savefile_name+'best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

	class Save_weights(keras.callbacks.Callback):
		def on_epoch_begin(self, epoch, logs=None):
			model.save_weights(os.path.join(models_dir, savefile_name+'weights{:02d}.hdf5'.format(epoch)))

	history = model.fit({'input': xdata}, {'recon': xdata_blur, 'sampling': np.zeros((nsamp, n_sites)), 'ising': np.zeros((nsamp, ))},
	                validation_split=0.3,
	                initial_epoch=1,
	                epochs=1 + nb_epochs_train,
	                batch_size=batch_size,
	                verbose=1,
	                callbacks=[checkpoint])#,


	modelname = os.path.join(models_dir, savefile_name+'.h5')
	model.save_weights(modelname)
	w = csv.writer(open(os.path.join(models_dir, 'history_'+savefile_name+'.csv'), 'w'))
	for key, val in history.history.items():
		w.writerow([key, val])

	del history

	return model


def Train_IsingCpAmpNet(eht_array, target, fov_param, flux_label, blur_param, sefd_param, lr, nb_epochs_train, sample_weight, ising_weight, batch_size = 32, n_ising_layers = 5, models_dir='../joint_opt/models/anti-aliasing/12302019/', savefile_name='nn_params', weather=False):
	
	###############################################################################
	# prepare the training data
	###############################################################################
	xdata, xdata_blur, t1, t2, F, tc1, tc2, tc3, F_cphase, cphase_proj, sigma = Prepare_EHT_Data(fov_param, flux_label, blur_param, sefd_param, eht_array, target, data_augmentation=True)
	n_sites = np.unique(np.concatenate([t1, t2])).shape[0] + 1
	###############################################################################
	# define the model
	###############################################################################
	# nsamp = 10000

	vis = np.matmul(xdata.reshape((-1, 32*32)), np.transpose(F)).astype(np.complex64)
	vis1 = np.matmul(xdata.reshape((-1, 32*32)), np.transpose(F_cphase[:, :, 0])).astype(np.complex64)
	vis2 = np.matmul(xdata.reshape((-1, 32*32)), np.transpose(F_cphase[:, :, 1])).astype(np.complex64)
	vis3 = np.matmul(xdata.reshape((-1, 32*32)), np.transpose(F_cphase[:, :, 2])).astype(np.complex64)



	vis_concat = np.concatenate([vis.real, vis.imag], -1)
	cphase = np.angle(vis1 * vis2 * vis3) * 180 / np.pi
	vis_amp = np.abs(vis)
	vis_angle = np.angle(vis) * 180 / np.pi

	if weather:
		# model = IsingVisNet(t1, t2, F, n_ising_layers=n_ising_layers, slope_const=3)
		model = IsingCpAmpNet(t1, t2, tc1, tc2, tc3, F, F_cphase, cphase_proj, n_ising_layers=n_ising_layers, slope_const=3, sigma=sigma, obs_prob=obs_prob)
	else:
		model = IsingCpAmpNet(t1, t2, tc1, tc2, tc3, F, F_cphase, cphase_proj, n_ising_layers=n_ising_layers, slope_const=3, sigma=sigma)



	adam_opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

	recon_weight = 1.0
	vis_weight = 0.0 #2e-2
	cphase_weight = 1e-1
	# model.compile(optimizer=adam_opt, loss={'recon': 'mean_absolute_error', 'sampling': site_sparsity, 'ising': energy}, 
	# 			loss_weights={'recon': recon_weight, 'sampling': sample_weight, 'ising': ising_weight})

	
	model.compile(optimizer=adam_opt, loss={'recon': Lambda_similarity, 'sampling': site_sparsity, 'ising': energy, 'vis_angle_pred': Lambda_angle_diff, 'cphase_pred': Lambda_angle_diff}, 
				loss_weights={'recon': recon_weight, 'sampling': sample_weight, 'ising': ising_weight, 'vis_angle_pred': vis_weight, 'cphase_pred': cphase_weight})

	# modelname = os.path.join(models_dir, savefile_name+'best.h5')

	checkpoint = ModelCheckpoint(models_dir+savefile_name+'best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

	class Save_weights(keras.callbacks.Callback):
		def on_epoch_begin(self, epoch, logs=None):
			model.save_weights(os.path.join(models_dir, savefile_name+'weights{:02d}.hdf5'.format(epoch)))

	
	history = model.fit({'input': xdata}, {'recon': xdata_blur, 'sampling': np.zeros((nsamp, n_sites)), 'ising': np.zeros((nsamp, )), 'vis_angle_pred': vis_angle, 'cphase_pred': cphase},
	                validation_split=0.3,
	                initial_epoch=1,
	                epochs=1 + nb_epochs_train,
	                batch_size=batch_size,
	                verbose=1,
	                callbacks=[checkpoint])


	modelname = os.path.join(models_dir, savefile_name+'.h5')
	model.save_weights(modelname)
	w = csv.writer(open(os.path.join(models_dir, 'history_'+savefile_name+'.csv'), 'w'))
	for key, val in history.history.items():
		w.writerow([key, val])

	K.clear_session()

	del history, checkpoint, adam_opt

	del xdata, xdata_blur, t1, t2, F, tc1, tc2, tc3, F_cphase, cphase_proj, sigma

	return model


def Train_IsingMutipleVisNet(eht_array, target_list, fov_param, flux_label, blur_param, sefd_param, lr, nb_epochs_train, sample_weight, ising_weight, batch_size = 32, n_ising_layers = 5, models_dir='../joint_opt/models/anti-aliasing/12302019/', savefile_name='nn_params'):
	
	###############################################################################
	# prepare the training data
	###############################################################################
	t1_list = []
	t2_list = []
	F_list = []
	tc1_list = []
	tc2_list = []
	tc3_list = []
	F_cphase_list = []
	cphase_proj_list = []

	for k in range(len(target_list)):
		target = target_list[k]
		if k == 0:
			xdata, xdata_blur, t1, t2, F, tc1, tc2, tc3, F_cphase, cphase_proj, sigma = Prepare_EHT_Data(fov_param, flux_label, blur_param, sefd_param, eht_array, target)
		else:
			_, _, t1, t2, F, tc1, tc2, tc3, F_cphase, cphase_proj, _ = Prepare_EHT_Data(fov_param, flux_label, blur_param, sefd_param, eht_array, target)
		t1_list.append(t1)
		t2_list.append(t2)
		F_list.append(F)
		tc1_list.append(tc1)
		tc2_list.append(tc2)
		tc3_list.append(tc3)
		F_cphase_list.append(F_cphase)
		cphase_proj_list.append(cphase_proj)

	n_sites = np.unique(np.concatenate([t1, t2])).shape[0] + 1
	###############################################################################
	# define the model
	###############################################################################
	# nsamp = 10000
	
	# model = IsingVisNet(t1, t2, F, n_ising_layers=n_ising_layers, slope_const=3, sigma=sigma)
	model = IsingMutipleVisNet(t1_list, t2_list, F_list, n_ising_layers=5, slope_const=1e2, sigma=sigma)
	
	adam_opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

	recon_weight = 1.0

	loss_dict = {'sampling': site_sparsity, 'ising': energy}
	for k in range(len(t1_list)):
		loss_dict['recon'+str(k)] = 'mean_absolute_error'

	loss_weights_dict = {'sampling': sample_weight, 'ising': ising_weight}
	for k in range(len(t1_list)):
		loss_weights_dict['recon'+str(k)] = recon_weight / len(t1_list)

	# model.compile(optimizer=adam_opt, loss={'recon': 'mean_absolute_error', 'sampling': site_sparsity, 'ising': energy}, 
	# 			loss_weights={'recon': recon_weight, 'sampling': sample_weight, 'ising': ising_weight})
	model.compile(optimizer=adam_opt, loss=loss_dict, loss_weights=loss_weights_dict)

	# modelname = os.path.join(models_dir, savefile_name+'best.h5')

	checkpoint = ModelCheckpoint(models_dir+savefile_name+'best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

	class Save_weights(keras.callbacks.Callback):
		def on_epoch_begin(self, epoch, logs=None):
			model.save_weights(os.path.join(models_dir, savefile_name+'weights{:02d}.hdf5'.format(epoch)))

	fitdata_dict = {'sampling': np.zeros((nsamp, n_sites)), 'ising': np.zeros((nsamp, ))}
	for k in range(len(t1_list)):
		fitdata_dict['recon'+str(k)] = xdata_blur

	history = model.fit({'input': xdata}, fitdata_dict,
	                validation_split=0.3,
	                initial_epoch=1,
	                epochs=1 + nb_epochs_train,
	                batch_size=batch_size,
	                verbose=1,
	                callbacks=[checkpoint])#,


	modelname = os.path.join(models_dir, savefile_name+'.h5')
	model.save_weights(modelname)
	w = csv.writer(open(os.path.join(models_dir, 'history_'+savefile_name+'.csv'), 'w'))
	for key, val in history.history.items():
		w.writerow([key, val])

	del history, checkpoint
	del xdata, xdata_blur, t1, t2, F, tc1, tc2, tc3, F_cphase, cphase_proj, sigma

	return model



def Train_IsingMutipleCpAmpNet(eht_array, target_list, fov_param, flux_label, blur_param, sefd_param, lr, nb_epochs_train, sample_weight, ising_weight, batch_size = 32, n_ising_layers = 5, models_dir='../joint_opt/models/anti-aliasing/12302019/', savefile_name='nn_params'):
	
	###############################################################################
	# prepare the training data
	###############################################################################
	t1_list = []
	t2_list = []
	F_list = []
	tc1_list = []
	tc2_list = []
	tc3_list = []
	F_cphase_list = []
	cphase_proj_list = []

	for k in range(len(target_list)):
		target = target_list[k]
		if k == 0:
			xdata, xdata_blur, t1, t2, F, tc1, tc2, tc3, F_cphase, cphase_proj, sigma = Prepare_EHT_Data(fov_param, flux_label, blur_param, sefd_param, eht_array, target, data_augmentation=True)
		else:
			_, _, t1, t2, F, tc1, tc2, tc3, F_cphase, cphase_proj, _ = Prepare_EHT_Data(fov_param, flux_label, blur_param, sefd_param, eht_array, target, data_augmentation=True)
		t1_list.append(t1)
		t2_list.append(t2)
		F_list.append(F)
		tc1_list.append(tc1)
		tc2_list.append(tc2)
		tc3_list.append(tc3)
		F_cphase_list.append(F_cphase)
		cphase_proj_list.append(cphase_proj)

	n_sites = np.unique(np.concatenate([t1, t2])).shape[0] + 1
	###############################################################################
	# define the model
	###############################################################################
	# nsamp = 10000
	
	model = IsingMutipleCpAmpNet(t1_list, t2_list, tc1_list, tc2_list, tc3_list, F_list, F_cphase_list, cphase_proj_list, n_ising_layers=n_ising_layers, slope_const=3, sigma=sigma)
	
	adam_opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

	recon_weight = 1.0

	loss_dict = {'sampling': site_sparsity, 'ising': energy}
	for k in range(len(t1_list)):
		loss_dict['recon'+str(k)] = Lambda_similarity

	loss_weights_dict = {'sampling': sample_weight, 'ising': ising_weight}
	for k in range(len(t1_list)):
		loss_weights_dict['recon'+str(k)] = recon_weight / len(t1_list)

	model.compile(optimizer=adam_opt, loss=loss_dict, loss_weights=loss_weights_dict)

	# model.compile(optimizer=adam_opt, loss={'recon': Lambda_similarity, 'sampling': site_sparsity, 'ising': energy}, 
	# 			loss_weights={'recon': recon_weight, 'sampling': sample_weight, 'ising': ising_weight})

	# modelname = os.path.join(models_dir, savefile_name+'best.h5')

	checkpoint = ModelCheckpoint(models_dir+savefile_name+'best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

	class Save_weights(keras.callbacks.Callback):
		def on_epoch_begin(self, epoch, logs=None):
			model.save_weights(os.path.join(models_dir, savefile_name+'weights{:02d}.hdf5'.format(epoch)))

	
	fitdata_dict = {'sampling': np.zeros((nsamp, n_sites)), 'ising': np.zeros((nsamp, ))}
	for k in range(len(t1_list)):
		fitdata_dict['recon'+str(k)] = xdata_blur

	history = model.fit({'input': xdata}, fitdata_dict,
	                validation_split=0.3,
	                initial_epoch=1,
	                epochs=1 + nb_epochs_train,
	                batch_size=batch_size,
	                verbose=1,
	                callbacks=[checkpoint])


	modelname = os.path.join(models_dir, savefile_name+'.h5')
	model.save_weights(modelname)
	w = csv.writer(open(os.path.join(models_dir, 'history_'+savefile_name+'.csv'), 'w'))
	for key, val in history.history.items():
		w.writerow([key, val])

	del history

	return model

def Prepare_BlackHole_Feature_Data(sefd_param = 0, target = 'm87'):
	###############################################################################
	# prepare the training data with features
	###############################################################################
	bh_sim_data = np.load('/home/groot/BoumanLab/joint_opt/data/bh_sim_data.npy', allow_pickle=True)
	image = bh_sim_data.item()['image']
	spin_label = bh_sim_data.item()['spin']
	class_label = bh_sim_data.item()['type']

	nsamp = image.shape[0]

	combined_class_label = [(class_label[k], spin_label[k]) for k in range(nsamp)]
	combined_class_unique = np.unique(combined_class_label, axis=0)
	n_combined_class = len(combined_class_unique)

	combined_class_array = np.zeros((nsamp, n_combined_class))

	for k in range(nsamp):
		for i in range(n_combined_class):
			if class_label[k] == combined_class_unique[i][0] and spin_label[k] == combined_class_unique[i][1]:
				combined_class_array[k, i] = 1.0
			else:
				combined_class_array[k, i] = 0.0

	spin_unique = np.unique(spin_label)
	spin_array = np.zeros((nsamp, len(spin_unique)))
	for k in range(nsamp):
		for i in range(len(spin_unique)):
			if spin_label[k] == spin_unique[i]:
				spin_array[k, i] = 1.0
			else:
				spin_array[k, i] = 0.0

	class_unique = np.unique(class_label)
	class_array = np.zeros((nsamp, len(class_unique)))
	for k in range(nsamp):
		for i in range(len(class_unique)):
			if class_label[k] == class_unique[i]:
				class_array[k, i] = 1.0
			else:
				class_array[k, i] = 0.0

	shuffle_order = np.random.RandomState(seed=42).permutation(nsamp)
	image_shuffle = image[shuffle_order]
	# spin_shuffle = spin_array[shuffle_order]
	# class_shuffle = class_array[shuffle_order]
	combined_class_shuffle = combined_class_array[shuffle_order]
	class_shuffle = class_array[shuffle_order]
	spin_shuffle = spin_array[shuffle_order]

	###############################################################################
	# generate eht observation data
	###############################################################################
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

	array = '/home/groot/BoumanLab/eht-imaging/arrays/EHT2019.txt'
	eht = eh.array.load_txt(array)
	fov = 160 * eh.RADPERUAS

	###############################################################################
	# define scientific target
	###############################################################################
	if target == 'm87':
		ra = 12.513728717168174
		dec = 12.39112323919932
	elif target == 'sgrA':
		ra = 19.414182210498385
		dec = -29.24170032236311
	# ra = 12.513728717168174
	# dec = 12.39112323919932

	rf = 230e9
	npix = 160
	mjd = 57853 # day of observation
	simim = eh.image.make_empty(npix, fov, ra, dec, rf=rf, source='random', mjd=mjd)
	simim.imvec = np.zeros((npix, npix, 1)).reshape((-1, 1))#xdata[0, :, :, :].reshape((-1, 1))
	obs = simim.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, add_th_noise=add_th_noise, ampcal=ampcal, phasecal=phasecal, 
				    stabilize_scan_phase=stabilize_scan_phase, stabilize_scan_amp=stabilize_scan_amp,
				    jones=jones,inv_jones=inv_jones,dcal=dcal, frcal=frcal, dterm_offset=dterm_offset)

	
	t1 = obs.data['t1']
	t2 = obs.data['t2']
	n_sites = np.unique(np.concatenate([t1, t2])).shape[0] + 1

	obs_data = obs.unpack(['u', 'v', 'vis', 'sigma'])
	uv = np.hstack((obs_data['u'].reshape(-1,1), obs_data['v'].reshape(-1,1)))
	F = ftmatrix(simim.psize, simim.xdim, simim.ydim, uv, pulse=simim.pulse)

	# complex visibilities
	vis_data = np.matmul(image_shuffle, F.T)


	# obs.add_cphase(count='min')
	obs.add_cphase(count='max')
	tc1 = obs.cphase['t1']
	tc2 = obs.cphase['t2']
	tc3 = obs.cphase['t3']

	cphase_map = np.zeros((len(obs.cphase['time']), 3))

	zero_symbol = 10000
	for k1 in range(cphase_map.shape[0]):
		for k2 in list(np.where(obs.data['time']==obs.cphase['time'][k1])[0]):
			if obs.data['t1'][k2] == obs.cphase['t1'][k1] and obs.data['t2'][k2] == obs.cphase['t2'][k1]:
				cphase_map[k1, 0] = k2
				if k2 == 0:
					cphase_map[k1, 0] = zero_symbol
			elif obs.data['t2'][k2] == obs.cphase['t1'][k1] and obs.data['t1'][k2] == obs.cphase['t2'][k1]:
				cphase_map[k1, 0] = -k2
				if k2 == 0:
					cphase_map[k1, 0] = -zero_symbol
			elif obs.data['t1'][k2] == obs.cphase['t2'][k1] and obs.data['t2'][k2] == obs.cphase['t3'][k1]:
				cphase_map[k1, 1] = k2
				if k2 == 0:
					cphase_map[k1, 1] = zero_symbol
			elif obs.data['t2'][k2] == obs.cphase['t2'][k1] and obs.data['t1'][k2] == obs.cphase['t3'][k1]:
				cphase_map[k1, 1] = -k2
				if k2 == 0:
					cphase_map[k1, 1] = -zero_symbol
			elif obs.data['t1'][k2] == obs.cphase['t3'][k1] and obs.data['t2'][k2] == obs.cphase['t1'][k1]:
				cphase_map[k1, 2] = k2
				if k2 == 0:
					cphase_map[k1, 2] = zero_symbol
			elif obs.data['t2'][k2] == obs.cphase['t3'][k1] and obs.data['t1'][k2] == obs.cphase['t1'][k1]:
				cphase_map[k1, 2] = -k2
				if k2 == 0:
					cphase_map[k1, 2] = -zero_symbol


	cphase_proj = np.zeros((cphase_map.shape[0], F.shape[0]), dtype=np.float32)
	for k in range(cphase_map.shape[0]):
		for j in range(cphase_map.shape[1]):
			if cphase_map[k][j] > 0:
				if int(cphase_map[k][j]) == zero_symbol:
					cphase_map[k][j] = 0
				cphase_proj[k, int(cphase_map[k][j])] = 1
			else:
				if np.abs(int(cphase_map[k][j])) == zero_symbol:
					cphase_map[k][j] = 0
				cphase_proj[k, int(-cphase_map[k][j])] = -1
	###############################################################################
	# thermal noises: 0 represents no thermal noises, 1 represents site-varying thermal noises, 2 represents site-equivalent thermal noises
	###############################################################################
	if sefd_param == 1:
		sigma = 224.46 * np.concatenate([np.expand_dims(obs.data['sigma'], -1), np.expand_dims(obs.data['sigma'], -1)], -1)
	elif sefd_param == 2:
		sigma = 224.46 * np.concatenate([np.expand_dims(obs.data['sigma'], -1), np.expand_dims(obs.data['sigma'], -1)], -1)
		sigma = np.mean(sigma.reshape((-1, ))) * np.ones(sigma.shape)
	else:
		sigma = None

	# return vis_data, spin_shuffle, class_shuffle, t1, t2, sigma
	return vis_data, combined_class_shuffle, class_shuffle, spin_shuffle, t1, t2, tc1, tc2, tc3, cphase_proj, sigma, combined_class_unique, class_unique, spin_unique

def Train_IsingFeatureNet(eht_array, target, sefd_param, lr, nb_epochs_train, sample_weight, ising_weight, batch_size = 32, n_ising_layers = 5, models_dir='../joint_opt/models/anti-aliasing/12302019/', savefile_name='nn_params', network='vis', feature_name='both'):
	
	###############################################################################
	# prepare the training data
	###############################################################################
	fov_param = 160
	flux_label = 1
	blur_param = 1	
	# vis_shuffle, spin_shuffle, class_shuffle, t1, t2, sigma = Prepare_BlackHole_Feature_Data(sefd_param)
	vis_shuffle, combined_class_shuffle, class_shuffle, spin_shuffle, t1, t2, tc1, tc2, tc3, cphase_proj, sigma, combined_class_unique, class_unique, spin_unique = Prepare_BlackHole_Feature_Data(sefd_param, target)
	n_sites = np.unique(np.concatenate([t1, t2])).shape[0] + 1

	nsamp = vis_shuffle.shape[0]


	###############################################################################
	# define the model
	###############################################################################
	if network ==  'vis':
		model = IsingVisFeatureNet(t1, t2, n_ising_layers=n_ising_layers, slope_const=3, n_layers=3, n_hid=200, sigma=sigma, feature_name=feature_name)
	elif network == 'cpamp':
		model = IsingCpAmpFeatureNet(t1, t2, tc1, tc2, tc3, cphase_proj, n_ising_layers=n_ising_layers, slope_const=3, n_layers=3, n_hid=200, sigma=sigma, feature_name=feature_name)

	adam_opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

	class_weight = 1.0
	# model.compile(optimizer=adam_opt, loss={'bh_class': 'categorical_crossentropy', 'spin_class': 'categorical_crossentropy', 'sampling': site_sparsity, 'ising': energy}, 
	# 			loss_weights={'bh_class': class_weight, 'spin_class': class_weight, 'sampling': sample_weight, 'ising': ising_weight})
	if feature_name == 'class':
		model.compile(optimizer=adam_opt, loss={'bh_class': 'categorical_crossentropy', 'sampling': site_sparsity, 'ising': energy}, 
					loss_weights={'bh_class': class_weight, 'sampling': sample_weight, 'ising': ising_weight})
	elif feature_name == 'spin':
		model.compile(optimizer=adam_opt, loss={'spin_class': 'categorical_crossentropy', 'sampling': site_sparsity, 'ising': energy}, 
					loss_weights={'spin_class': class_weight, 'sampling': sample_weight, 'ising': ising_weight})
	elif feature_name == 'both':
		model.compile(optimizer=adam_opt, loss={'combined_class': 'categorical_crossentropy', 'sampling': site_sparsity, 'ising': energy}, 
					loss_weights={'combined_class': class_weight, 'sampling': sample_weight, 'ising': ising_weight})

	# modelname = os.path.join(models_dir, savefile_name+'best.h5')

	checkpoint = ModelCheckpoint(models_dir+savefile_name+'best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

	class Save_weights(keras.callbacks.Callback):
		def on_epoch_begin(self, epoch, logs=None):
			model.save_weights(os.path.join(models_dir, savefile_name+'weights{:02d}.hdf5'.format(epoch)))

	# history = model.fit({'input': vis_shuffle}, {'bh_class': class_shuffle, 'spin_class': spin_shuffle, 'sampling': np.zeros((nsamp, n_sites)), 'ising': np.zeros((nsamp, ))},
	#                 validation_split=0.3,
	#                 initial_epoch=1,
	#                 epochs=1 + nb_epochs_train,
	#                 batch_size=batch_size,
	#                 verbose=1,
	#                 callbacks=[checkpoint])

	if feature_name == 'class':
		history = model.fit({'input': vis_shuffle}, {'bh_class': class_shuffle, 'sampling': np.zeros((nsamp, n_sites)), 'ising': np.zeros((nsamp, ))},
		                validation_split=0.3,
		                initial_epoch=1,
		                epochs=1 + nb_epochs_train,
		                batch_size=batch_size,
		                verbose=1,
		                callbacks=[checkpoint])
	elif feature_name == 'spin':
		history = model.fit({'input': vis_shuffle}, {'spin_class': spin_shuffle, 'sampling': np.zeros((nsamp, n_sites)), 'ising': np.zeros((nsamp, ))},
		                validation_split=0.3,
		                initial_epoch=1,
		                epochs=1 + nb_epochs_train,
		                batch_size=batch_size,
		                verbose=1,
		                callbacks=[checkpoint])
	elif feature_name == 'both':
		history = model.fit({'input': vis_shuffle}, {'combined_class': combined_class_shuffle, 'sampling': np.zeros((nsamp, n_sites)), 'ising': np.zeros((nsamp, ))},
		                validation_split=0.3,
		                initial_epoch=1,
		                epochs=1 + nb_epochs_train,
		                batch_size=batch_size,
		                verbose=1,
		                callbacks=[checkpoint])


	modelname = os.path.join(models_dir, savefile_name+'.h5')
	model.save_weights(modelname)
	w = csv.writer(open(os.path.join(models_dir, 'history_'+savefile_name+'.csv'), 'w'))
	for key, val in history.history.items():
		w.writerow([key, val])

	del history

	return model


if __name__ == '__main__':	
	###############################################################################
	# record the command line inputs
	###############################################################################

	script = str(sys.argv[0])
	eht_array = str(sys.argv[1])
	target = str(sys.argv[2])
	lr = float(sys.argv[3])
	nb_epochs_train = int(sys.argv[4])
	sample_weight = float(sys.argv[5])
	ising_weight = float(sys.argv[6])
	blur_param = float(sys.argv[7])
	fov_param = float(sys.argv[8])
	sefd_param = int(sys.argv[9])
	flux_label = int(sys.argv[10])
	file_index = sys.argv[11]
	weather = str(sys.argv[12])

	if weather == 'True':
		weather = True
	else:
		weather = False

	# eht_array = 'eht2019'
	# target = 'm87'#'sgrA'#'both'#
	# lr = 0.001#0.001
	# nb_epochs_train = 50
	# sample_weight = 0.05
	# ising_weight = 0.0001
	# blur_param = 0.25#0.75
	# fov_param = 100.0
	# sefd_param = 0
	# flux_label = 1
	# file_index = 'cpamp1'#'featurecpamp1'#'featurevis1'
	# weather = False

	savefile_name = eht_array+'_'+target+'_'+file_index+'_sample'+str(sample_weight)+'_ising'+str(ising_weight)+'_blur'+str(blur_param)+'_fov'+str(fov_param)+'_sefd'+str(sefd_param)+'_flux'+str(flux_label)+'_weather'+str(weather)

	models_dir = './data/20200606/'#'../joint_opt/models/anti-aliasing/02012020/'
	###############################################################################
	# complex visibility
	###############################################################################
	if file_index[0:3] == 'vis':
		if target == 'both':
			target_list = ['sgrA', 'm87']
			model = Train_IsingMutipleVisNet(eht_array, target_list, fov_param, flux_label, blur_param, sefd_param, lr, nb_epochs_train, sample_weight, ising_weight, batch_size = 32, n_ising_layers = 5, models_dir=models_dir, savefile_name=savefile_name)
		else:
			model = Train_IsingVisNet(eht_array, target, fov_param, flux_label, blur_param, sefd_param, lr, nb_epochs_train, sample_weight, ising_weight, batch_size = 64, n_ising_layers = 5, models_dir=models_dir, savefile_name=savefile_name, weather=weather)
			# model = Train_IsingVisNet(eht_array, target, fov_param, flux_label, blur_param, sefd_param, lr, nb_epochs_train, sample_weight, ising_weight, batch_size = 32, n_ising_layers = 10, models_dir=models_dir, savefile_name=savefile_name, weather=weather)
	###############################################################################
	# closure phase and amplitude
	###############################################################################
	if file_index[0:5] == 'cpamp':
		if target == 'both':
			target_list = ['sgrA', 'm87']
			model = Train_IsingMutipleCpAmpNet(eht_array, target_list, fov_param, flux_label, blur_param, sefd_param, lr, nb_epochs_train, sample_weight, ising_weight, batch_size = 32, n_ising_layers = 5, models_dir=models_dir, savefile_name=savefile_name)
		else:
			model = Train_IsingCpAmpNet(eht_array, target, fov_param, flux_label, blur_param, sefd_param, lr, nb_epochs_train, sample_weight, ising_weight, batch_size = 64, n_ising_layers = 5, models_dir=models_dir, savefile_name=savefile_name, weather=weather)
			# model = Train_IsingCpAmpNet(eht_array, target, fov_param, flux_label, blur_param, sefd_param, lr, nb_epochs_train, sample_weight, ising_weight, batch_size = 32, n_ising_layers = 5, models_dir=models_dir, savefile_name=savefile_name, weather=weather)


	###############################################################################
	# complex visibility feature
	###############################################################################
	if file_index[0:10] == 'featurevis':
		model = Train_IsingFeatureNet(eht_array, target, sefd_param, lr, nb_epochs_train, sample_weight, ising_weight, batch_size = 32, n_ising_layers = 5, models_dir=models_dir, savefile_name=savefile_name, network='vis', feature_name=file_index[10:-1])

	###############################################################################
	# closure phase and amplitude feature
	###############################################################################
	if file_index[0:12] == 'featurecpamp':
		model = Train_IsingFeatureNet(eht_array, target, sefd_param, lr, nb_epochs_train, sample_weight, ising_weight, batch_size = 32, n_ising_layers = 5, models_dir=models_dir, savefile_name=savefile_name, network='cpamp', feature_name=file_index[12:-1])


	del model
	for k in range(20):
		gc.collect()


	# eht_array = 'eht2019'
	# target = 'm87'#'both'#
	# lr = 0.001#0.001
	# nb_epochs_train = 200
	# sample_weight = 0.2#0.005
	# ising_weight = 0.2#0.005
	# blur_param = 0.75
	# fov_param = 100.0
	# sefd_param = 0
	# flux_label = 1
	# file_index = 'featurevis1'#'featurecpamp2'#

	# savefile_name = eht_array+'_'+target+'_'+file_index+'_sample'+str(sample_weight)+'_ising'+str(ising_weight)+'_blur'+str(blur_param)+'_fov'+str(fov_param)+'_sefd'+str(sefd_param)+'_flux'+str(flux_label)

	# model = IsingCpAmpFeatureNet(t1, t2, tc1, tc2, tc3, cphase_proj, n_ising_layers=5, slope_const=3, n_layers=3, n_hid=200, sigma=sigma)
	# model = IsingVisFeatureNet(t1, t2, n_ising_layers=3, slope_const=3, n_layers=3, n_hid=200, sigma=sigma)
	# models_dir = '../joint_opt/models/anti-aliasing/01032020/'#'../models/anti-aliasing/11212019/'#'../models/anti-aliasing/11202019/'#'../models/anti-aliasing/11162019/'#'../models/anti-aliasing/11142019/'
	# modelname = os.path.join(models_dir, savefile_name+'best'+'.hdf5')
	# model.load_weights(modelname)
	# Q_vec = model.get_layer('ising').get_weights()[0]                                                                                                                
	# delta = model.get_layer('ising').get_weights()[1] 
	# n_sites = len(delta)
	# Q = np.zeros((n_sites, n_sites))
	# k = 0
	# for k1 in range(n_sites):
	# 	for k2 in range(k1):
	# 		Q[k1, k2] = Q_vec[k]
	# 		k += 1
	# Q = Q + Q.T

	# tlib1 = {'PV': 0, 'PDB': 1, 'ALMA': 2, 'APEX': 3, 'LMT': 4, 'SMT': 5, 'SPT': 6, 'OVRO': 7, 'JCMT': 8, 'SMA': 9, 'KP': 10, 'GLT': 11}
	# tlib2 = {'ALMA': 0, 'APEX': 1, 'PDB': 2, 'PV': 3, 'LMT': 4, 'SMT': 5, 'KP': 6, 'OVRO': 7, 'JCMT': 8, 'SMA': 9, 'GLT': 10, 'SPT': 11}

	# tlib_list = list(tlib1)
	# tlib2_order = []
	# for k in range(len(tlib_list)):
	# 	tlib2_order.append(tlib2[tlib_list[k]])

	# if target == 'm87':
	# 	Q_new = np.array(Q)
	# 	delta_new = np.array(delta)
	# 	for j1 in range(n_sites):
	# 		delta_new[j1] = delta[tlib2_order[j1]]
	# 		for j2 in range(n_sites):
	# 			Q_new[j1, j2] = Q[tlib2_order[j1], tlib2_order[j2]]
	# 	Q = Q_new
	# 	delta = delta_new

	# plt.figure(), 
	# plt.plot(np.arange(12), delta, 'ro')
	# plt.ylim([-0.1, 0.5])
	# plt.title(r'Ising model parameter ($\theta_{jj}$)', fontsize=18)
	# plt.xticks(range(len(tlib_list)), list(tlib_list), size='small', rotation=35, fontsize=12)
	# plt.yticks(fontsize=14)
	# # plt.savefig(os.path.join(data_dir, 'delta2_'+savefile_name+'.pdf'))


	# plt.figure(), plt.imshow(Q), plt.set_cmap('seismic')
	# cb = plt.colorbar()
	# cb.ax.tick_params(labelsize=14)
	# plt.title(r'Ising model parameter ($\theta_{jk}$)', fontsize=18)
	# plt.clim(-0.6, 0.6)
	# plt.xticks(range(len(tlib_list)), list(tlib_list), size='small', rotation=35, fontsize=12)
	# plt.yticks(range(len(tlib_list)), list(tlib_list), size='small', fontsize=12)
	# plt.ylim(-0.5, n_sites-0.5)


	# combined_class_prob, mask_pred, energy_pred = model.predict(np.concatenate([vis_shuffle[-900::]]*5, 0))

	# class_true = np.argmax(np.concatenate([combined_class_shuffle[-900::]]*5, 0), 1)
	# class_pred = np.argmax(np.concatenate([combined_class_prob[-900::]]*5, 0), 1)

	# val_mask_pred = mask_pred
	# class_accuracy = np.sum(class_pred == class_true)/len(class_pred)
	# print('Prediction Accuracy {}'.format(class_accuracy))

	# val_true_index = np.where(class_pred == class_true)[0]
	# val_false_index = np.where(class_pred != class_true)[0]

	# mask_true_pred = np.sum(mask_pred, 1)[val_true_index]
	# mask_false_pred = np.sum(mask_pred, 1)[val_false_index]

	# from sklearn.metrics import confusion_matrix

	# class_label = [combined_class_unique[k][0]+combined_class_unique[k][1] for k in range(len(combined_class_unique))]
	# cmat = confusion_matrix(class_true, class_pred)

	# cmat_norm = np.array(1.0*cmat)
	# for k in range(cmat.shape[0]):
	# 	cmat_norm[k] = 1.0 * cmat[k] / np.sum(cmat[k])

	# plt.figure(), plt.imshow(cmat_norm), plt.set_cmap('jet')
	# cb = plt.colorbar()
	# cb.ax.tick_params(labelsize=14)
	# plt.title(r'Confusion Matrix', fontsize=18)
	# plt.clim(0, 1)
	# plt.xticks(range(len(class_label)), class_label, size='small', rotation=35, fontsize=12)
	# plt.yticks(range(len(class_label)), class_label, size='small', fontsize=12)
	# plt.xlabel('Predicted Label')
	# plt.ylabel('True Label')
	# plt.ylim(-0.5, cmat.shape[0]-0.5)


	# plt.figure(), plt.imshow(cmat), plt.set_cmap('jet')
	# cb = plt.colorbar()
	# cb.ax.tick_params(labelsize=14)
	# plt.title(r'Confusion Matrix', fontsize=18)
	# # plt.clim(0, 1)
	# plt.xticks(range(len(class_label)), class_label, size='small', rotation=35, fontsize=12)
	# plt.yticks(range(len(class_label)), class_label, size='small', fontsize=12)
	# plt.xlabel('Predicted Label')
	# plt.ylabel('True Label')
	# plt.ylim(-0.5, cmat.shape[0]-0.5)


	# train_class_true = np.argmax(combined_class_shuffle[0:-900], 1)
	# train_class_count = [np.sum(train_class_true==k) for k in range(13)]
	# plt.figure(), plt.plot(train_class_count)



	# Q_vec = model.get_layer('ising').get_weights()[0]                                                                                                                
	# delta = model.get_layer('ising').get_weights()[1] 
	# n_sites = len(delta)
	# Q = np.zeros((n_sites, n_sites))
	# k = 0
	# for k1 in range(n_sites):
	# 	for k2 in range(k1):
	# 		Q[k1, k2] = Q_vec[k]
	# 		k += 1
	# Q = Q + Q.T

	# tlib1 = {'PV': 0, 'PDB': 1, 'ALMA': 2, 'APEX': 3, 'LMT': 4, 'SMT': 5, 'SPT': 6, 'JCMT': 7, 'SMA': 8, 'KP': 9, 'GLT': 10}
	# tlib2 = {'ALMA': 0, 'APEX': 1, 'PDB': 2, 'PV': 3, 'LMT': 4, 'SMT': 5, 'KP': 6, 'JCMT': 7, 'SMA': 8, 'GLT': 9, 'SPT': 10}

	# tlib_list = list(tlib1)
	# tlib2_order = []
	# for k in range(len(tlib_list)):
	# 	tlib2_order.append(tlib2[tlib_list[k]])

	# if target == 'm87':
	# 	Q_new = np.array(Q)
	# 	delta_new = np.array(delta)
	# 	for j1 in range(n_sites):
	# 		delta_new[j1] = delta[tlib2_order[j1]]
	# 		for j2 in range(n_sites):
	# 			Q_new[j1, j2] = Q[tlib2_order[j1], tlib2_order[j2]]
	# 	Q = Q_new
	# 	delta = delta_new

	# plt.figure(), 
	# plt.plot(np.arange(11), delta, 'ro')
	# plt.ylim([-0.1, 0.5])
	# plt.title(r'Ising model parameter ($\theta_{jj}$)', fontsize=18)
	# plt.xticks(range(len(tlib_list)), list(tlib_list), size='small', rotation=35, fontsize=12)
	# plt.yticks(fontsize=14)
	# # plt.savefig(os.path.join(data_dir, 'delta2_'+savefile_name+'.pdf'))


	# plt.figure(), plt.imshow(Q), plt.set_cmap('seismic')
	# cb = plt.colorbar()
	# cb.ax.tick_params(labelsize=14)
	# plt.title(r'Ising model parameter ($\theta_{jk}$)', fontsize=18)
	# plt.clim(-0.6, 0.6)
	# plt.xticks(range(len(tlib_list)), list(tlib_list), size='small', rotation=35, fontsize=12)
	# plt.yticks(range(len(tlib_list)), list(tlib_list), size='small', fontsize=12)
	# plt.ylim(-0.5, n_sites-0.5)

	# sample_mean_list = np.zeros((5, 11))
	# correlation_list = np.zeros((5, 11, 11))
	# for k in range(5):
	# 	print('cpamp'+str(k+1))
	# 	file_index = 'cpamp'+str(k+1)
	# 	savefile_name = eht_array+'_'+target+'_'+file_index+'_sample'+str(sample_weight)+'_ising'+str(ising_weight)+'_blur'+str(blur_param)+'_fov'+str(fov_param)+'_sefd'+str(sefd_param)+'_flux'+str(flux_label)+'_weather'+str(weather)
	# 	modelname = os.path.join(models_dir, savefile_name+'.h5')
	# 	model.load_weights(modelname)

	# 	# # recon, sample, logprob, _, _ = model.predict(np.concatenate([xdata]*20, 0))
	# 	# recon, sample, logprob, _, _ = model.predict(xdata)
	# 	# sample_mean = np.mean(sample, 0)
	# 	# sample_c = sample - sample_mean

	# 	# cov = np.matmul(sample_c.T, sample_c) / sample.shape[0]
	# 	# var = np.diag(cov) 
	# 	# std = np.sqrt(var) 
	# 	# correlation = cov / np.outer(std, std)
	# 	# correlation[np.arange(sample.shape[1]), np.arange(sample.shape[1])] = 0

	# 	Q_vec = model.get_layer('ising').get_weights()[0]                                                                                                                
	# 	delta = model.get_layer('ising').get_weights()[1] 
	# 	n_sites = len(delta)
	# 	Q = np.zeros((n_sites, n_sites))
	# 	j = 0
	# 	for k1 in range(n_sites):
	# 		for k2 in range(k1):
	# 			Q[k1, k2] = Q_vec[j]
	# 			j += 1
	# 	Q = Q + Q.T

	# 	sample_mean = delta
	# 	correlation = Q

	# 	if target == 'm87':
	# 		correlation_new = np.array(correlation)
	# 		sample_mean_new = np.array(sample_mean)
	# 		for j1 in range(n_sites):
	# 			sample_mean_new[j1] = sample_mean[tlib2_order[j1]]
	# 			for j2 in range(n_sites):
	# 				correlation_new[j1, j2] = correlation[tlib2_order[j1], tlib2_order[j2]]
	# 		correlation = correlation_new
	# 		sample_mean = sample_mean_new

	# 	sample_mean_list[k] = sample_mean
	# 	correlation_list[k] = correlation

	# 	del recon, sample, logprob

	# delta_mean = np.mean(sample_mean_list, 0)
	# delta_std = np.std(sample_mean_list, 0)


	# plt.figure(), 
	# (_, caps, _) = plt.errorbar(np.arange(11), delta_mean, delta_std, fmt='ro', ecolor='r', capthick=2, elinewidth=2, barsabove=True, lolims=True, uplims=True)
	# caps[0].set_marker('_')
	# caps[1].set_marker('_')
	# plt.ylim([0.0, 1.0])
	# plt.title(r'Sampling activity', fontsize=18)
	# plt.xticks(range(len(tlib_list)), list(tlib_list), size='small', rotation=35, fontsize=12)
	# plt.yticks(fontsize=14)




	# recon, sample, logprob = model.predict(np.concatenate([xdata]*20, 0))
	# sample_mean = np.mean(sample, 0)
	# sample_c = sample - sample_mean

	# cov = np.matmul(sample_c.T, sample_c) / sample.shape[0]
	# var = np.diag(cov) 
	# std = np.sqrt(var) 
	# correlation = cov / np.outer(std, std)
	# correlation[np.arange(sample.shape[1]), np.arange(sample.shape[1])] = 0


	# if target == 'm87':
	# 	correlation_new = np.array(correlation)
	# 	sample_mean_new = np.array(sample_mean)
	# 	for j1 in range(n_sites):
	# 		sample_mean_new[j1] = sample_mean[tlib2_order[j1]]
	# 		for j2 in range(n_sites):
	# 			correlation_new[j1, j2] = correlation[tlib2_order[j1], tlib2_order[j2]]
	# 	correlation = correlation_new
	# 	sample_mean = sample_mean_new


	# plt.figure(), 
	# plt.plot(np.arange(11), sample_mean, 'ro')
	# plt.ylim([0.0, 1.0])
	# plt.title(r'Sampling activity', fontsize=18)
	# plt.xticks(range(len(tlib_list)), list(tlib_list), size='small', rotation=35, fontsize=12)
	# plt.yticks(fontsize=14)
	# # plt.savefig(os.path.join(data_dir, 'delta2_'+savefile_name+'.pdf'))


	# plt.figure(), plt.imshow(correlation), plt.set_cmap('seismic')
	# cb = plt.colorbar()
	# cb.ax.tick_params(labelsize=14)
	# plt.title(r'Sampling correlation', fontsize=18)
	# plt.clim(-0.3, 0.3)
	# plt.xticks(range(len(tlib_list)), list(tlib_list), size='small', rotation=35, fontsize=12)
	# plt.yticks(range(len(tlib_list)), list(tlib_list), size='small', fontsize=12)
	# plt.ylim(-0.5, n_sites-0.5)


	# history_name = os.path.join(models_dir, 'history_'+savefile_name+'.csv')
	# with open(history_name) as csvfile:
	# 	train_history = csv.reader(csvfile)
	# 	loss_history = []
	# 	for row in train_history:
	# 		loss_history.append(row)
	# loss = [np.float(num) for num in loss_history[4][1][1:-1].split(', ')]
	# val_loss = [np.float(num) for num in loss_history[0][1][1:-1].split(', ')]

	# plt.figure(), plt.plot(loss)
	# plt.plot(val_loss)
	# plt.ylim([0.02, 0.1])
	# plt.legend(['loss', 'val_loss'])