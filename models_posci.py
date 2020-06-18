import os
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomUniform, Constant
import keras.models
import keras.layers
import keras.initializers
import keras.regularizers
import keras.callbacks
from keras import backend as K
from keras import losses

import helpers_posci as hp
from layers_posci import _unet_from_tensor, Ising_sampling, Ising_sampling2, Site_mask_prob
from layers_posci import realnvp_encoder, realnvp_decoder, Lambda_Gaussian, Lambda_logdet_sigmoid, STE_layer, Gaussian_sampling

########################################################################################
# joint sensing and imaging network with complex visibility
########################################################################################
def IsingVisNet(t1, t2, F, n_ising_layers=5, slope_const=1e2, sigma=None, binary_slope=10, obs_prob=None):
	filt = 64
	kern = 3
	acti = None

	tlib = {}                                                              
	tind = 0
	telescopes = np.concatenate([t1, t2])                                                               
	for k in range(len(telescopes)):
		if telescopes[k] not in tlib:
			tlib[telescopes[k]] = tind 
			tind += 1

	if 'SPT'  not in telescopes:
		tlib['SPT'] = tind
		tind += 1
	elif 'GLT'  not in telescopes:
		tlib['GLT'] = tind
		tind += 1

	n_sites = tind

	Fm1 = np.zeros((len(t1), n_sites))
	Fm2 = np.zeros((len(t1), n_sites))
	for k in range(len(t1)):
		Fm1[k, tlib[t1[k]]] = 1
		Fm2[k, tlib[t2[k]]] = 1
	Fm1 = tf.constant(Fm1, dtype=tf.float32)
	Fm2 = tf.constant(Fm2, dtype=tf.float32)


	input_shape = (32, 32, 1)
	input_im = keras.layers.Input(shape=input_shape, name='input')

	vis= keras.layers.Lambda(hp.Lambda_dft(F))(input_im)
	if sigma is not None:
		vis = keras.layers.GaussianNoise(sigma)(vis)

	ising_sample, energy= Ising_sampling2(output_dim=n_sites, name='ising',
									my_initializer=Constant(0.1))(input_im, n_layers=n_ising_layers, const=slope_const)
	# ising_sample, energy= Ising_sampling(output_dim=n_sites, name='ising',
	# 								my_initializer=Constant(0.1))(input_im, n_layers=n_ising_layers, const=slope_const, xi=0.3, L=1)

	if obs_prob is None:
		site_mask = keras.layers.Lambda(hp.Lambda_binary_convert(binary_slope), name='sampling')(ising_sample)
		vis_mask = keras.layers.Lambda(hp.Lambda_vis_mask2(Fm1, Fm2), name='vis_mask')(site_mask)
	else:
		site_mask = keras.layers.Lambda(hp.Lambda_binary_convert(binary_slope), name='sampling')(ising_sample)
		
		prob_mask = np.zeros(n_sites)
		for k in obs_prob.keys():
			prob_mask[tlib[k]] = obs_prob[k]

		site_mask_weather = Site_mask_prob(prob_mask)(site_mask)
		vis_mask = keras.layers.Lambda(hp.Lambda_vis_mask2(Fm1, Fm2), name='vis_mask')(site_mask_weather)
	
	# site_mask = keras.layers.Lambda(hp.Lambda_binary_convert(binary_slope), name='sampling')(ising_sample)

	# vis_mask = keras.layers.Lambda(hp.Lambda_vis_mask2(Fm1, Fm2), name='vis_mask')(site_mask)

	vis_selected = keras.layers.Lambda(hp.Lambda_select0)([vis_mask, vis])

	vis_reshape = keras.layers.Reshape((2*F.shape[0], ))(vis_selected)
	dirty_im = keras.layers.Dense(32*32*1, activation=acti, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_dirtyimage')(vis_reshape)
	dirty_im_reshape = keras.layers.Reshape((32, 32, 1))(dirty_im)
	outputs = _unet_from_tensor(dirty_im_reshape, filt, kern, acti)
	outputs = keras.layers.ReLU(name='recon')(outputs)
	model = keras.models.Model(inputs=input_im, outputs=[outputs, site_mask, energy])
	return model


########################################################################################
# joint sensing and imaging network with amplitude and closure phase
########################################################################################
def IsingCpAmpNet(t1, t2, tc1, tc2, tc3, F, F_cphase, cphase_proj, n_ising_layers=5, slope_const=1e1, sigma=None, binary_slope=10, obs_prob=None):
	filt = 64
	kern = 3
	acti = None

	n_vis = F.shape[0]
	n_cphase = F_cphase.shape[0]
	F = tf.constant(F, dtype=tf.complex64)
	F_cphase = tf.constant(F_cphase, dtype=tf.complex64)
	cphase_proj = tf.constant(cphase_proj, dtype=tf.float32)

	tlib = {}                                                              
	tind = 0
	telescopes = np.concatenate([t1, t2])                                                               
	for k in range(len(telescopes)):
		if telescopes[k] not in tlib:
			tlib[telescopes[k]] = tind 
			tind += 1

	if 'SPT'  not in telescopes:
		tlib['SPT'] = tind
		tind += 1
	elif 'GLT'  not in telescopes:
		tlib['GLT'] = tind
		tind += 1
	n_sites = tind

	Fm1 = np.zeros((len(t1), n_sites))
	Fm2 = np.zeros((len(t1), n_sites))
	for k in range(len(t1)):
		Fm1[k, tlib[t1[k]]] = 1
		Fm2[k, tlib[t2[k]]] = 1
	Fm1 = tf.constant(Fm1, dtype=tf.float32)
	Fm2 = tf.constant(Fm2, dtype=tf.float32)


	Fcm1 = np.zeros((len(tc1), n_sites))
	Fcm2 = np.zeros((len(tc1), n_sites))
	Fcm3 = np.zeros((len(tc1), n_sites))
	for k in range(len(tc1)):
		Fcm1[k, tlib[tc1[k]]] = 1
		Fcm2[k, tlib[tc2[k]]] = 1
		Fcm3[k, tlib[tc3[k]]] = 1
	Fcm1 = tf.constant(Fcm1, dtype=tf.float32)
	Fcm2 = tf.constant(Fcm2, dtype=tf.float32)
	Fcm3 = tf.constant(Fcm3, dtype=tf.float32)


	input_shape = (32, 32, 1)
	input_im = keras.layers.Input(shape=input_shape, name='input')

	vis = keras.layers.Lambda(hp.Lambda_dft3(F), name='vis')(input_im)
	
	if sigma is not None:
		vis_split = keras.layers.Lambda(hp.Lambda_split)(vis)
		vis_split = keras.layers.GaussianNoise(sigma)(vis_split)
		vis = keras.layers.Lambda(hp.Lambda_combine)(vis_split)

		vis_amp = keras.layers.Lambda(hp.Lambda_amp, name='vis_amp')(vis)

		vis_angle_noisy = keras.layers.Lambda(hp.Lambda_angle)(vis)
		cphase = keras.layers.Lambda(hp.Lambda_cphase(cphase_proj))(vis_angle_noisy)

	else:
		vis_amp = keras.layers.Lambda(hp.Lambda_amp, name='vis_amp')(vis)
		cphase = keras.layers.Lambda(hp.Lambda_cphase2(F_cphase), name='cphase')(input_im)

	# Sampling section
	ising_sample, energy= Ising_sampling2(output_dim=n_sites, name='ising',
									my_initializer=Constant(0.1))(input_im, n_layers=n_ising_layers, const=slope_const)
	
	# ising_sample, energy= Ising_sampling(output_dim=n_sites, name='ising',
	# 								my_initializer=Constant(0.1))(input_im, n_layers=n_ising_layers, const=slope_const, xi=1, L=1)

	if obs_prob is None:
		site_mask = keras.layers.Lambda(hp.Lambda_binary_convert(binary_slope), name='sampling')(ising_sample)
		vis_mask = keras.layers.Lambda(hp.Lambda_vis_mask2(Fm1, Fm2), name='vis_mask')(site_mask)
		cphase_mask = keras.layers.Lambda(hp.Lambda_cphase_mask2(Fcm1, Fcm2, Fcm3), name='cphase_mask')(site_mask)
	else:
		site_mask = keras.layers.Lambda(hp.Lambda_binary_convert(binary_slope), name='sampling')(ising_sample)

		prob_mask = np.zeros(n_sites)
		for k in obs_prob.keys():
			prob_mask[tlib[k]] = obs_prob[k]

		site_mask_weather = Site_mask_prob(prob_mask)(site_mask)
		vis_mask = keras.layers.Lambda(hp.Lambda_vis_mask2(Fm1, Fm2), name='vis_mask')(site_mask_weather)
		cphase_mask = keras.layers.Lambda(hp.Lambda_cphase_mask2(Fcm1, Fcm2, Fcm3), name='cphase_mask')(site_mask_weather)


	# vis_mask = keras.layers.Lambda(hp.Lambda_vis_mask2(Fm1, Fm2), name='vis_mask')(site_mask)
	# cphase_mask = keras.layers.Lambda(hp.Lambda_cphase_mask2(Fcm1, Fcm2, Fcm3), name='cphase_mask')(site_mask)

	vis_amp_selected = keras.layers.Lambda(hp.Lambda_select)([vis_mask, vis_amp])
	cphase_selected = keras.layers.Lambda(hp.Lambda_select)([cphase_mask, cphase])

	# Reconstruction section
	inputs = keras.layers.concatenate([vis_amp_selected, cphase_selected], -1)

	vis = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_vis1')(inputs)
	vis = keras.layers.LeakyReLU(alpha=0.3, name='acti_vis1')(vis)
	vis = keras.layers.BatchNormalization()(vis)
	vis = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_vis2')(vis)
	vis = keras.layers.LeakyReLU(alpha=0.3, name='acti_vis2')(vis)
	vis = keras.layers.BatchNormalization()(vis)
	vis = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_vis3')(vis)
	vis = keras.layers.LeakyReLU(alpha=0.3, name='acti_vis3')(vis)
	vis = keras.layers.BatchNormalization()(vis)
	vis_angle = keras.layers.Dense(n_vis, activation=None, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='vis_angle_pred')(vis)
	vis_pred = keras.layers.Lambda(hp.Lambda_Vis, name='vis_pred')([vis_amp_selected, vis_angle])
	
	# vis_amp_pred = keras.layers.Dense(n_vis, activation=None, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='vis_amp_pred')(vis)
	# vis_pred = keras.layers.Lambda(hp.Lambda_Vis, name='vis_pred')([vis_amp_pred, vis_angle])
	

	dirty_im = keras.layers.Dense(32*32*1, activation=acti, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_dirtyimage')(vis_pred)
	dirty_im_reshape = keras.layers.Reshape((32, 32, 1))(dirty_im)

	recon = _unet_from_tensor(dirty_im_reshape, filt, kern, acti)
	recon = keras.layers.ReLU(name='recon')(recon)
	cphase_pred = keras.layers.Lambda(hp.Lambda_cphase(cphase_proj), name='cphase_pred')(vis_angle)

	# model = keras.models.Model(inputs=input_im, outputs=[recon, site_mask, energy, vis_pred, cphase_pred])
	model = keras.models.Model(inputs=input_im, outputs=[recon, site_mask, energy, vis_angle, cphase_pred])
	return model

########################################################################################
# multiple targets: joint sensing and imaging network with complex visibility
########################################################################################
def IsingMutipleVisNet(t1_list, t2_list, F_list, n_ising_layers=5, slope_const=1e2, sigma=None):
	filt = 64
	kern = 3
	acti = None


	Fm1_list = []
	Fm2_list = []
	for k in range(len(t1_list)):
		t1 = t1_list[k]
		t2 = t2_list[k]
		F = F_list[k]

		if k == 0:
			tlib = {}                                                              
			tind = 0
			telescopes = np.concatenate([t1, t2])                                                               
			for k in range(len(telescopes)):
				if telescopes[k] not in tlib:
					tlib[telescopes[k]] = tind 
					tind += 1

			if 'SPT'  not in telescopes:
				tlib['SPT'] = tind
				tind += 1
			elif 'GLT'  not in telescopes:
				tlib['GLT'] = tind
				tind += 1

			n_sites = tind

		Fm1 = np.zeros((len(t1), n_sites))
		Fm2 = np.zeros((len(t1), n_sites))
		for k in range(len(t1)):
			Fm1[k, tlib[t1[k]]] = 1
			Fm2[k, tlib[t2[k]]] = 1
		Fm1 = tf.constant(Fm1, dtype=tf.float32)
		Fm2 = tf.constant(Fm2, dtype=tf.float32)

		Fm1_list.append(Fm1)
		Fm2_list.append(Fm2)


	input_shape = (32, 32, 1)
	input_im = keras.layers.Input(shape=input_shape, name='input')


	ising_sample, energy = Ising_sampling2(output_dim=n_sites, name='ising',
									my_initializer=Constant(0.1))(input_im, n_layers=n_ising_layers, const=slope_const)
	
	site_mask = keras.layers.Lambda(hp.Lambda_binary_convert(10), name='sampling')(ising_sample)

	outputs_list = []
	for k in range(len(t1_list)):
		F = F_list[k]
		Fm1 = Fm1_list[k]
		Fm2 = Fm2_list[k]
		vis= keras.layers.Lambda(hp.Lambda_dft(F))(input_im)
		if sigma is not None:
			vis = keras.layers.GaussianNoise(sigma)(vis)

		vis_mask = keras.layers.Lambda(hp.Lambda_vis_mask2(Fm1, Fm2), name='vis_mask'+str(k))(site_mask)

		vis_selected = keras.layers.Lambda(hp.Lambda_select0)([vis_mask, vis])

		vis_reshape = keras.layers.Reshape((2*F.shape[0], ))(vis_selected)
		dirty_im = keras.layers.Dense(32*32*1, activation=acti, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_dirtyimage'+str(k))(vis_reshape)
		dirty_im_reshape = keras.layers.Reshape((32, 32, 1))(dirty_im)
		outputs = _unet_from_tensor(dirty_im_reshape, filt, kern, acti)
		outputs = keras.layers.ReLU(name='recon'+str(k))(outputs)
		outputs_list.append(outputs)
	# model = keras.models.Model(inputs=input_im, outputs=[outputs, site_mask, energy])
	model = keras.models.Model(inputs=input_im, outputs=outputs_list+[site_mask, energy])
	return model

########################################################################################
# multiple targets: joint sensing and imaging network with amplitude and closure phase
########################################################################################
def IsingMutipleCpAmpNet(t1_list, t2_list, tc1_list, tc2_list, tc3_list, F_list, F_cphase_list, cphase_proj_list, n_ising_layers=5, slope_const=1e1, sigma=None):
	filt = 64
	kern = 3
	acti = None



	Fm1_list = []
	Fm2_list = []
	Fcm1_list = []
	Fcm2_list = []
	Fcm3_list = []
	for k in range(len(t1_list)):
		t1 = t1_list[k]
		t2 = t2_list[k]
		tc1 = tc1_list[k]
		tc2 = tc2_list[k]
		tc3 = tc3_list[k]
		F = F_list[k]
		F_cphase = F_cphase_list[k]
		cphase_proj = cphase_proj_list[k]

		if k == 0:
			tlib = {}                                                              
			tind = 0
			telescopes = np.concatenate([t1, t2])                                                               
			for k in range(len(telescopes)):
				if telescopes[k] not in tlib:
					tlib[telescopes[k]] = tind 
					tind += 1

			if 'SPT'  not in telescopes:
				tlib['SPT'] = tind
				tind += 1
			elif 'GLT'  not in telescopes:
				tlib['GLT'] = tind
				tind += 1

			n_sites = tind


		Fm1 = np.zeros((len(t1), n_sites))
		Fm2 = np.zeros((len(t1), n_sites))
		for k in range(len(t1)):
			Fm1[k, tlib[t1[k]]] = 1
			Fm2[k, tlib[t2[k]]] = 1
		Fm1 = tf.constant(Fm1, dtype=tf.float32)
		Fm2 = tf.constant(Fm2, dtype=tf.float32)


		Fcm1 = np.zeros((len(tc1), n_sites))
		Fcm2 = np.zeros((len(tc1), n_sites))
		Fcm3 = np.zeros((len(tc1), n_sites))
		for k in range(len(tc1)):
			Fcm1[k, tlib[tc1[k]]] = 1
			Fcm2[k, tlib[tc2[k]]] = 1
			Fcm3[k, tlib[tc3[k]]] = 1
		Fcm1 = tf.constant(Fcm1, dtype=tf.float32)
		Fcm2 = tf.constant(Fcm2, dtype=tf.float32)
		Fcm3 = tf.constant(Fcm3, dtype=tf.float32)


		Fm1_list.append(Fm1)
		Fm2_list.append(Fm2)
		Fcm1_list.append(Fcm1)
		Fcm2_list.append(Fcm2)
		Fcm3_list.append(Fcm3)


	input_shape = (32, 32, 1)
	input_im = keras.layers.Input(shape=input_shape, name='input')

	# Sampling section
	ising_sample, energy= Ising_sampling2(output_dim=n_sites, name='ising',
									my_initializer=Constant(0.1))(input_im, n_layers=n_ising_layers, const=slope_const)
	
	site_mask = keras.layers.Lambda(hp.Lambda_binary_convert(10), name='sampling')(ising_sample)

	outputs_list = []
	for k in range(len(t1_list)):
		F = F_list[k]
		F_cphase = F_cphase_list[k]
		cphase_proj = cphase_proj_list[k]
		Fm1 = Fm1_list[k]
		Fm2 = Fm2_list[k]
		Fcm1 = Fcm1_list[k]
		Fcm2 = Fcm2_list[k]
		Fcm3 = Fcm3_list[k]

		n_vis = F.shape[0]
		n_cphase = F_cphase.shape[0]
		F = tf.constant(F, dtype=tf.complex64)
		F_cphase = tf.constant(F_cphase, dtype=tf.complex64)
		cphase_proj = tf.constant(cphase_proj, dtype=tf.float32)

		vis = keras.layers.Lambda(hp.Lambda_dft3(F), name='vis'+str(k))(input_im)
		
		if sigma is not None:
			vis_split = keras.layers.Lambda(hp.Lambda_split)(vis)
			vis_split = keras.layers.GaussianNoise(sigma)(vis_split)
			vis = keras.layers.Lambda(hp.Lambda_combine)(vis_split)

			vis_amp = keras.layers.Lambda(hp.Lambda_amp, name='vis_amp'+str(k))(vis)

			vis_angle_noisy = keras.layers.Lambda(hp.Lambda_angle)(vis)
			cphase = keras.layers.Lambda(hp.Lambda_cphase(cphase_proj))(vis_angle_noisy)

		else:
			vis_amp = keras.layers.Lambda(hp.Lambda_amp, name='vis_amp'+str(k))(vis)
			cphase = keras.layers.Lambda(hp.Lambda_cphase2(F_cphase), name='cphase'+str(k))(input_im)


		vis_mask = keras.layers.Lambda(hp.Lambda_vis_mask2(Fm1, Fm2), name='vis_mask'+str(k))(site_mask)
		cphase_mask = keras.layers.Lambda(hp.Lambda_cphase_mask2(Fcm1, Fcm2, Fcm3), name='cphase_mask'+str(k))(site_mask)

		vis_amp_selected = keras.layers.Lambda(hp.Lambda_select)([vis_mask, vis_amp])
		cphase_selected = keras.layers.Lambda(hp.Lambda_select)([cphase_mask, cphase])

		# Reconstruction section
		inputs = keras.layers.concatenate([vis_amp_selected, cphase_selected], -1)

		vis = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_vis1'+str(k))(inputs)
		vis = keras.layers.LeakyReLU(alpha=0.3, name='acti_vis1'+str(k))(vis)
		vis = keras.layers.BatchNormalization()(vis)
		vis = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_vis2'+str(k))(vis)
		vis = keras.layers.LeakyReLU(alpha=0.3, name='acti_vis2'+str(k))(vis)
		vis = keras.layers.BatchNormalization()(vis)
		vis_angle = keras.layers.Dense(n_vis, activation=None, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_vis3'+str(k))(vis)
		vis_pred = keras.layers.Lambda(hp.Lambda_Vis, name='vis_pred'+str(k))([vis_amp_selected, vis_angle])

		dirty_im = keras.layers.Dense(32*32*1, activation=acti, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_dirtyimage'+str(k))(vis_pred)
		dirty_im_reshape = keras.layers.Reshape((32, 32, 1))(dirty_im)

		recon = _unet_from_tensor(dirty_im_reshape, filt, kern, acti)
		recon = keras.layers.ReLU(name='recon'+str(k))(recon)
		cphase_pred = keras.layers.Lambda(hp.Lambda_cphase(cphase_proj), name='cphase_pred'+str(k))(vis_angle)
		outputs_list.append(recon)
		outputs_list.append(vis_pred)
		outputs_list.append(cphase_pred)
	model = keras.models.Model(inputs=input_im, outputs=outputs_list+[site_mask, energy])
	return model


########################################################################################
# joint sensing and feature extraction network with complex visibility
########################################################################################
def IsingVisFeatureNet(t1, t2, n_ising_layers=5, slope_const=1e2, n_layers=3, n_hid=1000, sigma=None, feature_name='both', binary_slope=10):
	filt = 64
	kern = 3
	acti = None

	tlib = {}                                                              
	tind = 0
	telescopes = np.concatenate([t1, t2])                                                               
	for k in range(len(telescopes)):
		if telescopes[k] not in tlib:
			tlib[telescopes[k]] = tind 
			tind += 1

	if 'SPT'  not in telescopes:
		tlib['SPT'] = tind
		tind += 1
	elif 'GLT'  not in telescopes:
		tlib['GLT'] = tind
		tind += 1

	n_sites = tind

	Fm1 = np.zeros((len(t1), n_sites))
	Fm2 = np.zeros((len(t1), n_sites))
	for k in range(len(t1)):
		Fm1[k, tlib[t1[k]]] = 1
		Fm2[k, tlib[t2[k]]] = 1
	Fm1 = tf.constant(Fm1, dtype=tf.float32)
	Fm2 = tf.constant(Fm2, dtype=tf.float32)


	input_shape = (len(t1), )
	vis = keras.layers.Input(shape=input_shape, dtype=tf.complex64, name='input')
	vis_split = keras.layers.Lambda(hp.Lambda_split)(vis)

	if sigma is not None:
		vis_split = keras.layers.GaussianNoise(sigma)(vis_split)

	ising_sample, energy= Ising_sampling2(output_dim=n_sites, name='ising',
									my_initializer=Constant(0.1))(vis, n_layers=n_ising_layers, const=slope_const)

	site_mask = keras.layers.Lambda(hp.Lambda_binary_convert(binary_slope), name='sampling')(ising_sample)

	vis_mask = keras.layers.Lambda(hp.Lambda_vis_mask2(Fm1, Fm2), name='vis_mask')(site_mask)

	vis_selected = keras.layers.Lambda(hp.Lambda_select0)([vis_mask, vis_split])

	
	vis_reshape = keras.layers.Reshape((2*len(t1), ))(vis_selected)

	layer_input = vis_reshape
	for k in range(n_layers):
		hidden = keras.layers.Dense(n_hid, activation=acti, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None))(layer_input)
		hidden_act = keras.layers.ReLU()(hidden)
		layer_input = keras.layers.BatchNormalization()(hidden_act)

	if feature_name == 'class':
		bh_class = keras.layers.Dense(2, activation=acti, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None))(layer_input)
		bh_class = keras.layers.Softmax(axis=-1, name='bh_class')(bh_class)
		model = keras.models.Model(inputs=vis, outputs=[bh_class, site_mask, energy])
	elif feature_name == 'spin':
		spin_class = keras.layers.Dense(7, activation=acti, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None))(layer_input)
		spin_class = keras.layers.Softmax(axis=-1, name='spin_class')(spin_class)
		model = keras.models.Model(inputs=vis, outputs=[spin_class, site_mask, energy])
	elif feature_name == 'both':
		combined_class = keras.layers.Dense(13, activation=acti, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None))(layer_input)
		combined_class = keras.layers.Softmax(axis=-1, name='combined_class')(combined_class)
		model = keras.models.Model(inputs=vis, outputs=[combined_class, site_mask, energy])
	# model = keras.models.Model(inputs=vis, outputs=[bh_class, spin_class, site_mask, energy])
	# model = keras.models.Model(inputs=vis, outputs=[combined_class, site_mask, energy])

	return model


########################################################################################
# joint sensing and feature extraction network with amplitude and closure phase
########################################################################################
def IsingCpAmpFeatureNet(t1, t2, tc1, tc2, tc3, cphase_proj, n_ising_layers=5, slope_const=1e2, n_layers=3, n_hid=1000, sigma=None, feature_name='both', binary_slope=10):
	filt = 64
	kern = 3
	acti = None

	tlib = {}                                                              
	tind = 0
	telescopes = np.concatenate([t1, t2])                                                               
	for k in range(len(telescopes)):
		if telescopes[k] not in tlib:
			tlib[telescopes[k]] = tind 
			tind += 1

	if 'SPT'  not in telescopes:
		tlib['SPT'] = tind
		tind += 1
	elif 'GLT'  not in telescopes:
		tlib['GLT'] = tind
		tind += 1

	n_sites = tind


	Fm1 = np.zeros((len(t1), n_sites))
	Fm2 = np.zeros((len(t1), n_sites))
	for k in range(len(t1)):
		Fm1[k, tlib[t1[k]]] = 1
		Fm2[k, tlib[t2[k]]] = 1
	Fm1 = tf.constant(Fm1, dtype=tf.float32)
	Fm2 = tf.constant(Fm2, dtype=tf.float32)


	Fcm1 = np.zeros((len(tc1), n_sites))
	Fcm2 = np.zeros((len(tc1), n_sites))
	Fcm3 = np.zeros((len(tc1), n_sites))
	for k in range(len(tc1)):
		Fcm1[k, tlib[tc1[k]]] = 1
		Fcm2[k, tlib[tc2[k]]] = 1
		Fcm3[k, tlib[tc3[k]]] = 1
	Fcm1 = tf.constant(Fcm1, dtype=tf.float32)
	Fcm2 = tf.constant(Fcm2, dtype=tf.float32)
	Fcm3 = tf.constant(Fcm3, dtype=tf.float32)


	input_shape = (len(t1), )
	vis = keras.layers.Input(shape=input_shape, dtype=tf.complex64, name='input')
	
	if sigma is not None:
		vis_split = keras.layers.Lambda(hp.Lambda_split)(vis)
		vis_split = keras.layers.GaussianNoise(sigma)(vis_split)
		vis = keras.layers.Lambda(hp.Lambda_combine)(vis_split)

		vis_amp = keras.layers.Lambda(hp.Lambda_amp, name='vis_amp')(vis)

		vis_angle_noisy = keras.layers.Lambda(hp.Lambda_angle)(vis)
		cphase = keras.layers.Lambda(hp.Lambda_cphase(cphase_proj))(vis_angle_noisy)

	else:
		vis_amp = keras.layers.Lambda(hp.Lambda_amp, name='vis_amp')(vis)
		vis_angle = keras.layers.Lambda(hp.Lambda_angle)(vis)
		cphase = keras.layers.Lambda(hp.Lambda_cphase(cphase_proj))(vis_angle)



	ising_sample, energy= Ising_sampling2(output_dim=n_sites, name='ising',
									my_initializer=Constant(0.1))(vis, n_layers=n_ising_layers, const=slope_const)

	site_mask = keras.layers.Lambda(hp.Lambda_binary_convert(binary_slope), name='sampling')(ising_sample)


	vis_mask = keras.layers.Lambda(hp.Lambda_vis_mask2(Fm1, Fm2), name='vis_mask')(site_mask)
	cphase_mask = keras.layers.Lambda(hp.Lambda_cphase_mask2(Fcm1, Fcm2, Fcm3), name='cphase_mask')(site_mask)

	vis_amp_selected = keras.layers.Lambda(hp.Lambda_select)([vis_mask, vis_amp])
	cphase_selected = keras.layers.Lambda(hp.Lambda_select)([cphase_mask, cphase])


	# Reconstruction section
	inputs = keras.layers.concatenate([vis_amp_selected, cphase_selected], -1)

	vis_pred = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_vis1')(inputs)
	vis_pred = keras.layers.LeakyReLU(alpha=0.3, name='acti_vis1')(vis_pred)
	vis_pred = keras.layers.BatchNormalization()(vis_pred)
	vis_pred = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_vis2')(vis_pred)
	vis_pred = keras.layers.LeakyReLU(alpha=0.3, name='acti_vis2')(vis_pred)
	vis_pred = keras.layers.BatchNormalization()(vis_pred)
	vis_angle = keras.layers.Dense(len(t1), activation=None, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_vis3')(vis_pred)
	layer_input = keras.layers.Lambda(hp.Lambda_Vis, name='vis_pred')([vis_amp_selected, vis_angle])

	for k in range(n_layers):
		hidden = keras.layers.Dense(n_hid, activation=acti, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None))(layer_input)
		hidden_act = keras.layers.ReLU()(hidden)
		layer_input = keras.layers.BatchNormalization()(hidden_act)

	if feature_name == 'class':
		bh_class = keras.layers.Dense(2, activation=acti, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None))(layer_input)
		bh_class = keras.layers.Softmax(axis=-1, name='bh_class')(bh_class)
		model = keras.models.Model(inputs=vis, outputs=[bh_class, site_mask, energy])
	elif feature_name == 'spin':
		spin_class = keras.layers.Dense(7, activation=acti, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None))(layer_input)
		spin_class = keras.layers.Softmax(axis=-1, name='spin_class')(spin_class)
		model = keras.models.Model(inputs=vis, outputs=[spin_class, site_mask, energy])
	elif feature_name == 'both':
		combined_class = keras.layers.Dense(13, activation=acti, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None))(layer_input)
		combined_class = keras.layers.Softmax(axis=-1, name='combined_class')(combined_class)
		model = keras.models.Model(inputs=vis, outputs=[combined_class, site_mask, energy])

	# model = keras.models.Model(inputs=vis, outputs=[bh_class, spin_class, site_mask, energy])
	# model = keras.models.Model(inputs=vis, outputs=[combined_class, site_mask, energy])

	return model


########################################################################################
# flow-based joint sensing and imaging network with complex visibility
########################################################################################
def FlowVisNet(t1, t2, F, n_coupling_layers=5, slope_const=1e2, sigma=None, binary_slope=10, obs_prob=None, generator='gaussian'):
	filt = 64
	kern = 3
	acti = None

	tlib = {}                                                              
	tind = 0
	telescopes = np.concatenate([t1, t2])                                                               
	for k in range(len(telescopes)):
		if telescopes[k] not in tlib:
			tlib[telescopes[k]] = tind 
			tind += 1

	if 'SPT'  not in telescopes:
		tlib['SPT'] = tind
		tind += 1
	elif 'GLT'  not in telescopes:
		tlib['GLT'] = tind
		tind += 1

	n_sites = tind

	Fm1 = np.zeros((len(t1), n_sites))
	Fm2 = np.zeros((len(t1), n_sites))
	for k in range(len(t1)):
		Fm1[k, tlib[t1[k]]] = 1
		Fm2[k, tlib[t2[k]]] = 1
	Fm1 = tf.constant(Fm1, dtype=tf.float32)
	Fm2 = tf.constant(Fm2, dtype=tf.float32)


	input_shape = (32, 32, 1)
	input_im = keras.layers.Input(shape=input_shape, name='input')

	vis= keras.layers.Lambda(hp.Lambda_dft(F))(input_im)
	if sigma is not None:
		vis = keras.layers.GaussianNoise(sigma)(vis)

	if generator == 'realnvp':
		encoder, resnet_list, permute_list = realnvp_encoder(output_dim=n_sites, n_coupling=n_coupling_layers)
		decoder = realnvp_decoder(resnet_list, permute_list, output_dim=n_sites, n_coupling=n_coupling_layers) 

		z = keras.layers.Lambda(Lambda_Gaussian(output_dim=n_sites), name='random_seeds')(input_im)
		flow_sample, logdet = decoder(z)
	elif generator == 'gaussian':
		flow_sample, logdet = Gaussian_sampling(output_dim=n_sites, name='gaussian_samples')(input_im)

	site_mask = STE_layer(name='sampling')(flow_sample)

	logdet2 = keras.layers.Lambda(Lambda_logdet_sigmoid(1.0))(flow_sample)

	logprob = keras.layers.Add(name='logprob')([logdet, logdet2])


	if obs_prob is None:
		vis_mask = keras.layers.Lambda(hp.Lambda_vis_mask2(Fm1, Fm2), name='vis_mask')(site_mask)
	else:
		
		prob_mask = np.zeros(n_sites)
		for k in obs_prob.keys():
			prob_mask[tlib[k]] = obs_prob[k]

		site_mask_weather = Site_mask_prob(prob_mask)(site_mask)
		vis_mask = keras.layers.Lambda(hp.Lambda_vis_mask2(Fm1, Fm2), name='vis_mask')(site_mask_weather)
	
	# site_mask = keras.layers.Lambda(hp.Lambda_binary_convert(binary_slope), name='sampling')(ising_sample)

	# vis_mask = keras.layers.Lambda(hp.Lambda_vis_mask2(Fm1, Fm2), name='vis_mask')(site_mask)

	vis_selected = keras.layers.Lambda(hp.Lambda_select0)([vis_mask, vis])

	vis_reshape = keras.layers.Reshape((2*F.shape[0], ))(vis_selected)
	dirty_im = keras.layers.Dense(32*32*1, activation=acti, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_dirtyimage')(vis_reshape)
	dirty_im_reshape = keras.layers.Reshape((32, 32, 1))(dirty_im)
	outputs = _unet_from_tensor(dirty_im_reshape, filt, kern, acti)
	outputs = keras.layers.ReLU(name='recon')(outputs)
	model = keras.models.Model(inputs=input_im, outputs=[outputs, site_mask, logprob])
	return model


########################################################################################
# flow-based joint sensing and imaging network with amplitude and closure phase
########################################################################################
def FlowCpAmpNet(t1, t2, tc1, tc2, tc3, F, F_cphase, cphase_proj, n_coupling_layers=5, slope_const=1e1, sigma=None, binary_slope=10, obs_prob=None, generator='gaussian'):
	filt = 64
	kern = 3
	acti = None

	n_vis = F.shape[0]
	n_cphase = F_cphase.shape[0]
	F = tf.constant(F, dtype=tf.complex64)
	F_cphase = tf.constant(F_cphase, dtype=tf.complex64)
	cphase_proj = tf.constant(cphase_proj, dtype=tf.float32)

	tlib = {}                                                              
	tind = 0
	telescopes = np.concatenate([t1, t2])                                                               
	for k in range(len(telescopes)):
		if telescopes[k] not in tlib:
			tlib[telescopes[k]] = tind 
			tind += 1

	if 'SPT'  not in telescopes:
		tlib['SPT'] = tind
		tind += 1
	elif 'GLT'  not in telescopes:
		tlib['GLT'] = tind
		tind += 1
	n_sites = tind

	Fm1 = np.zeros((len(t1), n_sites))
	Fm2 = np.zeros((len(t1), n_sites))
	for k in range(len(t1)):
		Fm1[k, tlib[t1[k]]] = 1
		Fm2[k, tlib[t2[k]]] = 1
	Fm1 = tf.constant(Fm1, dtype=tf.float32)
	Fm2 = tf.constant(Fm2, dtype=tf.float32)


	Fcm1 = np.zeros((len(tc1), n_sites))
	Fcm2 = np.zeros((len(tc1), n_sites))
	Fcm3 = np.zeros((len(tc1), n_sites))
	for k in range(len(tc1)):
		Fcm1[k, tlib[tc1[k]]] = 1
		Fcm2[k, tlib[tc2[k]]] = 1
		Fcm3[k, tlib[tc3[k]]] = 1
	Fcm1 = tf.constant(Fcm1, dtype=tf.float32)
	Fcm2 = tf.constant(Fcm2, dtype=tf.float32)
	Fcm3 = tf.constant(Fcm3, dtype=tf.float32)


	input_shape = (32, 32, 1)
	input_im = keras.layers.Input(shape=input_shape, name='input')

	vis = keras.layers.Lambda(hp.Lambda_dft3(F), name='vis')(input_im)
	
	if sigma is not None:
		vis_split = keras.layers.Lambda(hp.Lambda_split)(vis)
		vis_split = keras.layers.GaussianNoise(sigma)(vis_split)
		vis = keras.layers.Lambda(hp.Lambda_combine)(vis_split)

		vis_amp = keras.layers.Lambda(hp.Lambda_amp, name='vis_amp')(vis)

		vis_angle_noisy = keras.layers.Lambda(hp.Lambda_angle)(vis)
		cphase = keras.layers.Lambda(hp.Lambda_cphase(cphase_proj))(vis_angle_noisy)

	else:
		vis_amp = keras.layers.Lambda(hp.Lambda_amp, name='vis_amp')(vis)
		cphase = keras.layers.Lambda(hp.Lambda_cphase2(F_cphase), name='cphase')(input_im)

	# Sampling section
	if generator == 'realnvp':
		encoder, resnet_list, permute_list = realnvp_encoder(output_dim=n_sites, n_coupling=n_coupling_layers)
		decoder = realnvp_decoder(resnet_list, permute_list, output_dim=n_sites, n_coupling=n_coupling_layers) 

		z = keras.layers.Lambda(Lambda_Gaussian(output_dim=n_sites), name='random_seeds')(input_im)
		flow_sample, logdet = decoder(z)
	elif generator == 'gaussian':
		flow_sample, logdet = Gaussian_sampling(output_dim=n_sites, name='gaussian_samples')(input_im)

	site_mask = STE_layer(name='sampling')(flow_sample)

	logdet2 = keras.layers.Lambda(Lambda_logdet_sigmoid(1.0))(flow_sample)

	logprob = keras.layers.Add(name='logprob')([logdet, logdet2])

	if obs_prob is None:
		vis_mask = keras.layers.Lambda(hp.Lambda_vis_mask2(Fm1, Fm2), name='vis_mask')(site_mask)
		cphase_mask = keras.layers.Lambda(hp.Lambda_cphase_mask2(Fcm1, Fcm2, Fcm3), name='cphase_mask')(site_mask)
	else:
		prob_mask = np.zeros(n_sites)
		for k in obs_prob.keys():
			prob_mask[tlib[k]] = obs_prob[k]

		site_mask_weather = Site_mask_prob(prob_mask)(site_mask)
		vis_mask = keras.layers.Lambda(hp.Lambda_vis_mask2(Fm1, Fm2), name='vis_mask')(site_mask_weather)
		cphase_mask = keras.layers.Lambda(hp.Lambda_cphase_mask2(Fcm1, Fcm2, Fcm3), name='cphase_mask')(site_mask_weather)


	# vis_mask = keras.layers.Lambda(hp.Lambda_vis_mask2(Fm1, Fm2), name='vis_mask')(site_mask)
	# cphase_mask = keras.layers.Lambda(hp.Lambda_cphase_mask2(Fcm1, Fcm2, Fcm3), name='cphase_mask')(site_mask)

	vis_amp_selected = keras.layers.Lambda(hp.Lambda_select)([vis_mask, vis_amp])
	cphase_selected = keras.layers.Lambda(hp.Lambda_select)([cphase_mask, cphase])

	# Reconstruction section
	inputs = keras.layers.concatenate([vis_amp_selected, cphase_selected], -1)

	vis = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_vis1')(inputs)
	vis = keras.layers.LeakyReLU(alpha=0.3, name='acti_vis1')(vis)
	vis = keras.layers.BatchNormalization()(vis)
	vis = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_vis2')(vis)
	vis = keras.layers.LeakyReLU(alpha=0.3, name='acti_vis2')(vis)
	vis = keras.layers.BatchNormalization()(vis)
	vis = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_vis3')(vis)
	vis = keras.layers.LeakyReLU(alpha=0.3, name='acti_vis3')(vis)
	vis = keras.layers.BatchNormalization()(vis)
	vis_angle = keras.layers.Dense(n_vis, activation=None, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='vis_angle_pred')(vis)
	vis_pred = keras.layers.Lambda(hp.Lambda_Vis, name='vis_pred')([vis_amp_selected, vis_angle])
	
	# vis_amp_pred = keras.layers.Dense(n_vis, activation=None, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='vis_amp_pred')(vis)
	# vis_pred = keras.layers.Lambda(hp.Lambda_Vis, name='vis_pred')([vis_amp_pred, vis_angle])
	

	dirty_im = keras.layers.Dense(32*32*1, activation=acti, use_bias=True, kernel_initializer=RandomUniform(minval=-5e-4, maxval=5e-4, seed=None), name='dense_dirtyimage')(vis_pred)
	dirty_im_reshape = keras.layers.Reshape((32, 32, 1))(dirty_im)

	recon = _unet_from_tensor(dirty_im_reshape, filt, kern, acti)
	recon = keras.layers.ReLU(name='recon')(recon)
	cphase_pred = keras.layers.Lambda(hp.Lambda_cphase(cphase_proj), name='cphase_pred')(vis_angle)

	# model = keras.models.Model(inputs=input_im, outputs=[recon, site_mask, energy, vis_pred, cphase_pred])
	model = keras.models.Model(inputs=input_im, outputs=[recon, site_mask, logprob, vis_angle, cphase_pred])
	return model