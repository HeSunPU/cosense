"""

created on Wed Oct. 31, 2019

@author: He Sun, Caltech

help trainging the joint sensing and imaging network

member functions include discrete fourier transforms (dft), spliting & concatenating complex matrices and generating observation mask


"""


import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer, Lambda
from keras.initializers import RandomUniform, Constant
import keras.layers


def Lambda_dft(F):
	def func(input_im):
		input_im = tf.cast(input_im, tf.complex64)
		input_im_reshape = tf.reshape(input_im, (-1, F.shape[1], 1))
		# compute the Fourier domain measurements
		F_tensor = tf.constant(F, dtype=tf.complex64)
		vis = tf.matmul(F_tensor, input_im_reshape)
		vis_r = tf.math.real(vis)
		vis_i = tf.math.imag(vis)
		# concatenate the real and imag parts of the visibility
		vis_con = tf.concat([vis_r, vis_i], -1)
		return tf.cast(vis_con, dtype=tf.float32)
	return func


def Lambda_dft3(F):
	def func(input_im):
		input_im = tf.cast(input_im, tf.complex64)
		input_im_reshape = tf.reshape(input_im, (-1, F.shape[1], 1))
		# compute the Fourier domain measurements
		vis = tf.squeeze(tf.matmul(F, input_im_reshape), -1)
		return vis
	return func


def Lambda_vis_mask2(F1, F2):
	def func(x):
		return tf.matmul(x, tf.transpose(F1)) * tf.matmul(x, tf.transpose(F2))
	return func


def Lambda_cphase_mask2(F1, F2, F3):
	def func(x):
		return tf.matmul(x, tf.transpose(F1)) * tf.matmul(x, tf.transpose(F2)) * tf.matmul(x, tf.transpose(F3))
	return func


def Lambda_select0(x):
	return tf.matmul(tf.matrix_diag(x[0]), x[1])


def Lambda_select(x):
	return x[0] * x[1]


def Lambda_binary_convert(const=10):
	def func(x):
		# return K.sigmoid(const * x)
		return 0.5 * (x + 1)
	return func


def Lambda_split(vis):
	vis_r = tf.math.real(vis)
	vis_i = tf.math.imag(vis)
	vis_con = tf.concat([tf.expand_dims(vis_r, -1), tf.expand_dims(vis_i, -1)], -1)
	return tf.cast(vis_con, dtype=tf.float32)


def Lambda_combine(vis_con):
	vis_r = vis_con[:, :, 0]
	vis_i = vis_con[:, :, 1]
	vis = tf.cast(vis_r, dtype=tf.complex64) + 1j * tf.cast(vis_i, dtype=tf.complex64)
	return vis


def Lambda_amp(x):
	return tf.abs(x)


def Lambda_angle(vis):
	return tf.math.angle(vis)


def Lambda_Vis(x):
	amp, angle = x
	amp = tf.cast(amp, tf.complex64)
	angle = tf.cast(angle, tf.complex64) * np.pi / 180
	vis = amp * tf.exp(1j * angle)
	return tf.concat([tf.math.real(vis), tf.math.imag(vis)], -1)


def Lambda_cphase(cphase_proj):
	def func(x):
		return tf.matmul(x, tf.transpose(cphase_proj))
	return func


def Lambda_cphase2(F_cphase):
	def func(input_im):
		input_im = tf.cast(input_im, tf.complex64)
		input_im_reshape = tf.reshape(input_im, (-1, F_cphase.shape[1], 1))
		# compute the Fourier domain measurements
		vis1 = tf.squeeze(tf.matmul(F_cphase[:, :, 0], input_im_reshape), -1)
		vis2 = tf.squeeze(tf.matmul(F_cphase[:, :, 1], input_im_reshape), -1)
		vis3 = tf.squeeze(tf.matmul(F_cphase[:, :, 2], input_im_reshape), -1)
		bispec = vis1 * vis2 * vis3
		cphase = tf.math.angle(bispec) * 180 / np.pi
		return cphase
	return func

