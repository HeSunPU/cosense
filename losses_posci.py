import numpy as np
import tensorflow as tf
from keras import backend as K
import keras.layers


def site_sparsity(y_true, y_pred):
	return K.mean(K.abs(y_pred))

def energy(y_true, y_pred):
	return y_pred

def entropy_loss(_, y_pred):
		return -y_pred

def Lambda_cross_correlation(x):
	x_true0, x_pred0 = x
	x_true = tf.transpose(x_true0, [1, 2, 0, 3])
	x_pred = tf.transpose(x_pred0, [3, 1, 2, 0])
	cross_correlation = tf.nn.depthwise_conv2d(x_pred, x_true, strides=[1, 1, 1, 1], padding='SAME')
	cross_correlation = tf.transpose(cross_correlation, [3, 1, 2, 0])
	norm_prod = ((tf.sqrt(tf.reduce_sum(tf.square(x_pred0), [1, 2])) + 1e-5) * (tf.sqrt(tf.reduce_sum(tf.square(x_true0), [1, 2])) + 1e-5))
	norm_prod = tf.tensordot(norm_prod, tf.ones((1, 1, 1, 1)), [-1, 0])
	return  cross_correlation / norm_prod

def Lambda_similarity(y_true, y_pred):
	cross_correlation = keras.layers.Lambda(Lambda_cross_correlation)([y_true, y_pred])
	max_cross_corr = keras.layers.MaxPool2D((32, 32))(cross_correlation)
	return 1-K.mean(max_cross_corr)

def Lambda_angle_diff(y_true, y_pred):
	angle_true = y_true * np.pi / 180
	angle_pred = y_pred * np.pi / 180
	return K.mean(1 - K.cos(angle_true - angle_pred))
