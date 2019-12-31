import numpy as np

import skimage.transform

import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
# from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
	if random_state is None:
		random_state = np.random.RandomState(None)
	shape = image.shape
	shape_size = shape[:2]
	# Random affine
	center_square = np.float32(shape_size) // 2
	square_size = min(shape_size) // 3
	pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
	pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
	M = cv2.getAffineTransform(pts1, pts2)
	image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

	dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
	dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
	dz = np.zeros_like(dx)

	x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
	indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

	return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
