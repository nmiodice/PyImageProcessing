import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay
from scipy import ndimage
from scipy import misc
import numpy as np
import random
import sys
import os

class ImTools:

	def __init__(self, fp):
		self.mImg = self.read_image(fp)
		assert(self.mImg is not None)

	# returns an NDIMAGE of FP, or NONE if the file cannot be found
	def read_image(self, fp):
		try:
			self.mImg = misc.imread(fp)
		except (FileNotFoundError):
			return None
		return self.mImg

	# writes an IMG, an NDIMAGE, to the file path specified by FP. file extension
	# is determined by the extension in FP and can be any other supported by
	# SCIPY.MISC.IMWRITE
	def write_image(self, fp, img):
		try:
			self.mImg = misc.imsave(fp, img)
		except:
			print('error saving image to file')

	# prints IMG to STDOUT in the format specified by FORMAT. FORMAT can be 'jpg',
	# 'png', or any other extension supported by SCIPY.MISC.IMWRITE
	def write_image_stdout(self, format, img):
		tmp_file_name = 'zz_improc_tmp_img.' + format
		self.write_image(tmp_file_name, img)
		with open(tmp_file_name, 'rb') as f:
			bytes = f.read()
			sys.stdout.buffer.write(bytes)
			f.close()
		os.remove(tmp_file_name)

	# displays an arbitrary image in a new figure with no axis information. If no
	# image is supplied, SELF.MIMG is shown
	def show(self, img = None):
		if img is None:
			img = self.mImg
		plt.imshow(img, cmap = plt.cm.gray)
		plt.axis('off')
		plt.show()

	# takes an M x N x P image and returns an (M * N) x P array, where element
    # (i, j), contains the value of the j'th color channel in the i'th pixel.
    # Pixels are re-arranged in row-major order
	def get_rgb_features(self, rgb_scale = 1):
		shape = self.mImg.shape
		total_px = shape[0] * shape[1]
		features = self.mImg.reshape(1, total_px, shape[2])
		# removes unnecessary layer of array nesting
		features = features[0, :].astype(float)

		for color_chanel in range(shape[2]):
			features[color_chanel, :] *= rgb_scale

		return features

	# takes an M x N x P image and returns an (M * N) x 2 array, where element
    # (i, 0), contains the x location of the i'th pixel and element (i, 1)
    # contains the y location of the i'th pixel. Pixels are re-arranged in row-
	# major order. XY_SCALE is used to scale each feature value
	def get_xy_features(self, xy_scale = 1):
		shape = self.mImg.shape
		total_px = shape[0] * shape[1]

		x_coords = np.arange(shape[1])
		y_coords = np.arange(shape[0])

		features = np.meshgrid(x_coords, y_coords)
		features[0].resize(1, total_px)
		features[1].resize(1, total_px)

		# removes unnecessary layer of array nesting
		features[0] = features[0][0, :]
		features[1] = features[1][0, :]

		# normalize to 1 - 255 range
		for i in range(0, 2):
			features[i] = features[i].astype(float)
			features[i] *= xy_scale

		return np.transpose(features)

	# takes an M x N x P image and returns an (M * N) x (P + 2) array
	# containing RGB features concatenated with XY features
	def get_rgb_xy_features(self, rgb_scale = 1, xy_scale = 1):
		rgb_features = self.get_rgb_features(rgb_scale)
		xy_features = self.get_xy_features(xy_scale)
		features = np.concatenate((rgb_features, xy_features), axis = 1)
		return features

	# returns a K-MEANS cluster the IMG data member into K clusters. Each pixel in
	# each of the K clusters is replaced with the average color of the cluster it
    # belongs to
	def kmeans_cluster(self, k, features):
		k_means = KMeans(n_clusters = k, \
			init = 'k-means++', \
			precompute_distances = True, \
			n_init = 3)
		k_means.fit(features)
		
		shape = self.mImg.shape
		labels = k_means.labels_
		labels.resize(shape[0], shape[1])
		cluster_map = np.copy(self.mImg)

		# replace elements of each cluster with the mean value of the color
		# channels from the original image
		for clus in range(k):
			for col in range(shape[2]):
				cluster_map[labels == clus, col] = np.mean(cluster_map[labels == clus, col])

		return cluster_map

	# returns an array of corner points
	def _get_corner_points(self):
		shape = self.mImg.shape
		coords = []
		coords.append([0, 0])
		coords.append([shape[1] - 1, shape[0] - 1])
		coords.append([shape[1] - 1, 0])
		coords.append([0, shape[0] - 1])
		return coords

	# returns coordinates which are relatively evenly spaced across the image,
	# perturbed by Gaussian noise. Optionally, if INCLUDE_CORNERS is True, then
	# corner points are added as well
	def _get_distributed_points(self, spacing, sigma, include_corners = False):
		shape = self.mImg.shape
		n_x = int(shape[1] / spacing)
		n_y = int(shape[0] / spacing)

		coords = []
		if include_corners:
			coords = self._get_corner_points()

		# generates evenly spaced points, perturbed by Gaussian noise
		for x in range(-1, n_x + 1):
			mu_x = spacing * (x + 1)
			for y in range(-1, n_y + 1):
				mu_y = spacing * (y + 1)
				# clamp generated points to image boundary
				x_pt = min(int(random.gauss(mu_x, sigma)), shape[1])
				x_pt = max(0, x_pt)
				y_pt = min(int(random.gauss(mu_y, sigma)), shape[0])
				y_pt = max(0, y_pt)
				coords.append([x_pt, y_pt])
		return coords
	
	# creates a triangulation reduction of an image. SIZE is a parameter which
	# indicates the (approx) number of vertex points on the longest dimension
	def triangulate(self, size):
		shape = self.mImg.shape
		spacing = max(shape) / size
		sigma = spacing / 4
		coords = self._get_distributed_points(spacing, sigma, include_corners = True)
		
		tri = Delaunay(coords)
		im_pts = self.get_xy_features()
		# pt_tri_membership becomes a map which is the same size as the
		# original image (first two dimensions only). each element contains
		# the triangle ID of that point in the source image
		pt_tri_membership = tri.find_simplex(im_pts.astype(dtype = np.double))
		pt_tri_membership.resize(shape[0], shape[1])
		num_tri = np.max(pt_tri_membership)

		tri_map = np.copy(self.mImg) 
		# replace elements of each triangle with the mean value of the color
		# channels from the original image
		for tri in range(num_tri + 1):
			this_tri = pt_tri_membership == tri
			if not np.any(this_tri):
				continue
			for col in range(shape[2]):
				tri_map[this_tri, col] = np.mean(self.mImg[this_tri, col])
		return tri_map

	# converts an RGB image, IM, to gray-scale using the formula for luminosity:
	#	L = 0.21 * R + 0.72 * G + 0.07 * B
	def rgb2gray(self, im = None):
		if im is None:
			im = self.mImg
		return im[:, :, 0] * 0.21 + im[:, :, 1] * 0.72 + im[:, :, 2] * 0.07

	# returns an M x N energy map of the an M x N gradient map GRAD. The energy 
	# map is is aggregated from the top to bottom of the first dimension of
	# GRAD. The general algorithm is described here: 
	# 	http://en.wikipedia.org/wiki/Seam_carving
	def _get_energy_map(self, grad):
		energy_map = np.zeros(grad.shape)
		energy_map[0, :] = grad[0, :]
		
		# for each pixel in each row of the gradient map, we can find its
		# corresponding minimum energy value by summing that pixels gradient 
		# value with the minimum of the cumulative energy values from 
		# neighboring pixels in the preceding row.
		n_rows = grad.shape[0]
		for row in range(1, n_rows):
			row_orig = energy_map[row - 1, :]
			row_shift_r = np.roll(row_orig, 1)[1:-1]
			row_shift_l = np.roll(row_orig, -1)[1:-1]
			row_orig = row_orig[1:-1]
			# note: np.minimum can only compare 2 arrays at a time!
			min_energy_vals = np.minimum(row_shift_r, row_shift_l)
			min_energy_vals = np.minimum(min_energy_vals, row_orig)
			energy_map[row, 1:-1] = grad[row, 1:-1] + min_energy_vals

			# edge pixels cannot be vectorized like the other pixels because
			# they only look 2, not 3, pixels from the row above
			energy_map[row, 0] = grad[row, 0] + min(energy_map[row - 1, 0], energy_map[row - 1, 1])
			energy_map[row, -1] = grad[row, -1] + min(energy_map[row - 1, -1], energy_map[row - 1, -2])
		return energy_map
	
	# reconstructs the lowest cost energy path using GRAD and ENERGY_MAP and
	# returns a list of pixels to be removed. in the returned list, L, L[I] = J
	# if pixel J is in the lowest cost energy path from row I. it is assumed
	# that the path is a vertical slice of the image
	def _get_lowest_cost_path(self, grad, energy_map, exact = True):
		last_row = energy_map[-1, :]
		if exact:
			candidates = np.where(last_row == last_row.min())[0]
		else:
			min_val = last_row.min()
			std = last_row.std()
			thresh = min_val + std/float(16)
			candidates = np.where(last_row <= thresh)[0]

		low_energy_px = random.choice(candidates)
		print(low_energy_px)
		seam = [low_energy_px]
		n_rows = grad.shape[0]
		n_cols = grad.shape[1]

		# build the list of pixels in reverse order, using dynamic programming
		for row in reversed(range(n_rows - 1)):
			last_px = seam[-1]
			last_enegery_val = energy_map[row + 1, last_px]
			last_grad_val = grad[row + 1, last_px]

			# prevents out of bounds condition when checking near edge pixels
			left_idx = min(n_cols - 1, last_px + 1)
			right_idx = max(0, last_px - 1)

			if last_grad_val + energy_map[row, last_px] == last_enegery_val:
				seam.append(last_px)
			elif last_grad_val + energy_map[row, left_idx] == last_enegery_val:
				seam.append(left_idx)
			elif last_grad_val + energy_map[row, right_idx] == last_enegery_val:
				seam.append(right_idx)
			else:
				# this should absolutely never happen, and indicates an 
				# algorithmic error
				assert(True is False)
		return list(reversed(seam))

	# deletes a vertical seam from IM using a list, SEAM, where each element of
	# SEAM[I] = J if pixel J should be removed from seam I
	def _delete_seam(self, im, seam):
		h, w, d = im.shape
		new_im = np.zeros((h, w - 1, d))

		for dim in range(d):
   			for row in range(h):
   				rem_px = seam[row]
   				new_im[row, :, dim] = np.append(im[row, 0:rem_px, dim], im[row, (rem_px + 1):, dim])
		return new_im

	def smart_resize(self, axis, n_px):
		im_col = np.copy(self.mImg)
		# the algorithm is designed to work only in the x direction, so a simple
		# transpose will allow us to work around this
		if axis.lower() == 'y':
			im_col = np.transpose(im_col, (1, 0, 2))
		elif axis.lower() != 'x':
			raise ValueError('Axis parameter must be either "x" or "y"')
		
		# the general algorithm is to generate a gradient map, then use a
		# dynamic programming approach to find low energy seams in the gradient
		# map. these seems are candidates for removal
		while n_px is not 0:
			n_px -= 1
			im_gray = self.rgb2gray(im_col)
			
			grad = np.gradient(im_gray)
			grad = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
			energy_map = self._get_energy_map(grad)
			seam = self._get_lowest_cost_path(grad, energy_map, exact = False)
			im_col = self._delete_seam(im_col, seam)
		
		# see comments at beginning of function
		if axis.lower() == 'y':
			im_col = np.transpose(im_col, (1, 0, 2))
		return im_col
		


