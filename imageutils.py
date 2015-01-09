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
	def get_corner_points(self):
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
	def get_distributed_points(self, spacing, sigma, include_corners = False):
		shape = self.mImg.shape
		n_x = int(shape[1] / spacing)
		n_y = int(shape[0] / spacing)

		coords = []
		if include_corners:
			coords = self.get_corner_points()

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
		coords = self.get_distributed_points(spacing, sigma, include_corners = True)
		
		tri = Delaunay(coords)
		im_pts = self.get_xy_features()
		# pt_tri_membership becomes a map which is the same size as the
		# original image (first two dimensions only). each element contains
		# the triangle membership of that point in the source image
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
				tri_map[this_tri, col] = np.mean(tri_map[this_tri, col])
		return tri_map


