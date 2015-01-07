import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage
from scipy import misc
import numpy as np

class ImageML:

	def __init__(self, fp):
		self.mImg = self.read_image(fp)

	# returns an NDIMAGE of FP, or NONE if the file cannot be found
	def read_image(self, fp):
		try:
			self.mImg = misc.imread(fp)
		except (FileNotFoundError):
			return None
		return self.mImg

	# displays self.MIMG image in a new figure with no axis information
	def show(self):
		plt.imshow(self.mImg, cmap=plt.cm.gray)
		plt.axis('off')
		plt.show()

	# displays an arbitrary image in a new figure with no axis information
	def show(self, img):
		plt.imshow(img, cmap=plt.cm.gray)
		plt.axis('off')
		plt.show()

	# takes an M x N x P image and returns an (M * N) x P array, where element
    # (i, j), contains the value of the j'th color channel in the i'th pixel.
    # Pixels are re-arranged in row-major order
	def get_rgb_features(self, rgb_scale):
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
	def get_xy_features(self, xy_scale):
		shape = self.mImg.shape
		total_px = shape[0] * shape[1]

		x_coords = np.arange(shape[0])
		y_coords = np.arange(shape[1])

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
			# features[i] *= float(255)/features[i].max()
			# features[i] += 1

		return np.transpose(features)

	# takes an M x N x P image and returns an (M * N) x (P + 2) array
	# containing RGB features concatenated with XY features
	def get_rgb_xy_features(self, rgb_scale, xy_scale):
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


