from imageutils import ImageML
import numpy as np
import time

if __name__ == '__main__':
	im_ml = ImageML('img.jpg')
	features = im_ml.get_rgb_xy_features(1, .25)
	im_ml.show(im_ml.kmeans_cluster(10, features))