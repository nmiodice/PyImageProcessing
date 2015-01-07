from imageutils import ImTools
import numpy as np
import time

if __name__ == '__main__':
	im_tools = ImTools('img.jpg')
	im = im_tools.triangulate(50)
	im_tools.show(im)