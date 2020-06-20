from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt 
import pylab
import skimage as ski 
import skimage.segmentation as sks
import cv2
import numpy as np


img = ski.img_as_float(plt.imread('239239_faces.jpg')[::2,::2,:3])


#img = cv2.imread()
pylab.figure(figsize=(20,10))


segments_fz = sks.felzenszwalb(img, scale=100, sigma=0.01, min_size=100)
borders = sks.find_boundaries(segments_fz)
unique_colors = np.unique(segments_fz.ravel())
segments_fz[borders] = -1 
colors = [np.zeros(3)]
for color in unique_colors:
    colors.append(np.mean(img[segments_fz == color], axis=0)) 
cm = LinearSegmentedColormap.from_list('pallete', colors, N=len(colors))
pylab.subplot(121), pylab.imshow(img), pylab.title('Original', size=20), pylab.axis('off'),
pylab.subplot(122), pylab.imshow(segments_fz, cmap=cm), 
pylab.title('Segmented with Felzenszwalbs\'s method', size=20), pylab.axis('off'), 
pylab.show()