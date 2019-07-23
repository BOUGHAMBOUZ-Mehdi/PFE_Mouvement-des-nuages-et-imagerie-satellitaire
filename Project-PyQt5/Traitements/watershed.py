# USAGE
# python watershed.py --image images/coins_01.png

# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

path = './wetransfer-063323/'
path2 = './ImagesExtractionKMeans/'
fichiers = os.listdir(path)

i = 1
for fichier in fichiers:
        image=cv2.imread(path+fichier, -1)
        thresh = cv2.imread(path2+fichier, -1) 
        """
        plt.imshow(im, cmap=plt.get_cmap('gray'))
        plt.title('image ')
        plt.show()
        plt.imshow(thresh, cmap=plt.get_cmap('gray'))
        plt.title('image ')
        plt.show()
        """
        
        # convert the mean shift image to grayscale, then apply
        # Otsu's thresholding
        #thresh = np.uint8(thresh)
        # compute the exact Euclidean distance from every binary
        # pixel to the nearest zero pixel, then find peaks in this
        # distance map
        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=20,
        	labels=thresh)
        
        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)
        print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
        
        # loop over the unique labels returned by the Watershed
        # algorithm
        for label in np.unique(labels):
        	# if the label is zero, we are examining the 'background'
        	# so simply ignore it
            	if label == 0:
            		continue
        
        	# otherwise, allocate memory for the label region and draw
        	# it on the mask
        	mask = np.zeros(image.shape, dtype="uint8")
        	mask[labels == label] = 255
        
        	# detect contours in the mask and grab the largest one
        	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        		cv2.CHAIN_APPROX_SIMPLE)[-2]
        	c = max(cnts, key=cv2.contourArea)
        
        	# draw a circle enclosing the object
        	((x, y), r) = cv2.minEnclosingCircle(c)
        	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        
        # show the output image
        """
        plt.imshow(image, cmap=plt.get_cmap('gray'))
        plt.title('image ')
        plt.show()
        """
        cv2.imwrite("./WaterShed/"+fichier, image)
        print i
        i+=1
        
