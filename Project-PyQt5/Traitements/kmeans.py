#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Fonction permettant de faire une segmentation des nuages grâce à un opérateur de K-Means
# sur les intensités, avec K = 3 (terre, mer, nuages)


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Permet de charger un dossier
import os

import medpy.filter.smoothing as mfs
from skimage.filters import threshold_otsu

 
class Segment:
    def __init__(self,segments=3):
        #define number of segments, with default 3
        self.segments=segments
        
    def kmeans(self,image):
       # Etape de prétraitement : application d'un filtre anisotrope
       image=mfs.anisotropic_diffusion(image, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1)
       
       vectorized=image.flatten()
       vectorized=np.float32(vectorized)
       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
              10, 1.0)
       ret,label,center=cv2.kmeans(vectorized,self.segments,None,
              criteria,10,cv2.KMEANS_RANDOM_CENTERS)
       res = center[label.flatten()]
       segmented_image = res.reshape((image.shape))
       return label.reshape((image.shape[0],image.shape[1])), segmented_image.astype(np.uint8), center
   
    def extractComponent(self,image,label_image,label):
       component=np.zeros(image.shape,np.uint8)
       component[label_image==label]=image[label_image==label]
       return component
   

path = 'ImagesEgalisationAdapt/'
fichiers = os.listdir(path)

for fichier in fichiers:
    #image = cv2.imread('20170411101417_MSG2.tif', -1)
    image = cv2.imread(path+fichier, -1)
    
    seg = Segment()
    
    # On extrait une carte des labels, une image résultat des clusters, et un tableau des centres de clusters
    # (ici, 3 clusters donc 3 centres)
    label, result, center = seg.kmeans(image)
    
    
    maximum = max(center)
    
    # On extrait les pixels appartenant au cluster du centre ayant le niveau de gris le plus important
    # (les pixels appartenant donc à des nuages)
    extracted=seg.extractComponent(image,label, np.argmax(center))
    #plt.imshow(extracted, cmap=plt.get_cmap('gray'))
    
    # Binarisation de l'image résultat d'extraction des nuages
    ret, thresh = cv2.threshold(extracted, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Enregistrement des images résultats
    cv2.imwrite("ImagesKMeans/"+fichier, result)
    cv2.imwrite("ImagesExtractionKMeans/"+fichier, thresh)

    plt.show()

