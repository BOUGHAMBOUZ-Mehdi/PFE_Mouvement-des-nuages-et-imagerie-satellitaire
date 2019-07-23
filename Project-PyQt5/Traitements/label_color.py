#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Fonction permettant de renvoyer une carte de K-Means en couleur en fonction d'une carte de label
# (ici, appliqué aux images de LBP)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.color import label2rgb
import scipy.misc


 
class Segment:
    def __init__(self,segments=3):
        #define number of segments, with default 3
        self.segments=segments
        
    def kmeans(self,image):
       
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
   

# Dossier contenant les images de LBP

path = 'ImagesLBP/R1P8/'
fichiers = os.listdir(path)

for fichier in fichiers:
    image = cv2.imread(path+fichier, -1)
    
    seg = Segment()
    label, result, center = seg.kmeans(image)

    image_label_color = label2rgb(label)
    scipy.misc.imsave("ImagesKMeansLBPColor/"+fichier, image_label_color)

    
