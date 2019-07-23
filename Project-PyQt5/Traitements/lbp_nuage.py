#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Fichier permettant de lire et d'afficher des images TIFF, 
# et en calculer une carte de style LBP 

import cv2
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import numpy as np
import os
# Utilisation d'un filtre anisotrope

import medpy.filter.smoothing as mfs


path = './Base de données/wetransfer-063323/'
fichiers = os.listdir(path)

for fichier in fichiers:
    image = cv2.imread(path+fichier, -1)

    # Prétraitement = filtre anisotrope

    image = mfs.anisotropic_diffusion(image, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1)
    #plt.imshow(image, cmap=plt.get_cmap('gray'))
    #plt.show()
    
    
    # Paramètres du calcul de LBP 
    radius = 1
    n_points = 8 * radius
    L = local_binary_pattern(image, n_points, radius, method='default')

    cv2.imwrite("ImagesLBP/R1P8/"+fichier, L.astype("uint16"))
