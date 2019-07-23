#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Fonction permettant de faire la soustraction à partir des images segmentées de nuages
# (images issues de l'extraction par K-Means)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


path = 'ImagesExtractionKMeans/'
fichiers = os.listdir(path)
fichiers = sorted(fichiers)


for i in range(0, len(fichiers) - 1):
    
    image1 = cv2.imread(path+fichiers[i], -1)
    #print(image1)
    image2 = cv2.imread(path+fichiers[i+1], -1)
    
    sous = image2 - image1
    cv2.imwrite("ImagesSoustraction/"+fichiers[i], sous)
    



