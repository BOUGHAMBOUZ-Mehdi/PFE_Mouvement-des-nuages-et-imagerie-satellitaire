#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Fonction permettant de faire une égalisation adapatative d'histogramme
# Permet d'obtenir des images avec un meilleur contraste, pour avoir de meilleurs résultats
# par la suite sur une segmentation de nuages

import cv2
import numpy as np
from skimage import exposure
import os


path = 'Base de données/wetransfer-063323/'
fichiers = os.listdir(path)

for fichier in fichiers:
    image = cv2.imread(path+fichier, -1)
    im_eq = exposure.equalize_adapthist(image)
    im_eq = im_eq * 255
    
    # Enregistrement des résultats
    cv2.imwrite("ImagesEgalisationAdapt/"+fichier, im_eq)