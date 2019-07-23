#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Fonction permettant, à partir d'une image segmentée de nuages, d'en
# calculer les composantes connexes (= nuage) et d'étudier le mouvement des nuages par le calcul 
# de caractéristiques (centre de gravité, taille...)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Insertion des rectangles englobants
import matplotlib.patches as mpatches

# Calcul des parties connexes
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import scipy.ndimage.measurements as snm
import os


path = 'ImagesExtractionKMeans/'
fichiers = os.listdir(path)
fichiers = sorted(fichiers)


# Tableaux réunissant les centres de gravité et les tailles de tous les nuages, sur toutes 
# nos images
all_centroid = []
all_tailles = []


for fichier in fichiers:
        image = cv2.imread(path+fichier, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Enregistrement des images 
        #cv2.imwrite("ImagesKMeansOpen/"+fichier, image)
        
        # "Labellisation" des parties connexes de l'image
        label_image = label(image)
        image_label_color = label2rgb(label_image, image=image)

        # Affichage de l'image et des rectangles englobants
        
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.imshow(image_label_color)
        
        centroid = []
        tailles = []
        
        for region in regionprops(label_image):
            # Nous considérons uniquement les nuages ayant une certaine aire
            if region.area >= 3000:

                taille = region.area
                tailles.append(taille)
                # Définition de la boîte englobante
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                i = region.label
                centre = snm.center_of_mass(image, label_image, i)
                centroid.append(centre)
                #ax.plot(centre[0], centre[1], 'b', markersize=12)
                
        all_centroid.append(centroid)
        all_tailles.append(tailles)
        
        ax.set_axis_off()
        plt.tight_layout()
        #cv2.imwrite("ImagesKMeans/"+fichier, image_label_color.astype("uint8"))
        fig.savefig("ImagesLabel/"+fichier)
        plt.show()
        
# On peut alors par exemple considérer un nuage, extraire son centre de gravité et sa taille
# et afficher leur progression sur une courbe
        
"""
tableau1 = all_centroid[11:59]
tableau2 = all_tailles[11:59]
t1 = []
t2 = []
for i in range (0,len(tableau1)):
    t1.append(tableau1[i][0])
    t2.append(tableau2[i][0])

#t2 = tableau1[1][0]
t1 = np.column_stack((t1))

plt.title('Evolution du centre de gravité du nuage au cours de la séquence')
plt.xlabel("Largeur de l'image")
plt.ylabel("Hauteur de l'image")
plt.ylim(max(t1[0]), 0)
plt.plot(t1[1], t1[0])
plt.show()
#plt.savefig('centre.png')

plt.title('Evolution de la taille du nuage au cours de la séquence')
plt.xlabel("Séquence d'images")
plt.ylabel("Taille du nuage")
plt.plot(t2)
plt.show()
#plt.savefig('taille.png')
"""
