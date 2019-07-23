#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# référence : http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html

import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
import cv2

PATCH_SIZE = 21

# open the camera image
image = cv2.imread('image.jpg',0)

# select some patches from grassy areas of the image
cloud_locations = [(474, 450), (10, 800), (460, 170), (450, 900)]
cloud_patches = []
for loc in cloud_locations:
    cloud_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

# select some patches from sky areas of the image
earth_locations = [(150, 500), (450, 700), (374, 300), (360, 918)]
earth_patches = []
for loc in earth_locations:
    earth_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
for patch in (cloud_patches + earth_patches):
    glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])


# create the figure
fig = plt.figure(figsize=(8, 8))

"""
# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest',
          vmin=0, vmax=255)
for (y, x) in cloud_patches:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in earth_patches:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')
"""

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(cloud_patches)], ys[:len(cloud_patches)], 'go',
        label='Cloud')
ax.plot(xs[len(cloud_patches):], ys[len(cloud_patches):], 'bo',
        label='Earth')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(cloud_patches):
    ax = fig.add_subplot(3, len(cloud_patches), len(cloud_patches)*1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Cloud %d' % (i + 1))

for i, patch in enumerate(earth_patches):
    ax = fig.add_subplot(3, len(earth_patches), len(earth_patches)*2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Earth %d' % (i + 1))


# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14)
plt.show()
