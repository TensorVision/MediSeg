#!/usr/bin/env python

"""Get superpixels of an image."""

from skimage.segmentation import slic, quickshift  # , felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.data import coffee

import matplotlib.pyplot as plt

import Image
import numpy

img = coffee()
drip = ("/home/moose/GitHub/MediSeg/DATA/Segmentation_Rigid_Training/"
        "Training/OP4/Raw/")


for i in range(10, 40):
    im = Image.open(drip + "img_%i_raw.png" % i)
    img = numpy.array(im)

    w, h, d = original_shape = tuple(img.shape)

    segments = slic(img,
                    n_segments=50,
                    compactness=20)
    b1 = mark_boundaries(img, segments, color=(1, 1, 0))
    segments = quickshift(img, ratio=0.5, max_dist=10, sigma=0.0)
    b2 = mark_boundaries(img, segments, color=(1, 1, 0))
    segments = quickshift(img, ratio=0.5, max_dist=10, sigma=0.1)
    b3 = mark_boundaries(img, segments, color=(1, 1, 0))
    segments = quickshift(img, ratio=0.5, max_dist=10, sigma=1.0)
    b4 = mark_boundaries(img, segments, color=(1, 1, 0))
    # Display all results, alongside original image
    fig = plt.figure()

    a = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(b1)
    a.set_title('slic')

    a = fig.add_subplot(2, 2, 2)
    imgplot = plt.imshow(b2)
    a.set_title('20,30')

    a = fig.add_subplot(2, 2, 3)
    imgplot = plt.imshow(b3)
    a.set_title('30,20')

    a = fig.add_subplot(2, 2, 4)
    imgplot = plt.imshow(b4)
    a.set_title('30,30')

    # plt.figure(1)
    # plt.clf()
    # ax = plt.axes([0, 0, 1, 1])
    # plt.axis('off')
    # plt.title('Original image (96,615 colors)')
    # plt.imshow(b1)

    # plt.figure(2)
    # plt.clf()
    # ax = plt.axes([0, 0, 1, 1])
    # plt.axis('off')
    # plt.title('Quantized image (64 colors, K-Means)')
    # plt.imshow(b2)

    # n_colors = 500
    # kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(img)
    # labels = kmeans.predict(img)
    # plt.figure(3)
    # plt.clf()
    # ax = plt.axes([0, 0, 1, 1])
    # plt.axis('off')
    # plt.title('Quantized image (64 colors, Random)')
    # plt.imshow(b3)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()
