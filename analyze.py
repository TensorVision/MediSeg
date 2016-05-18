#!/usr/bin/env python

"""Analyze how well a segmentation is."""

import numpy as np
import scipy
import logging
import time
import os.path

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from seg_utils import overlay_images, replace_colors


def evaluate(data,
             data_dir,
             model,
             elements,
             get_segmentation,
             load_label_seg,
             color_changes,
             verbose=True):
    """
    Analyze how well a model does on given data.

    Parameters
    ----------
    data : tuple
        (x_files, y_files) where x_files and y_files are lists of strings
    data_dir : str
        Path to the directory of the data
    model : object
    elements : iterable
        A list / set or another iterable which contains the possible
        segmentation classes (commonly 0 and 1)
    get_segmentation : function
        Takes a string and a model the model as input and returns a numpy
        array. The string is the path to an image file.
    verbose : bool
        Print messages when in verbose mode.
    """
    # Initialize confusion matrix
    cm = {}
    for i in elements:
        cm[i] = {}
        for j in elements:
            cm[i][j] = 0
    # Initialize timings for segmentation
    times = []
    for xfile, yfile in zip(*data):
        if verbose:
            logging.info("Segmentation of '%s'...", xfile)
            logging.info("    label='%s'...", yfile)
        t0 = time.time()
        segmentation = get_segmentation(xfile, model)
        t1 = time.time()
        times.append(t1 - t0)
        correct_seg = load_label_seg(yfile)
        cm_tmp = get_confusion_matrix(correct_seg, segmentation)
        out_dir = os.path.join(data_dir, 'out')
        gen_img_and_overlay(xfile, segmentation, out_dir, color_changes)
        cm = merge_cms(cm, cm_tmp)
        if verbose:
            show_metrics(cm, indentation=4)
            print("    time: %0.4fs" % (t1 - t0))
    if verbose:
        show_metrics(cm)
        print("Average time: %0.4f" % (sum(times) / len(times)))
    return {'cm': cm, 'times': times}


def gen_img_and_overlay(source_file, segmentation, out_dir, color_changes):
    """
    Generate the segmentation image and the overlay.

    Parameters
    ----------
    source_file : str
        Name of the file of which the segmentation was done.
    segmentation : numpy array
    out_dir : str
        Directory in which the segmentation image and the overlay should be
        put.
    color_changes : dict
        Encode which classes (key) of 'segmentation' should get which color
        (value). The key and value have to be in (0, 0, 0)
    """
    basename = os.path.splitext(os.path.basename(source_file))[0]
    seg_name = "%s-segmentation.png" % basename
    seg_path = os.path.join(out_dir, seg_name)
    overlay_name = "%s-overlay.png" % basename
    overlay_path = os.path.join(out_dir, overlay_name)
    scipy.misc.imsave(seg_path, segmentation)
    segmentation = replace_colors(segmentation, color_changes)
    overlay_images(source_file, segmentation, overlay_path)
    logging.info("Created output for '%s'", source_file)


def get_confusion_matrix(correct_seg, segmentation, elements=None):
    """
    Get the accuracy data of a segmentation called 'confuscation matrix'.

    The confuscation matrix is a detailed count of which classes i were
    classifed as classes j, where i and j take all (elements) names.

    Parameters
    ----------
    correct_seg : numpy array
        Representing the ground truth.
    segmentation : numpy array
    elements : iterable
        A list / set or another iterable which contains the possible
        segmentation classes (commonly 0 and 1)

    Returns
    -------
    dict
        A confusion matrix m[correct][classified] = number of pixels in this
        category.
    """
    height, width = correct_seg.shape

    # Get classes
    if elements is None:
        elements = set(np.unique(correct_seg))
        elements = elements.union(set(np.unique(segmentation)))
        logging.debug("elements parameter not given to get_confusion_matrix")
        logging.debug("  assume '%s'", elements)

    # Initialize confusion matrix
    confusion_matrix = {}
    for i in elements:
        confusion_matrix[i] = {}
        for j in elements:
            confusion_matrix[i][j] = 0

    for x in range(width):
        for y in range(height):
            confusion_matrix[correct_seg[y][x]][segmentation[y][x]] += 1
    return confusion_matrix


def get_accuracy(n):
    """Get the accuracy from a confusion matrix n."""
    return (float(n[0][0] + n[1][1]) /
            (n[0][0] + n[1][1] + n[0][1] + n[1][0]))


def get_mean_accuracy(n):
    """Get the mean accuracy from a confusion matrix n."""
    t = []
    k = len(n[0])
    for i in range(k):
        t.append(sum([n[i][j] for j in range(k)]))
    return (1.0 / k) * sum([float(n[i][i]) / t[i] for i in range(k)])


def get_mean_iou(n):
    """Get mean intersection over union from a confusion matrix n."""
    t = []
    k = len(n[0])
    for i in range(k):
        t.append(sum([n[i][j] for j in range(k)]))
    return (1.0 / k) * sum([float(n[i][i]) / (t[i] - n[i][i] +
                            sum([n[j][i] for j in range(k)]))
                            for i in range(k)])


def get_frequency_weighted_iou(n):
    """Get frequency weighted intersection over union."""
    t = []
    k = len(n[0])
    for i in range(k):
        t.append(sum([n[i][j] for j in range(k)]))
    a = sum(t)**(-1)
    b = sum([(t[i] * n[i][i]) /
             (t[i] - n[i][i] + sum([n[j][i] for j in range(k)]))
             for i in range(k)])
    return a * b


def merge_cms(cm1, cm2):
    """Merge two confusion matrices."""
    assert 0 in cm1
    assert len(cm1[0]) == len(cm2[0])

    cm = {}
    k = len(cm1[0])
    for i in range(k):
        cm[i] = {}
        for j in range(k):
            cm[i][j] = cm1[i][j] + cm2[i][j]

    return cm


def show_metrics(cm, indentation=0):
    """Show a couple of metrics derived from the confusion matrix."""
    indent = " " * indentation
    print("%sAccuracy: %0.4f" % (indent, get_accuracy(cm)))
    print("%sMean Accuracy: %0.4f" % (indent, get_mean_accuracy(cm)))
    print("%sMean IoU: %0.4f" % (indent, get_mean_iou(cm)))
    print("%sFreq. weighted IoU: %0.4f" %
          (indent, get_frequency_weighted_iou(cm)))
