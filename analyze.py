#!/usr/bin/env python

"""Analyze how well a segmentation is."""

import scipy
import logging
import time
import os.path
import struct

from tensorvision.analyze import merge_cms, get_accuracy, get_mean_accuracy
from tensorvision.analyze import get_mean_iou, get_frequency_weighted_iou
from tensorvision.analyze import get_confusion_matrix
from tensorvision.utils import load_segmentation_mask
from tensorvision.utils import overlay_segmentation


def evaluate(hypes,
             data,
             out_dir,
             model,
             elements,
             get_segmentation,
             verbose=True):
    """
    Analyze how well a model does on given data.

    Parameters
    ----------
    hypes : dict
        Hyperparameters (model specific information)
    data : tuple
        (x_files, y_files) where x_files and y_files are lists of strings
    out_dir : str
        Path to the directory where the output gets stored
    model : object
    elements : iterable
        A list / set or another iterable which contains the possible
        segmentation classes (commonly 0 and 1)
    get_segmentation : function
        Takes a string and a model the model as input and returns a numpy
        array. The string is the path to an image file.
    load_label_seg : function
        Takes a path as a string and returns a segmentation mask as a numpy
        array.
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
    i = 1
    for xfile, yfile in zip(*data):
        if verbose:
            logging.info("Segmentation of '%s'...", xfile)
            logging.info("    label='%s'...", yfile)
        t0 = time.time()
        segmentation = get_segmentation(hypes, xfile, model)
        t1 = time.time()
        times.append(t1 - t0)
        correct_seg = load_segmentation_mask(hypes, yfile)
        cm_tmp = get_confusion_matrix(correct_seg, segmentation, elements)
        color_changes = get_color_changes_dict(hypes)
        gen_img_and_overlay(xfile, segmentation, out_dir, color_changes, i)
        cm = merge_cms(cm, cm_tmp)
        if verbose:
            show_metrics(cm, indentation=4)
            print("    time: %0.4fs" % (t1 - t0))
        i += 1
    if verbose:
        show_metrics(cm)
        print("Average time: %0.4f" % (sum(times) / len(times)))
    return {'cm': cm, 'times': times}


def gen_img_and_overlay(source_file, segmentation, out_dir, color_changes, i):
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
    i : int
        Counter
    """
    # Paths
    basename = os.path.splitext(os.path.basename(source_file))[0]
    seg_name = "%i-%s-segmentation.png" % (i, basename)
    seg_path = os.path.join(out_dir, seg_name)
    overlay_name = "%i-%s-overlay.png" % (i, basename)
    overlay_path = os.path.join(out_dir, overlay_name)

    # Logic
    scipy.misc.imsave(seg_path, segmentation)  # Store segmentation
    input_image = scipy.misc.imread(source_file, mode='RGB')  # Load original
    overlayed = overlay_segmentation(input_image, segmentation, color_changes)
    scipy.misc.imsave(overlay_path, overlayed)
    logging.info("Created output for '%s'", source_file)


def get_color_changes_dict(hypes):
    """
    Get a dict which maps class indices to colors.

    It has the class index as key and the color (r, g, b, a) as value.

    Parameters
    ----------
    hypes : dict

    Returns
    -------
    dict
    """
    color_changes = {}
    for i, cl in enumerate(hypes['classes']):
        color_changes[i] = get_color_tuple(cl['output'])
    return color_changes


def get_color_tuple(color):
    """Get a (r, g, b, a) tuple as color."""
    if isinstance(color, tuple):
        return color
    if isinstance(color, basestring):
        if color.startswith('#'):
            color = color[1:]
            color = struct.unpack('BBBB', color.decode('hex'))
            return color
    raise Exception("Wrong color code %s." % str(color))


def show_metrics(cm, indentation=0):
    """Show a couple of metrics derived from the confusion matrix."""
    indent = " " * indentation
    print("%sAccuracy: %0.4f" % (indent, get_accuracy(cm)))
    print("%sMean Accuracy: %0.4f" % (indent, get_mean_accuracy(cm)))
    print("%sMean IoU: %0.4f" % (indent, get_mean_iou(cm)))
    print("%sFreq. weighted IoU: %0.4f" %
          (indent, get_frequency_weighted_iou(cm)))
    print("%s%s" % (indent, cm))
