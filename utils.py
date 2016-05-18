#!/usr/bin/env python

"""Utility functions for the MediSeg project."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import isfile, join


def get_file_list(phase, data_dir):
    """
    Get a list of tuples (input, output) of files.

    Parameters
    ----------
    phase : {'train', 'test'}
    data_dir : str
        Path to a directory which contains 'Segmentation_Rigid_Training'
    """
    data_dir = os.path.join(data_dir, 'Segmentation_Rigid_Training')
    data_dir = os.path.join(data_dir, 'Training')
    ops = []
    if phase == 'train':
        ops.append(os.path.join(data_dir, 'OP1'))
        ops.append(os.path.join(data_dir, 'OP2'))
        ops.append(os.path.join(data_dir, 'OP3'))
    else:
        ops.append(os.path.join(data_dir, 'OP4'))

    x_files, y_files = [], []
    for op in ops:
        image_dir = os.path.join(op, "Raw")
        label_dir = os.path.join(op, "Masks")
        images = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                  if isfile(join(image_dir, f))]
        labels = [os.path.join(label_dir, f) for f in os.listdir(label_dir)
                  if isfile(join(label_dir, f)) and "instrument" in f]
        for x, y in zip(sorted(images), sorted(labels)):
            x_files.append(x)
            y_files.append(y)
    return x_files, y_files
