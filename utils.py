#!/usr/bin/env python

"""Utility functions for the MediSeg project."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json


def get_file_list(hypes, phase):
    """
    Get a list of tuples (input, output) of files.

    Parameters
    ----------
    hypes : dict
    phase : {'train', 'test'}
    """
    x_files, y_files = [], []
    jsonfile = hypes['data'][phase]

    with open(jsonfile) as data_file:
        data = json.load(data_file)

    for item in data:
        x_files.append(item['raw'])
        y_files.append(item['mask'])
    return x_files, y_files
