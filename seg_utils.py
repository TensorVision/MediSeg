#!/usr/bin/env python

"""Utility functions for segmentation tasks."""

from PIL import Image
import numpy


def get_image(image_path, force_mode=None):
    """
    Get a numpy array of an image so that one can access values[y][x].

    Parameters
    ----------
    image_path : str
    force_mode : {None, 'L', 'RGB', 'RGBA', ...}

    Returns
    -------
    numpy array
    """
    image = Image.open(image_path, 'r')
    if force_mode is not None:
        image = image.convert(mode=force_mode)
    width, height = image.size
    pixel_values = list(image.getdata())
    if image.mode == 'RGB':
        channels = 3
    elif image.mode == 'RGBA':
        image = Image.open(image_path).convert('RGB')
        pixel_values = list(image.getdata())
        channels = 3
    elif image.mode == 'L':
        channels = 1
    else:
        print("Unknown mode: %s" % image.mode)
        print("image_path: %s" % image_path)
        return None
    pixel_values = numpy.array(pixel_values).reshape((height, width, channels))
    return pixel_values
