#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Make all images in a folder binary."""

import os
from PIL import Image
import numpy
import scipy.misc
from keras.preprocessing.image import img_to_array


def main(directory):
    """Find all png images in directory and make them binary."""
    files = [os.path.join(directory, f)
             for f in sorted(os.listdir(directory))
             if f.endswith('.png')]

    for file_path in files:
        img = get_image(file_path, 'L')
        img = numpy.squeeze(img_to_array(img))
        img = normalize_labels(img)
        scipy.misc.imsave(file_path, img)


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


def normalize_labels(segmentation):
    """Set all labels which are not 0 to 1."""
    return segmentation.astype(bool).astype(int)


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--directory",
                        dest="directory",
                        help="directory of images which get binarized",
                        metavar="DIR",
                        required=True)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.directory)
