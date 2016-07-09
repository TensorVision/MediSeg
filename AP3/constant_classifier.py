#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The constant classifier which always guesses the most common class."""

import json
import os
import logging
from PIL import Image
import numpy

from tensorvision.utils import load_segmentation_mask

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils import get_file_list
import analyze


def main(hypes_file, output_dir, override):
    """
    Orchestrate.

    Parameters
    ----------
    hypes_file : str
        Path to a JSON file
    output_dir : str
        Path where the output should be stored
    override : bool
        If True, then override the model if it exists.
    """
    # Load hyperparameters
    with open(hypes_file, 'r') as f:
        hypes = json.load(f)

    # Set serialization path
    base = os.path.dirname(hypes_file)
    model_file_path = os.path.join(base, '%s.json' % hypes['model']['name'])
    model_file_path = os.path.abspath(model_file_path)

    if not os.path.isfile(model_file_path) or override:
        if not os.path.isfile(model_file_path):
            logging.info("Did not find '%s'. Start training...",
                         model_file_path)
        else:
            logging.info("Override '%s'. Start training...",
                         model_file_path)

        # Get training data
        x_files, y_files = get_file_list(hypes, 'train')

        # "Train" "classifier" (it just counts the classes)
        model = {}
        for i in range(len(hypes['classes'])):
            model[i] = 0

        for y_file in y_files:
            logging.info("Read '%s'...", y_file)
            mask = load_segmentation_mask(hypes, y_file)
            for row in mask:
                for pixel in row:
                    model[pixel] += 1

        # save model as json file
        with open(model_file_path, 'w') as f:
            json.dump(model, f)
    else:
        # load model from json file
        with open(model_file_path) as f:
            model = json.load(f)
    # Evaluate
    data = get_file_list(hypes, 'test')
    analyze.evaluate(hypes,
                     data,
                     output_dir,
                     model,
                     elements=[0, 1],
                     get_segmentation=get_segmentation)


def get_segmentation(hypes, image_path, model):
    """
    Get a segmentation.

    Path
    ----
    hypes : dict
        Hyperparameters (model specific information)
    image_path : str
        Path to a file which gets segmented.
    model : object

    Returns
    -------
    Numpy array of the same width and height as input.
    """
    im = Image.open(image_path, 'r')
    width, height = im.size
    segmentation = numpy.zeros((height, width), dtype=int)

    most_common_class = sorted(model.items(),
                               key=lambda x: x[1],
                               reverse=True)[0][0]

    for x in range(width):
        for y in range(height):
            segmentation[y][x] = most_common_class
    return segmentation


def is_valid_file(parser, arg):
    """
    Check if arg is a valid file that already exists on the file system.

    Parameters
    ----------
    parser : argparse object
    arg : str

    Returns
    -------
    arg
    """
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


def get_parser():
    """Get parser object for basic local classifier."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out",
                        dest="data",
                        help=("output directory"),
                        required=True)
    parser.add_argument("--hypes",
                        dest="hypes_file",
                        help=("Configuration file in JSON format"),
                        type=lambda x: is_valid_file(parser, x),
                        metavar="FILE",
                        required=True)
    parser.add_argument("--override",
                        action="store_true", dest="override", default=False,
                        help="override old model, if it exists")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.hypes_file, args.data, args.override)
