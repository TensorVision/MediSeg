#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepare the data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os.path
from sklearn.cross_validation import train_test_split
import sys
import zipfile


import logging
import utils

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def make_val_split(data_folder, test_size=20):
    """
    Split the images in train and test.

    Assumes a file all.txt in data_folder.

    Parameters
    ----------
    data_folder : str
        Path to the folder with the data
    """
    all_file = "all.txt"
    train_file = "train.txt"
    test_file = "val.txt"

    filename = os.path.join(data_folder, all_file)
    assert os.path.exists(filename), ("File not Found %s"
                                      % filename)

    files = [line for line in open(filename)]
    train, test = train_test_split(files, test_size=test_size)

    train_file = os.path.join(data_folder, train_file)
    test_file = os.path.join(data_folder, test_file)

    with open(train_file, 'w') as file:
        for label in train:
            file.write(label)

    with open(test_file, 'w') as file:
        for label in test:
            file.write(label)


def main():
    """Run the generation of required intermediate files."""
    data_dir = utils.cfg.data_dir
    zip_file = "Segmentation_Rigid_Training.zip"
    zip_file = os.path.join(data_dir, zip_file)
    if not os.path.exists(zip_file):
        logging.error("File not found: %s", zip_file)
        sys.exit(1)
    zipfile.ZipFile(zip_file, 'r').extractall(data_dir)

    make_val_split(data_dir)


def get_parser():
    """Get parser object for prepare_data.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    return parser


if __name__ == '__main__':
    _ = get_parser().parse_args()
    main()
