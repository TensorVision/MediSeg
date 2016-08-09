#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Get the correlation of F1 score and accuracy."""

import os
import csv
from scipy.stats.stats import pearsonr
from os import listdir
from os.path import isfile, join


def main(filename):
    onlyfiles = [f for f in listdir(filename) if isfile(join(filename, f)) and f.endswith(".csv")]
    f1s = []
    acs = []
    for filename in onlyfiles:
        print(filename)
        with open(filename, 'rb') as csvfile:
            next(csvfile)
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')

            for row in spamreader:
                f1, accuracy = row[1], row[2]
                f1s.append(float(f1))
                acs.append(float(accuracy))
    print(pearsonr(f1s, acs))


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
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="CSV file with data",
                        metavar="FILE")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.filename)
