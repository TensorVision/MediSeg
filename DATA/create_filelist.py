#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create filelists to use for training and testing."""

import os
import json
import sys


def main(data_path, store_txt):
    """Create the filelists."""
    train_data, test_data = [], []
    for opi in [i for i in range(1, 6 + 1)]:
        op = "OP%i" % opi
        directory = os.path.abspath(os.path.join(data_path, op))
        if not os.path.isdir(directory):
            print("[ERROR] Directory '%s' was not found." % directory)
            sys.exit(-1)
        files_data = [os.path.join(directory, f)
                      for f in sorted(os.listdir(directory))
                      if f.endswith('.png')]
        for i, el in enumerate(sorted(files_data)):
            if i < 40 * 2 and opi <= 4:
                if i % 2 == 0:
                    train_data.append({'raw': el, 'mask': None})
                else:
                    train_data[-1]['mask'] = el
            else:
                if i % 2 == 0:
                    test_data.append({'raw': el, 'mask': None})
                else:
                    test_data[-1]['mask'] = el

    # write data
    if not store_txt:
        with open('trainfiles.json', 'w') as outfile:
            json.dump(train_data, outfile, indent=4, separators=(',', ': '))

        with open('testfiles.json', 'w') as outfile:
            json.dump(test_data, outfile, indent=4, separators=(',', ': '))
    else:
        with open('train.txt', 'w') as outfile:
            for data in train_data:
                outfile.write("%s %s\n" % (data['raw'], data['mask']))

        with open('val.txt', 'w') as outfile:
            for data in train_data:
                outfile.write("%s %s\n" % (data['raw'], data['mask']))


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dir",
                        dest="data_path",
                        help="directory where the OP1 - OP6 directorys are",
                        metavar="DIRECTORY",
                        required=True)
    parser.add_argument("--txt",
                        action="store_true",
                        dest="store_txt",
                        default=False,
                        help="Store files as TXT instead of JSON")
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args.data_path, args.store_txt)
