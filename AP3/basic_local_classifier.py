#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A basic classifier which uses only local features."""

import os.path
from PIL import Image
import numpy
import scipy.misc
import scipy.ndimage

import logging
import sys
import time
import numpy as np

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


from keras.models import Sequential
from keras.layers import Dense, Dropout
import sklearn
from keras.models import model_from_yaml

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils import get_file_list
import analyze


def get_features(x, y, image):
    """Get features at position (x, y) from image."""
    height, width, _ = image.shape
    return (image[y][x][0],
            image[y][x][1],
            image[y][x][2],
            x,   # Model 2
            y)  # Model 2


def inputs(hypes, _, phase, data_dir):
    """
    Get data.

    Parameters
    ----------
    hypes : dict
    _ : ignore this
    phase : {'train', 'val'}
    data_dir : str

    Returns
    -------
    tuple
        (xs, ys), where xs and ys are lists of the same length.
        xs are paths to the input images and ys are paths to the expected
        output
    """
    x_files, y_files = get_file_list(phase, data_dir)
    x_files, y_files = sklearn.utils.shuffle(x_files,
                                             y_files,
                                             random_state=0)
    # x_files = x_files[:40]  # reducing data
    # y_files = y_files[:40]  # reducing data

    xs, ys = [], []
    for x, y in zip(x_files, y_files):
        logging.info("Read '%s' for data...", x)
        image = get_image(x, 'RGB')
        label = get_image(y, 'L')
        label = normalize_labels(label)
        im = Image.open(x, 'r')
        width, height = im.size
        for x in range(width):
            for y in range(height):
                image_val = get_features(x, y, image)
                label_val = (label[y][x][0] == 0)  # only 0 is background

                xs.append(image_val)
                ys.append(label_val)
    return xs, numpy.array(ys, dtype=int)


def get_image(image_path, force_mode=None):
    """Get a numpy array of an image so that one can access values[y][x]."""
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


def get_segmentation(image_path, model):
    """
    Get a segmentation.

    Path
    ----
    image_path : str
        Path to a file which gets segmented.
    model : object

    Returns
    -------
    Numpy array of the same width and height as input.
    """
    image = get_image(image_path, 'RGB')
    im = Image.open(image_path, 'r')
    width, height = im.size
    segmentation = numpy.zeros((height, width), dtype=int)

    x_test = []
    for x in range(width):
        for y in range(height):
            x_test.append(get_features(x, y, image))

    classes = model.predict_classes(numpy.array(x_test, dtype=int),
                                    batch_size=1024)
    i = 0
    for x in range(width):
        for y in range(height):
            segmentation[y][x] = classes[i]
            i += 1
    # segmentation = morphological_operations(segmentation)  # Model 3
    # Set all labels which are 1 to 0 and vice versa.
    segmentation = np.invert(segmentation.astype(bool)).astype(int)
    return segmentation


def morphological_operations(segmentation):
    """Apply morphological operations to improve the segmentation."""
    size = 3
    segmentation = scipy.ndimage.morphology.binary_erosion(segmentation,
                                                           iterations=size)
    segmentation = scipy.ndimage.morphology.binary_dilation(segmentation,
                                                            iterations=size)
    return segmentation


def normalize_labels(segmentation):
    """Set all labels which are not 0 to 1."""
    return segmentation.astype(bool).astype(int)


def load_label_seg(yfile):
    """
    Load the segmentation from a file.

    Parameters
    ----------
    yfile : str
        Path to a segmentation mask image.
    """
    correct_seg = get_image(yfile, 'L')
    correct_seg = normalize_labels(correct_seg)
    correct_seg = np.squeeze(correct_seg)
    return correct_seg


def main(data_dir):
    """Orchestrate."""
    model_file_path = 'basic_local_classifier.yaml'
    weights_file_path = 'basic_local_classifier.hdf5'

    color_changes = {(255, 255, 255): (0, 255, 0),
                     'default': (0, 0, 0)}

    if not os.path.isfile(model_file_path):
        # Get data
        x_train, y_train = inputs({}, None, 'train', data_dir)
        x_train = x_train
        y_train = y_train
        x_train, y_train = sklearn.utils.shuffle(x_train,
                                                 y_train,
                                                 random_state=0)

        # Reduce data
        logging.info("Start reducing data...")
        n = sum(y_train)
        print("n=%i" % n)
        true_count, false_count = 0, 0
        x_train_n, y_train_n = [], []
        for x, y in zip(x_train, y_train):
            if y == 1 and true_count < n:
                x_train_n.append(x)
                y_train_n.append(y)
                true_count += 1
            elif y == 0 and false_count < n:
                x_train_n.append(x)
                y_train_n.append(y)
                false_count += 1
            else:
                break
        x_train = numpy.array(x_train_n)
        y_train = numpy.array(y_train_n)
        logging.info("Reduced data data...")

        # Make model
        model = Sequential()
        model.add(Dense(64, input_dim=5, init='uniform', activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adagrad',  # rmsprop
                      metrics=['accuracy'])

        t0 = time.time()
        model.fit(x_train, y_train, batch_size=128, nb_epoch=1)
        t1 = time.time()
        print("Training Time: %0.4f" % (t1 - t0))

        # save as YAML
        yaml_string = model.to_yaml()
        with open(model_file_path, 'w') as f:
            f.write(yaml_string)
        model.save_weights(weights_file_path)

        # Evaluate
        data = get_file_list('test', data_dir)
        logging.info("Start segmentation")
        analyze.evaluate(data,
                         data_dir,
                         model,
                         elements=[0, 1],
                         load_label_seg=load_label_seg,
                         color_changes=color_changes,
                         get_segmentation=get_segmentation)
    else:
        with open(model_file_path) as f:
            yaml_string = f.read()
        model = model_from_yaml(yaml_string)
        model.load_weights(weights_file_path)
        model.compile(optimizer='adagrad', loss='binary_crossentropy')
        data = get_file_list('test', data_dir)
        analyze.evaluate(data,
                         data_dir,
                         model,
                         elements=[0, 1],
                         load_label_seg=load_label_seg,
                         color_changes=color_changes,
                         get_segmentation=get_segmentation)


def get_parser():
    """Get parser object for basic local classifier."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data",
                        dest="data",
                        help=("data directory which contains "
                              "'Segmentation_Rigid_Training'"),
                        required=True)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.data)
