#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A basic classifier which uses only local features."""

import os.path
import PIL.ImageEnhance
from PIL import Image
import numpy
import scipy.misc
import scipy.ndimage

import logging
import sys
import time
import numpy as np
import json

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.optimizers
import sklearn
from keras.models import model_from_yaml
from keras.preprocessing.image import img_to_array

from skimage.segmentation import quickshift, slic

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils import get_file_list
import analyze
from seg_utils import get_image


def get_features(x, y, image, model_nr=2):
    """Get features at position (x, y) from image."""
    height, width, _ = image.shape
    p = get_pos_colors(image, x, y)
    if model_nr in [1, "1.1"]:
        return p
    elif model_nr in [2, 3]:
        return (p[0], p[1], p[2], x, y)
    elif model_nr in [4]:
        left = get_pos_colors(image, x - 1, y)
        return (p[0], p[1], p[2], left[0], left[1], left[2], x, y)
    elif model_nr in [5]:
        left = get_pos_colors(image, x - 1, y)
        right = get_pos_colors(image, x + 1, y)
        top = get_pos_colors(image, x, y + 1)
        bottom = get_pos_colors(image, x, y - 1)
        return (p[0], p[1], p[2],
                left[0], left[1], left[2],
                right[0], right[1], right[2],
                top[0], top[1], top[2],
                bottom[0], bottom[1], bottom[2])
    else:
        print("model_nr '%s' unknown" % str(model_nr))
        sys.exit(-1)


def get_pos_colors(image, x, y):
    """Get the color at a position or 0-vector, if the position is invalid."""
    if x > 0 and y > 0 and len(image) > y and len(image[0]) > x:
        return (image[y][x][0], image[y][x][1], image[y][x][2])
    else:
        return (0, 0, 0)


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
    x_files, y_files = get_file_list(hypes, 'train')
    x_files, y_files = sklearn.utils.shuffle(x_files,
                                             y_files,
                                             random_state=0)

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
                image_val = get_features(x, y, image, hypes['model_nr'])
                label_val = (label[y][x][0] == 0)  # only 0 is background

                xs.append(image_val)
                ys.append(label_val)
    return xs, numpy.array(ys, dtype=int)


def shuffle_in_unison_inplace(a, b):
    """Shuffle both, a and b, the same way."""
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]


def generate_training_data(hypes, x_files, y_files):
    """
    Generate training data.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    x_files : list
        Paths to raw data files
    y_files : list
        Paths to segmentation masks

    Yields
    ------
    tuple
        (xs, ys) - training batch of feature list xs and label list ys
    """
    x_files, y_files = sklearn.utils.shuffle(x_files,
                                             y_files,
                                             random_state=0)
    i = 0
    xs, ys = get_traindata_single_file(hypes, x_files[i], y_files[i])
    i = (i + 1) % len(x_files)
    while True:
        while len(xs) < hypes['solver']['batch_size']:
            xs_tmp, ys_tmp = get_traindata_single_file(hypes,
                                                       x_files[i],
                                                       y_files[i])
            i = (i + 1) % len(x_files)
            xs = np.concatenate((xs, xs_tmp), axis=0)
            ys = np.concatenate((ys, ys_tmp), axis=0)
            if hypes['training']['make_equal']:
                xs, ys = reduce_data_equal(xs, ys)

        # xs, ys = shuffle_in_unison_inplace(xs, ys)
        # print("sum(ys)=%i / %i" % (np.sum(ys), len(ys) - np.sum(ys)))
        # print("sum(ys[s])=%i" % np.sum(ys[:hypes['solver']['batch_size']]))
        yield (xs[:hypes['solver']['batch_size']],
               ys[:hypes['solver']['batch_size']])
        xs = xs[hypes['solver']['batch_size']:]
        ys = ys[hypes['solver']['batch_size']:]


def get_traindata_single_file(hypes, x, y):
    """Get trainingdata for a single file x with segmentation file y."""
    xs, ys = [], []
    logging.info("Read '%s' for data...", x)
    image = get_image(x, 'RGB')
    label = get_image(y, 'L')
    label = normalize_labels(label)
    im = Image.open(x, 'r')
    width, height = im.size
    for x in range(width):
        for y in range(height):
            image_val = get_features(x, y, image, hypes['model_nr'])
            label_val = (label[y][x][0] == 0)  # only 0 is background

            xs.append(image_val)
            ys.append(label_val)
    return numpy.array(xs), numpy.array(ys, dtype=int)


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
    image = get_image(image_path, 'RGB')

    # Preprocess
    # import skimage.exposure
    # image = skimage.exposure.equalize_hist(image)
    # image = Image.fromarray(image, 'RGB')
    # converter = PIL.ImageEnhance.Color(image)
    # image = converter.enhance(2)
    # image = img_to_array(image)
    # scipy.misc.imshow(image)

    im = Image.open(image_path, 'r')
    width, height = im.size
    segmentation = numpy.zeros((height, width), dtype=int)

    x_test = []
    for x in range(width):
        for y in range(height):
            x_test.append(get_features(x, y, image, hypes['model_nr']))

    classes = model.predict_classes(numpy.array(x_test, dtype=int),
                                    batch_size=1024)
    i = 0
    for x in range(width):
        for y in range(height):
            segmentation[y][x] = classes[i]
            i += 1
    if hypes['model_nr'] in [3, "1.1"]:
        segmentation = morphological_operations(segmentation)
    # Set all labels which are 1 to 0 and vice versa.
    segmentation = np.invert(segmentation.astype(bool)).astype(int)
    # segmentation = superpixel_majority_vote(image, segmentation)
    return segmentation


def superpixel_majority_vote(image, segmentation):
    """Mark superpixels by majority vote."""
    image = image.astype(float)
    segments = quickshift(image, ratio=0.5, max_dist=10, sigma=1.0)
    # segments = slic(image, n_segments=50, compactness=20)
    # watershed -
    ("http://scikit-image.org/docs/dev/auto_examples/segmentation/"
     "plot_marked_watershed.html")
    # http://scikit-image.org/docs/dev/auto_examples/
    height, width = segments.shape
    segment_count = {}
    for x in range(width):
        for y in range(height):
            s = segments[y][x]
            if s not in segment_count:
                segment_count[s] = {0: 0, 1: 0}  # binary
            segment_count[s][segmentation[y][x]] += 1
    for x in range(width):
        for y in range(height):
            s = segments[y][x]
            class_ = int(segment_count[s][1] > segment_count[s][0])
            segmentation[y][x] = class_
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


def main(hypes_file, data_dir, override):
    """Orchestrate."""
    with open(hypes_file, 'r') as f:
        hypes = json.load(f)
    if 'training' not in hypes:
        hypes['training'] = {}
    if 'make_equal' not in hypes['training']:
        hypes['training']['make_equal'] = False

    base = os.path.dirname(hypes_file)
    model_file_path = os.path.join(base, '%s.yaml' % hypes['model']['name'])
    model_file_path = os.path.abspath(model_file_path)
    weights_file_path = os.path.join(base, '%s.hdf5' % hypes['model']['name'])
    weights_file_path = os.path.abspath(weights_file_path)

    color_changes = {0: (0, 0, 0, 0),
                     1: (0, 255, 0, 127),
                     'default': (0, 0, 0, 0)}

    if not os.path.isfile(model_file_path) or override:
        if not os.path.isfile(model_file_path):
            logging.info("Did not find '%s'. Start training...",
                         model_file_path)
        else:
            logging.info("Override '%s'. Start training...",
                         model_file_path)

        # Get data
        # x_files, y_files = inputs(hypes, None, 'train', data_dir)
        x_files, y_files = get_file_list(hypes, 'train')
        x_files, y_files = sklearn.utils.shuffle(x_files,
                                                 y_files,
                                                 random_state=0)

        x_train, y_train = get_traindata_single_file(hypes,
                                                     x_files[0],
                                                     y_files[0])

        nb_features = x_train[0].shape[0]
        logging.info("Input gets %i features", nb_features)

        # Make model
        model = Sequential()
        model.add(Dense(64,
                  input_dim=nb_features,
                  init='uniform',
                  activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adagrad',  # rmsprop
                      metrics=['accuracy'])

        generator = generate_training_data(hypes, x_files, y_files)
        t0 = time.time()
        sep = hypes['solver']['samples_per_epoch']
        if True:
            model.fit_generator(generator,
                                samples_per_epoch=sep,
                                nb_epoch=hypes['solver']['epochs'],
                                verbose=1,
                                # callbacks=[callb],
                                validation_data=(x_train, y_train))
        else:
            logging.info("Fit with .fit")
            x_train, y_train = inputs(hypes, None, 'train', data_dir)
            model.fit(x_train, y_train, batch_size=128, nb_epoch=1)
        t1 = time.time()
        print("Training Time: %0.4f" % (t1 - t0))

        # save as YAML
        yaml_string = model.to_yaml()
        with open(model_file_path, 'w') as f:
            f.write(yaml_string)
        model.save_weights(weights_file_path)

        # Evaluate
        data = get_file_list(hypes, 'test')
        logging.info("Start segmentation")
        analyze.evaluate(hypes,
                         data,
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
        data = get_file_list(hypes, 'test')
        analyze.evaluate(hypes,
                         data,
                         data_dir,
                         model,
                         elements=[0, 1],
                         load_label_seg=load_label_seg,
                         color_changes=color_changes,
                         get_segmentation=get_segmentation)


def reduce_data_equal(x_train, y_train, max_per_class=None):
    """
    Reduce the amount of data to get the same number per class.

    This script assumes that y_train is a list of binary labels {0, 1}.
    """
    n = min(sum(y_train), abs(len(y_train) - sum(y_train)))
    if max_per_class is not None:
        n = min(n, max_per_class)
    true_count, false_count = 0, 0
    x_train_n, y_train_n = [], []
    x_train = list(x_train)
    y_train = list(y_train)
    for x, y in zip(x_train, y_train):
        if y == 1 and true_count < n:
            x_train_n.append(x)
            y_train_n.append(y)
            true_count += 1
        elif y == 0 and false_count < n:
            x_train_n.append(x)
            y_train_n.append(y)
            false_count += 1
    x_train = numpy.array(x_train_n)
    y_train = numpy.array(y_train_n)
    return x_train, y_train


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
