#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Get patches of the raw image as input."""

import json
import os
import numpy as np
import logging
import itertools
import random
import sys
import time

from PIL import Image
import numpy
import math

# ML
from keras.models import model_from_yaml
from keras.preprocessing.image import img_to_array
import scipy.misc
import sklearn

# Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, Reshape
from keras.layers import MaxPooling2D, Dropout
import keras.optimizers

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils import get_file_list

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)
import analyze
from seg_utils import get_image


def generate_batch(hypes, data_dir, phase):
    """Generate patches."""
    x_files, y_files = get_file_list(phase, data_dir)
    x_files, y_files = sklearn.utils.shuffle(x_files,
                                             y_files,
                                             random_state=0)
    batch_x, batch_y = [], []
    while True:
        for x, y in zip(x_files, y_files):
            logging.info("Read '%s' for data...", x)
            image = get_image(x, 'RGB')
            label = get_image(y, 'L')
            label = normalize_labels(label)
            im = Image.open(x, 'r')
            width, height = im.size
            image_vals = get_features(hypes, image, 'data')
            label_vals = get_features(hypes, label, 'label')
            for patch, label_ in zip(image_vals, label_vals):
                patch = img_to_array(patch)
                label_ = img_to_array(label_)
                _, w, h = label_.shape
                label_ = label_.reshape((w, h))
                if phase == 'val' and 1.0 not in label_:
                    continue
                # scipy.misc.imshow(patch)
                # scipy.misc.imshow(label_)
                batch_x.append(patch)
                batch_y.append(label_)  # .flatten()
                if len(batch_x) == hypes['solver']['batch_size']:
                    yield (np.array(batch_x), np.array(batch_y))
                    batch_x, batch_y = [], []


def get_features(hypes, image, img_type):
    """
    Get features from image.

    Parameters
    ----------
    hypes : dict
        Hyperparamters such as arch>stride, arch>patch_size
    image : numpy array
    img_type : {'data', 'label'}

    Yields
    ------
    numpy array
        patch of size stride x stride from image
    """
    width, height, _ = image.shape
    stride = hypes['arch']['stride']
    patch_size = hypes['arch']['patch_size']
    if img_type == 'data':
        window_width = patch_size
        left_pad = 0
        top_pad = 0
    else:
        window_width = stride
        left_pad = int(math.floor(patch_size - stride) / 2)
        top_pad = left_pad
    width, height, _ = image.shape
    for x in range(left_pad, width - window_width, stride):
        for y in range(top_pad, height - window_width, stride):
            res = image[x:(x + window_width), y:(y + window_width)]
            assert res.shape[0] == window_width, "width"
            assert res.shape[1] == window_width, "heigth"
            yield res


def normalize_labels(segmentation):
    """Set all labels which are not 0 to 1."""
    return segmentation.astype(bool).astype(int)


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
            image_val = get_features(x, y)
            label_val = (label[y][x][0] == 0)  # only 0 is background

            xs.append(image_val)
            ys.append(label_val)
    return numpy.array(xs), numpy.array(ys, dtype=int)


def main(hypes_file, data_dir, override):
    """Orchestrate."""
    with open(hypes_file, 'r') as f:
        hypes = json.load(f)

    model_file_path = '%s.yaml' % hypes['model']['name']
    weights_file_path = '%s.hdf5' % hypes['model']['name']

    color_changes = {(255, 255, 255): (0, 255, 0),
                     'default': (0, 0, 0)}

    if not os.path.isfile(model_file_path) or override:
        patch_size = hypes['arch']['patch_size']
        img_channels = hypes['arch']['num_channels']
        nb_out = hypes['arch']['stride']**2  # number of classes is 2

        model = Sequential()
        model.add(Convolution2D(64, 3, 3, border_mode='valid',
                                init='glorot_normal',
                                activation='sigmoid',
                                input_shape=(img_channels,
                                             patch_size,
                                             patch_size)))
        model.add(Convolution2D(32, 3, 3,
                                activation='relu',
                                init='glorot_normal'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        # model.add(Convolution2D(64, 3, 3, border_mode='same'))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(64, 3, 3))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add(Flatten())
        # model.add(Dense(64, activation='sigmoid'))
        # # model.add(Dropout(0.5))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(nb_out,
                        activation='sigmoid',
                        init='glorot_normal'))
        model.add(Reshape((hypes['arch']['stride'], hypes['arch']['stride'])))

        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        opt = keras.optimizers.Adadelta(lr=hypes['solver']['learning_rate'],
                                        rho=0.95,
                                        epsilon=1e-08)
        model.compile(loss=hypes['solver']['loss'],
                      optimizer=opt)  # hypes['solver']['optimizer']

        # while 1:
        #     b = generate_batch(hypes, data_dir, 'train')

        # for e in range(10):
        #     print 'Epoch', e
        #     batches = 0
        #     for X_batch, Y_batch in generate_batch(hypes, data_dir, 'train'):
        #         Y_batch = np.reshape(Y_batch, (-1, 400))
        #         loss = model.fit(X_batch,
        #                          Y_batch,
        #                          batch_size=hypes['solver']['batch_size'])
        #         print(loss)
        #         batches += 1
        #         if e > 2:
        #             # we need to break the loop by hand because
        #             # the generator loops indefinitely
        #             break

        # # Train
        g = generate_batch(hypes, data_dir, 'val')
        X_test, Y_test = g.next()

        # print("#" * 80)
        # print(X_test.shape)
        # print(Y_test.shape)

        model.fit_generator(generate_batch(hypes, data_dir, 'train'),
                            samples_per_epoch=hypes['solver']['samples_per_epoch'],
                            nb_epoch=hypes['solver']['epochs'],
                            verbose=1,
                            validation_data=(X_test, Y_test))
        x_files, y_files = get_file_list('train', data_dir)
        x_files, y_files = sklearn.utils.shuffle(x_files,
                                                 y_files,
                                                 random_state=0)
        # ij = 0
        # for epoch in range(1, hypes['solver']['epochs'] + 1):
        #     print("#" * 80)
        #     print("# Epoch %i" % epoch)
        #     print("#" * 80)
        #     x_files, y_files = sklearn.utils.shuffle(x_files,
        #                                              y_files,
        #                                              random_state=epoch)
        #     for x_train_file, y_train_file in zip(x_files, y_files):
        #         x_train, y_train = get_traindata_single_file(hypes,
        #                                                      x_train_file,
        #                                                      y_train_file)
        #         # Reduce data
        #         # x_train, y_train = reduce_data_equal(x_train,
        #         #                                      y_train)

        #         t0 = time.time()
        #         model.fit(x_train, y_train,
        #                   batch_size=128,
        #                   nb_epoch=1,
        #                   # callbacks=[callb]
        #                   )
        #         ij += 1
        #         print("%i of %i" %
        #               (ij, hypes['solver']['epochs'] * len(x_files)))
        #         t1 = time.time()
        #         print("Training Time: %0.4f" % (t1 - t0))
        print("done with fit_generator")
        # save as YAML
        yaml_string = model.to_yaml()
        with open(model_file_path, 'w') as f:
            f.write(yaml_string)
        model.save_weights(weights_file_path)

        # Evaluate
        data = get_file_list('val', data_dir)
        analyze.evaluate(hypes,
                         data,
                         data_dir,
                         model,
                         elements=[0, 1],
                         get_segmentation=get_segmentation,
                         load_label_seg=load_label_seg,
                         color_changes=color_changes,
                         verbose=True)
    else:
        with open(model_file_path) as f:
            yaml_string = f.read()
        model = model_from_yaml(yaml_string)
        model.load_weights(weights_file_path)
        model.compile(optimizer=hypes['solver']['optimizer'],
                      loss='binary_crossentropy')
        data = get_file_list('test', data_dir)
        analyze.evaluate(hypes,
                         data,
                         data_dir,
                         model,
                         elements=[0, 1],
                         get_segmentation=get_segmentation,
                         load_label_seg=load_label_seg,
                         color_changes=color_changes,
                         verbose=True)


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
    # Load raw image
    image = get_image(image_path, 'RGB')
    height, width, _ = image.shape

    # Make the image an appropriate shape
    stride = hypes['arch']['stride']
    patch_size = hypes['arch']['patch_size']

    # How often does the window get applied in the different directions?
    width_n = int(math.ceil(width / stride))
    height_n = int(math.ceil(height / stride))

    assert patch_size >= stride
    left_pad = int(math.floor(patch_size - stride) / 2)
    right_pad = (((width_n * stride - width) % stride) +
                 (patch_size - stride - left_pad))
    top_pad = left_pad
    bottom_pad = (((height_n * stride - height) % stride) +
                  (patch_size - stride - top_pad))
    pad_width = ((top_pad, bottom_pad),
                 (left_pad, right_pad),
                 (0, 0))
    image = numpy.pad(image,
                      pad_width=pad_width,
                      mode='constant')
    segmentation = numpy.zeros(shape=(height, width))

    # Generate input patches of image
    patches = []
    coords = []
    for i in range(width_n):
        for j in range(height_n):
            x = stride * i
            y = stride * j
            patch = image[y:(y + patch_size), x:(x + patch_size)]
            patch = img_to_array(patch)
            assert patch.shape == (3, patch_size, patch_size), \
                "patch had shape %s" % str(patch.shape)
            patches.append(patch)
            coords.append({'x': i, 'y': j})
            if len(patches) == hypes['solver']['batch_size']:
                patches = numpy.array(patches)
                # Run model on input patches and collect output patches
                res = model.predict(patches)

                # Assemble output patches to segmentation image
                for coords, res in zip(coords, res):
                    res = res.reshape(stride, stride)
                    for m in range(stride):
                        for n in range(stride):
                            value = res[n][m]
                            x = coords['x'] * stride + m
                            y = coords['y'] * stride + n
                            segmentation[y][x] = value

                # Cleanup for next batch
                patches, coords = [], []
    # scipy.misc.imshow(segmentation)
    print("amax=%0.4f, mean=%0.4f, median=%0.4f, 70%%=%0.4f, 95%%=%0.4f" %
          (np.amax(segmentation),
           np.mean(segmentation),
           np.median(segmentation),
           np.percentile(segmentation, 70),
           np.percentile(segmentation, 95)))
    threshold = np.percentile(segmentation, 95)
    return segmentation > threshold


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
    parser.add_argument("--data",
                        dest="data",
                        help=("data directory which contains "
                              "'Segmentation_Rigid_Training'"),
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


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args.hypes_file, args.data, args.override)
