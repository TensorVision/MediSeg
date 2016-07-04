#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from keras.models import model_from_yaml


def generate_nnet(feats):
    """Generate a neural network.

    Parameters
    ----------
    feats : list with at least one feature vector

    Returns
    -------
    Neural network object
    """
    # Load it here to prevent crash of --help when it's not present
    from keras.models import Sequential
    from keras.layers import Dense, Convolution2D, Flatten

    model = Sequential()
    input_shape = (feats[0].shape[0],
                   feats[0].shape[1],
                   feats[0].shape[2])
    logging.info("input shape: %s", input_shape)
    model.add(Convolution2D(10, 3, 3,
                            border_mode='same',
                            input_shape=input_shape))

    model.add(Convolution2D(10, 3, 3, activation='relu', border_mode='same'))
    # model.add(Convolution2D(10, 3, 3, activation='relu', border_mode='same'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


def serialize_model(hypes, model):
    """Serialize a model."""
    yaml_path = hypes["segmenter"]["serialized_model_path"] + ".yaml"
    hdf5_path = hypes["segmenter"]["serialized_model_path"] + ".hdf5"
    model.save_weights(hdf5_path)
    with open(yaml_path, 'w') as f:
        f.write(model.to_yaml())


def load_model(hypes):
    """Load a serialized model."""
    yaml_path = hypes["segmenter"]["serialized_model_path"] + ".yaml"
    hdf5_path = hypes["segmenter"]["serialized_model_path"] + ".hdf5"
    with open(yaml_path) as f:
        yaml_string = f.read()
    model = model_from_yaml(yaml_string)
    model.load_weights(hdf5_path)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy')
    return model
