#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


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
    import tflearn

    tflearn.init_graph(num_cores=2, gpu_memory_fraction=0.6)

    input_shape = (None,
                   feats[0].shape[0],
                   feats[0].shape[1],
                   feats[0].shape[2])
    logging.info("input shape: %s", input_shape)
    net = tflearn.input_data(shape=input_shape)
    net = tflearn.conv_2d(net, 10, 3, activation='relu', regularizer="L2")
    net = tflearn.conv_2d(net, 10, 3, activation='relu', regularizer="L2")
    net = tflearn.fully_connected(net, 2, activation='sigmoid')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.01,
                             loss='categorical_crossentropy', name='target')
    return tflearn.DNN(net)
