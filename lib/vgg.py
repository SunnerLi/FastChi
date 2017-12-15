# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import tensorflow as tf
import numpy as np
import scipy.io
import pdb

""" 
    This program is revised from here:
    https://github.com/lengstrom/fast-style-transfer/blob/master/src/vgg.py
"""

MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939])

def net(data_path, input_image, reduce=False, reuse=False):
    """
        The wrapper function to return the network dict object

        Arg:    data_path   - The path of pre-trained model
                input_image - The placeholder object of input
                reduce      - If the function should adopt our revision or not
                reuse       - If the model should share weight
        Ret:    The dict object which contain VGG network graph
    """
    if reduce == False:
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )
        return __net(data_path, input_image, layers, reuse=False)        
    else:
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4'
        )   
        return __net(data_path, input_image, layers, reuse=reuse)

def __net(data_path, input_image, layers, reuse=False):
    """ 
        Construct the model by the given list of layer name

        Arg:    data_path   - The path of pre-trained model
                input_image - The placeholder object of input
                layers      - The tuple which contains the order name list of network
                reuse       - If the model should share weight
        Ret:    The dict object which contain VGG network graph
    """
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]

    net = {}
    current = input_image
    with tf.variable_scope("VGG", reuse=reuse):
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                current = _conv_layer(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current)
            elif kind == 'pool':
                current = _pool_layer(current)
            net[name] = current

        assert len(net) == len(layers)
        return net

def _conv_layer(tensor, weights, bias):
    """
        Return the conv2d & add tensor for the given parameter

        Arg:    tensor  - The previous tensor object
                weights - The ndnumpy weight object to initialized filter
                bias    - The ndnumpy weight object to initialized bias
        Ret:    The result tensor object
    """
    conv = tf.nn.conv2d(tensor, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)

def _pool_layer(tensor):
    """
        Return the max pooling tensor

        Arg:    tensor  - The previous tensor object
        Ret:    The result tensor object
    """
    return tf.nn.max_pool(tensor, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')

def preprocess(image):
    """
        Normalized to fit ImageNet normalization format

        Arg:    image   - The previous tensor object
        Ret:    The result tensor object
    """
    return image - MEAN_PIXEL

def unprocess(image):
    """
        Unnormalized the tensor with ImageNet normalization format

        Arg:    image   - The previous tensor object
        Ret:    The result tensor object
    """
    return image + MEAN_PIXEL