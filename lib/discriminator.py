import tensorflow as tf
import numpy as np

MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939])

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
        self.epsilon  = epsilon
        self.momentum = momentum
        self.name = name

  def __call__(self, x, train=False):
    print('[ Discriminator ] - BatchNorm')
    return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, is_training=train, scope=self.name)

def net(image_ph, base_filter = 32, reuse=False):
    layers = (
        'conv1_1', 'lrelu1_1',
        'conv2_1', 'lrelu2_1',
        'conv3_1', 'lrelu3_1',
        'conv4_1', 'lrelu4_1',
    )

    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        net = {}
        current = image_ph
        for i, name in enumerate(layers):
            kind = name[:-3]
            if kind == 'conv':
                exp_of_filter = int(name[4]) - 1
                current = _conv_layer(current, 5, base_filter * (2 ** exp_of_filter))
            elif kind == 'lrelu':
                current = _lrelu_layer(current)
            elif kind == 'bn':
                batch_norm_layer = batch_norm()
                current = batch_norm_layer(current)
            net[name] = current
        # return tf.nn.sigmoid(h4), h4
        return net

def _conv_layer(tensor, filter_size, output_channel, strides = (1, 2, 2, 1)):
    # Conv
    batch, height, width, channel = tensor.get_shape()
    weight = tf.Variable(tf.truncated_normal([filter_size, filter_size, int(channel), output_channel]), trainable=False)
    conv = tf.nn.conv2d(tensor, weight, strides=strides, padding='SAME')
    print('[ Discriminator ] - Convolution \t filter num: ', output_channel, '\tfilter size: [', filter_size, ',', filter_size, '] \t stride: ', strides)

    # Add bias
    bias_shape = tf.stack([conv.get_shape()[1], conv.get_shape()[2], conv.get_shape()[3]])
    bias = tf.Variable(tf.truncated_normal(bias_shape), trainable=False)
    _ = conv + bias

    # Reshape
    shape = tf.stack([tf.shape(conv)[0], conv.get_shape()[1], conv.get_shape()[2], conv.get_shape()[3]])
    conv = tf.reshape(_, shape)
    return conv

def _lrelu_layer(tensor, alpha = 0.2):
    print('[ Discriminator ] - Leaky ReLU \t\t AlphaL ', alpha)
    return tf.maximum(alpha * tensor, tensor)

def preprocess(image):
    return image - MEAN_PIXEL


def unprocess(image):
    return image + MEAN_PIXEL

if __name__ == '__main__':
    image_ph = tf.placeholder(tf.float32, [None, 224, 400, 3])
    network = net(image_ph)