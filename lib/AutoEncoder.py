import tensorlayer as tl
import tensorflow as tf
import numpy as np

def lrelu(tensor, alpha=0.2):
    return tf.maximum(alpha * tensor, tensor)

class AutoEncoder(object):
    def __init__(self):
        self.lrelu_list = []

    def preprocess(self, tensor):
        return tensor - np.array([123.68, 116.779, 103.939])

    def conv(self, layer, n_filter=32, filter_size=(3, 3), strides=(1, 1), act = None, name = 'conv2d'):
        """
            The wrapper of inception of convolution
        """
        network = tl.layers.Conv2d(layer, n_filter=n_filter / 2, filter_size=(1, 1), strides=(1, 1), act = None, name = name + '_shrink')
        network = tl.layers.Conv2d(network, n_filter=n_filter, filter_size=filter_size, strides=(1, 1), act = act, name = name)
        network = tl.layers.Conv2d(network, n_filter=n_filter * 2, filter_size=(1, 1), strides=(1, 1), act = None, name = name + '_expand')
        return network

    def deconv(self, layer, n_filter=32, filter_size=(3, 3), strides=(1, 1), act = None, name = 'conv2d'):
        """
            The wrapper of inception of convolution
        """
        batch, height, width, channel = layer.outputs.get_shape()
        network = tl.layers.DeConv2d(layer, n_out_channel = 32)
        network = tl.layers.DeConv2d(layer, n_filter=n_filter / 2, filter_size=(1, 1), strides=(1, 1), act = None, name = name + '_shrink')
        network = tl.layers.Conv2d(layer, n_filter=n_filter, filter_size=filter_size, strides=(1, 1), act = act, name = name)
        network = tl.layers.Conv2d(layer, n_filter=n_filter * 2, filter_size=(1, 1), strides=(1, 1), act = None, name = name + '_expand')
        return network

    def residual_block(self, layer, n_filter=64, filter_size = (3, 3)):
        tmp = self.conv(layer, n_filter=n_filter)
        return layer + self.conv(tmp, n_filter=n_filter)

    def build(self, image_ph):
        network = tl.layers.InputLayer(image_ph, 'autoencoder_input_layer')
        network = self.conv(network, n_filter=16, filter_size=(9, 9), act = lrelu, name = 'autoencoder_conv2d_1')
        network = self.conv(network, n_filter=32, filter_size=(3, 3),strides=(2, 2), act = lrelu, name = 'autoencoder_conv2d_1')
        network = self.conv(network, n_filter=64, filter_size=(3, 3),strides=(2, 2), act = lrelu, name = 'autoencoder_conv2d_1')
        network = self.residual_block(network)
        network = self.residual_block(network)
        network = self.residual_block(network)
        network = self.residual_block(network)
        network = self.residual_block(network)
    