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
        batch, height, width, channel = layer.outputs.get_shape()
        height, width, channel = int(height), int(width), int(channel)
        network = tl.layers.Conv2d(layer, n_filter=int(channel / 2), filter_size=(1, 1), strides=(1, 1), act = None, name = name + '_shrink')
        network = tl.layers.Conv2d(network, n_filter=int(channel / 2), filter_size=filter_size, strides=strides, act = act, name = name)
        network = tl.layers.Conv2d(network, n_filter=n_filter, filter_size=(1, 1), strides=(1, 1), act = None, name = name + '_expand')
        return network

    def deconv(self, layer, n_out_channel=32, filter_size=(3, 3), strides=(1, 1), act = None, name = 'conv2d'):
        """
            The wrapper of inception of convolution
        """
        # Define shape
        batch, height, width, channel = layer.outputs.get_shape()
        height, width, channel = int(height), int(width), int(channel)
        scaler = strides[0]
        shrink_shape = (height, width)
        deconv_shape = (height * scaler, width * scaler)
        output_shape = deconv_shape

        # Define module
        network = tl.layers.DeConv2d(layer, n_out_channel = int(channel / 2), filter_size=(1, 1), out_size=shrink_shape, strides = (1, 1), name =name + '_shrink')
        network = tl.layers.DeConv2d(network, n_out_channel = int(channel / 2), filter_size=filter_size, out_size=deconv_shape, strides = strides, act = act, name =name)
        network = tl.layers.DeConv2d(network, n_out_channel = n_out_channel, filter_size=(1, 1), out_size=output_shape, strides = (1, 1), name =name + '_expand')
        network = tl.layers.ReshapeLayer(network, shape = [tf.shape(layer.outputs)[0], height * scaler, width * scaler, n_out_channel], name = name + '_align')
        return network

    def residual_block(self, layer, n_filter=64, filter_size = (3, 3), name = 'res_block'):
        tmp = self.conv(layer, n_filter=n_filter, name = name + 'conv2d_1')
        return tl.layers.ElementwiseLayer([tmp, self.conv(tmp, n_filter=n_filter, name = name + 'conv2d_2')], combine_fn = tf.add, name =name + '_add')

    def build(self, image_ph):
        network = tl.layers.InputLayer(image_ph, 'autoencoder_input_layer')
        network = self.conv(network, n_filter=16, filter_size=(9, 9), act = lrelu, name = 'autoencoder_conv2d_1')
        network = self.conv(network, n_filter=32, filter_size=(3, 3),strides=(2, 2), act = lrelu, name = 'autoencoder_conv2d_2')
        network = self.conv(network, n_filter=64, filter_size=(3, 3),strides=(2, 2), act = lrelu, name = 'autoencoder_conv2d_3')
        network = self.residual_block(network, name = 'autoencoder_res_block1')
        network = self.residual_block(network, name = 'autoencoder_res_block2')
        network = self.residual_block(network, name = 'autoencoder_res_block3')
        network = self.residual_block(network, name = 'autoencoder_res_block4')
        network = self.residual_block(network, name = 'autoencoder_res_block5')
        network = self.deconv(network, n_out_channel=32, strides=(2, 2), act = lrelu, name = 'autoencoder_conv2d_transpose_1')
        network = self.deconv(network, n_out_channel=16, strides=(2, 2), act = lrelu, name = 'autoencoder_conv2d_transpose_2')
        network = self.conv(network, n_filter=3, filter_size=(9, 9), strides=(1, 1), name = 'autoencoder_conv2d_final')
        return network.outputs
    