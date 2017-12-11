import tensorlayer as tl
import tensorflow as tf
import numpy as np

def lrelu(tensor, alpha=0.2):
    return tf.maximum(alpha * tensor, tensor)

class DarkNet(object):
    def __init__(self):
        self.lrelu_list = []

    def preprocess(self, tensor):
        return tensor - np.array([123.68, 116.779, 103.939])

    def build(self, image_ph, base_filter=32):
        network = tl.layers.InputLayer(image_ph, name='darknet_input_layer')

        # Conv1
        network = tl.layers.Conv2d(network, n_filter=base_filter, filter_size=(3, 3), strides=(1, 1), name ='darknet_conv2d_1')
        network = tl.layers.BatchNormLayer(network, act = lrelu, name ='darknet_batchnorm_layer_1')
        self.lrelu_list.append(network)
        network = tl.layers.MaxPool2d(network, name='darknet_maxpool_1')

        # Conv2
        network = tl.layers.Conv2d(network, n_filter=base_filter * 2, filter_size=(3, 3), strides=(1, 1), name ='darknet_conv2d_2')
        network = tl.layers.BatchNormLayer(network, act = lrelu, name ='darknet_batchnorm_layer_2')
        self.lrelu_list.append(network)
        network = tl.layers.MaxPool2d(network, name='darknet_maxpool_2')

        # Conv3, 4, 5
        network = tl.layers.Conv2d(network, n_filter=base_filter * 4, filter_size=(3, 3), strides=(1, 1), name ='darknet_conv2d_3')
        network = tl.layers.Conv2d(network, n_filter=base_filter * 2, filter_size=(1, 1), strides=(1, 1), name ='darknet_conv2d_4')
        network = tl.layers.Conv2d(network, n_filter=base_filter * 4, filter_size=(3, 3), strides=(1, 1), name ='darknet_conv2d_5')
        network = tl.layers.BatchNormLayer(network, act = lrelu, name ='darknet_batchnorm_layer_5')
        self.lrelu_list.append(network)
        network = tl.layers.MaxPool2d(network, name='darknet_maxpool_5')

        # Conv6, 7, 8
        network = tl.layers.Conv2d(network, n_filter=base_filter * 8, filter_size=(3, 3), strides=(1, 1), name ='darknet_conv2d_6')
        network = tl.layers.Conv2d(network, n_filter=base_filter * 4, filter_size=(1, 1), strides=(1, 1), name ='darknet_conv2d_7')
        network = tl.layers.Conv2d(network, n_filter=base_filter * 8, filter_size=(3, 3), strides=(1, 1), name ='darknet_conv2d_8')
        network = tl.layers.BatchNormLayer(network, act = lrelu, name ='darknet_batchnorm_layer_8')
        self.lrelu_list.append(network)
        network = tl.layers.MaxPool2d(network, name='darknet_maxpool_8')

        # Conv9, 10, 11, 12, 13
        network = tl.layers.Conv2d(network, n_filter=base_filter * 16, filter_size=(3, 3), strides=(1, 1), name ='darknet_conv2d_9')
        network = tl.layers.Conv2d(network, n_filter=base_filter * 8, filter_size=(1, 1), strides=(1, 1), name ='darknet_conv2d_10')
        network = tl.layers.Conv2d(network, n_filter=base_filter * 16, filter_size=(3, 3), strides=(1, 1), name ='darknet_conv2d_11')
        network = tl.layers.Conv2d(network, n_filter=base_filter * 8, filter_size=(1, 1), strides=(1, 1), name ='darknet_conv2d_12')
        network = tl.layers.Conv2d(network, n_filter=base_filter * 16, filter_size=(3, 3), strides=(1, 1), name ='darknet_conv2d_13')
        network = tl.layers.BatchNormLayer(network, act = lrelu, name ='darknet_batchnorm_layer_13')
        self.lrelu_list.append(network)
        network = tl.layers.MaxPool2d(network, name='darknet_maxpool_13')

        # Conv14, 15, 16, 17, 18
        network = tl.layers.Conv2d(network, n_filter=base_filter * 32, filter_size=(3, 3), strides=(1, 1), name ='darknet_conv2d_14')
        network = tl.layers.Conv2d(network, n_filter=base_filter * 16, filter_size=(1, 1), strides=(1, 1), name ='darknet_conv2d_15')
        network = tl.layers.Conv2d(network, n_filter=base_filter * 32, filter_size=(3, 3), strides=(1, 1), name ='darknet_conv2d_16')
        network = tl.layers.Conv2d(network, n_filter=base_filter * 16, filter_size=(1, 1), strides=(1, 1), name ='darknet_conv2d_17')
        network = tl.layers.Conv2d(network, n_filter=base_filter * 32, filter_size=(3, 3), strides=(1, 1), name ='darknet_conv2d_18')
        network = tl.layers.BatchNormLayer(network, act = lrelu, name ='darknet_batchnorm_layer_18')
        self.lrelu_list.append(network)
        network = tl.layers.MaxPool2d(network, name='darknet_maxpool_18')

        return network.outputs