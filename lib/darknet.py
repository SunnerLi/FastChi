import tensorlayer as tl
import tensorflow as tf
import numpy as np

def lrelu(tensor, alpha=0.2):
    return tf.maximum(alpha * tensor, tensor)

class DarkNet(object):
    def __init__(self, name):
        self.style_list = []
        self.content_list = []
        self.name = name

    def preprocess(self, tensor):
        return tensor - np.array([123.68, 116.779, 103.939])

    def convBlock(self, layer, n_filter = 32, filter_size = (1, 1), strides = (1, 1), act = lrelu, name = None):
        network = tl.layers.Conv2d(layer, n_filter=n_filter, filter_size=(3, 3), strides=(1, 1), name =name + 'conv')
        network = tl.layers.BatchNormLayer(network, act = act, name = name + '_BN')
        return network        

    def getContentLayer(self, idx):
        return self.content_list[idx]

    def getStyleLayer(self, idx):
        return self.style_list[idx]

    def build(self, image_ph, base_filter=16):
        with tf.variable_scope(self.name):
            network = tl.layers.InputLayer(image_ph, name='darknet_input_layer')

            # Conv1
            network = self.convBlock(network, n_filter=base_filter, filter_size = (3, 3), name = 'darknet_conv_block_1')
            self.style_list.append(network)
            network = tl.layers.MaxPool2d(network, name='darknet_maxpool_1')

            # Conv2
            network = self.convBlock(network, n_filter=base_filter * 2, filter_size = (3, 3), name = 'darknet_conv_block_2')
            self.style_list.append(network)
            network = tl.layers.MaxPool2d(network, name='darknet_maxpool_2')

            # Conv3, 4, 5
            network = self.convBlock(network, n_filter=base_filter * 4, filter_size = (3, 3), name = 'darknet_conv_block_3')
            self.style_list.append(network)
            network = self.convBlock(network, n_filter=base_filter, filter_size = (1, 1), name = 'darknet_conv_block_4')
            network = self.convBlock(network, n_filter=base_filter * 4, filter_size = (3, 3), name = 'darknet_conv_block_5')           
            network = tl.layers.MaxPool2d(network, name='darknet_maxpool_5')

            # Conv6, 7, 8
            network = self.convBlock(network, n_filter=base_filter * 8, filter_size = (3, 3), name = 'darknet_conv_block_6')
            self.style_list.append(network)
            network = self.convBlock(network, n_filter=base_filter * 2, filter_size = (1, 1), name = 'darknet_conv_block_7')
            self.content_list.append(network)
            network = self.convBlock(network, n_filter=base_filter * 8, filter_size = (3, 3), name = 'darknet_conv_block_8')  
            network = tl.layers.MaxPool2d(network, name='darknet_maxpool_8')

            # Conv9, 10, 11, 12, 13
            network = self.convBlock(network, n_filter=base_filter * 16, filter_size = (3, 3), name = 'darknet_conv_block_9')
            self.style_list.append(network)
            network = self.convBlock(network, n_filter=base_filter * 4, filter_size = (1, 1), name = 'darknet_conv_block_10')
            self.content_list.append(network)
            network = self.convBlock(network, n_filter=base_filter * 16, filter_size = (3, 3), name = 'darknet_conv_block_11')  
            network = self.convBlock(network, n_filter=base_filter * 4, filter_size = (1, 1), name = 'darknet_conv_block_12')
            network = self.convBlock(network, n_filter=base_filter * 16, filter_size = (3, 3), name = 'darknet_conv_block_13')  
            network = tl.layers.MaxPool2d(network, name='darknet_maxpool_13')
            return network.outputs