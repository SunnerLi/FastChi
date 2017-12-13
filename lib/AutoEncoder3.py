import tensorlayer as tl
import tensorflow as tf

WEIGHTS_INIT_STDEV = .1

def net(image_ph):
    network = tl.layers.InputLayer(image_ph, 'autoencoder_input_layer')
    print('type: ', type(network))
    network = conv(network, 32, 9, 1, name ='autoencoder_conv2d_1')
    network = conv(network, 64, 3, 2, name ='autoencoder_conv2d_2')
    network = conv(network, 128, 3, 2, name = 'autoencoder_conv2d_3')
    network = residual_block(network, name = 'autoencoder_res_block1')
    network = residual_block(network, name = 'autoencoder_res_block2')
    network = residual_block(network, name = 'autoencoder_res_block3')
    network = residual_block(network, name = 'autoencoder_res_block4')
    network = residual_block(network, name = 'autoencoder_res_block5')
    network = deconv(network, 64, 3, 2, name = 'autoencoder_conv2d_transpose_1')
    network = deconv(network, 32, 3, 2, name = 'autoencoder_conv2d_transpose_2')
    network = conv(network, 3, 9, 1, act = tf.identity, name = 'autoencoder_conv2d_final')
    network = tf.nn.tanh(network.outputs) * 150.0 + 255.0 / 2
    return network

def conv(layer, n_filter=32, filter_size= 3 , strides= 1, act = tf.nn.relu, name = 'conv2d'):
    """
        The wrapper of inception of convolution
    """
    print('type: ', type(layer))
    network = tl.layers.Conv2d(layer, n_filter=n_filter, filter_size=(filter_size, filter_size), strides=(strides, strides), act = act, name = name)
    network = tl.layers.BatchNormLayer(network, name = name + '_batchnorm_layer')
    return network

def deconv(layer, n_out_channel = 32, filter_size = 3, strides = 1, act = tf.nn.relu, name = 'deconv2d'):
    """
        The wrapper of inception of convolution
    """
    batch, height, width, channel = layer.outputs.get_shape()
    height, width, channel = int(height), int(width), int(channel)
    output_shape = (height * strides, width * strides)
    network = tl.layers.DeConv2d(layer, n_out_channel = n_out_channel, filter_size=(filter_size, filter_size), out_size=output_shape, strides = (strides, strides), name =name + '_shrink')
    network = tl.layers.BatchNormLayer(network, name = name + '_batchnorm_layer')
    return network

def residual_block(network, filter_size=3, name = 'residual_block'):
    middle1 = tl.layers.Conv2d(network, n_filter=128, filter_size=(filter_size, filter_size), strides=(1, 1), act = tf.nn.relu, name = name + '_conv2d_1')
    middle2 = tl.layers.Conv2d(network, n_filter=128, filter_size=(filter_size, filter_size), strides=(1, 1), act = tf.nn.relu, name = name + '_conv2d_2')    
    return tl.layers.ElementwiseLayer([middle1, middle2], combine_fn = tf.add, name = name + '_elementwise_layer')