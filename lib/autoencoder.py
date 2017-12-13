import tensorflow as tf, pdb

WEIGHTS_INIT_STDEV = .1

def net(image, base_filter = 32):
    conv1 = _conv_layer(image, base_filter, 9, 1)
    conv2 = _conv_layer(conv1, base_filter * 2, 3, 2)
    conv3 = _conv_layer(conv2, base_filter * 4, 3, 2)
    resid1 = _residual_block(conv3, n_filter = base_filter * 4)
    resid2 = _residual_block(resid1, n_filter = base_filter * 4)
    resid3 = _residual_block(resid2, n_filter = base_filter * 4)
    resid4 = _residual_block(resid3, n_filter = base_filter * 4)
    resid5 = _residual_block(resid4, n_filter = base_filter * 4)
    conv_t1 = _conv_tranpose_layer(resid5, base_filter * 2, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, base_filter, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds

def small_net(image, base_filter = 16):
    conv1 = _conv_layer_incept(image, base_filter, 9, 1)
    conv2 = _conv_layer_incept(conv1, base_filter * 2, 3, 2)
    conv3 = _conv_layer_incept(conv2, base_filter * 4, 3, 2)
    resid1 = _residual_block_incept(conv3, n_filter = base_filter * 4)
    resid2 = _residual_block_incept(resid1, n_filter = base_filter * 4)
    resid3 = _residual_block_incept(resid2, n_filter = base_filter * 4)
    resid4 = _residual_block_incept(resid3, n_filter = base_filter * 4)
    resid5 = _residual_block_incept(resid4, n_filter = base_filter * 4)
    conv_t1 = _conv_tranpose_layer_incept(resid5, base_filter * 2, 3, 2)
    conv_t2 = _conv_tranpose_layer_incept(conv_t1, base_filter, 3, 2)
    conv_t3 = _conv_layer_incept(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds

def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)
    return net

def _conv_layer_incept(net, num_filters, filter_size, strides, relu=True):
    # Shrinking
    weights_init = _conv_init_vars(net, num_filters // 2, 1)
    strides_shape = [1, 1, 1, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')

    # Working
    weights_init = _conv_init_vars(net, num_filters // 2, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')

    # Expanding
    weights_init = _conv_init_vars(net, num_filters, 1)
    strides_shape = [1, 1, 1, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)
    return net

def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)
    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]

    new_rows, new_cols = int(rows * strides), int(cols * strides)
    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)

    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = tf.reshape(net, new_shape)

    net = _instance_norm(net)
    return tf.nn.relu(net)


def _conv_tranpose_layer_incept(net, num_filters, filter_size, strides):
    # Shrinking
    weights_init = _conv_init_vars(net, num_filters // 2, 1, transpose=True)
    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * 1), int(cols * 1)
    new_shape = [batch_size, new_rows, new_cols, num_filters // 2]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1, 1, 1, 1]
    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = tf.reshape(net, new_shape)

    # Working
    weights_init = _conv_init_vars(net, num_filters // 2, filter_size, transpose=True)
    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)
    new_shape = [batch_size, new_rows, new_cols, num_filters // 2]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = tf.reshape(net, new_shape)

    # Expanding
    weights_init = _conv_init_vars(net, num_filters, 1, transpose=True)
    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * 1), int(cols * 1)
    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1, 1, 1, 1]
    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = tf.reshape(net, new_shape)
    net = _instance_norm(net)
    return tf.nn.relu(net)

def _residual_block(net, n_filter = 128, filter_size = 3):
    tmp = _conv_layer(net, n_filter, filter_size, 1)
    return net + _conv_layer(tmp, n_filter, filter_size, 1, relu=False)

def _residual_block_incept(net, n_filter = 128, filter_size = 3):
    tmp = _conv_layer_incept(net, n_filter, filter_size, 1)
    return net + _conv_layer_incept(tmp, n_filter, filter_size, 1, relu=False)

def _instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init