import tensorlayer as tl
import tensorflow as tf
import numpy as np

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
        self.epsilon  = epsilon
        self.momentum = momentum
        self.name = name

  def __call__(self, x, train=False):
    print('[ Discriminator ] - BatchNorm')
    return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, is_training=train, scope=self.name)

def net(image_ph, base_filter=16, reuse=True):
    """
        Since it should load the pre-trained model, the structure should be the same as the origin
        Refer site: https://github.com/gliese581gg/YOLO_tensorflow/blob/master/YOLO_tiny_tf.py
    """
    layers = (
        'conv_1', 'bnorm_1', 'lrelu_1', 'pool_2',
        'conv_3', 'bnorm_3', 'lrelu_3', 'pool_4',
        'conv_5', 'bnorm_5', 'lrelu_5', 'pool_6', 'conv_7', 'bnorm_7', 'lrelu_7', 'pool_8',
        'conv_9', 'bnorm_9', 'lrelu_9', 'pool_10', 'conv_11', 'bnorm_11', 'lrelu_11', 'pool_12',
        'conv_13', 'bnorm_13', 'lrelu_13', 'conv_14', 'bnorm_14', 'lrelu_14', 'conv_15', 'bnorm_15', 'lrelu_15'
    )
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        net = {}
        current = image_ph
        filter_num = base_filter
        for i, name in enumerate(layers):
            kind = name[:4]
            idx = int(name[name.index('_')+1:])
            if kind == 'conv':
                current = conv_layer(idx, current, filter_num, 3, 1)
                if filter_num < 1024:
                    filter_num *= 2
            elif kind == 'lrel':
                current = lrelu_layer(current)
            elif kind == 'pool':
                current = pooling_layer(idx, current, 2, 2)
            # elif kind == 'bnor':
                # batch_norm_layer = batch_norm(name = str(i) + '_bn')
                # current = batch_norm_layer(current)
            net[name] = current
        return net

def conv_layer(idx, inputs, filters, size,stride, alpha = 0.1):
    channels = inputs.get_shape()[3]
    weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1), trainable=False)
    biases = tf.Variable(tf.constant(0.1, shape=[filters]), trainable=False)
    pad_size = size//2
    pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
    inputs_pad = tf.pad(inputs,pad_mat)
    conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')	
    conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')	
    print('    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx,size,size,stride,filters,int(channels)))
    return tf.maximum(alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')

def pooling_layer(idx,inputs,size,stride):
    print('    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx,size,size,stride))
    return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')

def lrelu_layer(tensor, alpha=0.2):
    return tf.maximum(alpha * tensor, tensor)

def fc_layer(idx,inputs,hiddens,flat = False,linear = False):
    input_shape = inputs.get_shape().as_list()		
    if flat:
    	dim = input_shape[1]*input_shape[2]*input_shape[3]
    	inputs_transposed = tf.transpose(inputs,(0,3,1,2))
    	inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
    else:
    	dim = input_shape[1]
    	inputs_processed = inputs
    weight = tf.Variable(tf.truncated_normal([dim,hiddens], stddev=0.1), trainable=False)
    biases = tf.Variable(tf.constant(0.1, shape=[hiddens]), trainable=False)	
    print('    Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (idx,hiddens,int(dim),int(flat),1-int(linear))	)
    if linear : return tf.add(tf.matmul(inputs_processed,weight),biases,name=str(idx)+'_fc')
    ip = tf.add(tf.matmul(inputs_processed,weight),biases)
    return tf.maximum(alpha*ip,ip,name=str(idx)+'_fc')

def preprocess(tensor):
    return tensor - np.array([123.68, 116.779, 103.939])


def restore(sess, pretrained_path):
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, pretrained_path)
    print("Restore complete!")

if __name__ == '__main__':
    image_ph = tf.placeholder(tf.float32, [None, 224, 400, 3])
    network = net(image_ph)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, 'YOLO_tiny.ckpt')