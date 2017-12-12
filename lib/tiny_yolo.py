import tensorlayer as tl
import tensorflow as tf
import numpy as np

def lrelu(tensor, alpha=0.2):
    return tf.maximum(alpha * tensor, tensor)

class Tiny_YOLO(object):
    """
        Since it should load the pre-trained model, the structure should be the same as the origin
        Refer site: https://github.com/gliese581gg/YOLO_tensorflow/blob/master/YOLO_tiny_tf.py
    """
    alpha = 0.1

    def __init__(self):
        self.style_list = []
        self.content_list = []

    def preprocess(self, tensor):
        return tensor - np.array([123.68, 116.779, 103.939])

    def getContentLayer(self, idx):
        return self.content_list[idx]

    def getStyleLayer(self, idx):
        return self.style_list[idx]

    def build(self, image_ph, pretrained_path, base_filter=16):
        # Conv1
        self.conv_1 = self.conv_layer(1,image_ph,16,3,1)
        self.relu_1 = lrelu(self.conv_1)
        self.pool_2 = self.pooling_layer(2,self.conv_1,2,2)

        # Conv2
        self.conv_3 = self.conv_layer(3,self.pool_2,32,3,1)
        self.relu_2 = lrelu(self.conv_3)
        self.pool_4 = self.pooling_layer(4,self.conv_3,2,2)

        # Conv3
        self.conv_5 = self.conv_layer(5,self.pool_4,64,3,1)
        self.relu_3 = lrelu(self.conv_5)
        self.pool_6 = self.pooling_layer(6,self.conv_5,2,2)
        self.conv_7 = self.conv_layer(7,self.pool_6,128,3,1)
        self.relu_4 = lrelu(self.conv_7)
        self.pool_8 = self.pooling_layer(8,self.conv_7,2,2)

        # Conv4
        self.conv_9 = self.conv_layer(9,self.pool_8,256,3,1)
        self.relu_5 = lrelu(self.conv_9)
        self.pool_10 = self.pooling_layer(10,self.conv_9,2,2)
        self.conv_11 = self.conv_layer(11,self.pool_10,512,3,1)
        self.relu_6 = lrelu(self.conv_11)
        self.pool_12 = self.pooling_layer(12,self.conv_11,2,2)

        # Conv5
        self.conv_13 = self.conv_layer(13,self.pool_12,1024,3,1)
        self.relu_7 = lrelu(self.conv_13)
        self.conv_14 = self.conv_layer(14,self.conv_13,1024,3,1)
        self.relu_8 = lrelu(self.conv_14)
        self.conv_15 = self.conv_layer(15,self.conv_14,1024,3,1)
        self.relu_9 = lrelu(self.conv_15)

        # Append style and content list
        self.style_list.append(self.relu_1)
        self.style_list.append(self.relu_2)
        self.style_list.append(self.relu_3)
        self.style_list.append(self.relu_5)
        self.style_list.append(self.relu_7)
        self.content_list.append(self.relu_4)
        self.content_list.append(self.relu_6)

        # Load
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, pretrained_path)

    def conv_layer(self,idx,inputs,filters,size,stride):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))

        pad_size = size//2
        pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
        inputs_pad = tf.pad(inputs,pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')	
        conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')	
        print('    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx,size,size,stride,filters,int(channels)))
        return tf.maximum(self.alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')

    def pooling_layer(self,idx,inputs,size,stride):
        print('    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx,size,size,stride))
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')

    def fc_layer(self,idx,inputs,hiddens,flat = False,linear = False):
        input_shape = inputs.get_shape().as_list()		
        if flat:
        	dim = input_shape[1]*input_shape[2]*input_shape[3]
        	inputs_transposed = tf.transpose(inputs,(0,3,1,2))
        	inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
        else:
        	dim = input_shape[1]
        	inputs_processed = inputs
        weight = tf.Variable(tf.truncated_normal([dim,hiddens], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))	
        print('    Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (idx,hiddens,int(dim),int(flat),1-int(linear))	)
        if linear : return tf.add(tf.matmul(inputs_processed,weight),biases,name=str(idx)+'_fc')
        ip = tf.add(tf.matmul(inputs_processed,weight),biases)
        return tf.maximum(self.alpha*ip,ip,name=str(idx)+'_fc')

if __name__ == '__main__':
    image_ph = tf.placeholder(tf.float32, [None, 224, 400, 3])

    with tf.Session() as sess:
        net = Tiny_YOLO()
        net.build(image_ph, 'YOLO_tiny.ckpt')