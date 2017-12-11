import _init_path
from darknet import DarkNet
from config import *
import tensorflow as tf
import numpy as np

saver = None
style_features = {}     # Store the gram matrix of style

def train(style_img):
    global style_features

    # Precompute gram matrix of style frature
    style_shape = (1,) + style_img.shape
    with tf.Graph().as_default():
        style_ph = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        net = DarkNet()
        style_logits = net.build(net.preprocess(style_ph))
        
        # Run
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for layer in net.lrelu_list:
                feature = layer.outputs.eval(feed_dict={
                    style_ph: [style_img]
                })
                feature = np.reshape(feature[0], (-1, feature.shape[3]))
                gram = np.matmul(feature.T, feature) / feature.size
                style_features[layer.name] = gram

    # Deal with content image
    with tf.Graph.as_default():
        content_ph = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, 3))
        net = DarkNet()
        content_logits = net.build(content_logits) 
            

if __name__ == '__main__':
    # style_image = get_img() ## ????
    style_image = np.random.random([224, 224, 3])


    train(style_image)

    # Store model
    # saver = tf.train.Saver()
    # saver.save(sess, model_path + model_name)
