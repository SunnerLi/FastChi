import _init_path
from data_helper
from AutoEncoder import AutoEncoder
from darknet import DarkNet
from config import *
import tensorlayer as tl
import tensorflow as tf
import numpy as np
import datetime

saver = None
style_features = []     # Store the gram matrix of style
content_feature = []    # Store the gram matrix of content

def train(content_imgs, style_img):
    global style_features
    global content_feature

    # Precompute gram matrix of style frature
    style_shape = (1,) + style_img.shape
    with tf.Graph().as_default():
        style_ph = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        net = DarkNet(name='style_cnn')
        style_logits = net.build(net.preprocess(style_ph))
        
        # Run
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for layer in net.style_list:
                feature = layer.outputs.eval(feed_dict={
                    style_ph: [style_img]
                })
                feature = np.reshape(feature[0], (-1, feature.shape[3]))
                gram = np.matmul(feature.T, feature) / feature.size
                style_features.append(gram)

    # Deal with content image
    tl.layers.clear_layers_name()
    with tf.Graph().as_default():
        content_ph = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, 3))
        net = DarkNet(name='content_cnn')
        content_logits = net.build(net.preprocess(content_ph))
        for layer in net.content_list:
            content_feature.append(layer)

        # Construct render network and loss network
        transfer_net = AutoEncoder()
        transfer_logits = transfer_net.build(content_ph / 255.0)
        net = DarkNet(name='examine_cnn')
        content_normalized_logits = net.build(net.preprocess(transfer_logits))

        # -----------------------------------------------------------------------------------------------
        # Define loss
        # -----------------------------------------------------------------------------------------------    
        # Content loss
        batch, height, width, channel = content_feature[0].outputs.get_shape()
        content_size = int(height) * int(width) * int(channel)
        content_loss = content_weight * (
            tf.reduce_sum(tf.square(net.getContentLayer().outputs - content_feature[0].outputs) / content_size)
        )
        
        # Style loss
        style_loss = None
        for i, layer in enumerate(style_features):
            content_lrelu_layer = net.getStyleLayer(i)
            batch, height, width, channel = content_lrelu_layer.outputs.get_shape()
            flat_size = tensor_size_prod(content_lrelu_layer.outputs)
            feats = tf.reshape(content_lrelu_layer.outputs, [int(batch), int(height) * int(width), int(channel)])
            feats_T = tf.transpose(feats, perm=[0, 2, 1])
            grams = tf.matmul(feats_T, feats) / flat_size
            style_gram = style_features[i]
            if style_loss is None:
                style_loss = tf.reduce_sum(tf.square(grams - style_gram)) / flat_size
            else:
                style_loss += tf.reduce_sum(tf.square(grams - style_gram)) / flat_size
        
        # TV denoising
        tv_y_size = tensor_size_prod(transfer_logits[:, 1:, :, :])
        tv_x_size = tensor_size_prod(transfer_logits[:, :, 1:, :])
        y_tv = tf.reduce_sum(tf.square(transfer_logits[:, 1:, :, :] - transfer_logits[:, :255, :, :]))
        x_tv = tf.reduce_sum(tf.square(transfer_logits[:, :, 1:, :] - transfer_logits[:, :, :255, :]))
        tv_loss = tv_weight * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size
        
        # Total loss and optimizer
        loss = content_loss + style_loss + tv_loss
        train_op = tf.train.AdamOptimizer().minimize(loss)

        # Train
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                num_iter = num_example // batch_size
                for j in range(num_iter):
                    img_batch = get_img_batch(content_imgs, j)
                    _ = sess.run([train_op], feed_dict={
                        content_ph: img_batch
                    })
                _style_loss, _content_loss, _tv_loss = sess.run([style_loss, content_loss, tv_loss], feed_dict={
                    content_ph: img_batch
                })
                print("epoch: ", i, '\tstyle loss: ', _style_loss, '\tcontent loss: ', _content_loss, '\tTV loss: ', _tv_loss, '\ttime: ', datetime.datetime.now().time())
            saver = tf.train.Saver()
            saver.save(sess, model_path + model_name)


            

if __name__ == '__main__':
    # style_image = get_img()       # ????
    # content_image = get_files()   # ????
    style_image = np.random.random([224, 224, 3])
    content_image = np.random.random([1000, 224, 224, 3])
    train(content_image, style_image)