import _init_path
# from AutoEncoder import AutoEncoder
from AutoEncoder2 import net as AutoEncoder
from multiprocessing import Process, Queue
from tiny_yolo import Tiny_YOLO
from darknet import DarkNet
from config import *
from utils import *
import tensorlayer as tl
import tensorflow as tf
import numpy as np
import datetime

saver = None
style_features = []     # Store the gram matrix of style
content_feature = []    # Store the gram matrix of content

def train(content_image_name_list, style_img):
    global style_features
    global content_feature

    # Precompute gram matrix of style frature
    style_shape = (1,) + style_img.shape
    with tf.Graph().as_default():
        style_ph = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        # net = DarkNet(name='style_cnn')
        
        # Run
        with tf.Session() as sess:
            net = Tiny_YOLO()
            style_logits = net.build(net.preprocess(style_ph))
            net.restore(sess, pretrained_path='./lib/YOLO_tiny.ckpt')
            for layer in net.style_list:
                feature = layer.eval(feed_dict={
                    style_ph: np.asarray([style_img])
                })
                feature = np.reshape(feature[0], (-1, feature.shape[3]))
                gram = np.matmul(feature.T, feature) / feature.size
                style_features.append(gram)

    # Deal with content image
    tl.layers.clear_layers_name()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            content_ph = tf.placeholder(tf.float32, shape=(batch_size, 224, 400, 3))
            net2 = Tiny_YOLO()
            content_logits = net2.build(net2.preprocess(content_ph))
            for layer in net2.content_list:
                content_feature.append(layer)
            net2.restore(sess, './lib/YOLO_tiny.ckpt')

            # Construct render network and loss network
            # transfer_net = AutoEncoder()
            # transfer_logits = transfer_net.build(content_ph)
            transfer_logits = AutoEncoder(content_ph)
            net = Tiny_YOLO()
            content_normalized_logits = net.build(net.preprocess(transfer_logits))

            # -----------------------------------------------------------------------------------------------
            # Define loss
            # -----------------------------------------------------------------------------------------------    
            # Content loss
            content_loss = None
            for i in range(len(net.content_list)):
                if content_loss is None:
                    content_loss = content_weight * tf.nn.l2_loss(net.getContentLayer(i) - content_feature[i])
                else:
                    content_loss += content_weight * tf.nn.l2_loss(net.getContentLayer(i) - content_feature[i])

            # Style loss
            style_loss = None
            for i, layer in enumerate(style_features):
                content_lrelu_layer = net.getStyleLayer(i)
                batch, height, width, channel = content_lrelu_layer.get_shape()
                flat_size = tensor_size_prod(content_lrelu_layer)
                feats = tf.reshape(content_lrelu_layer, [int(batch), int(height) * int(width), int(channel)])
                feats_T = tf.transpose(feats, perm=[0, 2, 1])
                grams = tf.matmul(feats_T, feats) / flat_size
                style_gram = style_features[i]
                if style_loss is None:
                    style_loss = style_weight * tf.nn.l2_loss(grams - style_gram) / batch_size / style_gram.size
                else:
                    style_loss += style_weight * tf.nn.l2_loss(grams - style_gram) / batch_size / style_gram.size

            # TV denoising
            tv_y_size = tensor_size_prod(transfer_logits[:, 1:, :, :])
            tv_x_size = tensor_size_prod(transfer_logits[:, :, 1:, :])
            y_tv = tf.nn.l2_loss(transfer_logits[:, 1:, :, :] - transfer_logits[:, :223, :, :])
            x_tv = tf.nn.l2_loss(transfer_logits[:, :, 1:, :] - transfer_logits[:, :, :399, :])
            tv_loss = tv_weight * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size

            # Total loss and optimizer
            loss = content_loss + style_loss + tv_loss
            train_op = tf.train.AdamOptimizer(0.002).minimize(content_loss)

            # Train
            try:
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, './lib/YOLO_tiny2.ckpt')
            except:
                pass
        
            # iteration = len(content_image_name_list) * epoch
            iteration = 2
            # if adopt_multiprocess == True:
                # img_queue = Queue()
                # get_img_proc = Process(target=get_img_batch_proc, args=(content_image_name_list, img_queue, iteration, batch_size))
                # get_img_proc.start()
# 
            img_batch = get_img_batch_random(content_image_name_list, batch_size=batch_size)


            for i in range(iteration):

                # Get batch image
                # if adopt_multiprocess == True:
                    # img_batch = img_queue.get()
                # else:
                    # img_batch = get_img_batch_random(content_image_name_list, batch_size=batch_size)

                # Update
                _ = sess.run([train_op], feed_dict={
                    content_ph: img_batch / 255.0
                })

                # Verbose
                if i % evaluate_period == 0:
                    _style_loss, _content_loss, _tv_loss, _loss = sess.run([style_loss, content_loss, tv_loss, loss], feed_dict={
                        content_ph: img_batch
                    })
                    print("epoch: ", i, '\tstyle: ', _style_loss, '\tcontent: ', _content_loss, '\tTV: ', _tv_loss, '\ttotal: ', _loss, '\ttime: ', datetime.datetime.now().time())

                    _style_result = sess.run([transfer_logits,], feed_dict={
                        content_ph: img_batch / 255.0
                    })
                    print('max: ', np.max(_style_result[0][0]))                    
                    _style_result = np.concatenate((img_batch[0], _style_result[0][0]), axis=1)
                    save_img(str(i) + '.png', _style_result)

            if adopt_multiprocess == True:
                get_img_proc.join()
            saver = tf.train.Saver()
            saver.save(sess, model_path + model_name)    

if __name__ == '__main__':
    style_image = get_img(style_path + style_name)
    content_image_name_list = get_content_imgs(content_path)
    # style_image = np.random.random([224, 224, 3])
    # content_image = np.random.random([1000, 224, 224, 3])
    train(content_image_name_list, style_image)