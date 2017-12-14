import _init_path
from autoencoder import small_net as SmallAutoEncoder
# from autoencoder import net as AutoEncoder

from multiprocessing import Process, Queue
from functools import reduce
from config import *
from utils import *
import tensorflow as tf
import numpy as np
import datetime
import discriminator as vgg
# import vgg_shape as vgg
# import vgg

saver = None
style_features = {}     # Store the gram matrix of style
content_feature = {}    # Store the gram matrix of content

# STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')
# CONTENT_LAYER = ('relu4_2',)
STYLE_LAYERS = ('lrelu1_1', 'lrelu2_1', 'lrelu3_1')
CONTENT_LAYER = ('lrelu4_1',)

def train(content_image_name_list, style_img):
    global style_features
    global content_feature

    # Precompute gram matrix of style frature
    style_shape = (1,) + style_img.shape
    with tf.Graph().as_default():
        style_ph = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        with tf.Session() as sess:

            net = vgg.net(vgg_path, vgg.preprocess(style_ph))
            # net = vgg.net(vgg.preprocess(style_ph))

            sess.run(tf.global_variables_initializer())
            for layer_name in STYLE_LAYERS:
                feature = net[layer_name].eval(feed_dict={
                    style_ph: np.asarray([style_img])
                })
                feature = np.reshape(feature[0], (-1, feature.shape[3]))
                gram = np.matmul(feature.T, feature) / feature.size
                style_features[layer_name] =  gram
        sess.close()

    # Build network
    with tf.Graph().as_default():
        soft_config = tf.ConfigProto(allow_soft_placement=True)
        soft_config.gpu_options.allow_growth = True
        soft_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        with tf.Session(config=soft_config) as sess:
            content_ph = tf.placeholder(tf.float32, shape=(batch_size, image_shape[1], image_shape[2], image_shape[3]))

            net = vgg.net(vgg_path, vgg.preprocess(content_ph))
            # net = vgg.net(vgg.preprocess(content_ph))

            for layer_name in CONTENT_LAYER:
                content_feature[layer_name] = net[layer_name]

            # Build the main path of the graph
            transfer_logits = SmallAutoEncoder(content_ph / 255.0)
            # net = vgg.net(vgg_path, vgg.preprocess(transfer_logits))
            net = vgg.net(vgg_path, vgg.preprocess(transfer_logits), reuse=True)
            # net = vgg.net(vgg.preprocess(transfer_logits))

            # -----------------------------------------------------------------------------------------------
            # Define loss
            # -----------------------------------------------------------------------------------------------    
            # Content loss
            content_loss = 0.0
            content_losses = []
            for layer_name in CONTENT_LAYER:
                content_losses.append(content_weight * tf.nn.l2_loss(net[layer_name] - content_feature[layer_name]))
            content_loss += reduce(tf.add, content_losses)

            # Style loss
            style_loss = 0.0
            style_losses = []
            for layer_name in STYLE_LAYERS:
                content_lrelu_layer = net[layer_name]
                batch, height, width, channel = content_lrelu_layer.get_shape()
                flat_size = tensor_size_prod(content_lrelu_layer)
                feats = tf.reshape(content_lrelu_layer, [int(batch), int(height) * int(width), int(channel)])
                feats_T = tf.transpose(feats, perm=[0, 2, 1])
                grams = tf.matmul(feats_T, feats) / flat_size
                style_gram = style_features[layer_name]
                style_losses.append(style_weight * tf.nn.l2_loss(grams - style_gram) / batch_size / style_gram.size)
            style_loss += reduce(tf.add, style_losses)

            # TV denoising
            tv_y_size = tensor_size_prod(transfer_logits[:, 1:, :, :])
            tv_x_size = tensor_size_prod(transfer_logits[:, :, 1:, :])
            y_tv = tf.nn.l2_loss(transfer_logits[:, 1:, :, :] - transfer_logits[:, :image_shape[1] - 1, :, :])
            x_tv = tf.nn.l2_loss(transfer_logits[:, :, 1:, :] - transfer_logits[:, :, :image_shape[2] - 1, :])
            tv_loss = tv_weight * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size

            # Total loss and optimizer
            loss = content_loss + style_loss + tv_loss
            train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
            style_train_op = tf.train.AdamOptimizer(0.002).minimize(style_loss)

            # Train
            # iteration = 1000
            

            sess.run(tf.global_variables_initializer())

            # Train style first
            for i in range(style_pretrain_iteration):
                img_batch = get_img_batch_random(content_image_name_list, batch_size=batch_size)
                
                # Update
                _ = sess.run([style_train_op], feed_dict={
                    content_ph: img_batch
                })
                # Verbose
                if i % evaluate_period == 0:
                    _style_loss, _content_loss, _tv_loss, _loss = sess.run([style_loss, content_loss, tv_loss, loss], feed_dict={
                        content_ph: img_batch
                    })
                    print("epoch: ", i, '\tstyle: ', _style_loss, '\ttime: ', datetime.datetime.now().time())

                    _style_result = sess.run([transfer_logits,], feed_dict={
                        content_ph: img_batch
                    })
                    _style_result = np.concatenate((img_batch[0], _style_result[0][0]), axis=1)
                    save_img('style_' + str(i) + '.png', _style_result)


            # if adopt_multiprocess == True:
                # img_queue = Queue()
                # get_img_proc = Process(target=get_img_batch_proc, args=(content_image_name_list, img_queue, iteration, batch_size))
                # get_img_proc.start()
            # img_batch = get_img_batch_random(content_image_name_list, batch_size=batch_size)

            # Train mixture result
            for i in range(iteration):

                # Get batch image
                if adopt_multiprocess == True:
                    img_batch = img_queue.get()
                else:
                    img_batch = get_img_batch_random(content_image_name_list, batch_size=batch_size)

                # Update
                _ = sess.run([train_op], feed_dict={
                    content_ph: img_batch
                })

                # Verbose
                if i % evaluate_period == 0:
                    _style_loss, _content_loss, _tv_loss, _loss = sess.run([style_loss, content_loss, tv_loss, loss], feed_dict={
                        content_ph: img_batch
                    })
                    print("epoch: ", i, '\tstyle: ', _style_loss, '\tcontent: ', _content_loss, '\tTV: ', _tv_loss, '\ttotal: ', _loss, '\ttime: ', datetime.datetime.now().time())

                    _style_result = sess.run([transfer_logits,], feed_dict={
                        content_ph: img_batch
                    })
                    _style_result = np.concatenate((img_batch[0], _style_result[0][0]), axis=1)
                    save_img(str(i) + '.png', _style_result)

            if adopt_multiprocess == True:
                get_img_proc.join()
            saver = tf.train.Saver()
            saver.save(sess, model_path + model_name)    

if __name__ == '__main__':
    style_image = get_img(style_path + style_name)
    content_image_name_list = get_content_imgs(content_path)
    train(content_image_name_list, style_image)