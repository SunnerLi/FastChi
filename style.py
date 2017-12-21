import _init_path
from utils import tensor_size_prod, get_img_batch, get_img, get_content_imgs, get_img_batch_random, save_img, get_img_batch_proc
from autoencoder import small_net as SmallAutoEncoder
from autoencoder import net as AutoEncoder
from multiprocessing import Process, Queue
from autoencoder import net as AutoEncoder
from functools import reduce
from config import *
import tensorflow as tf
import numpy as np
import datetime
import argparse
import time
import vgg

saver = None            # Model save object
style_features = {}     # Store the gram matrix of style
content_feature = {}    # Store the gram matrix of content
tower_grades=[]         # The list object to contain gradient_and_variations

# The list of layer name
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')
CONTENT_LAYER = ('relu4_2',)

# Flag to control if use inception structure & vgg revision
adopt_revision = False

def average_gradients(tower_grads):
    """
        Calculate the average gradient
        This function is lent from here: https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py

        Arg:    tower_grads - The list of the gradients
        Ret:    The average gradient (tuple of gradient and variance)
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            if g != None:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
        grad = tf.concat(grads,0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def buildGraphAneLoss(gpu_idx, optimizer):
    """
        Build the graph of the whole network and loss function

        Arg:    gpu_idx     - The index of the GPU
                optimizer   - Tensorflow optimizer object
        Ret:    1. The image placeholder object
                2. The render network result op
                3. The style loss op
                4. The content loss op
                5. The total variation loss op
                6. The total loss op
    """
    global tower_grades

    with tf.device('/gpu:%d'%(gpu_idx)):
        with tf.name_scope('tower_%d'%(gpu_idx))as scope:
            # Origin image path
            if adopt_multiprocess == True:
                content_ph = tf.placeholder(tf.float32, shape=(batch_size // len(device_list), image_shape[1], image_shape[2], image_shape[3]))
            else:
                content_ph = tf.placeholder(tf.float32, shape=(batch_size, image_shape[1], image_shape[2], image_shape[3]))

            # Actual image path
            if adopt_revision == True:
                net = vgg.net(vgg_path, vgg.preprocess(content_ph), reduce=True, reuse=False)
            else:
                net = vgg.net(vgg_path, vgg.preprocess(content_ph), reduce=False, reuse=False)
            for layer_name in CONTENT_LAYER:
                content_feature[layer_name] = net[layer_name]

            # Render image path
            if adopt_revision == True:
                transfer_logits = SmallAutoEncoder(content_ph / 255.0)
                net = vgg.net(vgg_path, vgg.preprocess(transfer_logits), reduce=True, reuse=True)
            else:
                transfer_logits = AutoEncoder(content_ph / 255.0)
                net = vgg.net(vgg_path, vgg.preprocess(transfer_logits), reduce=False, reuse=True)

            # ---------------------------------------------------------------------------------------------------------
            # Define loss
            # ---------------------------------------------------------------------------------------------------------
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
            grads=optimizer.compute_gradients(loss,gate_gradients=2)
            tower_grades.append(grads)
    return content_ph, transfer_logits, style_loss, content_loss, tv_loss, loss

def train(content_image_name_list, style_img):
    """
        Train the neural style transfer model

        Arg:    content_image_name_list - The name list of training data
                style_img               - The style image (ndarray type)
    """
    global style_features
    global content_feature
    global adopt_revision
    global tower_grades

    # -------------------------------------------------------------------------------------------------------------------
    # Precompute gram matrix of style frature
    # -------------------------------------------------------------------------------------------------------------------
    style_shape = (1,) + style_img.shape
    with tf.Graph().as_default():
        style_ph = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        with tf.Session() as sess:
            if adopt_revision == True:
                net = vgg.net(vgg_path, vgg.preprocess(style_ph), reduce=True, reuse=False)
            else:
                net = vgg.net(vgg_path, vgg.preprocess(style_ph), reduce=False, reuse=False)
            sess.run(tf.global_variables_initializer())
            for layer_name in STYLE_LAYERS:
                feature = net[layer_name].eval(feed_dict={
                    style_ph: np.asarray([style_img])
                })
                feature = np.reshape(feature[0], (-1, feature.shape[3]))
                gram = np.matmul(feature.T, feature) / feature.size
                style_features[layer_name] =  gram

    # -------------------------------------------------------------------------------------------------------------------
    # Build network
    # -------------------------------------------------------------------------------------------------------------------
    with tf.Graph().as_default():
        soft_config = tf.ConfigProto(allow_soft_placement=True)
        soft_config.gpu_options.allow_growth = True
        soft_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        optimizer = tf.train.AdamOptimizer()

        with tf.Session(config=soft_config) as sess:
            with tf.variable_scope(tf.get_variable_scope()):
                content_ph1, transfer_logits1, style_loss1, content_loss1, tv_loss1, loss1 = buildGraphAneLoss(0, optimizer)
                if adopt_multiprocess == True:                
                    content_ph2, transfer_logits2, style_loss2, content_loss2, tv_loss2, loss2 = buildGraphAneLoss(1, optimizer)

            grads=average_gradients(tower_grades)
            train_op = optimizer.apply_gradients(grads)
            init_op=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

            # ---------------------------------------------------------------------------------------------------------
            # Train
            # ---------------------------------------------------------------------------------------------------------
            # If use multi-process to load image
            if adopt_multiprocess == True:
                img_queue = Queue()
                get_img_proc = Process(target=get_img_batch_proc, args=(content_image_name_list, img_queue, iteration, batch_size))
                get_img_proc.start()

            # Run
            sess.run(tf.global_variables_initializer())
            time_cost = time.time()
            for i in range(iteration):
                # Get batch image
                if adopt_multiprocess == True:
                    img_batch = img_queue.get()
                    feed_dict = {
                        content_ph1: img_batch[:len(img_batch) // 2],
                        content_ph2: img_batch[len(img_batch) // 2:]
                    }
                else:
                    img_batch = get_img_batch_random(content_image_name_list, batch_size=batch_size)
                    feed_dict = {
                        content_ph1: img_batch[:len(img_batch)]
                    }

                # Update
                _ = sess.run([train_op], feed_dict=feed_dict)

                # Verbose
                if i % evaluate_period == 0:
                    _style_loss1, _content_loss1, _tv_loss1, _loss1 = sess.run([style_loss1, content_loss1, tv_loss1, loss1], feed_dict=feed_dict)
                    print("epoch: ", i, '\tstyle: ', _style_loss1, '\tcontent: ', _content_loss1, '\tTV: ', _tv_loss1, '\ttotal: ', _loss1, '\tgpu idx: 0', '\ttime: ', datetime.datetime.now().time())
                    if adopt_multiprocess == True: 
                        _style_loss2, _content_loss2, _tv_loss2, _loss2 = sess.run([style_loss2, content_loss2, tv_loss2, loss2], feed_dict=feed_dict)
                        print("epoch: ", i, '\tstyle: ', _style_loss2, '\tcontent: ', _content_loss2, '\tTV: ', _tv_loss2, '\ttotal: ', _loss2, '\tgpu idx: 1', '\ttime: ', datetime.datetime.now().time())
                    _style_result = sess.run([transfer_logits1,], feed_dict=feed_dict)                   
                    _style_result = np.concatenate((img_batch[0], _style_result[0][0]), axis=1)
                    save_img(str(i) + '.png', _style_result)

            print("Training Time cost (s): ", time.time() - time_cost)
            if adopt_multiprocess == True:
                get_img_proc.join()
            saver = tf.train.Saver()
            saver.save(sess, model_path + model_name)    

if __name__ == '__main__':
    # Deal with parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='inception', dest='mode', help='Use inception or original version')
    args = parser.parse_args()
    if args.mode == 'inception':
        adopt_revision = True
        print("------------- Adopt inception mode -------------------")
    else:
        adopt_revision = False
        print("------------- Adopt original  mode -------------------")

    # Load image and train
    style_image = get_img(style_path + style_name)
    content_image_name_list = get_content_imgs(content_path)
    train(content_image_name_list, style_image)