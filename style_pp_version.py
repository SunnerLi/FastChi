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
import vgg

saver = None            # Model save object
style_features = {}     # Store the gram matrix of style
content_feature = {}    # Store the gram matrix of content

# The list of layer name
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')
CONTENT_LAYER = ('relu4_2',)

# Flag to control if use inception structure & vgg revision
adopt_revision = False

def average_gradients(tower_grads):
    """Calculate average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been 
       averaged across all towers.
    """
    # for m in xrange(len(tower_grads)):
    #     for n in xrange(len(tower_grads[m])):
    #         print(type(tower_grads[0][n][0]))
    # for gg in tower_grads:
    #     for x in gg:
    #         print(type(x[0]))
    #     print(tower_grads)

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g != None:
            # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads,0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train(content_image_name_list, style_img):
    """
        Train the neural style transfer model

        Arg:    content_image_name_list - The name list of training data
                style_img               - The style image (ndarray type)
    """
    global style_features
    global content_feature
    global adopt_revision

    # -------------------------------------------------------------------------------------------------------------------
    # Precompute gram matrix of style frature
    # -------------------------------------------------------------------------------------------------------------------
    style_shape = (1,) + style_img.shape
    with tf.Graph().as_default(),tf.device('/cpu:0'):
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
        soft_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
        soft_config.gpu_options.allow_growth = True
        soft_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        train_step=tf.train.MomentumOptimizer(learning_rate,0.9,use_nesterov=True,use_locking=True)
        global_step=tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)
        tower_grades=[]
        #with tf.Session(config=soft_config) as sess:
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(1):
                with tf.device('/gpu:%d'%i):
                    with tf.name_scope('tower_%d'%i)as scope:
                        # Origin image path
                        content_ph = tf.placeholder(tf.float32, shape=(batch_size, image_shape[1], image_shape[2], image_shape[3]))
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
                        tf.get_collection('losses',scope)
                        summaries=tf.get_collection(tf.GraphKeys.SUMMARIES,scope)
                        grads=train_step.compute_gradients(loss,gate_gradients=2)
                        tower_grades.append(grads)
                        #train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            
        grads=average_gradients(tower_grades)
        train_op=train_step.apply_gradients(grads,global_step=global_step)
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
        with tf.Session(config=soft_config) as sess:
            sess.run(init_op)
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
            saver = tf.train.Saver(tf.global_variables(),sharded=True)
            res=saver.save(sess, model_path + model_name)

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