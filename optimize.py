from __future__ import print_function
import functools
import vgg, pdb, time
import tensorflow as tf, numpy as np, os
import transform
from utils import get_img

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'

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
        grad = tf.concat(0,grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

# np arr, np arr
def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False):
    if slow:
        batch_size = 1
    mod = len(content_targets) % batch_size
    if mod > 0:
        print("Train set has been trimmed slightly..")
        content_targets = content_targets[:-mod] 

    style_features = {}

    batch_shape = (batch_size,256,256,3)
    style_shape = (1,) + style_target.shape
    print(style_shape)

    # precompute style features
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        style_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    with tf.Graph().as_default(),tf.device('/cpu:0'):
        train_step = tf.train.MomentumOptimizer(learning_rate,0.9,use_nesterov=True,use_locking=True)
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False
            )
        tower_grades=[]
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(1):
                with tf.device('/gpu:%d'% i):
                    with tf.name_scope('tower_%d'%i)as scope:

                        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
                        X_pre = vgg.preprocess(X_content)

                        # precompute content features
                        content_features = {}
                        content_net = vgg.net(vgg_path, X_pre)
                        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

                        if slow:
                            preds = tf.Variable(tf.random_normal(X_content.get_shape()) * 0.256)
                            preds_pre = preds
                        else:
                            preds = transform.net(X_content/255.0)
                            preds_pre = vgg.preprocess(preds)

                        net = vgg.net(vgg_path, preds_pre)

                        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
                        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
                        content_loss = content_weight * (2 * tf.nn.l2_loss(net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size)

                        style_losses = []
                        for style_layer in STYLE_LAYERS:
                            layer = net[style_layer]
                            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
                            size = height * width * filters
                            feats = tf.reshape(layer, (bs, height * width, filters))
                            feats_T = tf.transpose(feats, perm=[0,2,1])
                            grams = tf.matmul(feats_T, feats) / size
                            style_gram = style_features[style_layer]
                            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

                        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

                        # total variation denoising
                        tv_y_size = _tensor_size(preds[:,1:,:,:])
                        tv_x_size = _tensor_size(preds[:,:,1:,:])
                        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
                        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
                        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size

                        loss = content_loss + style_loss + tv_loss
                        tf.get_collection('losses',scope)
                        #total_loss=tf.add_n(losses,name='total_loss')
                        summaries=tf.get_collection(tf.GraphKeys.SUMMARIES,scope)
                        grads=train_step.compute_gradients(loss,gate_gradients=2)
                        tower_grades.append(grads)
                        
        grads=average_gradients(tower_grades)
        train_op=train_step.apply_gradients(grads,global_step=global_step)
        #summary_op= tf.summary.merge(summaries)
        init_op=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

        gpu_options=tf.GPUOptions(allow_growth=True)
        
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False,gpu_options=gpu_options)) as sess:
            # overall loss
            #train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

            # sess.run(tf.global_variables_initializer())
            sess.run(init_op)

            import random
            uid = random.randint(1, 100)
            print("UID: %s" % uid)
            for epoch in range(epochs):
                num_examples = len(content_targets)
                iterations = 0
                num_examples=150
                while iterations * batch_size < num_examples:
                    start_time = time.time()
                    curr = iterations * batch_size
                    step = curr + batch_size
                    X_batch = np.zeros(batch_shape, dtype=np.float32)
                    for j, img_p in enumerate(content_targets[curr:step]):
                        X_batch[j] = get_img(img_p, (256,256,3)).astype(np.float32)

                    iterations += 1
                    assert X_batch.shape[0] == batch_size

                    feed_dict = {
                        X_content:X_batch
                    }
                    #print("X_content shape: ", np.shape(X_batch))

                    train_op.run(feed_dict=feed_dict)
                    end_time = time.time()
                    delta_time = end_time - start_time
                    if debug:
                        print("UID: %s, batch time: %s" % (uid, delta_time))

                        '''is_print_iter = int(iterations) % print_iterations == 0
                        if slow:
                            is_print_iter = epoch % print_iterations == 0
                        is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                        should_print = is_print_iter or is_last'''

                    if iterations%5==0:
                        to_get = [style_loss, content_loss, tv_loss, loss, preds]
                        test_feed_dict = {
                        X_content:X_batch
                        }

                        tup = sess.run(to_get, feed_dict = test_feed_dict)
                        _style_loss,_content_loss,_tv_loss,_loss,_preds = tup
                        losses = (_style_loss, _content_loss, _tv_loss, _loss)
                        if slow:
                            _preds = vgg.unprocess(_preds)
                        else:
                            saver = tf.train.Saver(tf.global_variables(),sharded=True)
                            res = saver.save(sess, save_path)
                        yield(_preds, losses, iterations, epoch)

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)