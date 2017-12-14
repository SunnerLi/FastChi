from __future__ import print_function
import functools
import vgg, pdb, time
import tensorflow as tf ,os
import numpy as np
import transform
from utils import get_img
import pp 
import sys

ppservers=()

#if len(sys.argv[1]):
    #ncpus=int(sys.argv[1])
    #job_server=pp.Server(ncpus,ppservers=ppservers)
#else:

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'
learning_rate=1e-3

def train_work(num):
    import tensorflow as tf 
    #train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #train = tf.global_variables_initializer()
    if num==1:
        train_step_1.run(feed_dict_1)
    if num==2:
        train_step_2.run(feed_dict_2)
    '''if num=0:
        train.run(feed_dict_1)
        train.run(feed_dict_2)
        return train'''

from djangosaml2.cache import DjangoSessionCacheAdapter

def patched_sync(self):
  self._set_objects(dict(self))
  self.session.modified = True
      
# np arr, np arr
def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=True):
    if slow:
        batch_size = 1
    mod = len(content_targets) % batch_size
    if mod > 0:
        print("Train set has been trimmed slightly..") 
        content_targets = content_targets[:-mod] 

    style_features = {}

    batch_shape = (batch_size/2,256,256,3)
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

    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        # precompute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        if slow:
            preds = tf.Variable(
                tf.random_normal(X_content.get_shape()) * 0.256
            )
            preds_pre = preds
        else:
            preds = transform.net(X_content/255.0)
            preds_pre = vgg.preprocess(preds)

        net = vgg.net(vgg_path, preds_pre)

        content_size = _tensor_size(content_features[CONTENT_LAYER])*(batch_size/2)
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        )

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

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / (batch_size/2)

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/(batch_size/2)
        global loss
        loss = content_loss + style_loss + tv_loss

        # overall loss
        #train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        global train_step_1,train_step_2
        train_step_1= tf.train.AdamOptimizer(learning_rate).minimize(loss)
        train_step_2= tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        import random
        uid = random.randint(1, 100)
        print("UID: %s" % uid)

        for epoch in range(epochs):
            
            num_examples = len(content_targets)
            iterations = 0
            num_examples=1500
            
            if num_examples%batch_size==0:
        	    run_times=num_examples//batch_size
            else:
                run_times=(num_examples//batch_size)+1
            
            for iterations in range(run_times):
                #print(run_times)
                #while iterations * batch_size < num_examples:
                start_time = time.time()
                job_server=pp.Server(2,ppservers=ppservers)
                curr = iterations * batch_size
                step = curr +batch_size
                #parallrl b/2
                X_batch_1 = np.zeros((batch_size//2,256,256,3), dtype=np.float32)
                X_batch_2 = np.zeros((batch_size//2,256,256,3), dtype=np.float32)
                for j, img_p in enumerate(content_targets[curr:step]):
                    if(j<batch_size/2):
                        X_batch_1[j] = get_img(img_p, (256,256,3)).astype(np.float32)
                    else:
                        temp=j-batch_size/2
                        X_batch_2[int(temp)]=get_img(img_p,(256,256,3)).astype(np.float32)
                    #X_batch_1[j]=get_img(img_p,(256,256,3)).astype(np.float32)
                #iterations += 1 parallel =batch/2
                assert X_batch_1.shape[0] == batch_size/2
                #----try to seperate the batch
                global feed_dict_1,feed_dict_2
                feed_dict_1= {
                    X_content:X_batch_1
                }    
                feed_dict_2={
                    X_content:X_batch_2
                }
                #---parallel run small 
                #train_step.run(feed_dict=feed_dict_1)
                #inputs=[(train_step_1,feed_dict_1),(train_step_2,feed_dict_2)]
                #jobs=[(input,job_server.submit(train_work,(input,)))for input in inputs]
                job_server.submit(train_work,(1,),())
                job_server.submit(train_work,(2,),())
                
                #print(feed_dict_1)
                #train_step = train_work(1,train_step)
                #train_step = train_work(2,train_step)
                end_time = time.time()
                delta_time = end_time - start_time
                
                if debug: 
                    print("UID: %s, batch time: %s" % (uid, delta_time))
   

                is_print_iter = int(iterations) % print_iterations == 0    

                if slow:
                    is_print_iter = epoch % print_iterations == 0
    
                is_last = epoch == 1 and iterations * batch_size >= num_examples
                #should_print=is_print_iter or is_last
                #print("%s %s" % (is_print_iter,is_last))
                #if should_print:
                job_server.wait()
                job_server.destroy()
                if iterations%5==0:
                    job_server.print_stats()
                    to_get= [style_loss, content_loss, tv_loss, loss, preds]

                    test_feed_dict_1= {
                        X_content:X_batch_1
                    }
                    test_feed_dict_2={
                        X_content:X_batch_2
                    }
                    test_feed_dict={
                        X_content:X_batch_1
                    }
                    #tup=sess.run(to_get,feed_dict=test_feed_dict)
                    tup_1=sess.run(to_get,feed_dict= test_feed_dict_1)
                    tup_2=sess.run(to_get,feed_dict=test_feed_dict_2)
                    #serial tup parallel tup_1
                    _style_loss,_content_loss,_tv_loss,_loss,_preds = tup_1
                    _s,_c,_t,_l,_p= tup_2
                    if _l>_loss:
                        continue
                        train_step_2=train_step_1
                    else:
                        _style_loss,_content_loss,_tv_loss,_loss,_preds=tup_2
                        train_step_1=train_step_2
                    losses = (_style_loss, _content_loss, _tv_loss, _loss)
                    if slow:
                        _preds = vgg.unprocess(_preds)
                    else:
                        saver = tf.train.Saver()
                        res = saver.save(sess, save_path)
                    print("%s %s %s %s" %losses)
                    job_server.print_stats()
                    yield (_preds, losses, iterations, epoch)


def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)