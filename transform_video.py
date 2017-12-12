import _init_path
from AutoEncoder import AutoEncoder
from multiprocessing import Process
from utils import list_files, get_img, save_img
from config import *
import tensorflow as tf
import numpy as np
import subprocess
import os

def work(in_files, out_files, device_id, total_device, device_idx):
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True

    # Construct graph
    shape = (1, 224, 400, 3)
    img_ph = tf.placeholder(tf.float32, shape=shape)
    net = AutoEncoder()
    logits = net.build(img_ph)

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_path + model_name)

        if total_device <= 1:
            start = 0
            end = int(len(in_files) // 1)
        else:
            start = device_idx * int(len(in_files) // 1) / total_device
            end = device_idx * int(len(in_files) // 1) / total_device + int(len(in_files) // 1) / total_device
        for i in range(start, end, 1):
            img_batch = np.ndarray(shape)
            for j, img_path in enumerate(in_files[i : i+1]):
                img = get_img(img_path)
                img_batch[j] = img
            _style_result = sess.run([logits,], feed_dict={
                img_ph: img_batch / 255.0
            })
            for j, img_path in enumerate(out_files[i : i+1]):
                save_img(img_path, _style_result[0][j])

def stylize_video(in_files, out_files):
    global device_list

    # Update GPU index
    if adopt_multiprocess == False:
        with tf.Graph().as_default():
            p = Process(target=work, args=(in_files, out_files, device_list[0], 1, 0))
            p.start()
            p.join()
    else:
        proc_list = []
        with tf.Graph().as_default():
            for i in range(len(device_list)):
                p = Process(target=work, args=(in_files, out_files, device_list[i], len(device_list, i)))
                proc_list.append(p)
                p.start()
            for i in range(len(device_list)):
                proc_list[i].join()            

if __name__ == '__main__':
    # Check the folder is exist
    if not os.path.exists(model_path):
        print('You should train style model first...')
        exit()
    if not os.path.exists(video_path):
        os.mkdir(video_path)
    in_dir = os.path.join(video_path, 'in')
    if not os.path.exists(in_dir):
        os.mkdir(in_dir)
    out_dir = os.path.join(video_path, 'out')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Decode video into images
    in_args = ['ffmpeg', '-i', video_path + video_input_name, '%s/frame_%%d.png' % in_dir]
    subprocess.call(" ".join(in_args), shell=True)

    # Assemble the list of the image name and transfer
    img_name_list = list_files(in_dir)
    in_files = [os.path.join(in_dir, x) for x in img_name_list]
    out_files = [os.path.join(out_dir, x) for x in img_name_list]
    stylize_video(in_files, out_files)

    # Encode as output video
    frame_per_second = 30
    out_args = ['ffmpeg', '-i', '%s/frame_%%d.png' % out_dir, '-f', 'mp4', '-q:v', '0', '-vcodec', 'mpeg4', '-r', str(frame_per_second), video_path + video_output_name]
    subprocess.call(" ".join(out_args), shell=True)