import _init_path
from autoencoder import small_net as SmallAutoEncoder
from utils import list_files, get_img, save_img
from autoencoder import net as AutoEncoder
from config import *
import tensorflow as tf
import numpy as np
import subprocess
import threading
import argparse
import os

# Flag to control if use inception structure & vgg revision
adopt_revision = False

def work(in_files, out_files, device_id, total_device, device_idx):
    """
        Stylized the images
        This function supports multi-GPU transformation

        Arg:    in_files        - The path list of input images
                out_files       - The path list of output images
                device_id       - The name of device which is following the rule that tensorflow makes
                total_device    - The total number of devices
                devices_idx     - The index of the current device
    """
    global adopt_revision

    with tf.Graph().as_default():
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True

        # Construct graph
        img_ph = tf.placeholder(tf.float32, shape=image_shape)
        if adopt_revision == True:
            logits = SmallAutoEncoder(img_ph)
        else:
            logits = AutoEncoder(img_ph)

        # Run
        with tf.Session(config=tf_config) as sess:
            with tf.device(device_id):              # Adopt multi-GPU to transfer the image
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.restore(sess, model_path + model_name)           

                if total_device <= 1:
                    start = 0
                    end = int(len(in_files) // 1)
                else:
                    start = device_idx * int(len(in_files) // total_device)
                    end = device_idx * int(len(in_files) // total_device) + int(len(in_files) // total_device)
                for i in range(start, end, 1):
                    print("progress: ", i, ' / ', end - start, '\t proc: ', device_idx)
                    img_batch = np.ndarray(image_shape)
                    for j, img_path in enumerate(in_files[i : i+1]):
                        img = get_img(img_path)
                        img_batch[j] = img
                    _style_result = sess.run([logits,], feed_dict={
                        img_ph: img_batch / 255.0
                    })
                    for j, img_path in enumerate(out_files[i : i+1]):
                        save_img(img_path, _style_result[0][j])
            
def stylize_video(in_files, out_files):
    """ 
        Stylized the video
        This function is a wrapper function, and it will call work function indirectly

        Arg:    in_files    - The path list of input images
                out_files   - The path list of output images
    """
    global device_list

    # Update GPU index
    if adopt_multiprocess == False:
        work(in_files, out_files, device_list[0], 1, 0)
    else:
        thread_list = []
        with tf.Graph().as_default():
            for i in range(len(device_list)):
                _thread = threading.Thread(target=work, args=(in_files, out_files, device_list[i], len(device_list), i))
                thread_list.append(_thread)
                _thread.start()
            for i in range(len(device_list)):
                thread_list[i].join()            

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