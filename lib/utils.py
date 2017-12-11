import scipy.misc, numpy as np, os, sys
import numpy as np
import time

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)

def scale_img(style_path, style_scale):
    scale = float(style_scale)
    o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = _get_img(style_path, img_size=new_shape)
    return style_target

def get_img(src, img_size=False):
   img = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
   if img_size != False:
       img = scipy.misc.imresize(img, img_size)
   return img

def exists(p, msg):
    assert os.path.exists(p), msg

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files

def get_content_imgs(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files]

def get_img_batch(arr, idx, batch_size=32):
    """
        Get batch data toward specific index
        I drop this function since the result cannot be slown quickly by this design
    """
    img_batch = np.ndarray([batch_size, 224, 400, 3])
    if idx * batch_size < len(arr):
        for j, img in enumerate(arr[idx * batch_size : idx * batch_size + batch_size]):
            img = get_img(img, (224, 400, 3))
            img = img.astype(np.float32)
            img_batch[j] = img
    else:
        for j, img in enumerate(arr[len(arr) - batch_size : ]):
            img = get_img(img, (224, 400, 3))
            img = img.astype(np.float32)
            img_batch[j] = img
    return img_batch

def get_img_batch_random(arr, batch_size=32):
    img_batch = np.ndarray([batch_size, 224, 400, 3])
    idx = np.arange(0 , len(arr))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    img_shuffle = [get_img(arr[i], (224, 400, 3)).astype(np.float32) for i in idx]
    return np.asarray(img_shuffle)

def get_img_batch_proc(arr, img_queue, times, batch_size=32):
    for i in range(times):
        # img_batch = get_img_batch(arr, i, batch_size=batch_size)
        img_batch = get_img_batch_random(arr, batch_size=batch_size)
        while True:
            if img_queue.qsize() > 20:
                time.sleep(5)
            else:
                img_queue.put(img_batch)
                break
    
def tensor_size_prod(tensor):
    batch, height, width, channel = tensor.get_shape()
    return int(height) * int(width) * int(channel)