import tensorflow as tf
import numpy as np

def get_img_batch(arr, idx):
    return np.zeros([32, 256, 256, 3])

def tensor_size_prod(tensor):
    batch, height, width, channel = tensor.get_shape()
    return int(height) * int(width) * int(channel)