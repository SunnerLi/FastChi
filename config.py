# Different loss weight
# --------------------------
# Inception hyper-parameter
# --------------------------
content_weight = 1e-3
style_weight = 2e4
tv_weight = 2e2

# --------------------------
# Random tiny-yolo hyper-parameter
# --------------------------
# content_weight = 1e-7
# style_weight = 1e0
# tv_weight = 2e-2

# Related input/output path
model_path = './model/'
model_name = 'chi.ckpt'
style_path = './img/style/'
style_name = 'star.jpg'
content_path = 'data/train2014'
video_path = './video/'
video_input_name = '20977333_328662897561035_2692550031910633472_n.mp4'
video_output_name = 'res.mp4'
vgg_path = './data/imagenet-vgg-verydeep-19.mat'
image_shape = (1, 224, 400, 3)

# Parameter about training
evaluate_period = 10
learning_rate = 0.002
num_example = 64
batch_size = 8
iteration = 75000

# Use multi-process
adopt_multiprocess = True
device_list = ['/gpu:0', '/gpu:0']