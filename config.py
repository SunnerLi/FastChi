epoch = 10
model_path = './model/'
model_name = 'chi.ckpt'
style_path = './img/style/'
style_name = 'star.jpg'
content_path = 'data/train2014'
batch_size = 32
num_example = 64

evaluate_period = 10

adopt_multiprocess = True

# Different loss weight
content_weight = 1.0
style_weight = 1.0
tv_weight = 1.0
