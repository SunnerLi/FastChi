# FastChi - The Simple Acceleration Version of Fast Style Transfer toward Chi-chi Video

[![Packagist](https://img.shields.io/badge/Tensorflow-1.3.0-yellow.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.5.2-blue.svg)]()

![](https://raw.githubusercontent.com/SunnerLi/FastChi/another_vgg_version/img/logo.png)

Abstraction
---
By style transfer mechanism, the style of another image can be illustrated toward another target picture. However, the speed of the computation is a tough problem. In this article, we purposed an accelerating version to enhance this disadvantage. We use pure CNN based method which contains render network and feature extractor network to transfer the chi-chi （柴柴） video.  The size of render network is reduced and the graphic process unit (GPU) is used to accelerate the computation. We also use multi-thread to speed up the process of image reading. By this improvement, the speed of reading can be accelerated up to about **15** times, the speed of tensorflow computation can speed up to around **1.7** times, and the speed of transformation can be accelerated up to **1.86** times.

Usage
---
Pre-install & download pre-train model and MS COCO training images:
```
$ sudo apt-get install ffmpeg
```

Train stylized model
```
$ python style.py --mode origin      # Don't adopt our revision
$ python style.py --mode inception   # Adopt our revision
```

Stylized the video
```
$ python transform_video.py --mode origin      # Don't adopt our revision
$ python transform_video.py --mode inception   # Adopt our revision
```

Notice
---
1. You should ensure the mode between stylized and transform is the same, or the graph cannot be loaded correctly    
2. The default scale of inception revision is 2. If you want to change the scale, you should revise `shrink_scale` variable which is in `./lib/autoencoder.py` directly     
3. To avoid using too much argument during using our code, only 1 parameter can be assigned in the command line interface. If you want to change if you want to use multi-thread (also multi-GPU mechanism) or not, you should revise `adopt_multiprocess` variable which is in `./config.py` directly.    

Environment
---
All experiments are testing under Ubuntu 16.04. The corresponding hardware is the server which have multiple GeForce **GTX 1080 Ti** graphic processing units.    

Participate
---
[YenLiu](https://github.com/YenLiu1020)    
[Peter Chuang](https://github.com/Peter654q)    
[SunnerLi](https://github.com/SunnerLi)
