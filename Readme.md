# FastChi – The Parallelized Fast Style Transfer toward Chi-chi video

[![Packagist](https://img.shields.io/badge/Tensorflow-1.3.0-yellow.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.5.2-blue.svg)]()

![](https://raw.githubusercontent.com/SunnerLi/FastChi/another_vgg_version/img/logo.png)

Abstraction
---
The style transfer is a very popular problem in recent year. Moreover, there’re some application which use such kind of transferring into their product. However, the speed and performance are important issues that should be conquered. In Prisma, the smart phone should spend amount of time to transfer the image. To accelerate the style transferring procedure, we purpose the parallel structure toward this task, and speed up the procedure of transforming. The network will split the video as two parts, and use two GPU to transform the style to the content image in parallelism. For our expectation, the video which contains chi-chi (柴柴) will be transferred in very quick speed. 

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
3. To avoid using too much argument during using our code, only 1 parameter can be assigned in the command line interface. If you want to change if you want to use multi-thread or not, you should revise `adopt_multiprocess` variable which is in `./config.py` directly.    

Participate
---
[YenLiu](https://github.com/YenLiu1020)    
[Peter Chuang](https://github.com/Peter654q)    
[SunnerLi](https://github.com/SunnerLi)
