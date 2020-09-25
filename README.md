# serach-by-image
实现以图搜图功能，连接mysql数据库
### 所需第三方依赖库以及文件介绍
1、IDE：Pycharm

2、Python：3.7.x

3、Packages：

Keras + theano  + Pillow + Numpy + flask + flask_httpauth + argparse

具体看requirements.txt文件。
一键安装：pip install -r requirements.txt

本项目使用到的keras框架是基于theano做为后端，将keras默认使用的tensorflow后端改为theano方法请自行百度。

4、文件内容：


-   image_utils

     
    - models
     

        - pretrained（预训练的模型权重文件（notop.h5文件））
       
          
            - densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5
          
          
            - resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
          
          
            - vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
          
       
        - extract_cnn_vgg16_keras.py
       
     
    - image_mysql.py：连接数据库文件
     
     
    - ImageRearch.py：以图搜图代码
     

 
- transform_data.py：数据类型转换文件

 
- app.py:以图搜图 搜索 & 录入图片 接口


### 运行

$ python app.py
     
或

$ run_app.sh

