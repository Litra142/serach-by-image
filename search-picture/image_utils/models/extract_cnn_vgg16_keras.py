import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.models import Model,Sequential,load_model
from numpy import linalg as LA
from PIL import Image as pil_image
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from keras.applications.densenet import preprocess_input as preprocess_input_densenet
import cv2

class VGGNet:
    def __init__(self,recognition_model_type="VGG16",model_path=None,weight=None):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (224, 224, 3)
        self.weight = weight
        self.pooling = 'max'
        self.recognition_model_type = recognition_model_type
        self.model_path = model_path
        # include_top：是否保留顶层的3个全连接网络
        # weights：None代表随机初始化，即不加载预训练权重。'imagenet'代表加载预训练权重
        # input_tensor：可填入Keras tensor作为模型的图像输出tensor
        # input_shape：可选，仅当include_top=False有效，应为长为3的tuple，指明输入图片的shape，图片的宽高必须大于48，如(200,200,3)
        # pooling：当include_top = False时，该参数指定了池化方式。None代表不池化，最后一个卷积层的输出为4D张量。‘avg’代表全局平均池化，‘max’代表全局最大值池化。
        # classes：可选，图片分类的类别数，仅当include_top = True并且不加载预训练权重时可用。
        self.model = self._model_build(self.input_shape,self.weight,self.pooling)
        
    def _model_build(self,input_shape=None, weight=None, pooling=None):
        """模型建立
        Parameters:
            input_shape:
            weight:
            pooling:
        Return:
        """
        if self.recognition_model_type == "VGG16":
            model = VGG16(weights=weight,
                          input_shape=input_shape,
                          pooling=pooling, include_top=False)
            model.load_weights(self.model_path)
            try:
                model.predict(np.zeros((1, 224, 224, 3)))
            except Exception as e:
                pass
            return model
        elif self.recognition_model_type == "ResNet50":
            model = ResNet50(weights=weight,
                             input_shape=input_shape,
                             pooling=pooling, include_top=False)
            model.load_weights(self.model_path)
            try:
                model.predict(np.zeros((1, 224, 224, 3)))
            except Exception as e:
                pass
            return model
        elif self.recognition_model_type == "DenseNet121":
            model = DenseNet121(weights=weight,
                                input_shape=input_shape,
                                pooling=pooling, include_top=False)
            model.load_weights(self.model_path)
            try:
                model.predict(np.zeros((1, 224, 224, 3)))
            except Exception as e:
                pass
            return model
        else:
            raise ValueError("The pretrained model is wrong")


    '''
   利用vgg16/Resnet模型提取特征归一化特征向量
    '''
    # 提取最后一层卷积特征
    def extract_feat(self, img):
        """
        最后一层卷积特征
        Parameters:
            img:图片
        Return:
            norm_feat:归一化后的特征数据
        """

        target_size = (self.input_shape[0], self.input_shape[1])
        img = cv2.resize(img,dsize=target_size)
        img = np.expand_dims(img, axis=0)
        img = img.astype('float64')
        # img = img.reshape((self.input_shape[0], self.input_shape[1],3))
        if self.recognition_model_type == "VGG16":
            # img = img.astype('float64')
            img = preprocess_input_vgg(img)
        elif self.recognition_model_type == "ResNet50":
            # img = img.astype('float64')
            img = preprocess_input_resnet(img)
        elif self.recognition_model_type == "DenseNet121":
            # img = img.astype('float64')
            img = preprocess_input_densenet(img)
        else:
            raise ValueError("The pretrained model is wrong")
        # print(feat.shape)
        feat = self.model.predict(img)
        norm_feat = feat[0] / LA.norm(feat[0])
        return norm_feat
