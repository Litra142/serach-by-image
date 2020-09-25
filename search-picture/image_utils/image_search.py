import traceback
import numpy as np
from image_utils.models.extract_cnn_vgg16_keras import VGGNet
from image_utils.image_mysql import Image_mysql
from transform_data import *
root_path = "image_utils/"


class ImageRocognition:
    def __init__(self, maxres=1,
                 score_threshold=0.75,
                 db_info=None,
                 table_name="image_information",
                 net_type="VGG16"):
        """
         Parameter:
            maxres: 检索出相似度最大的前 maxres 张图片,默认值为 1
            score_threshold: 相似度得分阈值，当相似度最大的图片的得分低于此值，则视为图像库中不包含查询图片，将查询图片录入到图像库中,默认值为 0.75
            db_info: 数据库信息
            table_name: 表格名称，默认：image_information
            net_type: 预训练好的提取图像特征网络,默认：VGG16网络,可选：VGG16、ResNet50、DenseNet121
        """
        self.maxres = maxres
        self.score_threshold = score_threshold
        self.db = Image_mysql(db_info, table_name)
        self.net_type = net_type
        self.net = self.load_reconition_model()

    def load_reconition_model(self):
        # 加载人脸特征提取模型
        if self.net_type == "VGG16":
            model_path = root_path + "models/pretrained/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
        elif self.net_type == "ResNet50":
            model_path = root_path + "models/pretrained/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
        elif self.net_type == "DenseNet121":
            model_path = root_path + "models/pretrained/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"
        else:
            raise ValueError("The pretrained model is wrong")

        recognition_net = VGGNet(self.net_type, model_path=model_path)
        return recognition_net

    def image_search_main(self, image):
        """
        图片查询主函数
        Parameter:
            image:查询图片（RGB通道格式）
        Return:
            self.statusCode,
            self.isSuccess,
            self.msg,
            uid:图片的uid
        """
        # 获得查询图片对应的特征向量
        queryvec = self.net.extract_feat(image)
        # 读取数据库的图片数据
        feature_mids, feature_uids, feature_strs, img_num, msg = self.db.load_img_info()
        feats = str_series_to_mat(feature_strs)
        feats = np.array(feats)
        try:
            status_code, issuccess, msg, uid_get = self.image_search_help(feats, feature_uids, queryvec, self.maxres)
            return status_code, issuccess, msg, str(uid_get)
        except Exception as e:
            print(traceback.format_exc())
            return 500, False, e.args, None

    def image_search_help(self, feats, feature_uids, queryvec, maxres):
        """
        图片查询帮助函数（计算相似度并返回相似度最大的1张(默认)图片
        Parameter:
            feats: 数据库中的图像特征
            feature_uids: 图像的索引
            queryvec:图像特征数组,ndarray形式
            maxres: 返回相似度前 maxres 张图片
        Returns:
            statusCode,isSuccess,msg,uids
        """
        try:
            # 计算相似度
            scores = np.dot(queryvec, feats.T)
            if maxres == 1:
                index = np.argmax(scores)
                uid_get = feature_uids[index]
            elif maxres > 1:
                rank_id = np.argsort(scores)[::-1]
                rank_score = scores[rank_id]
                uid_get = []
                for i, index in enumerate(rank_id[0:maxres]):
                    if rank_score[i] > self.score_threshold:
                        uid_get.append(str(feature_uids[index]))
                    else:
                        continue
            else:
                raise ValueError("the model of image search get 'maxres': %d" % maxres)
            msg = "Image query successful"
            return 200, True, msg, uid_get
        except Exception as e:
            print(traceback.format_exc())
            return 500, False, e.args,  None

    def image_insert(self, img, uid):
        """
        录入图片
        Parameters:
            img: 输入相片
            uid: 图片id
        Return: 录入结果（是否成功）
        """
        try:
            norm_feat = self.net.extract_feat(img)
            json = np_to_str(norm_feat)
            lastid = self.db.insert_img_info(uid, uid, json)
            if lastid is None or lastid < 0:
                msg = "An error occurred while inserting data into the database"
                return 500, False, msg
            else:
                return 200, True, None
        except Exception as e:
            print(traceback.format_exc())
            return 500, False, e.args
