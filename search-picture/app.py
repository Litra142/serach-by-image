import argparse
import requests
import numpy as np
import cv2
from flask import Flask, request, jsonify
from image_utils.image_search import ImageRocognition
from flask_httpauth import HTTPBasicAuth
from transform_data import *


"""
args参数获取
"""
parser = argparse.ArgumentParser(description="arguments of model")
parser.add_argument("--app_port", default=8000, type=int,
                    help='the port of the service')

# 图像查询参数
parser.add_argument("--image_search_maxres", default=1, type=int,
                    help="retrieves the maxres images with the highest similarity")
parser.add_argument("--image_search_threshold", default=0.75, type=int,
                    help="similarity score threshold")
parser.add_argument("--image_search_net_type", default="VGG16", type=str,
                    help="model is used to extract feature normalized feature vectors,optional: VGG16、ResNet50、DenseNet121")


# 数据库
parser.add_argument("--db_host", default="10.21.23.210", type=str,
                    help="database parameter")
parser.add_argument("--db_user", default="document-ml-test", type=str,
                    help="database parameter")
parser.add_argument("--db_password", default="topview", type=str,
                    help="database parameter")
parser.add_argument("--db_port", default=3306, type=int,
                    help="database parameter")
parser.add_argument("--db_database", default="document-ml-test", type=str,
                    help="database parameter")
parser.add_argument("--db_charset", default="utf8", type=str,

                    help="database parameter")
parser.add_argument("--image_table_name", default="image_information", type=str,
                    help="table name of image information")
args = parser.parse_args()


"""
初始化数据库和模型
"""
db_info = {}
db_info["host"] = args.db_host
db_info["user"] = args.db_user
db_info["password"] = args.db_password
db_info["port"] = args.db_port
db_info["database"] = args.db_database
db_info["charset"] = args.db_charset

model = ImageRocognition(maxres=args.image_search_maxres,
                         score_threshold=args.image_search_threshold,
                         db_info=db_info,
                         table_name=args.image_table_name,
                         net_type=args.image_search_net_type)

def get_cv2_image(url):
    # 获取图像（返回BGR的图像）
    r = requests.get(url)
    image = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    return image

# api
app = Flask(__name__)
auth = HTTPBasicAuth()

@app.route("/photo/entryInformation", methods=['POST'])
def photo_entryInformation():
    postJson = request.get_json()
    print("get the json:", postJson.keys)
    isSuccess = False
    status_code = 400
    # 传入url或image参数
    if "url" in postJson and postJson["url"]:
        url = postJson["url"]
        try:
            orig_image = get_cv2_image(url)
            # 将图像由BGR改为RGB
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        except:
            msg = "parameter 'url' error"
            return jsonify({"isSuccess": isSuccess,
                            "code": status_code,
                            "massage": msg})
    elif "image" in postJson and postJson["image"]:
        image_base64 = postJson["image"]
        try:
            image = base64_to_np(image_base64)
        except:
            msg = "input image format error"
            return jsonify({"isSuccess": isSuccess,
                            "code": status_code,
                            "massage": msg})
    else:
        msg = "missing required parameter 'url' or 'image' or the parameter is empty"
        return jsonify({"isSuccess": isSuccess,
                         "code": status_code,
                         "massage": msg})
    # 传入uid参数
    if "uid" not in postJson:
        msg = "missing required parameter 'uid'"
        return jsonify({"isSuccess": isSuccess,
                         "code": status_code,
                         "massage": msg})
    elif not postJson["uid"]:
        msg = "the parameter 'uid' is empty"
        return jsonify({"isSuccess": isSuccess,
                        "code": status_code,
                        "message": msg})
    else:
        uid = postJson["uid"]
    # 录入图片
    status_code, isSuccess, msg = model.image_insert(image, uid)
    return jsonify({"isSuccess": isSuccess,
                    "code": status_code,
                    "message": msg})

# 获取图片搜索接口
@app.route('/photo/photoRecognition', methods=['POST'])
def photo_photoRecognition():
    postJson = request.get_json()
    print("get the json:", postJson.keys)
    isSuccess = False
    status_code = 400
    # 传入url或image参数
    if "url" in postJson and postJson["url"]:
        url = postJson["url"]
        try:
            image = cv2.imread(url)
        except:
            msg = "parameter 'url' error"
            return jsonify({"isSuccess": isSuccess,
                     "code": status_code,
                    "message": msg,
                    "data": None
                            })
    elif "image" in postJson and postJson["image"]:
        image_base64 = postJson["image"]
        try:
            image = base64_to_np(image_base64)
        except:
            msg = "input image format error"
            return jsonify({"isSuccess": isSuccess,
                     "code": status_code,
                    "message": msg,
                    "data": None
                            })
    else:
        msg = "missing required parameter 'url' or 'image' or the parameter is empty"
        return jsonify({"isSuccess": isSuccess,
                     "code": status_code,
                    "message": msg,
                    "data": None
                            })
    status_code, isSuccess, msg, uid = model.image_search_main(image)
    return jsonify({"isSuccess": isSuccess,
                     "code": status_code,
                    "message": msg,
                    "data": uid
                            })

@auth.error_handler
def unauthorized():
    return jsonify({"isSuccess": False,
                     "code": 401,
                    "message": "Unauthorized access",})

@app.errorhandler(400)
def not_found(error):
    return jsonify({"isSuccess": False,
                     "code": 400,
                    "message": "Invalid data!",})

@app.errorhandler(500)
def intern_err(error):
    return jsonify({"isSuccess": False,
                    "code": 500,
                    "message": "Internal error!", })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=args.app_port, debug=False)
