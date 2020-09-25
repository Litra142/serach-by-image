#author:Boyle time:2020/7/31
# 存放用于转换数据类型的方法
import base64
import numpy as np
import torch
import cv2

# 将base64格式数据转换为NumPy数组
def base64_to_np(img_base64):
    try:
        img_data = base64.b64decode(img_base64)
        nparr = np.fromstring(img_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img_np
    except:
        return None

# 将NumPy数组转换为base64格式数据
def np_to_base64(img_np):
    retval, buffer = cv2.imencode('.jpg', img_np)
    img_base64 = base64.b64encode(buffer)
    img_base64 = img_base64.decode()
    return img_base64

def np_to_str(np_arr):
    str_arr = ",".join(str(value) for value in np_arr.tolist())
    return str_arr

def str_to_np(str_arr):
    list_arr = [float(value) for value in str_arr.split(',')]
    return np.array(list_arr)

def str_series_to_mat(str_series):
    res_list = [[float(value) for value in json.split(',')] for json in str_series.values]
    return np.mat(res_list)

def image_tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img


def image_np_to_tensor(img):
    tensor = torch.from_numpy(img.transpose((2,0,1)))
    tensor = tensor.float().div(255)
    return tensor