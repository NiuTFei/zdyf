import tempfile
import datetime
import random
import string
from minio import Minio, S3Error
from flask import Flask, request, jsonify

# 导入其余所需包
import os
import json
import numpy as np
import torch

from algorithm3.app import predict3

app = Flask(__name__)
app.debug = True
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def hello_world():
    return 'hello world!'


@app.route('/algorithm3', methods=["POST"])
def algorithm3():
    # 此处默认所需数据在predict函数中直接导入
    # 如需将各个算法的输入文件统一管理存放在某个文件夹下，并在调用接口时通过路径的方式进行调用，可自行更改，并将输入对应的文件移动至所需位置。
    response = predict3(None)

    return jsonify(response)



def pull_minio(file_dir):
    """
    从minio下载文件
    :param file_dir: minio对象地址
    :return: 本地文件下载路径
    """
    path = tempfile.gettempdir() + r"/"
    file_dir_array = file_dir.split(r"/")
    file_name = file_dir_array[-4] + r"/" + file_dir_array[-3] + r"/" + file_dir_array[-2] + r"/" + file_dir_array[-1]
    minioClient = Minio("117.73.3.232:9001",
                        access_key='inspur',
                        secret_key='inspuross',
                        secure=False)
    try:
        minioClient.fget_object(
            bucket_name="model",
            object_name=file_name,
            file_path=path + file_dir_array[-1]
        )
        return path + file_dir_array[-1]
    except S3Error as err:
        return None


def put_minio(file_path):
    """
    上传文件到minio
    :param file_path: 本地文件路径
    :return: 上传后的下载地址，不包含minio server地址
    """
    try:
        client = Minio("117.73.3.232:9001",
                       access_key='inspur',
                       secret_key='inspuross',
                       secure=False)
        today = datetime.datetime.today()
        if "." in file_path:
            suffix = file_path.split('.')[1]
        result = client.fput_object(bucket_name="model", object_name="dataset/" + str(today.year) + "/" + str(today.month)
                                                                     + "/" + generate_random_str(8) + "." + suffix,
                                    file_path=file_path)
        return result.bucket_name + "/" + result.object_name
    except Exception as e:
        return None


def generate_random_str(randomlength=16):
    """
    生成一个指定长度的随机字符串，其中
    string.digits=0123456789
    string.ascii_letters=abcdefghigklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
    """
    str_list = [random.choice(string.digits + string.ascii_letters) for i in range(randomlength)]
    random_str = ''.join(str_list)
    return random_str


if __name__ == '__main__':
    app.run(host='0.0.0.0')
