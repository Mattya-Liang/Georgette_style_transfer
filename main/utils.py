# -*- coding: utf-8 -*-
# @File    : utils.py
# @Author  : AaronJny
# @Time    : 2020/03/13
# @Desc    :
import tensorflow as tf
import settings
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 我们准备使用经典网络在imagenet数据集上的与训练权重，所以归一化时也要使用imagenet的平均值和标准差
image_mean = tf.constant([0.485, 0.456, 0.406])
image_std = tf.constant([0.299, 0.224, 0.225])


def normalization(x):
    """
    对输入图片x进行归一化，返回归一化的值
    """
    return (x - image_mean) / image_std


def load_images(image_path, width=settings.WIDTH, height=settings.HEIGHT):
    """
    加载并处理图片
    :param image_path:　图片路径
    :param width: 图片宽度
    :param height: 图片长度
    :return:　一个张量
    """
    # 加载文件
    x = tf.io.read_file(image_path)
    # 解码图片
    x = tf.image.decode_jpeg(x, channels=3)
    # 修改图片大小
    x = tf.image.resize(x, [height, width])
    x = x / 255.
    # 归一化
    x = normalization(x)
    x = tf.reshape(x, [1, height, width, 3])
    # 返回结果
    return x


def save_image(image, filename):
    x = tf.reshape(image, image.shape[1:])
    x = x * image_std + image_mean
    x = x * 255.
    x = tf.cast(x, tf.int32)
    x = tf.clip_by_value(x, 0, 255)
    x = tf.cast(x, tf.uint8)
    x = tf.image.encode_jpeg(x)
    tf.io.write_file(filename, x)

def load_images_from_directory(directory_path, target_size=(256, 256)):
    """
    从指定的文件夹加载所有图片，并将它们调整到指定的尺寸。
    :param directory_path: 文件夹路径
    :param target_size: 图片的目标尺寸，默认为(256, 256)
    :return: 加载并处理的图片列表
    """
    images = []
    for img_name in os.listdir(directory_path):
        img_path = os.path.join(directory_path, img_name)
        if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # 扩展维度以匹配模型的输入
            images.append(img_array)
    return images
