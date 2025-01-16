# -*- coding: utf-8 -*-
# @File    : train.py
# @Author  : AaronJny
# @Time    : 2020/03/13
# @Desc    :
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from model import NeuralStyleTransferModel
import settings
import utils

# 创建模型
model = NeuralStyleTransferModel()

# 加载内容图片（需要被风格迁移化的照片）
content_image = utils.load_images(settings.CONTENT_IMAGE_PATH)

# 加载风格图片 每次更改要记得在setting.py里面进行修改路径

# 单一图像
# style_image = utils.load_images(settings.STYLE_IMAGE_PATH)

# 加载文件夹中的所有图片

# def load_images_from_dir(dir_path):
#     image_paths = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path)
#                    if fname.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
#     images = [Image.open(path) for path in image_paths]
#     return images

def load_images_from_dir(dir_path):
    image_paths = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path)
                   if fname.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
    
    processed_images = []
    for path in image_paths:
        # 使用 utils.load_images 处理每张图片
        processed_image = utils.load_images(path)
        processed_images.append(processed_image)
    
    return processed_images

# 获取风格图片的路径列表
style_images = load_images_from_dir(settings.STYLE_IMAGE_DIR_PATHS)

# 多张图像（存储在一个固定文件夹中）
# style_images = [utils.load_images(path) for path in settings.STYLE_IMAGE_DIR_PATHS]

# 计算出目标内容图片的内容特征备用
target_content_features = model([content_image, ])['content']

# 计算目标风格图片的风格特征

# 单一图像
# target_style_features = model([style_image, ])['style']

# 多张图像（存储在一个固定文件夹中）
# target_style_features = [model([style_image, ])['style'] for style_image in style_images]
target_style_features = []
for style_image in style_images:
    style_features = model(style_image)['style']
    target_style_features.append(style_features)

# 这里的M和N是图片的宽高乘积和通道数，用于计算内容loss和风格loss时的系数
# M代表图片的宽高乘积，可以改变参数去调整图片的大小，这里默认为450*300，可以改动Setting.py里的WIDTH和HEIGHT去改变图片大小
# N代表图片的通道数，也就是图片的深度，比如RGB图片的通道数就是3

M = settings.WIDTH * settings.HEIGHT
N = 3

# 我们现在计算Content Loss和Style Loss
# Content Loss是目标图片和生成图片在某一层的特征之间的MSE，MSE越小，说明两者的内容越接近
# Style Loss是目标图片和生成图片在某一层的特征之间的Gram Matrix之间的MSE，Gram Matrix是特征之间的相关性，MSE越小，说明两者的风格越接近
# 我们想要生成的图片，内容和目标图片一样，风格和目标图片一样，所以我们的目标是最小化Content Loss和Style Loss
# Content Loss 和 Style Loss 的加权系数可以自己调整，不同的加权系数会生成不同的风格图片
# Content Loss 和 Style Loss 的加权系数越大，生成的图片越接近目标图片，但是也会导致过拟合
# Content Loss 和 Style Loss 之间存在关系，关系是Content Loss 和 Style Loss 之间的数值差距，数值差距越大，生成的图片越不接近目标图片
# 所以我们需要在Content Loss 和 Style Loss 之间取得平衡，使得数值差距不要太大，也不要太小

# Content 部分：
# 这个部分不需要改动，我们只会将所有风格迁移到一张照片上
# 从line 55 到line 88 是计算内容图片的内容loss，无需进行较大的修改

# 计算出目标内容图片的内容特征备用

def _compute_content_loss(noise_features, target_features):
    """
    计算指定层上两个特征之间的内容loss
    :param noise_features: 噪声图片在指定层的特征
    :param target_features: 内容图片在指定层的特征
    """
    content_loss = tf.reduce_sum(tf.square(noise_features - target_features))
    # 计算系数
    x = 2. * M * N
    content_loss = content_loss / x
    return content_loss

# 计算并当前图片的内容loss

def compute_content_loss(noise_content_features):
    """
    计算并当前图片的内容loss
    :param noise_content_features: 噪声图片的内容特征
    """
    # 初始化内容损失
    content_losses = []
    # 加权计算内容损失
    for (noise_feature, factor), (target_feature, _) in zip(noise_content_features, target_content_features):
        layer_content_loss = _compute_content_loss(noise_feature, target_feature)
        content_losses.append(layer_content_loss * factor)
    return tf.reduce_sum(content_losses)


# Style 部分：

# 核心工具：gram_matrix

# 计算风格图片中的loss：我们使用格拉姆矩阵来计算风格loss

def gram_matrix(feature):
    """
    计算给定特征的格拉姆矩阵
    """
    # 先交换维度，把channel维度提到最前面
    x = tf.transpose(feature, perm=[2, 0, 1])
    # reshape，压缩成2d
    x = tf.reshape(x, (x.shape[0], -1))
    # 计算x和x的逆的乘积
    return x @ tf.transpose(x)

# 单一图像：
# 
# 计算并返回图片的风格loss
# 计算指定层上两个特征之间的风格loss
def _compute_style_loss(noise_feature, target_feature):
    """
    计算指定层上两个特征之间的风格loss
    :param noise_feature: 噪声图片在指定层的特征
    :param target_feature: 风格图片在指定层的特征
    """
    noise_gram_matrix = gram_matrix(noise_feature)
    style_gram_matrix = gram_matrix(target_feature)
    style_loss = tf.reduce_sum(tf.square(noise_gram_matrix - style_gram_matrix))
    # 计算系数
    x = 4. * (M ** 2) * (N ** 2)
    return style_loss / x

# 多张图像：

# 多张图像，所有权重相同：
def compute_combined_style_features(style_images):
    """
    合成多张风格图像的风格特征，假设所有权重相同
    :param style_images: 风格图像列表
    :return: 合成后的风格特征
    """
    combined_style_features = []
    
    # 提取每张风格图像的风格特征
    for style_image in style_images:
        style_features = model([style_image])['style']  # 提取风格特征
        combined_style_features.append(style_features)
    
    # 计算所有风格图像特征的平均值
    combined_style_features = [tf.reduce_mean(tf.stack(features), axis=0) for features in zip(*combined_style_features)]
    
    return combined_style_features

# # 多张图像，权重不同：

# # 自选权重
# style_weights = []

# def compute_weighted_style_features(style_images, style_weights):
#     """
#     合成多张风格图像的风格特征，并根据指定的权重进行加权
#     :param style_images: 风格图像列表
#     :param style_weights: 风格图像对应的权重列表
#     :return: 合成后的加权风格特征
#     """
#     weighted_style_features = []
    
#     # 提取每张风格图像的风格特征
#     for style_image in style_images:
#         style_features = model([style_image])['style']  # 提取风格特征
#         weighted_style_features.append(style_features)
    
#     # 对每一层的特征进行加权
#     weighted_combined_style_features = []
#     for features_at_layer in zip(*weighted_style_features):
#         weighted_features_at_layer = [features * weight for features, weight in zip(features_at_layer, style_weights)]
#         weighted_combined_style_features.append(tf.reduce_sum(tf.stack(weighted_features_at_layer), axis=0))
    
#     return weighted_combined_style_features

# # 在多张照片且权重不同的情况下，计算风格图片中的loss，weighted_style_features会被用在后面的计算中，具体在compute_style_loss函数中
# weighted_style_features = compute_weighted_style_features(style_images, style_weights)

# 单一图像的 Style Loss 计算
# def compute_style_loss(noise_style_features):
#     """
#     计算并返回图片的风格loss
#     :param noise_style_features: 噪声图片的风格特征
#     """
#     style_losses = []
#     for (noise_feature, factor), (target_feature, _) in zip(noise_style_features, target_style_features):
#         layer_style_loss = _compute_style_loss(noise_feature, target_feature)
#         style_losses.append(layer_style_loss * factor)
#     return tf.reduce_sum(style_losses)

# 多个图像的 Style Loss 计算
# def compute_weighted_style_loss(noise_style_features):
#     style_losses = []
#     for (noise_feature, factor), (target_features, _) in zip(noise_style_features, target_style_features):
#         # 对每一层的风格特征计算损失
#         layer_style_loss = 0
#         for noise_feat, target_feat in zip(noise_feature, target_features):
#             layer_style_loss += _compute_style_loss(noise_feat, target_feat)
#         style_losses.append(layer_style_loss * factor)
#     return tf.reduce_sum(style_losses)

def compute_weighted_style_loss(noise_style_features):
    """
    计算加权风格损失
    """
    style_losses = []
    # 对每个风格特征层计算损失
    for (noise_feature, factor) in noise_style_features:
        layer_style_loss = 0
        # 对每个风格图片计算损失
        for target_style_feature in target_style_features:
            target_feature = target_style_feature[len(style_losses)][0]  # 获取对应层的特征
            layer_style_loss += _compute_style_loss(noise_feature, target_feature)
        # 取平均值
        layer_style_loss /= len(target_style_features)
        style_losses.append(layer_style_loss * factor)
    return tf.reduce_sum(style_losses)

# 计算总损失
# Content Loss 和 Style Loss 的加权和就是我们的总Loss
def total_loss(noise_features):
    """
    计算总损失
    :param noise_features: 噪声图片特征数据
    """
    content_loss = compute_content_loss(noise_features['content'])

    # 单一图像
    # style_loss = compute_style_loss(noise_features['style']) 

    # 多张图像
    style_loss = compute_weighted_style_loss(noise_features['style']) 

    return content_loss * settings.CONTENT_LOSS_FACTOR + style_loss * settings.STYLE_LOSS_FACTOR


# 使用Adma优化器
optimizer = tf.keras.optimizers.Adam(settings.LEARNING_RATE)

# 基于内容图片随机生成一张噪声图片
noise_image = tf.Variable((content_image + np.random.uniform(-0.2, 0.2, (1, settings.HEIGHT, settings.WIDTH, 3))) / 2)


# 使用tf.function加速训练
@tf.function
def train_one_step():
    """
    一次迭代过程
    """
    # 求loss
    with tf.GradientTape() as tape:
        noise_outputs = model(noise_image)
        loss = total_loss(noise_outputs)
    # 求梯度
    grad = tape.gradient(loss, noise_image)
    # 梯度下降，更新噪声图片
    optimizer.apply_gradients([(grad, noise_image)])
    return loss

# main.py中添加噪声控制
noise_image = tf.Variable((content_image + np.random.uniform(-0.1, 0.1, (1, settings.HEIGHT, settings.WIDTH, 3))) / 2)  # 减小噪声范围

# 添加图像预处理
def preprocess_image(image):
    # 增加对比度
    image = tf.image.adjust_contrast(image, 1.2)
    # 保持图像值在合理范围内
    return tf.clip_by_value(image, 0.0, 1.0)

def enhance_image_features(image):
    """增强图像特征"""
    # 增加对比度
    image = tf.image.adjust_contrast(image, 1.3)
    # 增加锐度
    image = tf.image.adjust_saturation(image, 1.2)
    # 值域裁剪
    return tf.clip_by_value(image, 0.0, 1.0)

# 修改噪声生成
noise_image = tf.Variable((content_image + np.random.uniform(-0.1, 0.1, (1, settings.HEIGHT, settings.WIDTH, 3))) / 2)
# 如果没有文件夹的话就创建保存生成图片的文件夹
if not os.path.exists(settings.OUTPUT_DIR):
    os.mkdir(settings.OUTPUT_DIR)

# 共训练  settings.EPOCHS  个epochs，慢慢在setting里面调整
for epoch in range(settings.EPOCHS):
    # 使用tqdm提示训练进度
    with tqdm(total=settings.STEPS_PER_EPOCH, desc='Epoch {}/{}'.format(epoch + 1, settings.EPOCHS)) as pbar:
        # 每个epoch训练  settings.STEPS_PER_EPOCH  次
        for step in range(settings.STEPS_PER_EPOCH):
            _loss = train_one_step()
            pbar.set_postfix({'loss': '%.4f' % float(_loss)})
            pbar.update(1)
        # 每个epoch保存一次图片,记住每个照片都要看一下，看看里面的噪声有没有减少，防止过拟合
        utils.save_image(noise_image, '{}/{}.jpg'.format(settings.OUTPUT_DIR, epoch + 1))
