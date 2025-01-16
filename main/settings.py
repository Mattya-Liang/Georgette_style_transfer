# -*- coding: utf-8 -*-
# @File    : settings.py
# @Author  : AaronJny
# @Time    : 2020/03/13
# @Desc    :

# 以防我忘记了

# Model采用的是VGG19，这里定义了VGG19的各层名称
# VGG的层数越低，提取的是图片的低层特征，如边缘等；层数越高，提取的是图片的高层特征，如纹理等
# 要根据Style的风格去选择合适的层，比如选择靠前的层，提取的是图片的低层特征，这样生成的图片会更加清晰；选择靠后的层，提取的是图片的高层特征，这样生成的图片会更加抽象
# VGG19一共有5个卷积块，每个卷积块有2-4个卷积层，一共是 19 层
# 第一个卷积块有2个卷积层，第二个卷积块有2个卷积层，第三个卷积块有4个卷积层，第四个卷积块有4个卷积层，第五个卷积块有4个卷积层
# VGG19的卷积层的名称是这样的：block{卷积块编号}_conv{卷积层编号}
# 第一个卷积层提取的是图片的低层特征，第五个卷积层提取的是图片的高层特征
# 这里的CONTENT_LAYERS和STYLE_LAYERS是用来定义内容图片和风格图片分别采用的VGG层
# CONTENT_LAYERS的加权系数意味着在计算内容loss时，不同层的特征对loss的影响程度
# 这里我们默认为block4_conv2和block5_conv2，这两层是VGG19的最后两层卷积层，越往后的卷基层提取的特征越抽象
# 关于VGG19的更多信息，可以参考：https://arxiv.org/abs/1409.1556
import os
# 内容特征层及loss加权系数
CONTENT_LAYERS = {'block4_conv2': 0.8, 'block5_conv2': 0.2}

# 风格特征层及loss加权系数
# STYLE_LAYERS = {'block1_conv1': 0.25, 
#                 'block2_conv1': 0.25, 
#                 'block3_conv1': 0.2, 
#                 'block4_conv1': 0.2, 
#                 'block5_conv1': 0.1}
STYLE_LAYERS = {
    'block1_conv1': 0.3,    # 增加低层权重，捕捉线条
    'block2_conv1': 0.3,    # 增加中低层权重，捕捉纹理
    'block3_conv1': 0.2,    # 保持中层权重
    'block4_conv1': 0.15,   # 降低高层权重
    'block5_conv1': 0.05    # 降低最高层权重
    }

# 内容loss总加权系数
CONTENT_LOSS_FACTOR = 3
# 风格loss总加权系数
STYLE_LOSS_FACTOR = 70

# 图片宽度
WIDTH = 512
# 图片高度
HEIGHT = 512

# 在settings.py中定义目标图像尺寸
STYLE_IMAGE_HEIGHT = 1008  # 或者你需要的其他高度
STYLE_IMAGE_WIDTH = 756   # 或者你需要的其他宽度
# 在settings.py中添加

# 图像预处理参数
IMAGE_PREPROCESSING = {
    'target_size': (HEIGHT, WIDTH),  # 目标尺寸
    'preserve_aspect_ratio': True,   # 保持宽高比
    'interpolation': 'bilinear'      # 插值方法
}

# 训练epoch数
EPOCHS = 5
# 每个epoch训练多少次
STEPS_PER_EPOCH = 300
# 学习率
LEARNING_RATE = 0.00008

# 内容图片路径
CONTENT_IMAGE_PATH = r'C:\Users\12978\Desktop\Model Selection\DeepLearningExamples-master\tf2-neural-style-transfer\images\train_photo.jpg'

# 风格图片路径

# 单一照片
# STYLE_IMAGE_PATH = 'C:\\Users\\12978\\Desktop\\Model Selection\\DeepLearningExamples-master\\tf2-neural-style-transfer\\images\\微信图片_20250115153546.png'

# 多张照片（存储在一个固定文件夹中）

# 在settings.py中定义风格图片文件夹路径
STYLE_IMAGE_DIR_PATHS = r"C:\Users\12978\Desktop\Model Selection\DeepLearningExamples-master\tf2-neural-style-transfer\style_data"

output_folder_name = f"content_{CONTENT_LOSS_FACTOR} style_{STYLE_LOSS_FACTOR} output_lr{LEARNING_RATE} epochs{EPOCHS}"
# 生成图片的保存目录
# OUTPUT_DIR = r'C:\Users\12978\Desktop\Model Selection\DeepLearningExamples-master\tf2-neural-style-transfer\output_images'
OUTPUT_DIR = os.path.join(r'C:\Users\12978\Desktop\Model Selection\DeepLearningExamples-master\tf2-neural-style-transfer\output_images', output_folder_name)