# More Advanced CNN Model: CIFAR-10
# ---------------------------------------
#
# In this example, we will download the CIFAR-10 images
# and build a CNN model with dropout and regularization
# 在这个例子中，我们会下载CIFAR-10图像数据集并且利用dropout和标准化创建一个CNN模型
#
# CIFAR is composed ot 50k train and 10k test
# CIFAR数据集包含5W训练图片,和1W测试图片。图片是32*32个像素点组成的。
# images that are 32x32.

import os
import sys
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import urllib
from tensorflow.python.framework import ops

ops.reset_default_graph()

# Change Directory
# 获取当前文件绝对地址
abspath = os.path.abspath(__file__)
# print(abspath)
# E:\GitHub\TF_Cookbook\08_Convolutional_Neural_Networks\03_CNN_CIFAR10\ostest.py

# 获取绝对地址所在文件夹地址
dname = os.path.dirname(abspath)
# print(dname)
# E:\GitHub\TF_Cookbook\08_Convolutional_Neural_Networks\03_CNN_CIFAR10

# 更换工作目录
os.chdir(dname)

# Start a graph session
# 初始化Session
sess = tf.Session()

# 设置模型超参数
batch_size = 128  # 批处理数量
data_dir = 'temp'  # 数据目录
output_every = 50  # 输出训练loss值
generations = 20000  # 迭代次数
eval_every = 500  # 输出测试loss值
image_height = 32  # 图片高度
image_width = 32  # 图片宽度
crop_height = 24  # 裁剪后图片高度
crop_width = 24  # 裁剪后图片宽度
num_channels = 3  # 图片通道数
num_targets = 10  # 标签数
extract_folder = 'cifar-10-batches-bin'

# 指数学习速率衰减参数
learning_rate = 0.1  # 学习率
lr_decay = 0.1  # 学习率衰减速度
num_gens_to_wait = 250.  # 学习率更新周期

# 提取模型参数
image_vec_length = image_height*image_width*num_channels  # 将图片转化成向量所需大小
record_length = 1 + image_vec_length  # ( + 1 for the 0-9 label)

# 读取数据
data_dir = 'temp'
if not os.path.exists(data_dir):  # 当前目录下是否存在temp文件夹
    os.makedirs(data_dir)  # 如果当前文件目录下不存在这个文件夹，创建一个temp文件夹
#  设定CIFAR10下载路径
cifar10_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# Check if file exists, otherwise download it
data_file = os.path.join(data_dir, 'cifar-10-binary.tar.gz')
# print(data_file)  # temp\cifar-10-binary.tar.gz
if os.path.isfile(data_file):
    pass
else:
    # Download file
    def progress(block_num, block_size, total_size):
        progress_info = [cifar10_url, float(block_num*block_size)/float(total_size)*100.0]
        print('\r Downloading {} - {:.2f}%'.format(*progress_info), end="")


    filepath, _ = urllib.request.urlretrieve(cifar10_url, data_file, progress)
    # Extract file
    tarfile.open(filepath, 'r:gz').extractall(data_dir)

