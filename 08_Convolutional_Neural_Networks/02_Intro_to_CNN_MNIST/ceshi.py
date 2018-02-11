# Introductory CNN Model: MNIST Digits
# ---------------------------------------
#
# 在这个例子中，我们会下载MNIST手写数字数据集并构造一个简单的卷积神经网络预测(0-9)之间的数字

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.python.framework import ops

ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# 下载并读取数据集
# 如果本地没有temp文件夹需要从Tensorflow官网下载
data_dir = 'temp'
mnist = read_data_sets(data_dir)

# 把mnist.train.images中的数据集变换成(28*28)的数据格式，原文件中以784维向量的形式保存
train_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])

# print(train_xdata.shape)  # 输出格式是numpy.ndarray的形式
# print(test_xdata.shape)
# # (55000, 28, 28)
# (10000, 28, 28)


train_labels = mnist.train.labels
test_labels = mnist.test.labels
print(train_labels.shape)  # (55000,)
print(train_labels[0:10])  # [7 3 4 6 1 8 1 0 9 8]
target_size = max(train_labels) + 1
print(target_size)  # 10

sess.close()
