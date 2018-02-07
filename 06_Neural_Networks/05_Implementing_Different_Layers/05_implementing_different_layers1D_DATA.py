# Implementing Different Layers
# ---------------------------------------
#
# We will illustrate how to use different types
# of layers in TensorFlow
#
# The layers of interest are:
#  (1) Convolutional Layer卷积层
#  (2) Activation Layer激活层
#  (3) Max-Pool Layer池化层
#  (4) Fully Connected Layer 全连接层
#
# We will generate two different data sets for this
#  script, a 1-D data set (row of data) and
#  a 2-D data set (similar to picture)

import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import os
import random
import numpy as np
import random
from tensorflow.python.framework import ops

ops.reset_default_graph()

# ---------------------------------------------------|
# -------------------1D-data-------------------------|
# ---------------------------------------------------|

# Create graph session 创建初始图结构
ops.reset_default_graph()
sess = tf.Session()

# parameters for the run运行参数
data_size = 25
conv_size = 5  # 卷积核宽度方向的大小
maxpool_size = 5  # 池化层核宽度方向上的大小
stride_size = 1  # 卷积核宽度方向上的步长

# ensure reproducibility 确保复现性
seed = 13
np.random.seed(seed)
tf.set_random_seed(seed)

# Generate 1D data 生成一维数据
data_1d = np.random.normal(size=data_size)

# Placeholder
x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])


# --------Convolution--------
def conv_layer_1d(input_1d, my_filter, stride):
    # TensorFlow's 'conv2d()' function only works with 4D arrays:
    # [batch, height, width, channels], we have 1 batch, and
    # width = 1, but height = the length of the input, and 1 channel.
    # So next we create the 4D array by inserting dimension 1's.
    # 关于数据维度的处理十分关键，因为tensorflow中卷积操作只支持四维的张量，
    # 所以要人为的把数据补充为4维数据[1,1,25,1]
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Perform convolution with stride = 1, if we wanted to increase the stride,
    # to say '2', then strides=[1,1,2,1]
    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1, 1, stride, 1], padding="VALID")
    # Get rid of extra dimensions 去掉多余的层数，只保留数字
    conv_output_1d = tf.squeeze(convolution_output)
    return (conv_output_1d)


# Create filter for convolution.
my_filter = tf.Variable(tf.random_normal(shape=[1, conv_size, 1, 1]))
# Create convolution layer
my_convolution_output = conv_layer_1d(x_input_1d, my_filter, stride=stride_size)


# --------Activation--------
def activation(input_1d):
    return (tf.nn.relu(input_1d))


# Create activation layer
my_activation_output = activation(my_convolution_output)


# --------Max Pool--------
def max_pool(input_1d, width, stride):
    # Just like 'conv2d()' above, max_pool() works with 4D arrays.
    # [batch_size=1, width=1, height=num_input, channels=1]
    # 因为在处理卷积层的结果时，使用squeeze函数对结果输出进行降维，所以此处要将最大池化层的维度提升为4维
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Perform the max pooling with strides = [1,1,1,1]
    # If we wanted to increase the stride on our data dimension, say by
    # a factor of '2', we put strides = [1, 1, 2, 1]
    # We will also need to specify the width of the max-window ('width')
    pool_output = tf.nn.max_pool(input_4d, ksize=[1, 1, width, 1],
                                 strides=[1, 1, stride, 1],
                                 padding='VALID')
    # Get rid of extra dimensions
    pool_output_1d = tf.squeeze(pool_output)
    return (pool_output_1d)


my_maxpool_output = max_pool(my_activation_output, width=maxpool_size, stride=stride_size)


# --------Fully Connected--------
def fully_connected(input_layer, num_outputs):
    # First we find the needed shape of the multiplication weight matrix:
    # The dimension will be (length of input) by (num_outputs)
    weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer), [num_outputs]]))
    # squeeze函数用于去掉维度为1的维度。保留数据。

    # Initialize such weight
    # 初始化weight
    weight = tf.random_normal(weight_shape, stddev=0.1)
    # Initialize the bias
    # 初始化bias
    bias = tf.random_normal(shape=[num_outputs])
    # Make the 1D input array into a 2D array for matrix multiplication
    # 将一维的数组添加一维成为2维数组
    input_layer_2d = tf.expand_dims(input_layer, 0)
    # Perform the matrix multiplication and add the bias
    full_output = tf.add(tf.matmul(input_layer_2d, weight), bias)
    # Get rid of extra dimensions
    # 去掉多余的维度只保留数据
    full_output_1d = tf.squeeze(full_output)
    return (full_output_1d)


my_full_output = fully_connected(my_maxpool_output, 5)

# Run graph
# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_input_1d: data_1d}

print('>>>> 1D Data <<<<')

# Convolution Output
print('Input = array of length %d'%(x_input_1d.shape.as_list()[0]))  # 25
print('Convolution w/ filter, length = %d, stride size = %d, results in an array of length %d:'%
      (conv_size, stride_size, my_convolution_output.shape.as_list()[0]))  # 21
print(sess.run(my_convolution_output, feed_dict=feed_dict))

# Activation Output
print('\nInput = above array of length %d'%(my_convolution_output.shape.as_list()[0]))  # 21
print('ReLU element wise returns an array of length %d:'%(my_activation_output.shape.as_list()[0]))  # 21
print(sess.run(my_activation_output, feed_dict=feed_dict))

# Max Pool Output
print('\nInput = above array of length %d'%(my_activation_output.shape.as_list()[0]))  # 21
print('MaxPool, window length = %d, stride size = %d, results in the array of length %d'%
      (maxpool_size, stride_size, my_maxpool_output.shape.as_list()[0]))  # 17
print(sess.run(my_maxpool_output, feed_dict=feed_dict))

# Fully Connected Output
print('\nInput = above array of length %d'%(my_maxpool_output.shape.as_list()[0]))  # 17
print('Fully connected layer on all 4 rows with %d outputs:'%
      (my_full_output.shape.as_list()[0]))  # 5
print(sess.run(my_full_output, feed_dict=feed_dict))

# >>>> 1D Data <<<<
# Input = array of length 25
# Convolution w/ filter, length = 5, stride size = 1, results in an array of length 21:
# [-2.63576341 -1.11550486 -0.95571411 -1.69670296 -0.35699379  0.62266493
#   4.43316031  2.01364899  1.33044648 -2.30629659 -0.82916248 -2.63594174
#   0.76669347 -2.46465087 -2.2855041   1.49780679  1.6960566   1.48557389
#  -2.79799461  1.18149185  1.42146575]
#
# Input = above array of length 21
# ReLU element wise returns an array of length 21:
# [ 0.          0.          0.          0.          0.          0.62266493
#   4.43316031  2.01364899  1.33044648  0.          0.          0.
#   0.76669347  0.          0.          1.49780679  1.6960566   1.48557389
#   0.          1.18149185  1.42146575]
#
# Input = above array of length 21
# MaxPool, window length = 5, stride size = 1, results in the array of length 17
# [ 0.          0.62266493  4.43316031  4.43316031  4.43316031  4.43316031
#   4.43316031  2.01364899  1.33044648  0.76669347  0.76669347  1.49780679
#   1.6960566   1.6960566   1.6960566   1.6960566   1.6960566 ]
#
# Input = above array of length 17
# Fully connected layer on all 4 rows with 5 outputs:
# [ 1.71536088 -0.72340977 -1.22485089 -2.5412786  -0.16338299]
