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

# ---------------------------------------------------|
# -------------------2D-data-------------------------|
# ---------------------------------------------------|

# Reset Graph 重置图模型
ops.reset_default_graph()
sess = tf.Session()

# parameters for the run 运行参数
row_size = 10  # 2D图形高
col_size = 10  # 2D图形长
conv_size = 2
conv_stride_size = 2  # 卷积步长
maxpool_size = 2
maxpool_stride_size = 1  # 池化步长

# ensure reproducibility 确保复现性
seed = 13
np.random.seed(seed)
tf.set_random_seed(seed)

# Generate 2D data生成随机二维数据
data_size = [row_size, col_size]
data_2d = np.random.normal(size=data_size)

# --------Placeholder--------
x_input_2d = tf.placeholder(dtype=tf.float32, shape=data_size)


# Convolution 卷积层
def conv_layer_2d(input_2d, my_filter, stride_size):
    # TensorFlow's 'conv2d()' function only works with 4D arrays:
    # [batch, height, width, channels], we have 1 batch, and
    # 1 channel, but we do have width AND height this time.
    # So next we create the 4D array by inserting dimension 1's.
    # Tensorflow的卷积操作默认输入有四个维度[batch_size, height, width, channels]
    # 此处我们将维度增加到四维
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter,
                                      strides=[1, stride_size, stride_size, 1], padding="VALID")
    # filter表示卷积核，数据维度为[卷积核高度，卷积核长度，输入通道数，输出通道数]
    # stride_size表示步长，数据维度为[批处理数据大小,步长高，步长宽，通道数]，其中批处理数据大小和通道数一般跨度都为1，不需要修改。
    # Get rid of unnecessary dimensions
    # 将维数为1的维度去掉，保留数值数据。
    conv_output_2d = tf.squeeze(convolution_output)
    return (conv_output_2d)


# Create Convolutional Filter
my_filter = tf.Variable(tf.random_normal(shape=[conv_size, conv_size, 1, 1]))
# Create Convolutional Layer
my_convolution_output = conv_layer_2d(x_input_2d, my_filter, stride_size=conv_stride_size)


# --------Activation--------
def activation(input_1d):
    return (tf.nn.relu(input_1d))


# Create Activation Layer 激活层
my_activation_output = activation(my_convolution_output)


# --------Max Pool--------
def max_pool(input_2d, width, height, stride):
    # Just like 'conv2d()' above, max_pool() works with 4D arrays.
    # [batch_size=1, height=given, width=given, channels=1]
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Perform the max pooling with strides = [1,1,1,1]
    # If we wanted to increase the stride on our data dimension, say by
    # a factor of '2', we put strides = [1, 2, 2, 1]
    pool_output = tf.nn.max_pool(input_4d, ksize=[1, height, width, 1],
                                 strides=[1, stride, stride, 1],
                                 padding='VALID')
    # Get rid of unnecessary dimensions
    pool_output_2d = tf.squeeze(pool_output)
    return (pool_output_2d)


# Create Max-Pool Layer 最大池化层
# 即选择窗口中的最大值作为输出，池化层的窗口格式定义和卷积核的不一样
# 其四维数组格式定义和卷积步长，池化步长一致，和输入格式相同，即
# [batch_size, height, width, channels]
my_maxpool_output = max_pool(my_activation_output,
                             width=maxpool_size, height=maxpool_size, stride=maxpool_stride_size)


# --------Fully Connected--------
def fully_connected(input_layer, num_outputs):
    # 扁平化/光栅化处理使最大池化层输出为一维向量形式
    flat_input = tf.reshape(input_layer, [-1])
    # 设定weight的形状
    weight_shape = tf.squeeze(tf.stack([tf.shape(flat_input), [num_outputs]]))
    # 初始化weight
    weight = tf.random_normal(weight_shape, stddev=0.1)
    # 初始化bias
    bias = tf.random_normal(shape=[num_outputs])
    # 将一维输入还原为二维
    input_2d = tf.expand_dims(flat_input, 0)
    # 计算输出
    full_output = tf.add(tf.matmul(input_2d, weight), bias)
    # 降维，去掉维度为1的维度，便于进行观察
    full_output_2d = tf.squeeze(full_output)
    return (full_output_2d)


# Create Fully Connected Layer
my_full_output = fully_connected(my_maxpool_output, 5)

# Run graph
# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_input_2d: data_2d}

print('\n>>>> 2D Data <<<<')

# Convolution Output
print('Input = %s array'%(x_input_2d.shape.as_list()))
print('%s Convolution, stride size = [%d, %d] , results in the %s array'%
      (my_filter.get_shape().as_list()[:2], conv_stride_size, conv_stride_size, my_convolution_output.shape.as_list()))
print(sess.run(my_convolution_output, feed_dict=feed_dict))

# Activation Output
print('\nInput = the above %s array'%(my_convolution_output.shape.as_list()))
print('ReLU element wise returns the %s array'%(my_activation_output.shape.as_list()))
print(sess.run(my_activation_output, feed_dict=feed_dict))

# Max Pool Output
print('\nInput = the above %s array'%(my_activation_output.shape.as_list()))
print('MaxPool, stride size = [%d, %d], results in %s array'%
      (maxpool_stride_size, maxpool_stride_size, my_maxpool_output.shape.as_list()))
print(sess.run(my_maxpool_output, feed_dict=feed_dict))

# Fully Connected Output 全连接层
print('\nInput = the above %s array'%(my_maxpool_output.shape.as_list()))
# 全连接层输出除掉batch_size和channel维度后剩余维度为[4,4]
print('Fully connected layer on all %d rows results in %s outputs:'%
      (my_maxpool_output.shape.as_list()[0], my_full_output.shape.as_list()[0]))
# 由于全连接层神经元为5个所以输出维度为5
print(sess.run(my_full_output, feed_dict=feed_dict))

'''对于卷积层输出维度[batch_size,output_size,output_size',channels]
其中output_size及output_size'表示为对应维度上通过(W-F+2P)/S+1得到的结果
W为数据维度，F为卷积核或池化窗口的宽或高，P为Padding大小，其中设置卷积为Valid时，Padding为0若设置为SAME卷积，则会有Padding，S是步长大小
本例子中卷积层计算公式为[(10-2)+0]/2+1=5，池化层计算公式为[(5-2)+0]/1+1=4'''

# >>>> 2D Data <<<<
# Input = [10, 10] array
# [2, 2] Convolution, stride size = [2, 2] , results in the [5, 5] array
# [[ 0.14431179  0.72783369  1.51149166 -1.28099763  1.78439188]
#  [-2.54503059  0.76156765 -0.51650006  0.77131093  0.37542343]
#  [ 0.49345911  0.01592223  0.38653135 -1.47997665  0.6952765 ]
#  [-0.34617192 -2.53189754 -0.9525758  -1.4357065   0.66257358]
#  [-1.98540258  0.34398788  2.53760481 -0.86784822 -0.3100495 ]]
#
# Input = the above [5, 5] array
# ReLU element wise returns the [5, 5] array
# [[ 0.14431179  0.72783369  1.51149166  0.          1.78439188]
#  [ 0.          0.76156765  0.          0.77131093  0.37542343]
#  [ 0.49345911  0.01592223  0.38653135  0.          0.6952765 ]
#  [ 0.          0.          0.          0.          0.66257358]
#  [ 0.          0.34398788  2.53760481  0.          0.        ]]
#
# Input = the above [5, 5] array
# MaxPool, stride size = [1, 1], results in [4, 4] array
# [[ 0.76156765  1.51149166  1.51149166  1.78439188]
#  [ 0.76156765  0.76156765  0.77131093  0.77131093]
#  [ 0.49345911  0.38653135  0.38653135  0.6952765 ]
#  [ 0.34398788  2.53760481  2.53760481  0.66257358]]
#
# Input = the above [4, 4] array
# Fully connected layer on all 4 rows results in 5 outputs:
# [ 0.08245847 -0.16351229 -0.55429065 -0.24322605 -0.99900764]