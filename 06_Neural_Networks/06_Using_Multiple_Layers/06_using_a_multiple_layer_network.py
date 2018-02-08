# Using a Multiple Layer Network
# ---------------------------------------
# 使用多层神经网络层构造神经网络
# We will illustrate how to use a Multiple
# Layer Network in TensorFlow
#
# Low Birthrate data:
#
# Columns(列)   Variable（值）                             Abbreviation
# -----------------------------------------------------------------------------
# Low Birth Weight (0 = Birth Weight >= 2500g,            LOW
#                          1 = Birth Weight < 2500g)
# 低出生体重
# Age of the Mother in Years                              AGE
# 母亲妊娠年龄
# Weight in Pounds at the Last Menstrual Period           LWT
# 在最后一次月经期间体重增加。
# Race (1 = White, 2 = Black, 3 = Other)                  RACE
# 肤色
# Smoking Status During Pregnancy (1 = Yes, 0 = No)       SMOKE
# 怀孕期间吸烟状态
# History of Premature Labor (0 = None  1 = One, etc.)    PTL
# 早产的历史
# History of Hypertension (1 = Yes, 0 = No)               HT
# 高血压历史
# Presence of Uterine Irritability (1 = Yes, 0 = No)      UI
# 子宫刺激性的存在
# Birth Weight in Grams                                   BWT
# 以克为单位的体重
# ------------------------------
# The multiple neural network layer we will create will be composed of
# three fully connected hidden layers, with node sizes 25, 10, and 5

import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import os
import random
import numpy as np
import random
import requests
from tensorflow.python.framework import ops

# name of data file
# 数据集名称
birth_weight_file = 'birth_weight.csv'

# download data and create data file if file does not exist in current directory
# 如果当前文件夹下没有birth_weight.csv数据集则下载dat文件并生成csv文件
if not os.path.exists(birth_weight_file):
    birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
    birth_file = requests.get(birthdata_url)
    birth_data = birth_file.text.split('\r\n')
    # split分割函数,以一行作为分割函数，windows中换行符号为'\r\n',每一行后面都有一个'\r\n'符号。
    birth_header = birth_data[0].split('\t')
    # 每一列的标题，标在第一行，即是birth_data的第一个数据。并使用制表符作为划分。
    birth_data = [[float(x) for x in y.split('\t') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
    # 数组第一维表示遍历行从第一行开始，所以不包含标题，数组第二维遍历列(使用制表符进行分割)
    # print(np.array(birth_data).shape)
    # (189, 9)不包含标题
    # 此为list数据形式不是numpy数组不能使用np,shape函数,但是我们可以使用np.array函数将list对象转化为numpy数组后使用shape属性进行查看。
    # 注意，向其中写入文件时一定要去掉换行等操作符号，如果在csv中有换行符，也会作为一行数据的。
    # 读文件时，我们把csv文件读入列表中，写文件时会把列表中的元素写入到csv文件中。
    #
    # list = ['1', '2', '3', '4']
    # out = open(outfile, 'w')
    # csv_writer = csv.writer(out)
    # csv_writer.writerow(list)
    # 可能遇到的问题：直接使用这种写法会导致文件每一行后面会多一个空行。
    #
    # 解决办法如下：
    #
    # out = open(outfile, 'w', newline='')
    # csv_writer = csv.writer(out, dialect='excel')
    # csv_writer.writerow(list)

    with open(birth_weight_file, "w", newline='') as f:
        # 创建当前目录下birth_weight.csv文件
        writer = csv.writer(f)
        writer.writerows([birth_header])
        writer.writerows(birth_data)
        f.close()

# read birth weight data into memory将出生体重数据读进内存
birth_data = []
with open(birth_weight_file, newline='') as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    birth_header = next(csv_reader)  # 读取第一行每一列的标题
    for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
        birth_data.append(row)

birth_data = [[float(x) for x in row] for row in birth_data]

# Extract y-target (birth weight)
y_vals = np.array([x[8] for x in birth_data])

# Filter for features of interest
cols_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI']
x_vals = np.array(
    [[x[ix] for ix, feature in enumerate(birth_header) if feature in cols_of_interest] for x in birth_data])

# reset the graph for new run
ops.reset_default_graph()

# Create graph session 
sess = tf.Session()

# set batch size for training
batch_size = 100

# make results reproducible
seed = 3
np.random.seed(seed)
tf.set_random_seed(seed)

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


# Normalize by column (min-max norm to be between 0 and 1)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min)/(col_max - col_min)


x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))


# Define Variable Functions (weights and bias)
def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return (weight)


def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return (bias)


# Create Placeholders
x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)


# Create a fully connected layer:
def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return (tf.nn.relu(layer))


# --------Create the first layer (50 hidden nodes)--------
weight_1 = init_weight(shape=[7, 25], st_dev=10.0)
bias_1 = init_bias(shape=[25], st_dev=10.0)
layer_1 = fully_connected(x_data, weight_1, bias_1)

# --------Create second layer (25 hidden nodes)--------
weight_2 = init_weight(shape=[25, 10], st_dev=10.0)
bias_2 = init_bias(shape=[10], st_dev=10.0)
layer_2 = fully_connected(layer_1, weight_2, bias_2)

# --------Create third layer (5 hidden nodes)--------
weight_3 = init_weight(shape=[10, 3], st_dev=10.0)
bias_3 = init_bias(shape=[3], st_dev=10.0)
layer_3 = fully_connected(layer_2, weight_3, bias_3)

# --------Create output layer (1 output value)--------
weight_4 = init_weight(shape=[3, 1], st_dev=10.0)
bias_4 = init_bias(shape=[1], st_dev=10.0)
final_output = fully_connected(layer_3, weight_4, bias_4)

# Declare loss function (L1)
loss = tf.reduce_mean(tf.abs(y_target - final_output))

# Declare optimizer
my_opt = tf.train.AdamOptimizer(0.05)
train_step = my_opt.minimize(loss)

# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
test_loss = []
for i in range(200):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_loss.append(test_temp_loss)
    if (i + 1)%25 == 0:
        print('Generation: ' + str(i + 1) + '. Loss = ' + str(temp_loss))

# Plot loss (MSE) over time
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
