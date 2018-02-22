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

# 更改工作目录
abspath = os.path.abspath(__file__)  # 获取当前文件绝对地址
# E:\GitHub\TF_Cookbook\08_Convolutional_Neural_Networks\03_CNN_CIFAR10\ostest.py
dname = os.path.dirname(abspath)  # 获取文件所在文件夹地址
# E:\GitHub\TF_Cookbook\08_Convolutional_Neural_Networks\03_CNN_CIFAR10
os.chdir(dname)  # 转换目录文件夹到上层

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

# 检查这个文件是否存在，如果不存在下载这个文件
data_file = os.path.join(data_dir, 'cifar-10-binary.tar.gz')
# temp\cifar-10-binary.tar.gz
if os.path.isfile(data_file):
    pass
else:
    # 回调函数，当连接上服务器、以及相应的数据块传输完毕时会触发该回调，我们可以利用这个回调函数来显示当前的下载进度。
    # block_num已经下载的数据块数目，block_size数据块大小，total_size下载文件总大小

    def progress(block_num, block_size, total_size):
        progress_info = [cifar10_url, float(block_num*block_size)/float(total_size)*100.0]
        print('\r Downloading {} - {:.2f}%'.format(*progress_info), end="")


    # urlretrieve(url, filename=None, reporthook=None, data=None)
    # 参数 finename 指定了保存本地路径（如果参数未指定，urllib会生成一个临时文件保存数据。）
    # 参数 reporthook 是一个回调函数，当连接上服务器、以及相应的数据块传输完毕时会触发该回调，我们可以利用这个回调函数来显示当前的下载进度。
    # 参数 data指 post 到服务器的数据，该方法返回一个包含两个元素的(filename, headers)元组，filename 表示保存到本地的路径，header 表示服务器的响应头。
    # 此处 url=cifar10_url,filename=data_file,reporthook=progress

    filepath, _ = urllib.request.urlretrieve(cifar10_url, data_file, progress)
    # 解压文件
    tarfile.open(filepath, 'r:gz').extractall(data_dir)


# Define CIFAR reader
# 定义CIFAR读取器
def read_cifar_files(filename_queue, distort_images=True):
    reader = tf.FixedLengthRecordReader(record_bytes=record_length)
    # 返回固定长度的文件记录 record_length函数参数为一条图片信息即1+32*32*3
    key, record_string = reader.read(filename_queue)
    # 此处调用tf.FixedLengthRecordReader.read函数返回键值对
    record_bytes = tf.decode_raw(record_string, tf.uint8)
    # 读出来的原始文件是string类型，此处我们需要用decode_raw函数将String类型转换成uint8类型
    image_label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)
    # 见slice函数用法，取从0号索引开始的第一个元素。并将其转化为int32型数据。其中存储的是图片的标签

    # Extract image
    # 截取图像
    image_extracted = tf.reshape(tf.slice(record_bytes, [1], [image_vec_length]),
                                 [num_channels, image_height, image_width])
    # 从1号索引开始提取图片信息。这和此数据集存储图片信息的格式相关。
    # CIFAR-10数据集中
    """第一个字节是第一个图像的标签，它是一个0-9范围内的数字。接下来的3072个字节是图像像素的值。
       前1024个字节是红色通道值，下1024个绿色，最后1024个蓝色。值以行优先顺序存储，因此前32个字节是图像第一行的红色通道值。 
       每个文件都包含10000个这样的3073字节的“行”图像，但没有任何分隔行的限制。因此每个文件应该完全是30730000字节长。"""

    # Reshape image
    image_uint8image = tf.transpose(image_extracted, [1, 2, 0])
    # 详见tf.transpose函数，将[channel,image_height,image_width]转化为[image_height,image_width,channel]的数据格式。
    reshaped_image = tf.cast(image_uint8image, tf.float32)
    # 将图片剪裁或填充至合适大小
    final_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, crop_width, crop_height)

    if distort_images:
        # 将图像水平随机翻转，改变亮度和对比度。
        final_image = tf.image.random_flip_left_right(final_image)
        final_image = tf.image.random_brightness(final_image, max_delta=63)
        final_image = tf.image.random_contrast(final_image, lower=0.2, upper=1.8)

        # 对图片做标准化处理
        """Linearly scales `image` to have zero mean and unit norm.
        This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
        of all values in image, and `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.
        `stddev` is the standard deviation of all values in `image`. 
        It is capped away from zero to protect against division by 0 when handling uniform images."""
    final_image = tf.image.per_image_standardization(final_image)
    return (final_image, image_label)


# Create a CIFAR image pipeline from reader
# 从阅读器中构造CIFAR图片管道
def input_pipeline(batch_size, train_logical=True):
    # train_logical标志用于区分读取训练和测试数据集
    if train_logical:
        files = [os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(i)) for i in range(1, 6)]
    #  data_dir=tmp
    # extract_folder=cifar-10-batches-bin
    else:
        files = [os.path.join(data_dir, extract_folder, 'test_batch.bin')]
    filename_queue = tf.train.string_input_producer(files)
    image, label = read_cifar_files(filename_queue)

    # min_after_dequeue defines how big a buffer we will randomly sample
    # from -- bigger means better shuffling but slower start up and more
    # memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    # determines the maximum we will prefetch.  Recommendation:
    # min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3*batch_size
    example_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)

    return (example_batch, label_batch)


# Define the model architecture, this will return logits from images
# 定义模型架构，返回图片的元素
def cifar_cnn_model(input_images, batch_size, train_logical=True):
    # 截断高斯函数初始化化
    def truncated_normal_var(name, shape, dtype):
        return (tf.get_variable(name=name, shape=shape, dtype=dtype,
                                initializer=tf.truncated_normal_initializer(stddev=0.05)))

    # 0初始化
    def zero_var(name, shape, dtype):
        return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

    # 第一卷积层
    with tf.variable_scope('conv1') as scope:
        # Conv_kernel is 5x5 for all 3 colors and we will create 64 features
        # 第一层卷积层是5*5在3通道上进行卷积，并且构造64个feature map
        conv1_kernel = truncated_normal_var(name='conv_kernel1', shape=[5, 5, 3, 64], dtype=tf.float32)
        # We convolve across the image with a stride size of 1
        # 我们使用步长为1在原有图像上进行卷积
        conv1 = tf.nn.conv2d(input_images, conv1_kernel, [1, 1, 1, 1], padding='SAME')
        # Initialize and add the bias term
        # 初始化bias,并且加上偏置项
        conv1_bias = zero_var(name='conv_bias1', shape=[64], dtype=tf.float32)
        conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
        # ReLU element wise
        # 对结果使用ReLU非线性激活函数
        relu_conv1 = tf.nn.relu(conv1_add_bias)

    # Max Pooling
    # 池化层/下采样层
    pool1 = tf.nn.max_pool(relu_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer1')

    # Local Response Normalization (parameters from paper)
    # 局部响应归一化
    # http://blog.csdn.net/mao_xiao_feng/article/details/53488271
    # paper: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
    norm1 = tf.nn.lrn(pool1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm1')

    # 第二个卷积层
    with tf.variable_scope('conv2') as scope:
        # Conv kernel is 5x5, across all prior 64 features and we create 64 more features
        # 卷积核大小为5*5，输入通道数为64，输出通道数也为64
        conv2_kernel = truncated_normal_var(name='conv_kernel2', shape=[5, 5, 64, 64], dtype=tf.float32)
        # Convolve filter across prior output with stride size of 1
        # 卷积步长为1
        conv2 = tf.nn.conv2d(norm1, conv2_kernel, [1, 1, 1, 1], padding='SAME')
        # Initialize and add the bias
        # 初始化和添加偏置值
        conv2_bias = zero_var(name='conv_bias2', shape=[64], dtype=tf.float32)
        conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
        # ReLU element wise
        # 对结果使用ReLU非线性激活函数
        relu_conv2 = tf.nn.relu(conv2_add_bias)

    # Max Pooling
    # 池化层/下采样层
    pool2 = tf.nn.max_pool(relu_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer2')

    # Local Response Normalization (parameters from paper)
    # 局部响应归一化
    norm2 = tf.nn.lrn(pool2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')

    # Reshape output into a single matrix for multiplication for the fully connected layers
    # 光栅化处理，将其打平方便和全连接层进行连接
    reshaped_output = tf.reshape(norm2, [batch_size, -1])
    reshaped_dim = reshaped_output.get_shape()[1].value

    # First Fully Connected Layer
    # 全连接层1
    with tf.variable_scope('full1') as scope:
        # Fully connected layer will have 384 outputs.
        # 第一个全连接层有384个输出
        full_weight1 = truncated_normal_var(name='full_mult1', shape=[reshaped_dim, 384], dtype=tf.float32)
        full_bias1 = zero_var(name='full_bias1', shape=[384], dtype=tf.float32)
        full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output, full_weight1), full_bias1))

    # Second Fully Connected Layer
    # 全连接层2
    with tf.variable_scope('full2') as scope:
        # Second fully connected layer has 192 outputs.
        # 第二个全连接层有192个输出
        full_weight2 = truncated_normal_var(name='full_mult2', shape=[384, 192], dtype=tf.float32)
        full_bias2 = zero_var(name='full_bias2', shape=[192], dtype=tf.float32)
        full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))

    # Final Fully Connected Layer -> 10 categories for output (num_targets)
    # 最后的全连接层只有10个输出
    with tf.variable_scope('full3') as scope:
        # Final fully connected layer has 10 (num_targets) outputs.
        full_weight3 = truncated_normal_var(name='full_mult3', shape=[192, num_targets], dtype=tf.float32)
        full_bias3 = zero_var(name='full_bias3', shape=[num_targets], dtype=tf.float32)
        final_output = tf.add(tf.matmul(full_layer2, full_weight3), full_bias3)

    return (final_output)


# Loss function损失函数
def cifar_loss(logits, targets):
    # Get rid of extra dimensions and cast targets into integers
    # 去掉多余的维度并且将标签全部转换为int类型
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Calculate cross entropy from logits and targets
    # 计算预测结果和标签值的交叉熵函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    # Take the average loss across batch size
    # 计算出整个batch中的平均误差值
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return (cross_entropy_mean)


# 训练阶段
def train_step(loss_value, generation_num):
    # Our learning rate is an exponential decay after we wait a fair number of generations
    # 自适应学习率递减
    model_learning_rate = tf.train.exponential_decay(learning_rate, generation_num,
                                                     num_gens_to_wait, lr_decay, staircase=True)
    # Create optimizer
    # 创建优化器
    my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
    # Initialize train step
    # 初始化训练迭代器
    train_step = my_optimizer.minimize(loss_value)
    return (train_step)


# Accuracy function
# 精准度函数
def accuracy_of_batch(logits, targets):
    # 去除多余的维度并确保target为int类型
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Get predicted values by finding which logit is the greatest
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    # Check if they are equal across the batch
    predicted_correctly = tf.equal(batch_predictions, targets)
    # Average the 1's and 0's (True's and False's) across the batch size
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return (accuracy)


# 获取数据
print('Getting/Transforming Data.')
# 初始化数据通道
images, targets = input_pipeline(batch_size, train_logical=True)
# Get batch test images and targets from pipline
test_images, test_targets = input_pipeline(batch_size, train_logical=False)

# Declare Model
# 声明模型
print('Creating the CIFAR10 Model.')
with tf.variable_scope('model_definition') as scope:
    # Declare the training network model
    model_output = cifar_cnn_model(images, batch_size)
    # 这非常重要，我们必须设置scope重用变量
    # 否则，当我们设置测试网络模型，它会设置新的随机变量，这会使在测试批次上进行随机评估，影响评估结果
    scope.reuse_variables()
    test_output = cifar_cnn_model(test_images, batch_size)

# Declare loss function
# 声明损失函数
print('Declare Loss Function.')
loss = cifar_loss(model_output, targets)

# Create accuracy functio
# 创建精准度函数
accuracy = accuracy_of_batch(test_output, test_targets)

# Create training operations
print('Creating the Training Operation.')
generation_num = tf.Variable(0, trainable=False)
train_op = train_step(loss, generation_num)

# Initialize Variables
print('Initializing the Variables.')
init = tf.global_variables_initializer()
sess.run(init)

# Initialize queue (This queue will feed into the model, so no placeholders necessary)
# 初始化队列，所以不再需要使placeholders占位符提供数据
tf.train.start_queue_runners(sess=sess)

# Train CIFAR Model
# 训练CIFAR模型
print('Starting Training')
train_loss = []
test_accuracy = []
for i in range(generations):
    _, loss_value = sess.run([train_op, loss])

    if (i + 1)%output_every == 0:
        train_loss.append(loss_value)
        output = 'Generation {}: Loss = {:.5f}'.format((i + 1), loss_value)
        print(output)

    if (i + 1)%eval_every == 0:
        [temp_accuracy] = sess.run([accuracy])
        test_accuracy.append(temp_accuracy)
        acc_output = ' --- Test Accuracy = {:.2f}%.'.format(100.*temp_accuracy)
        print(acc_output)

# Print loss and accuracy
# 打印损失函数和精准度函数
# Matlotlib code to plot the loss and accuracies
eval_indices = range(0, generations, eval_every)
output_indices = range(0, generations, output_every)

# Plot loss over time
plt.plot(output_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()

# Plot accuracy over time
plt.plot(eval_indices, test_accuracy, 'k-')
plt.title('Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.show()
