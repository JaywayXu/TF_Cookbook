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
import tarfile
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
    # 参数 data 指 post 到服务器的数据，该方法返回一个包含两个元素的(filename, headers)元组，filename 表示保存到本地的路径，header 表示服务器的响应头。
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
def input_pipeline(batch_size, train_logical=False):
    # train_logical标志用于区分读取训练和测试数据集
    if train_logical:
        files = [os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(i)) for i in range(1, 6)]
    #  data_dir=tmp
    # extract_folder=cifar-10-batches-bin
    else:
        files = [os.path.join(data_dir, extract_folder, 'test_batch.bin')]
    filename_queue = tf.train.string_input_producer(files)
    image, label = read_cifar_files(filename_queue)
    print(train_logical, 'after read_cifar_files ops image', sess.run(tf.shape(image)))
    print(train_logical, 'after read_cifar_files ops label', sess.run(tf.shape(label)))
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3*batch_size
    # 批量读取图片数据
    example_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    print(train_logical, 'after shuffle_batch ops image', sess.run(tf.shape(image)))
    print(train_logical, 'after shuffle_batch ops example_batch', sess.run(tf.shape(example_batch)))
    print(train_logical, 'after shuffle_batch ops label', sess.run(tf.shape(label)))
    print(train_logical, 'after shuffle_batch ops label_batch', sess.run(tf.shape(label_batch)))


    return (example_batch, label_batch)


# 获取数据
print('Getting/Transforming Data.')
# 初始化数据管道获取训练数据和对应标签
images, targets = input_pipeline(batch_size, train_logical=True)
# 获取测试数据和对应标签
test_images, test_targets = input_pipeline(batch_size, train_logical=False)

# True after read_cifar_files ops image [24 24  3]
# True after read_cifar_files ops label [1]
# True after shuffle_batch ops image [24 24  3]
# True after shuffle_batch ops example_batch [128  24  24   3]
# True after shuffle_batch ops label [1]
# True after shuffle_batch ops label_batch [128   1]
# False after read_cifar_files ops image [24 24  3]
# False after read_cifar_files ops label [1]
# False after shuffle_batch ops image [24 24  3]
# False after shuffle_batch ops example_batch [128  24  24   3]
# False after shuffle_batch ops label [1]
# False after shuffle_batch ops label_batch [128   1]

sess.close()