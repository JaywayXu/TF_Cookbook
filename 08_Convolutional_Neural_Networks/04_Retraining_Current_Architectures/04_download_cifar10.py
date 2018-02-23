# Download/Saving CIFAR-10 images in Inception format
# 以Inception网络要求的格式下载和保存CIFAR-10数据集
#  ---------------------------------------
# 迁移学习
#  ---------------------------------------
# In this script, we download the CIFAR-10 images and
# transform/save them in the Inception Retrianing Format
# 在此脚本中，我们下载CIFAR-10数据集并且将其转化并保存为Inception 再训练的格式
# The end purpose of the files is for retrianing the
# Google Inception tensorflow model to work on the CIFAR-10.
# 在CIFAR-10训练集上重训练Google Inception tensorflow模型

# https://github.com/Asurada2015/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
# http://blog.csdn.net/tianzhaixing2013/article/details/73527771
# http://blog.csdn.net/daydayup_668819/article/details/68060483
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/tutorials/image_retraining.md


"""从原始数据集开始训练一个全新的图像识别模型需要耗费大量时间个计算力，如果我们可以重用预训练好的网络训练图片
将会缩短计算时间。本节将展示如何使用预训练好的Tensorflow图像模型，微调后训练其他图片数据集"""

# 迁移学习的基本思路是重用预训练模型的卷积层的权重和结构，然后重新训练全连接层。
import os
import tarfile
import _pickle as cPickle
import numpy as np
import urllib.request
import scipy.misc

cifar_link = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# 使用的是CIFAR-10 python版本;
# 该存档包含文件data_batch_1，data_batch_2，…，data_batch_5以及test_batch;
# 这些文件中的每一个都是用cPickle生成的Python"pickled"对象
data_dir = 'temp'
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

# 下载CIFAR-10 压缩文件
target_file = os.path.join(data_dir, 'cifar-10-python.tar.gz')
if not os.path.isfile(target_file):
    print('CIFAR-10 file not found. Downloading CIFAR data (Size = 163MB)')
    print('This may take a few minutes, please wait.')
    filename, headers = urllib.request.urlretrieve(cifar_link, target_file)

# 解压缩文件
tar = tarfile.open(target_file)
tar.extractall(path=data_dir)
tar.close()
# 解压缩后得到cifat-10-batches-py文件夹
objects = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 创建训练图片文件夹
# 创建train_dir与下属10个文件夹
train_folder = 'train_dir'
if not os.path.isdir(os.path.join(data_dir, train_folder)):
    for i in range(10):
        folder = os.path.join(data_dir, train_folder, objects[i])
        os.makedirs(folder)
# 创建测试图片文件夹
# 创建validation_dir与下属10个文件夹
test_folder = 'validation_dir'
if not os.path.isdir(os.path.join(data_dir, test_folder)):
    for i in range(10):
        folder = os.path.join(data_dir, test_folder, objects[i])
        os.makedirs(folder)

# 提取相应图片
data_location = os.path.join(data_dir, 'cifar-10-batches-py')
train_names = ['data_batch_' + str(x) for x in range(1, 6)]
test_names = ['test_batch']


# 定义提取数据集字典
def load_batch_from_file(file):
    file_conn = open(file, 'rb')
    # 打开并使用二进制方式读取文件
    image_dictionary = cPickle.load(file_conn, encoding='latin1')
    # 因为图片文件用cPickle函数进行保存，此处使用cPickle函数进行读取
    file_conn.close()
    return (image_dictionary)


# 提取字典数据并保存图片
def save_images_from_dict(image_dict, folder='data_dir'):
    # image_dict.keys() = 'labels', 'filenames', 'data', 'batch_label'
    for ix, label in enumerate(image_dict['labels']):
        # 从image_dict文件中提取['labels']信息，并将其交给迭代器，ix为数字索引，label为int型数据类别
        folder_path = os.path.join(data_dir, folder, objects[label])
        filename = image_dict['filenames'][ix]
        # 将图片数据转换成可视图片数据格式
        image_array = image_dict['data'][ix]
        image_array.resize([3, 32, 32])
        # 保存图片
        output_location = os.path.join(folder_path, filename)
        scipy.misc.imsave(output_location, image_array.transpose())


# 保存训练图片集
for file in train_names:
    print('Saving images from file: {}'.format(file))
    file_location = os.path.join(data_dir, 'cifar-10-batches-py', file)
    image_dict = load_batch_from_file(file_location)
    # print(image_dict)
    # image_dict为dict字典类型数据
    # 所以此时通过简单的操作不能知道其中数据类型的保存形式
    # 使用print()尝试其中数据保存格式
    # {'filenames': ['leptodactylus_pentadactylus_s_000004.png', 'camion_s_000148.png'...'']保存有一万张图片文件名称
    #  'labels': [6, 9, 9, 4, 1, 1, 2, 7, 8, 3, ...]保存有一万张图片所属类型
    #   'data': array([[ 59,  43,  50, ..., 140,  84,  72],
    #   [154, 126, 105, ..., 139, 142, 144],
    #    [255, 253, 253, ...,  83,  83,  84],
    #    ...,
    #    [ 71,  60,  74, ...,  68,  69,  68],
    #    [250, 254, 211, ..., 215, 255, 254],
    #    [ 62,  61,  60, ..., 130, 130, 131]], dtype=uint8),
    #    保存有一万张图片数据形状为(10000, 3072)的数组
    #   'batch_label': 'training batch 1 of 5'此属性包含属于文件批次信息,含义为5个文件中的第一个文件}

    # print(image_dict['data'].shape)   # (10000, 3072)
    save_images_from_dict(image_dict, folder=train_folder)  # train_dir

# 保存测试图片集
for file in test_names:
    print('Saving images from file: {}'.format(file))
    file_location = os.path.join(data_dir, 'cifar-10-batches-py', file)
    image_dict = load_batch_from_file(file_location)
    save_images_from_dict(image_dict, folder=test_folder)  # validation_dir

# 创建标签文件
cifar_labels_file = os.path.join(data_dir, 'cifar10_labels.txt')
print('Writing labels file, {}'.format(cifar_labels_file))
with open(cifar_labels_file, 'w') as labels_file:
    for item in objects:
        labels_file.write("{}\n".format(item))
