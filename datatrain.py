import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import warnings
import os
import shutil
import random
import glob

warnings.simplefilter(action='ignore',category=FutureWarning)

#使用GPU
'''physical_device=tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ",len(physical_device))
tf.config.experimental.set_memory_growth(physical_device[0],True)'''

'''os.chdir('./dataset')#修改文件路径
#修改设置文件路径
if os.path.isdir('training_set/dogs') is False:
    os.makedirs('training_set/dogs')
    os.makedirs('training_set/cats')
    os.makedirs('valid_set/dogs')
    os.makedirs('valid_set/cats')
    os.makedirs('test_set/dogs')
    os.makedirs('test_set/cats')
    #random.sample(pop,k)随机返回从总体序列中选择k长度的唯一元素列表
    #用glob查找指定符合规则的文件路径列表，匹配符包括“*”、“?”和"[]"，其中“*”表示匹配任意字符串，“?”匹配任意单个字符，[0-9]与[a-z]表示匹配0-9的单个数字与a-z的单个字符
    for c in random.sample(glob.glob('cat*'),500):
        #shutil实现文件复制和移动
        shutil.move(c,'training_set/cats')
    for c in random.sample(glob.glob('dog*'),500):
        shutil.move(c,'training_set/dogs')
    for c in random.sample(glob.glob('cat*'),100):
        shutil.move(c,'valid_set/cats')
    for c in random.sample(glob.glob('dog*'),100):
        shutil.move(c,'valid_set/dogs')
    for c in random.sample(glob.glob('cat*'),50):
        shutil.move(c,'test_set/cats')
    for c in random.sample(glob.glob('dog*'),50):
        shutil.move(c,'test_set/dogs')
'''

train_path='dataset/training_set'
valid_path='dataset/valid_set'
test_path='dataset/test_set'

#将图像转化为keras生成器的格式，设置为来自目录的图像数据生成器点流，它将返回一个目录迭代器
#它将从数据集所在的目录创建数据批，这些数据批可以使用fit函数传递到序列模型里面
#ImageDataGenerator()是keras.preprocessing.image模块中的图片生成器，可以每一次给模型“喂”一个batch_size大小的样本数据，同时也可以在每一个批次中对这batch_size个样本数据进行增强，扩充数据集大小，增强模型的泛化能力
#总结就是两点：图片生成器，负责生成一个批次一个批次的图片，以生成器的形式给模型训练；对每一个批次的训练图片，适时地进行数据增强处理

#preprocessing_function: 用户自定义函数预处理函数，将被应用于每个输入的函数。该函数将在图片缩放和数据提升之后运行。该函数接受一个参数，为一张图片（秩为3的numpy array），并且输出一个具有相同shape的numpy array
#\是换行标志 调用vgg16(基于cnn的网络)，preprocess_input是转为vgg16可以识别的图像输出

#flow_from_directory(directory): 传递实际数据并且指定如何处理这些数据的地方，以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
#directory: 目标文件夹路径,target_size:  图像将被resize成该尺寸,classes: 可选参数,为子文件夹的列表,默认为None,batch_size: batch数据的大小,默认32

train_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path,target_size=(224,224),classes=['cats','dogs'],batch_size=10)
valid_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path,target_size=(224,224),classes=['cats','dogs'],batch_size=10)
test_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path,target_size=(224,224),classes=['cats','dogs'],batch_size=10,shuffle=False)
#验证上边是不是有这么多张图,断言，即使得出结论是train_bathces=0也要继续
'''assert train_batches.n==1000
assert valid_batches.n==200
assert test_batches.n==100
assert train_batches.num_classes==valid_batches.num_classes==test_batches.num_classes==2
'''
imgs,labels=next(train_batches)

#绘制测试批的函数
def plotImages(images_arr):
    fig,axes=plt.subplots(1,10,figsize=(20,20) )#plt 利用subplot 实现在一张画布同时画多张图figsize=(20,20)窗口大小
    axes=axes.flatten()
    #zip()返回iterable，只能就行一次遍历，就是将矩阵一一对应
    for img,ax in zip(images_arr,axes):
        ax.imshow(img)
        ax.axis('off')#去掉坐标轴
    plt.tight_layout()#自动调整子图参数
    plt.show()

plotImages(imgs)
print(labels)
#fit函数：
#steps_per_epoch设置为等于从数据集生成训练集的步数或者样本批数，然后在训练中声明一个epoch已完成
#通常那指定为训练集中的样本数量除以批量(batch_size)大小
#validation_steps也是同上，只不过这是验证集有关

