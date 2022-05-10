from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten,BatchNormalization,Conv2D,MaxPool2D
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import itertools
import datatrain as data
import os
model=Sequential([
    #Conv2D是二维输入，padding默认是0填充，filter卷积核的个数即输出的维度,增加输出维度即深度
    Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same',input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2,2),strides=2), #降低输出维度（宽和高）
    Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'),
    MaxPool2D(pool_size=(2,2),strides=2),
    Flatten(),#平展成为一个一维向量，然后传为全连接层
    Dense(units=2,activation='softmax')#会出现两个节点就是对应猫和狗两种情况的
])
model.summary()
model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=data.train_batches,validation_data=data.valid_batches,epochs=10,verbose=2)
if os.path.isfile('models/predict_model.h5') is False:
    model.save('models/predict_model.h5')