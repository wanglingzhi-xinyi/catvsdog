import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
import datatrain as data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
new_model=load_model('models/predict_model.h5')

#获取测试集合
test_img,test_labels=next(data.test_batches)
data.plotImages(test_img)
print(test_labels)

#预测数据集
data.test_batches.classes#具有测试集中每个图像的所有对应标签
predictions=new_model.predict(x=data.test_batches,verbose=0)
np.round(predictions)#取整数，遵守四舍五入原则
#调用混淆矩阵函数,np.argmax算出数组最大值的下标，axis=0比较列(数组a[c][d][e])就是d,e不变c改变来比较，axis=1就是在只改变d
#在a为二维数组的时候，axis=-1与axis=1结果一致，当a为三维数组的时候，axis=-1与axis=2结果一致
cm=confusion_matrix(y_true=data.test_batches.classes,y_pred=np.argmax(predictions,axis=-1))#比较行
#创建函数，使得可以看到混淆矩阵函数可=可视化输出
def plot_confusion_matrix(cm,classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]#[:, np.newaxis]增加一个维度
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh=cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment="center",
                 color="white" if cm[i,j]>thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

data.test_batches.class_indices#查看标签对应，一般是猫的下标是0，狗是1
cm_plot_labels=['cat','dog']
plot_confusion_matrix(cm=cm,classes=cm_plot_labels)



