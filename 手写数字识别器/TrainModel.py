import glob
import os.path
import numpy as np
import cv2
from sklearn.svm import SVC
import pickle
def extract_feature(image):
    #将2d数组转化为1d数组
    image = image.reshape(-1)
    #图像归一化
    image = image / 255
    return image

#数据处理
def load_data(data_type='train'):
    # 1.1 读取图像数据，提取图像特征
    # 1.2 解析图像标签
    image_fnames = glob.glob((f'data/{data_type}/[0-9]-[0-9]*.png'))
    images, labels = [], []

    for fname in image_fnames:
        print(fname)
        #读取图像数据
        image = cv2.imread(fname)
        #转换为灰度图
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #图像特征提取
        image = extract_feature(image)
        images.append(image)
        #提取图像标签
        fname = os.path.basename(fname)
        label = int(fname.split('-')[0])
        #标签的获取
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    print(images)
    print(labels)
    return images, labels

# 2.训练评估
def estimator_train():
    #加载训练数据
    images, labels = load_data(data_type='train')
    #算法模型训练
    estimator = SVC()
    estimator.fit(images, labels)
    #支持向量机在训练集上的准确率
    acc = estimator.score(images, labels)
    print('训练集 Acc:', '%.2f' % acc)
    # 存储模型
    pickle.dump(estimator, open('model/estimator.pth', 'wb'))

def estimator_test():
    images, labels = load_data(data_type='test')
    estimator = pickle.load(open('model/estimator.pth', 'rb'))
    acc = estimator.score(images, labels)
    print('测试集 Acc: %.2f' % acc)

if __name__ == '__main__':
     estimator_test()
    # load_data()