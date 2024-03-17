import pickle
from sklearn.svm import SVC
import cv2
import glob
import os
import numpy as np


class Estimator1:

    def __init__(self):
        self.estimator = pickle.load(open('model/estimator.pth', 'rb'))

    def extract_feature(self, image):
        image = image.reshape(-1)
        image = image / 255
        return image

    def load_data(self, data_type='train'):
        image_fnames = glob.glob(f'data/{data_type}/[0-9]-[0-9]*.png')
        images, labels = [], []
        for fname in image_fnames:
            # 读取图像数据
            image = cv2.imread(fname)
            # 转换灰度图
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 提取图像特征
            image = self.extract_feature(image)
            images.append(image)

            # 解析标签
            fname = os.path.basename(fname)
            label = int(fname.split('-')[0])
            labels.append(label)

        images = np.array(images)
        labels = np.array(labels)

        return images, labels

    def train(self):
        images, labels = self.load_data(data_type='train')
        estimator = SVC()
        estimator.fit(images, labels)
        train_acc = estimator.score(images, labels)

        images, labels = self.load_data(data_type='test')
        test_acc = estimator.score(images, labels)
        pickle.dump(estimator, open('model/estimator.pth', 'wb'))
        self.estimator = estimator

        return train_acc, test_acc

    def predict(self):
        image = cv2.imread('data/train/temp.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = self.extract_feature(image)
        label = self.estimator.predict([image])

        return label[0]