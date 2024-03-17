import pickle
from sklearn.svm import SVC
import cv2
import glob
import os
import numpy as np
from skimage import feature
from skimage import transform
from skimage import io


class Estimator2:

    def __init__(self):
        estimator_path = 'model/estimator.pth'
        if os.path.exists(estimator_path):
            self.estimator = pickle.load(open(estimator_path, 'rb'))

    # 使用 HOG 特征
    def extract_feature(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = feature.hog(image,orientations=9,pixels_per_cell=(20, 20),cells_per_block=(3, 3),visualize=False,channel_axis=2)
        return image


    def load_data(self, data_type='train'):
        image_fnames = glob.glob(f'data/{data_type}/[0-9]-[0-9]*.png')
        images, labels = [], []
        for fname in image_fnames:
            # 读取图像数据
            image = cv2.imread(fname)

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
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = self.image_center(image)
        image = self.extract_feature(image)
        label = self.estimator.predict([image])

        return label[0]

# if __name__ == '__main__':
#     model = Estimator2()
#     model.train()

