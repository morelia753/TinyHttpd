import pickle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import cv2
import glob
import os
import numpy as np
from skimage import feature
from skimage import transform
from skimage import io
import matplotlib.pyplot as plt

class Estimator3:

    def __init__(self):
        estimator_path = 'model/estimator.pth'
        if os.path.exists(estimator_path):
            self.estimator = pickle.load(open(estimator_path, 'rb'))

    # 使用 HOG 特征
    def extract_feature(self, image):

        image = feature.hog(image,orientations=9,pixels_per_cell=(20, 20),cells_per_block=(3, 3),visualize=False)
        return image

    def image_center(self, image):

        # 背景黑色，前景色（数字）白色
        image = np.where(image == np.min(image), 255, 0).astype(np.uint8)

        # 第一个参数：图像
        # 第二个参数：轮廓的搜索模式
        # 第三个参数：轮廓的逼近方法
        contours, _ = cv2.findContours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for contour in contours:
            # 获得给定的轮廓内最小的矩形
            x, y, w, h = cv2.boundingRect(contour.squeeze())
            points.append((x, y, x + w, y + h))
        points = np.array(points)

        x1 = np.min(points[:, 0])
        y1 = np.min(points[:, 1])
        x2 = np.max(points[:, 2])
        y2 = np.max(points[:, 3])

        # 轮廓内数字图像
        image = image[y1: y2, x1: x2]

        # 对数字进行缩放
        # 计算宽高缩放系数，取最小的系数作为宽高的缩放因子
        scale_factor = np.min(300 / np.array(image.shape)).astype(np.float32)
        # 使用 transform.rescale 进行缩放
        # preserve_range: 保留原来像素值的范围
        image = transform.rescale(image, scale_factor, mode='constant', cval=0, preserve_range=True)

        # 对图像进行填充（设置到500x500）
        # 计算图像需要填充多少个像素点
        pad_shape = np.array([500, 500]) - image.shape
        pad_before = (pad_shape / 2).astype(np.int32)
        pad_afer = pad_shape - pad_before

        # 使用 np.pad 在图像的上下左右进行填充（保证数字在图像中心位置）
        image = np.pad(image, [(pad_before[0], pad_afer[0]), (pad_before[1], pad_afer[1])])
        # plt.imshow(image, cmap='gray')
        # plt.show()
        return image

    def load_data(self, data_type='train'):
        image_fnames = glob.glob(f'data/{data_type}/[0-9]-[0-9]*.png')
        images, labels = [], []
        for fname in image_fnames:
            # 读取图像数据
            image = cv2.imread(fname)
            #转换成灰度图
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            #图像中心化
            image = self.image_center(image)
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

        #搜索最优参数
        param_grid = {
            'C': [0.5, 0.8, 1.0, 1.2 ],
            'gamma': [0.01, 0.1, 0.5, 1.0]
        }
        estimator = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=3, n_jobs=-1)
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
        #图像的中心化
        image = self.image_center(image)
        image = self.extract_feature(image)
        label = self.estimator.predict([image])

        return label[0]

# if __name__ == '__main__':
#     model = Estimator3()
#     model.train()

