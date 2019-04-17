import numpy as np
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate
import cv2
import os
import random

import CNN
from data_processing import DataProcessing

random.seed(42)

dir = 'dataset\chars74k-lite'


def load_dataset(dir):
    images = np.zeros([7112, 20, 20, 3])
    targets = np.zeros([7112, 26])
    chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    i = 0
    j = 0
    for c in chars:
        folder = os.path.join(dir, c)
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images[j] = img
                targets[j, i] = 1
                j += 1
        i += 1
    return images, targets


def rotate_training_data(img, edg, y):
    new_img = np.zeros([len(img)*4, len(img[0]), len(img[0, 0]), len(img[0, 0, 0])])
    new_edg = np.zeros([len(img) * 4, len(img[0]), len(img[0, 0]), len(img[0, 0, 0])])
    new_y = np.zeros([len(y)*4, len(y[0])])
    new_img[:len(img)] = img
    new_edg[:len(img)] = img
    new_y[:len(y)] = y
    for i in range(3):
        new_img[len(img)*(i+1):len(img)*(i+2)] = rotate(input=img, angle=90*(i+1), axes=(1, 2))
        new_edg[len(edg)*(i+1):len(edg)*(i+2)] = rotate(input=edg, angle=90*(i+1), axes=(1, 2))
        new_y[len(y)*(i+1):len(y)*(i+2)] = y
    return new_img, new_edg, new_y


def main(dir):
    print('Loading data...')
    x, y = load_dataset(dir)
    print('Preprocessing data..')
    data = DataProcessing(x, y)
    average_num = 10
    test_accuracy = np.zeros(average_num)
    for i in range(average_num):
        print('Training network ', i+1)
        random_state = 42+i
        img_train, img_test, y_train, y_test = train_test_split(data.x, data.y, test_size=0.2, random_state=random_state)
        edges_train, edges_test = train_test_split(data.edges, test_size=0.2, random_state=random_state)
        # img_train, edges_train, y_train = rotate_training_data(img_train, edges_train, y_train)
        test_accuracy[i] = CNN.run(img_train, img_test, edges_train, edges_test, y_train, y_test)
    print('Average test accuracy over ', average_num, ' trials: ', np.mean(test_accuracy))


if __name__ == "__main__":
    main(dir)
