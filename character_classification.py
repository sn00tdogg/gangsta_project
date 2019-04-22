import numpy as np
import cv2 as cv
import os
import random

from models import Naive_Bayes, SVM, CNN, KNN
import plots

random.seed(100)

dir = 'dataset\chars74k-lite'
model_weights = 'model_weights.hdf5'


def load_dataset(dir):
    images = np.zeros([7112, 20, 20, 3])
    targets = np.zeros([7112])
    chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    i = 0
    j = 0
    for c in chars:
        folder = os.path.join(dir, c)
        for filename in os.listdir(folder):
            img = cv.imread(os.path.join(folder, filename))
            if img is not None:
                images[j] = img
                targets[j] = i
                j += 1
        i += 1
    return images, targets


def character_classification(dir, model_weights):
    print('Loading data...')
    x, y = load_dataset(dir)
    print('Processing data..')
    print('Training data shape: ', x.shape)
    print('Test data shape: ', y.shape)
    plots.plot_filters(x[0])
    SVM.linear_svm(x, y)
    SVM.svm(x, y)
    Naive_Bayes.naive_bayes(x, y)
    KNN.knn(x, y)
    CNN.fit_cnn(x, y, model_weights, trials=1)


if __name__ == "__main__":
    character_classification(dir, model_weights)
