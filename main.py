import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from skimage.color import rgb2gray
from keras.utils import to_categorical
import cv2
import os
import random

import CNN

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
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images[j] = img
                targets[j] = i
                j += 1
        i += 1
    return images, targets


def data_processing(x):
    x = x.astype('float32')
    x /= 255  # Make between 0 and 1
    x = rgb2gray(x)  # Convert to grey scale
    for i in range(len(x)):
        x[i] = cv2.bilateralFilter(x[i], 2, 2, 15)  # Filter the images
    x = x.reshape(x.shape + (1,))  # Reshape the images
    return x


def train_cnn(x, y, model_weights, trials=10):
    print('=== Convolution Neural Network ===')
    test_accuracy = np.zeros(trials)
    y = to_categorical(y, 26)
    for i in range(trials):
        print('Training network ', i + 1)
        random_state = 100 + i
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                            random_state=random_state,
                                                            stratify=y)
        test_accuracy[i] = CNN.run(x_train, x_test, y_train, y_test, model_weights)
    print('Average test accuracy over ', trials, ' trials: ', np.mean(test_accuracy))


def train_svm(x, y):
    # Change this as it performs at 17% accuracy
    print('=== SVM classification ===')
    random_state = 100
    x = x.reshape(len(x), 400)
    # y = to_categorical(y, 26)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=random_state,
                                                        stratify=y)
    clf = LinearSVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(accuracy_score(y_test, y_pred))


def main(dir, model_weights):
    print('Loading data...')
    x, y = load_dataset(dir)
    print('Processing data..')
    x = data_processing(x)
    train_svm(x, y)
    train_cnn(x, y, model_weights)
    # network = CNN.CNN()
    # network.test(x, y, model_weights)


if __name__ == "__main__":
    main(dir, model_weights)
