import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler
import cv2
import os

import CNN


def load_images_from_folder(dir):
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


def load_dataset():
    dir = 'dataset\chars74k-lite'
    chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    x, y = load_images_from_folder(dir)
    return x, y


def main():
    x, y = load_dataset()
    x = x/255
    print(x)
    x_train, x_test, y_train, y_test = train_test_split(rescaledX, y, test_size=0.2, random_state=42)
    CNN.run(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
