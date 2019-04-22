import numpy as np
import cv2 as cv
import os

dir_chars = 'dataset\chars74k-lite'
dir_texts = 'dataset\detection-images'


def load_data_chars():
    images = np.zeros([7112, 20, 20, 3])
    targets = np.zeros([7112])
    chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    i = 0
    j = 0
    for c in chars:
        folder = os.path.join(dir_chars, c)
        for filename in os.listdir(folder):
            img = cv.imread(os.path.join(folder, filename))
            if img is not None:
                images[j] = img
                targets[j] = i
                j += 1
        i += 1
    return images, targets


def load_data_detection():
    j = 0
    img1 = None
    img2 = None
    for filename in os.listdir(dir_texts):
        img = cv.imread(os.path.join(dir_texts, filename))
        if img is not None:
            if j == 0:
                img1 = img
            else:
                img2 = img
            j += 1
    return img1, img2
