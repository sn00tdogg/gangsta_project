import numpy as np
import cv2 as cv
import os

dir = 'dataset\chars74k-lite'


def load_dataset():
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
