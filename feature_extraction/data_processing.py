import numpy as np
import random
from skimage.color import rgb2gray
from skimage.transform import rotate


def scale_input(x):
    """Scales the values between 0 and 1"""
    x = x.astype('float32')
    return x/255


def grey_scale(x):
    """Inverts channels to 1D gray scale"""
    return rgb2gray(x)


def add_dimension(x):
    """Adds dimension to input"""
    return x.reshape(x.shape + (1,))  # Reshape the images


def remove_dimension(x):
    """Flattens input form Nx20x20 -> Nx4000"""
    return x.reshape(len(x), 400)


def add_pictures_without_chars(x, y):
    """Adds empty pictures without """
    num = 500
    new_x = np.zeros([x.shape[0]+num, x.shape[1], x.shape[2], x.shape[3]])
    new_y = np.zeros(len(y)+num)
    new_x[:x.shape[0]] = x
    new_y[:y.shape[0]] = y+1
    for i in range(num):
        new_x[x.shape[0]+i] = np.zeros([x.shape[1], x.shape[2], x.shape[3]])
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                new_x[x.shape[0]+i, j, k, 0] = min(np.random.gamma(shape=0.8, scale=4, size=1), 1)
    return new_x, new_y


def random_rotation(image_array):
    """Rotates image randomly 25 degrees"""
    random_degree = random.uniform(-25, 25)
    return rotate(image_array, random_degree)


def invert_colors(x):
    """Inverts colors, so black->white, and white->black"""
    if np.random.random() > 0.5:
        return np.ones(x.shape)-x
    return x

