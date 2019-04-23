import numpy as np
import cv2 as cv
from skimage import filters, feature


def add_edge_detection_filter(x):
    """Adds filter that detect edges"""
    if len(x) > 20:
        for i in range(len(x)):
            x[i] = feature.canny(x[i])
    else:
        x = feature.canny(x)
    return x


def add_gaussian_filter(x):
    """Adds gaussian filter"""
    if len(x) > 20:
        for i in range(len(x)):
            x[i] = filters.gaussian(x[i])
    else:
        x = filters.gaussian(x)
    return x


def add_median_filter(x):
    """Adds median filter"""
    if len(x) > 20:
        for i in range(len(x)):
            x[i] = filters.median(x[i])
    else:
        x = filters.median(x)
    return x


def add_bilateral_filter(x):
    """Add bilateral filter"""
    if len(x) > 20:
        for i in range(len(x)):
            x[i] = cv.bilateralFilter(x[0], 2, 2, 15.)
    else:
        x = cv.bilateralFilter(x, 2, 2, 15.)
    return x


def add_otsu_filter(x):
    if len(x) > 20:
        binary = np.zeros(x.shape)
        for i in range(len(x)):
            threshold = filters.threshold_otsu(x[i])
            binary[i] = x[i] > threshold
    else:
        threshold = filters.threshold_otsu(x)
        binary = x > threshold
    return binary
