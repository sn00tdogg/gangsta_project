import numpy as np
import cv2 as cv
from skimage import filters, feature


def add_edge_detection_filter(x):  # Applies Canny edge detection algorithm row by row
    """Adds filter that detect edges"""
    if len(x) > 20:
        for i in range(len(x)):
            x[i] = feature.canny(x[i])
    else:
        x = feature.canny(x)
    return x


def add_gaussian_filter(x):        # Applies Gaussian noise filter
    """Adds gaussian filter"""
    if len(x) > 20:
        for i in range(len(x)):
            x[i] = filters.gaussian(x[i])
    else:
        x = filters.gaussian(x)
    return x


def add_median_filter(x):       # Applies the edge detection filter
    """Adds median filter"""
    if len(x) > 20:
        for i in range(len(x)):
            x[i] = filters.median(x[i])
    else:
        x = filters.median(x)
    return x


def add_bilateral_filter(x):    # Applies bilateral filter 
    """Add bilateral filter"""
    if len(x) > 20:
        for i in range(len(x)):
            x[i] = cv.bilateralFilter(x[0], 2, 2, 15.)
    else:
        x = cv.bilateralFilter(x, 2, 2, 15.)
    return x


def add_otsu_filter(x):     # Applies Otsu's method. Clusters the data and colors them entirely white or black
    if len(x) > 20:
        binary = np.zeros(x.shape)      # Creates binary output array
        for i in range(len(x)):
            threshold = filters.threshold_otsu(x[i])      # Defines Otsu threshold
            binary[i] = x[i] > threshold                  # Checks if above or below threshold, saves in binary array
    else:
        threshold = filters.threshold_otsu(x)
        binary = x > threshold
    return binary
