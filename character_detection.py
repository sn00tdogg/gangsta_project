from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import random
import numpy as np

from models import CNN
from load_data import load_data_detection, load_data_chars

random.seed(100)                  # Sets seed to get the same results across runs       

model_weights = 'model_weights.hdf5'


def data_processing(x):           # Grayscale, normalization
    x = x.astype('float32')
    x /= 255  # Make between 0 and 1
    x = rgb2gray(x)  # Convert to grey scale
    x = x.reshape((1,) + x.shape + (1,))  # Reshape the images
    return x


def check_if_all_zero(img):       # Checks if sliding window is all white/all black
    img = img.reshape(400)
    percentage_zero = 0
    for i in range(len(img)):
        if img[i] < 10**(-5):
            percentage_zero += 1
        elif img[i] > 0.999999:
            percentage_zero += 1
    if percentage_zero/len(img) < 0.5:
        return True
    else:
        return False


def slicing_window(img, model_weights):     # Implements sliding windows with constant width/height
    stride_width = 5
    stride_height = 5
    width = 20
    height = 20
    #Import model weights from fitted CNN characterisation task
    network = CNN.CNN(num_classes=27, sample=img[0, 0:0 + height, 0:0 + height], model_weights=model_weights,
                      network_type='simple')

    predictions = []
    for i in range(0, img.shape[1]-height, stride_height):  #Slides windows across image, saves letters to prediction list
        for j in range(0, img.shape[2]-width, stride_width):
            predicted_prob = network.predict_character(img[:, i:i + height, j:j + width])
            if np.amax(predicted_prob) > 0.7 and np.argmax(predicted_prob) != 0 and check_if_all_zero(img[:, i:i + height, j:j + width]):
                predictions.append([np.argmax(predicted_prob), j, i])
                print(str(chr(96+np.argmax(predicted_prob))), j, i)
    fig, ax = plt.subplots(1)   #Make figure
    ax.imshow(img[0, :, :, 0], cmap='gray') # Show image
    for i in range(len(predictions)):       # Add red frames onto letter predictions
        rect = patches.Rectangle((predictions[i][1], predictions[i][2]), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()


def character_detection(model_weights):     # Perform detection task on both test images
    img1, img2 = load_data_detection()
    img1 = data_processing(img1)
    img2 = data_processing(img2)
    plt.imshow(img1[0, :, :, 0], cmap='gray')
    plt.show()
    slicing_window(img1, model_weights)
    slicing_window(img2, model_weights)


if __name__ == "__main__":
    character_detection(model_weights)
