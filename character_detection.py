from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import random
import numpy as np

from models import CNN
from load_data import load_data_detection

random.seed(100)

model_weights = 'model_weights.hdf5'


def data_processing(x):
    x = x.astype('float32')
    x /= 255  # Make between 0 and 1
    x = rgb2gray(x)  # Convert to grey scale
    x = cv2.bilateralFilter(x, 2, 2, 15)  # Filter the images
    x = x.reshape((1,) + x.shape + (1,))  # Reshape the images
    return x


def slicing_window(img, model_weights):
    stride = 10
    width = 20
    height = 20
    network = CNN.CNN()
    img_slices = np.zeros([int((img.shape[1]-height)/stride)*int((img.shape[2]-width)/stride), 20, 20, 1])
    for i in range(0, img.shape[1]-height, stride):
        for j in range(0, img.shape[2]-width, stride):
            print(i, j)
            img_slices[int(i/stride*j/stride+j/stride)] = img[:, i:i + height, j:j + height]
    predicted_prob = network.predict_character(img_slices, model_weights)
    predictions = np.zeros([int((img.shape[1]-height)/stride), int((img.shape[2]-width)/stride)])
    for i in range(len(predictions[0])):
        for j in range(len(predictions[1])):
            if np.amax(predicted_prob[i*j+j]) > 0.7:
                predictions[i, j] = np.argmax(predicted_prob[i*j+j])
    print(predictions)
    fig, ax = plt.subplots(1)
    ax.imshow(img[0, :, :, 0])
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if predictions[i, j] != 0:
                rect = patches.Rectangle((i*5, j*5), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)
    plt.show()


def character_detection(model_weights):
    img1, img2 = load_data_detection()
    plt.imshow(img1)
    plt.show()
    img1 = data_processing(img1)
    img2 = data_processing(img2)
    plt.imshow(img1[0, :, :, 0])
    plt.show()
    slicing_window(img1, model_weights)
    # slicing_window(img2, model_weights)


if __name__ == "__main__":
    character_detection(model_weights)
