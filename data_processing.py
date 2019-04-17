import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma
from skimage.color import rgb2gray
from skimage import feature
import matplotlib.pyplot as plt
import cv2


class DataProcessing:
    def __init__(self, x, y):
        self.x = self.convert_to_greyscale(x)
        self.x = self.reduce_noise(self.x)
        # self.otsu_mask = self.compute_otsu_mask(self.x)
        self.edges = self.extract_edges(self.x)
        self.y = y
        # self.plot_edges(self.x)
        self.reshape()

    def extract_edges(self, x):
        edges = np.zeros([len(x), len(x[0]), len(x[0, 0])])
        for i in range(len(x)):
            edges[i] = feature.canny(x[i], sigma=1)
        return edges

    def convert_to_greyscale(self, x):
        x = x.astype('float32')
        x /= 255
        x = rgb2gray(x)
        return x

    def reduce_noise(self, x):
        for i in range(len(x)):
            x[i] = cv2.bilateralFilter(x[i], 2, 2, 15)
        return x

    def compute_otsu_mask(self, x):
        otsu_mask = np.zeros([len(x), len(x[0]), len(x[0, 0])])
        for i in range(len(x)):
            img_grey = cv2.cvtColor(x[i], cv2.COLOR_BGR2GRAY)
            otsu_mask[i] = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return otsu_mask

    def reshape(self):
        self.x = self.x.reshape(self.x.shape+(1,))
        self.edges = self.edges.reshape(self.edges.shape+(1,))

    def plot_edges(self, x):
        print(x)
        im = x[100]
        edges1 = feature.canny(im, sigma=1)
        edges2 = feature.canny(im, sigma=3)
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

        ax1.imshow(im, cmap=plt.cm.gray)
        ax1.axis('off')
        ax1.set_title('noisy image', fontsize=20)

        ax2.imshow(edges1, cmap=plt.cm.gray)
        ax2.axis('off')
        ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

        ax3.imshow(edges2, cmap=plt.cm.gray)
        ax3.axis('off')
        ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

        fig.tight_layout()
        plt.show()
