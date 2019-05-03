import matplotlib.pyplot as plt

from feature_extraction.data_processing import grey_scale, scale_input
from feature_extraction.filters import add_gaussian_filter, add_bilateral_filter,\
    add_median_filter, add_edge_detection_filter, add_otsu_filter

from load_data import load_data_chars       # Load the classes


def plot_filters(x):        # Creates grid of plots of the filters applied one-by-one to the same letter image
    """Plots various filters used for some image"""
    img = None
    if len(x.shape) == 4:
        img = x[0]
    elif len(x.shape) == 3:
        img = x
    else:
        ValueError('Not valid input')
    img = scale_input(img)
    img = grey_scale(img)
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 3),
                           sharex='all', sharey='all')
    ax[0, 0].imshow(img, cmap='gray')       # Plot the raw data image, with no filter
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Image', fontsize=10)

    ax[0, 1].imshow(add_edge_detection_filter(img), cmap='gray')   #Plot image with Canny based edge detection filter
    ax[0, 1].axis('off')
    ax[0, 1].set_title('Edge detection filter', fontsize=10)

    ax[0, 2].imshow(add_otsu_filter(img), cmap='gray')      # Plot Otsu filter
    ax[0, 2].axis('off')
    ax[0, 2].set_title('Otsu filter', fontsize=10)

    ax[1, 0].imshow(add_gaussian_filter(img), cmap='gray')  # Plot Gaussian noise reduction filter
    ax[1, 0].axis('off')
    ax[1, 0].set_title('Gaussian filter', fontsize=10)

    ax[1, 1].imshow(add_bilateral_filter(img), cmap='gray')    # Plot bilateral filter
    ax[1, 1].axis('off')
    ax[1, 1].set_title('Bilateral filter', fontsize=10)

    ax[1, 2].imshow(add_median_filter(img), cmap='gray')       # Plot median filter
    ax[1, 2].axis('off')
    ax[1, 2].set_title('Median filter', fontsize=10)

    fig.tight_layout()          # Set grid-of-plots layout to "tight"
    plt.show()                  # Show grid-of-plots


if __name__ == '__main__':
    x, y = load_data_chars()
    plot_filters(x[2])

