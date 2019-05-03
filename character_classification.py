import random

from models import Naive_Bayes, SVM, CNN, KNN
from load_data import load_data_chars
import plots

random.seed(100)


def character_classification():             #Loads and shapes data
    print('Loading data...')
    x, y = load_data_chars()
    print('Processing data..')
    print('Training data shape: ', x.shape)
    print('Test data shape: ', y.shape)
    plots.plot_filters(x[0])                # Plots single filters
    SVM.svm(x, y)                           # Fits all 4 models
    Naive_Bayes.naive_bayes(x, y)
    KNN.knn(x, y)
    CNN.fit_cnn(x, y, trials=1, network_type='simple')


if __name__ == "__main__":
    character_classification()
