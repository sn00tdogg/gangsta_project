import random

from models import Naive_Bayes, SVM, CNN, KNN
import plots
import load_data

random.seed(100)


def character_classification():
    print('Loading data...')
    x, y = load_data.load_dataset()
    print('Processing data..')
    print('Training data shape: ', x.shape)
    print('Test data shape: ', y.shape)
    plots.plot_filters(x[0])
    SVM.linear_svm(x, y)
    SVM.svm(x, y)
    Naive_Bayes.naive_bayes(x, y)
    KNN.knn(x, y)
    CNN.fit_cnn(x, y, trials=10)


if __name__ == "__main__":
    character_classification()
