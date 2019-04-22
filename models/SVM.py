from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC

from feature_extraction import data_processing, filters
import load_data


def linear_svm(x, y):
    print('=== Linear SVM classification ===')
    random_state = 100

    # Data processing
    x = data_processing.scale_input(x)
    x = data_processing.grey_scale(x)
    x = filters.add_edge_detection_filter(x)
    x = data_processing.remove_dimension(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=random_state,
                                                        stratify=y)

    # Fit model
    clf = LinearSVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Test accuracy: ', accuracy_score(y_test, y_pred))


def svm(x, y):
    print('=== Non-linear SVM classification ===')
    random_state = 100
    # Data processing
    x = data_processing.scale_input(x)
    x = data_processing.grey_scale(x)
    x = filters.add_edge_detection_filter(x)
    x = data_processing.remove_dimension(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=random_state,
                                                        stratify=y)

    # Fit model
    clf = SVC(gamma='scale', kernel='rbf')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Test accuracy: ', accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    img, target = load_data.load_dataset()
    linear_svm(img, target)
    svm(img, target)
