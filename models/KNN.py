from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import data_processing


def linear_svm(x, y):
    print('=== Linear SVM classification ===')
    random_state = 100

    # Data processing
    x = data_processing.scale_input(x)
    x = data_processing.grey_scale(x)
    x = data_processing.add_edge_detection_filter(x)
    x = data_processing.remove_dimension(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=random_state,
                                                        stratify=y)

    # Fit model
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Test accuracy: ', accuracy_score(y_test, y_pred))
