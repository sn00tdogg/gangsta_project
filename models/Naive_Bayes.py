from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from feature_extraction import data_processing, filters
import load_data


def naive_bayes(x, y):
    print('=== Naive Bayes classification ===')
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
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('Test accuracy: ', accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    img, target = load_data.load_dataset()
    naive_bayes(img, target)
