import joblib
from joblib import load
from os import path
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocessing import load_dataset


def dividing_train_test(X, y):
    # Dividing dataset into training and test set
    return train_test_split(X, y, test_size=0.20, random_state=0)


def training(X_train, y_train):
    # Model fitting (Naive Bayes)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Exporting NB Classifier to later use in prediction
    resources_dir_classifier = path.join(path.dirname(__file__), 'c2_Classifier_Sentiment_Model')
    joblib.dump(classifier, resources_dir_classifier)

    return classifier


def predict(classifier, X_test):
    return classifier.predict(X_test)


def compute_accuracy_and_conf_matrix(classifier, X_test, y_test):
    y_pred = predict(classifier,X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracy_score(y_test, y_pred)


def main():
    _, dataset = load_dataset('assets/a1_RestaurantReviews_HistoricDump.tsv')
    X = load('preprocessed_data_training.joblib')
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = dividing_train_test(X, y)

    classifier = training(X_train, y_train)

    compute_accuracy_and_conf_matrix(classifier, X_test, y_test)


if __name__ == "__main__":
    main()

