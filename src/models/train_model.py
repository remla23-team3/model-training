import json
import pickle
import os
import sys
from pprint import pprint
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

#pylint: disable=wrong-import-position

current_directory = os.getcwd()
sys.path.append(current_directory)

from src.data.preprocess import load_dataset

#pylint: enable=wrong-import-position

def train_model(X_train, y_train):
    """
        Fits the model using the training set.
    """
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, 'data/processed/c2_Classifier_Sentiment_Model')

    return classifier


def train(state=42, test = False):
    """
        Model training function.
        Loads the dataset, trains the model and stores it.
        Models are stored remotely using dvc.
    """
    if test:
        _, dataset = load_dataset('tests_data/restaurant_reviews_with_rating.tsv')
        with open('tests_data/preprocessed_data_training', 'rb') as file:
            X = pickle.load(file)
    else:
        _, dataset = load_dataset('data/raw/restaurant_reviews_with_rating.tsv')
        with open('data/processed/preprocessed_data_training', 'rb') as file:
            X = pickle.load(file)

    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=state)

    classifier = train_model(X_train, y_train)

    return evaluate_score(classifier, X_test, y_test)


def evaluate_score(classifier, X_test, y_test):
    y_test_prediction = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_test_prediction)
    f1 = f1_score(y_test, y_test_prediction)
    precision = precision_score(y_test, y_test_prediction)
    recall = recall_score(y_test, y_test_prediction)

    return accuracy, f1, precision, recall


def save_metrics(accuracy, f1, precision, recall):

    with open('src/metrics.json', 'w', encoding='utf-8') as file:
        json.dump(
            {
                "train": {
                    "accuracy": accuracy,
                    "F1-Score": f1,
                    "precision": precision,
                    "recall": recall,
                }
            },
            file, ensure_ascii=False, indent=4)

    print(f'Accuracy: {accuracy:.3f}')
    