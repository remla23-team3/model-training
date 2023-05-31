import json
import os
import joblib
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data.preprocess import load_dataset

def train_model(X_train, y_train):
    """
        Fits the model using the training set.
    """
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, 'data/processed/c2_Classifier_Sentiment_Model')

    return classifier


def train():
    """
        Model training function.
        Loads the dataset, trains the model and stores it.
        Models are stored remotely using dvc.
    """
    _, dataset = load_dataset('data/raw/restaurant_reviews_with_rating.tsv')
    X = load('data/processed/preprocessed_data_training.joblib')
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    classifier = train_model(X_train, y_train)

    y_test_prediction = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_prediction)
    f1 = f1_score(y_test, y_test_prediction)
    precision = precision_score(y_test, y_test_prediction)
    recall = recall_score(y_test, y_test_prediction)

    if not os.path.exists('src/metrics'):
        os.makedirs('src/metrics')

    with open('data.json', 'w', encoding='utf-8') as file:
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
