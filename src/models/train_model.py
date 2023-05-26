import joblib
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from data.preprocess import load_dataset

def train_model(X_train, y_train):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, 'data/processed/c2_Classifier_Sentiment_Model')

    return classifier


def train():
    _, dataset = load_dataset('data/raw/restaurant_reviews_with_rating.tsv')
    X = load('data/processed/preprocessed_data_training.joblib')
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    classifier = train_model(X_train, y_train)

    y_test_prediction = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_prediction)

    print(f'Accuracy: {accuracy:.3f}')