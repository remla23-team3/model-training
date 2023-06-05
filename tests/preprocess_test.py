import json

from src.models.train_model import train
from src.data.preprocess import clean_review, preprocess_data, load_dataset, review_preprocess
from src.models.train_model import train, evaluate_score
import pytest
import pickle
import joblib
from sklearn.model_selection import train_test_split


def test_clean_review():
    "Test Feature and Data"
    reviews_and_cleaned = {
        'I love this restaurant and I will go back again.': 'love restaur go back',
        'I hate this restaurant and I would not go back again.': 'hate restaur would not go back',
        'This is an awful menu.': 'aw menu',
        'This chicken was goddamn awful, salad was not too bad.': 'chicken goddamn aw salad not bad',
        'Excellent kapsalon. Best hot sauce in town. Friendly staff and prices.': 'excel kapsalon best hot sauc town friendli staff price',
    }

    for review, cleaned_review in reviews_and_cleaned.items():
        assert clean_review(review) == cleaned_review


def test_nondeterminism_robustness():
    "Model Validation test"
    with open('src/metrics.json', 'r') as file:
        metrics = json.load(file)

    original_accuracy = metrics["train"]["accuracy"]

    for seed in [1,2,3,40,50,100]:
        accuracy_new_seed, _, _, _ = train(seed)
        print(f"Accuracy with seed {seed} " + str(accuracy_new_seed), seed)
        print("Original accuracy " + str(original_accuracy))
        assert abs(original_accuracy - accuracy_new_seed) <= 0.15


@pytest.fixture()
def trained_classifier():
    train()

    trained_classifier = joblib.load('data/processed/c2_Classifier_Sentiment_Model')
    yield trained_classifier


@pytest.fixture()
def dataset_with_labels():
    _, dataset_with_labels = load_dataset('data/raw/restaurant_reviews_with_rating.tsv')
    yield dataset_with_labels

@pytest.fixture()
def preprocessed_data(dataset_with_labels):
    with open('data/processed/preprocessed_data_training', 'rb') as file:
        preprocessed_data = pickle.load(file)

    yield preprocessed_data


def test_data_slice(trained_classifier, dataset_with_labels, preprocessed_data):

    y = dataset_with_labels.iloc[:, -1].values
    _, X_test_data, _, y_test_data = train_test_split(preprocessed_data, y, test_size=0.2, random_state=42)

    with open('data/processed/c1_BoW_Sentiment_Model.pkl', 'rb') as f:
        cv = pickle.load(f)

    original_score, _, _, _ = evaluate_score(trained_classifier, X_test_data, y_test_data)

    sliced_data = dataset_with_labels[dataset_with_labels['Liked'] == 1]
    sliced_data = sliced_data.reset_index(drop=True)

    corpus = review_preprocess(sliced_data, len(sliced_data))
    X_sliced_data = cv.transform(corpus).toarray()
    y_sliced_data = sliced_data.iloc[:, -1].values

    sliced_score, _, _, _ = evaluate_score(trained_classifier, X_sliced_data, y_sliced_data)

    assert abs(original_score - sliced_score) <= 0.1

    # sliced_data = test_data[test_data['Liked'] == 0]
    # sliced_score = evaluate_score(classifier, X_sliced_data, y_sliced_data)
    # assert abs(original_score - sliced_score) <= 0.05