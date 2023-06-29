import numpy as np

from model_training.src.data.preprocess import load_dataset
from scipy.stats import ks_2samp
import pytest
import pickle
from sklearn.model_selection import train_test_split

@pytest.fixture()
def dataset_with_labels():
    _, dataset_with_labels = load_dataset('tests_data/restaurant_reviews_with_rating.tsv')
    yield dataset_with_labels

@pytest.fixture()
def preprocessed_data(dataset_with_labels):
    with open('tests_data/preprocessed_data_training', 'rb') as file:
        preprocessed_data = pickle.load(file)

    yield preprocessed_data

def test_monitoring_invariants(dataset_with_labels, preprocessed_data):
    """
    Uses the two-sample Kolmogorov-Smirnov test to determine whether the two distributions are equal.
    """
    y = dataset_with_labels.iloc[:, -1].values
    X_train, X_test, _, _ = train_test_split(preprocessed_data, y, test_size=0.2, random_state=42)

    feature_sample_training = X_train[np.random.randint(0, X_train.shape[0])]
    feature_sample_serving = X_test[np.random.randint(0, X_test.shape[0])]

    ks_statistic = ks_2samp(feature_sample_training, feature_sample_serving)
    p_value = ks_statistic[1]

    assert p_value > 0.05

def test_monitoring_not_invariant(dataset_with_labels, preprocessed_data):
    """
    Bad-weather test
    Uses the two-sample Kolmogorov-Smirnov test to determine whether the two distributions are different.
    """
    y = dataset_with_labels.iloc[:, -1].values
    X_train, _, _, _ = train_test_split(preprocessed_data, y, test_size=0.2, random_state=42)

    feature_sample_training = X_train[np.random.randint(0, X_train.shape[0])]
    feature_sample_serving = np.random.normal(0.1, 1.5, 1000)

    ks_statistic = ks_2samp(feature_sample_training, feature_sample_serving)
    p_value = ks_statistic[1]

    assert p_value <= 0.05