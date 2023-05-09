import numpy as np
import re
import nltk
import pickle
import pandas as pd
from os import path
from joblib import dump
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


def load_dataset(file_name):
    dataset = pd.read_csv(
        file_name,
        delimiter='\t',
        quoting=3)
    return len(dataset), dataset


def clean_review(review, all_stopwords):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review


def review_preprocess(dataset, number_lines):
    nltk.download('stopwords')

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    print(dataset)
    corpus = []
    for i in range(0, number_lines):
        corpus.append(clean_review(dataset['Review'][i], all_stopwords))
    return corpus

def transformation(file_name):

    number_lines, dataset = load_dataset(file_name)
    corpus = review_preprocess(dataset, number_lines)

    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()

    # Saving BoW dictionary to later use in prediction
    bow_path = path.join(path.dirname(__file__), 'c1_BoW_Sentiment_Model.pkl')
    # bow_path = 'c1_BoW_Sentiment_Model.pkl'
    pickle.dump(cv, open(bow_path, "wb"))

    # Saved preprocess data
    dump(X, 'preprocessed_data_training.joblib')


if __name__ == "__main__":
    transformation('assets/a1_RestaurantReviews_HistoricDump.tsv')