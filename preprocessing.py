import numpy as np
import re
import nltk
import pickle
import pandas as pd
from joblib import dump
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


def load_dataset():
    dataset = pd.read_csv(
        'a1_RestaurantReviews_HistoricDump.tsv',
        delimiter='\t',
        quoting=3)
    return dataset


def clean_review(review, all_stopwords):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review


def review_preprocess(dataset):
    nltk.download('stopwords')

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus = []
    for i in range(0, 900):
        corpus.append(clean_review(dataset['Review'][i], all_stopwords))
    return corpus


def transformation():
    dataset = load_dataset()
    corpus = review_preprocess(dataset)

    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()

    # Saving BoW dictionary to later use in prediction
    bow_path = 'c1_BoW_Sentiment_Model.pkl'
    pickle.dump(cv, open(bow_path, "wb"))

    # Saved preprocess data
    dump(X, 'preprocessed_data.joblib')


if __name__ == "__main__":
    transformation()