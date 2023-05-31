import re
import pickle
import nltk
import pandas as pd
from joblib import dump
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


def load_dataset(file_name):
    """
        Loads the dataset from file with the given file_name.
    """
    dataset = pd.read_csv(file_name, delimiter='\t', dtype={'Review': str, 'Liked': int}) \
        [['Review', 'Liked']]
    return len(dataset), dataset

def clean_review(review, all_stopwords):
    """
        Cleans up reviews by removing the stopwords.
    """

    porter_stemmer = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower().split()
    review = [porter_stemmer.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review


def review_preprocess(dataset, number_lines):
    """
        Preprocesses the English reviews by removing the stopwords and negations.
    """

    nltk.download('stopwords')

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus = []
    for i in range(0, number_lines):
        corpus.append(clean_review(dataset['Review'][i], all_stopwords))
    return corpus


def preprocess_data(file_name):
    """
        Loads the dataset and preprocesses it.
    """
    number_lines, dataset = load_dataset(file_name)
    corpus = review_preprocess(dataset, number_lines)

    count_vectorizer = CountVectorizer(max_features=1420)
    X = count_vectorizer.fit_transform(corpus).toarray()

    with open('data/processed/c1_BoW_Sentiment_Model.pkl', 'wb') as file:
        pickle.dump(count_vectorizer, file)

    with open('data/processed/preprocessed_data_training', 'wb') as file:
        pickle.dump(X, file)
