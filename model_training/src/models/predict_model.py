import os
import sys
import pickle
import joblib
import json

from model_training.src.data.preprocess import load_dataset, review_preprocess

path2 = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(path2)

cvFile = path2 + r'\model_training\data\processed\c1_BoW_Sentiment_Model.pkl'
resources_dir_class = path2 + r'\model_training\data\processed\c2_Classifier_Sentiment_Model'

with open(cvFile, 'rb') as f:
    cv = pickle.load(f)

classifier = joblib.load(resources_dir_class)

def predict_dataset():
    number_lines, dataset = load_dataset('model_training/data/'+
                                         'raw/restaurant_reviews_without_rating.tsv')
    corpus = review_preprocess(dataset, number_lines)

    X_fresh = cv.transform(corpus).toarray()
    y_pred = classifier.predict(X_fresh)

    dataset['predicted_label'] = y_pred.tolist()

    output_name = 'model_training/data/processed/c3_preprocessed_no_rated_data.tsv'
    dataset.to_csv(output_name, sep='\t', encoding='UTF-8', index=False)


def predict_single(review) -> str:
    processed_input = cv.transform([review]).toarray()[0]
    prediction = classifier.predict([processed_input])[0]

    prediction_map = {
        0: 'negative',
        1: 'positive'
    }

    print(f'The model believes the review is {prediction_map[prediction]}.')
    return json.dumps(int(prediction))


def predict():

    review = input('Give me an input to perform a sentiment analysis.\n>')
    predict_single(review)

    predict_dataset()
