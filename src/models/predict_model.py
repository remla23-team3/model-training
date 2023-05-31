import pickle
import joblib

from data.preprocess import load_dataset, review_preprocess

cvFile = 'data/processed/c1_BoW_Sentiment_Model.pkl'
resources_dir_class = 'data/processed/c2_Classifier_Sentiment_Model'

def predict_dataset(classifier, cv):
    number_lines, dataset = load_dataset('data/raw/restaurant_reviews_without_rating.tsv')
    corpus = review_preprocess(dataset, number_lines)

    X_fresh = cv.transform(corpus).toarray()
    y_pred = classifier.predict(X_fresh)

    dataset['predicted_label'] = y_pred.tolist()

    output_name = 'data/processed/c3_Predicted_Sentiments_Fresh_Dump.tsv'
    dataset.to_csv(output_name, sep='\t', encoding='UTF-8', index=False)


def predict_single(classifier, review, cv) -> int:
    processed_input = cv.transform([review]).toarray()[0]
    prediction = classifier.predict([processed_input])[0]

    prediction_map = {
        0: 'negative',
        1: 'positive'
    }

    print(f'The model believes the review is {prediction_map[prediction]}.')
    return prediction


def predict():
    with open(cvFile, 'rb') as f:
        cv = pickle.load(f)

    classifier = joblib.load(resources_dir_class)

    review = input('Give me an input to perform a sentiment analysis.\n>')
    predict_single(classifier, review, cv)

    predict_dataset(classifier, cv)
