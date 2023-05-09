import pickle
import joblib
import json
from model_training.preprocessing import load_dataset, review_preprocess
from os import path

cvFile = path.join(path.dirname(__file__), 'c1_BoW_Sentiment_Model.pkl')
resources_dir_class = path.join(path.dirname(__file__), 'c2_Classifier_Sentiment_Model')
def predict_fresh_X(classifier, cv):
    number_lines, dataset = load_dataset('assets/a2_RestaurantReviews_FreshDump.tsv')
    corpus = review_preprocess(dataset, number_lines)

    X_fresh = cv.transform(corpus).toarray()
    y_pred = classifier.predict(X_fresh)

    dataset['predicted_label'] = y_pred.tolist()

    dataset.to_csv("assets/c3_Predicted_Sentiments_Fresh_Dump.tsv", sep='\t', encoding='UTF-8', index=False)


def predict_single(classifier, review, cv) -> int:
    processed_input = cv.transform([review]).toarray()[0]
    prediction = classifier.predict([processed_input])[0]

    prediction_map = {
        0: "negative",
        1: "positive"
    }
    print(f"The model believes the review is {prediction_map[prediction]}.")
    return prediction


def predict_single_review(review: str) -> int:
    cv = pickle.load(open(cvFile, "rb"))

    # Predictions (via sentiment classifier)
    classifier = joblib.load(resources_dir_class)
    return predict_single(classifier, review, cv)


def main():
    cv = pickle.load(open(cvFile, "rb"))

    # Predictions (via sentiment classifier)
    classifier = joblib.load(resources_dir_class)

    # Predicting single inputs
    review = input("Give me an input to perform a sentiment analysis.\n>")
    predict_single(classifier, review, cv)

    # Predicting whole dataset
    predict_fresh_X(classifier, cv)


if __name__ == "__main__":
    main()