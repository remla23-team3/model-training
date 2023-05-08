import pickle
import joblib
from preprocessing import load_dataset, review_preprocess


def predict_fresh_X(classifier, cv):
    number_lines, dataset = load_dataset('assets/a2_RestaurantReviews_FreshDump.tsv')
    corpus = review_preprocess(dataset, number_lines)

    X_fresh = cv.transform(corpus).toarray()
    y_pred = classifier.predict(X_fresh)

    dataset['predicted_label'] = y_pred.tolist()

    dataset.to_csv("assets/c3_Predicted_Sentiments_Fresh_Dump.tsv", sep='\t', encoding='UTF-8', index=False)


def predict_single(classifier, review, cv) -> float:
    processed_input = cv.transform([review]).toarray()[0]
    prediction = classifier.predict([processed_input])[0]

    prediction_map = {
        0: "negative",
        1: "positive"
    }
    print(f"The model believes the review is {prediction_map[prediction]}.")
    return prediction


def predict_single_review(review: str) -> float:
    cvFile = 'c1_BoW_Sentiment_Model.pkl'
    cv = pickle.load(open(cvFile, "rb"))

    # Predictions (via sentiment classifier)
    classifier = joblib.load('c2_Classifier_Sentiment_Model')
    return predict_single(classifier, review, cv)


def main():
    cvFile = 'assets/c1_BoW_Sentiment_Model.pkl'
    cv = pickle.load(open(cvFile, "rb"))

    # Predictions (via sentiment classifier)
    classifier = joblib.load('assets/c2_Classifier_Sentiment_Model')

    # Predicting single inputs
    review = input("Give me an input to perform a sentiment analysis.\n>")
    predict_single(classifier, review, cv)

    # Predicting whole dataset
    predict_fresh_X(classifier, cv)


if __name__ == "__main__":
    main()