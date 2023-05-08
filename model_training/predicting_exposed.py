import pickle
import joblib


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
    cvFile = 'c1_BoW_Sentiment_Model.pkl'
    cv = pickle.load(open(cvFile, "rb"))

    # Predictions (via sentiment classifier)
    classifier = joblib.load('c2_Classifier_Sentiment_Model')

    # Predicting single inputs
    review = input("Give me an input to perform a sentiment analysis.\n>")
    predict_single(classifier, review, cv)



if __name__ == "__main__":
    main()