import pickle
import joblib
from joblib import load
from preprocessing import load_dataset


def predict_fresh_X(classifier):
    dataset = load_dataset()
    X_fresh = load('preprocessed_data.joblib')

    y_pred = classifier.predict(X_fresh)
    print(y_pred)

    dataset['predicted_label'] = y_pred.tolist()

    dataset.to_csv("c3_Predicted_Sentiments_Fresh_Dump.tsv", sep='\t', encoding='UTF-8', index=False)


def predict_single(classifier, review, cv):
    processed_input = cv.transform([review]).toarray()[0]
    prediction = classifier.predict([processed_input])[0]

    prediction_map = {
        0: "negative",
        1: "positive"
    }
    print(f"The model believes the review is {prediction_map[prediction]}.")


def main():
    cvFile = 'c1_BoW_Sentiment_Model.pkl'
    cv = pickle.load(open(cvFile, "rb"))

    # Predictions (via sentiment classifier)
    classifier = joblib.load('c2_Classifier_Sentiment_Model')

    # Predicting single inputs
    review = input("Give me an input to perform a sentiment analysis.\n>")
    predict_single(classifier, review, cv)

    # Predicting whole dataset
    predict_fresh_X(classifier)


if __name__ == "__main__":
    main()