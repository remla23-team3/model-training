from src.data.preprocess import clean_review, preprocess_data

def test_clean_review():
    reviews_and_cleaned = {
        'I love this restaurant and I will go back again.': 'love restaur go back',
        'I hate this restaurant and I would not go back again.': 'hate restaur would not go back',
        'This is an awful menu.': 'aw menu',
        'This chicken was goddamn awful, salad was not too bad.': 'chicken goddamn aw salad not bad',
        'Excellent kapsalon. Best hot sauce in town. Friendly staff and prices.': 'excel kapsalon best hot sauc town friendli staff price',
    }

    for review, cleaned_review in reviews_and_cleaned.items():
        assert clean_review(review) == cleaned_review
