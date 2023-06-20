"""
Download and extract data.
"""
import urllib.request


# google drive auto-downloadable link
URL = 'https://drive.google.com/uc?export=download&id=18AJFXOVH8kmqeCRVAzr7SvBtnWNmbmmU'

zip_path, proba = urllib.request.urlretrieve(URL, filename="data/raw/restaurant_reviews_with_rating.tsv")
print(zip_path)
