from setuptools import find_packages, setup

setup(
    name='model_training',
    version='0.1',
    packages=find_packages(include=['model_training', 'model_training.*']),
    package_data={'model_training': ['c1_BoW_Sentiment_Model.pkl', 'c2_Classifier_Sentiment_Model']},
    include_package_data=True,
)