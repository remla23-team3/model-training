# Model-training

1. Install the requirements

```bash
pip install -r requirements.txt
```

2. Preprocess the training data, train the model and predict the fresh data with:

```bash
python model_training/preprocessing.py
python model_training/training.py
python model_training/predicting.py
```

To execute the pipelines with DVC:
```bash
dvc repro
```


To run metrics with DVC:

1. Run experiment
```bash
dvc exp run
```

2. See differences with 
```bash
dvc metrics diff
```

