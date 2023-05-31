# Model-training

## 1. Installing the requirements
To make use of the model, preferably create a 
<span style="background-color: green">**Python virtual envionment**</span>
 and follow the instructions below:


```bash
pip install -r requirements.txt
```

<!-- 2. Preprocess the training data, train the model and predict the fresh data with:

```bash
python model_training/preprocessing.py
python model_training/training.py
python model_training/predicting.py
``` -->

## 2. Running the pipelines
2.1 To **download the required resources** run:
```bash
dvc pull
```

2.2 To **execute the pipelines** with DVC run:
```bash
dvc repro
```

2.3 To run experiments with DVC:

If you want to **save the results as an experiment** run:
```bash
dvc exp run
```

See the differences with: 
```bash
dvc metrics diff
```

