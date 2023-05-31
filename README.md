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
2.1 To **execute the pipelines** with DVC run:
```bash
dvc repro
```


2.2 To run metrics with DVC:

To run the experiments and **calculate metrics**:
```bash
dvc exp run
```

See the differences with: 
```bash
dvc metrics diff
```

