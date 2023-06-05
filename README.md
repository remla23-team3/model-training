[![Coverage Status](https://coveralls.io/repos/github/remla23-team3/model-training/badge.svg)](https://coveralls.io/github/remla23-team3/model-training)

# Model-training

## 1. Installing the requirements
To make use of the model, preferably create a 
<span style="background-color: green">**Python virtual envionment**</span>
 and follow the instructions below:


```bash
pip install -r requirements.txt
```

## 2. Running the pipelines
You can run the pipelines using either of the 2.1 or 2.2 methods.

### 2.1 Without DVC
Preprocess the training data, train the model and predict the fresh data with:

```bash
python src/preprocess.py
python src/train.py
python src/predict.py
```


### 2.2 With DVC
#### 2.2.1 To **download the required resources** run:
```bash
dvc pull
```

#### 2.2.2 To **execute the pipelines** with DVC run:
```bash
dvc repro
```
It is a known issue that Goggle Drive sharing through links might cause issues with DVC: https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#url-format.

If experiencing errors accessing the remote, you can request direct access by sending us your gmail address at `daniela.toader07@gmail.com`.

#### 2.2.3 To run experiments with DVC:

If you want to **save the results as an experiment** run:
```bash
dvc exp run
```

See the differences with: 
```bash
dvc metrics diff
```

## 3. ML testing
To test using pytest, from the *virtual environment with the installed requirements* run:
```bash
pytest
```

## 4. Handling errors
### 4.1 **If encountering an error similar to:**
```bash
KeyError: "None of [Index(['Review', 'Liked'], dtype='object')] are in the [columns]"
```

### **and/or pulling the datasets with dvc fails:**
This means that the intput data is not correct. This might happen if the dvc commands fail to fetch it correctly. In this case, you can delete the `.dvc/cache`, the `data/processed` folder and the `.tsv` files from the`data/raw` folder and try following the steps in `2.2` again.


### 4.2 **If pulling the datasets with dvc fails and deleting the *.dvc/cache* and *data/processed* folders does not work:**

- 4.2.1: Copy the `restaurant_reviews_with_rating.tsv` and `restaurant_reviews_without_rating.tsv` located in the `data/backup` folder and add them to the `data/raw` folder. If the `.tsv` files are already there, please overwrite them.

- 4.2.2: Delete the `.dvc/cache` and `data/processed` folders as well to make sure incorrect input data does not get retrieved from the cache and that incorrectly preprocessed data does not get used. 

- 4.2.3 You can then try running the commands starting from `2.2.2`.

### 4.3 **If all fails:**
Repeat `4.2.1` and `4.2.2` and instead of running the pipelines using dvc, run them as described in `2.1`.