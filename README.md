# Model-training

## 1. Installing the requirements
To make use of the model, preferably create a 
<span style="background-color: green">**Python virtual envionment**</span>
 and follow the instructions below:


```bash
pip install -r requirements.txt
```

## 2. Running the pipelines
### 2.1 To **download the required resources** run:
```bash
dvc pull
```

### 2.2 To **execute the pipelines** with DVC run:
```bash
dvc repro
```
It is a known issue that Goggle Drive sharing through links might cause issues with DVC: https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#url-format.

If experiencing errors accessing the remote, you can request direct access by sending us your gmail address at `daniela.toader07@gmail.com`.

### 2.3 To run experiments with DVC:

If you want to **save the results as an experiment** run:
```bash
dvc exp run
```

See the differences with: 
```bash
dvc metrics diff
```

