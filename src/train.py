import os, sys
from pprint import pprint
from models.train_model import train, save_metrics

if __name__ == '__main__':
    current_directory = os.getcwd()
    print(current_directory)
    sys.path.append(current_directory)
    pprint(sys.path)
    accuracy, f1, precision, recall = train()
    save_metrics(accuracy, f1, precision, recall)
