from src.models.train_model import train, save_metrics

if __name__ == '__main__':
    accuracy, f1, precision, recall = train()
    save_metrics(accuracy, f1, precision, recall)
