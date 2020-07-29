import os
from zipfile import ZipFile
import pandas as pd


def environment():
    unzip_files()
    return csv_to_dataframe()


def unzip_files():
    if not os.path.isfile('train.csv'):
        ZipFile('train.csv.zip', 'r').extractall(os.getcwd())

    if not os.path.isfile('test.csv'):
        ZipFile('test.csv.zip', 'r').extractall(os.getcwd())

    if not os.path.isfile('test_labels.csv'):
        ZipFile('test_labels.csv.zip', 'r').extractall(os.getcwd())


def csv_to_dataframe():
    train_filename = os.path.join(os.getcwd(), 'train.csv')
    train = pd.read_csv(train_filename)
    test_filename = os.path.join(os.getcwd(), 'test.csv')
    test = pd.read_csv(test_filename)
    test_labels_filename = os.path.join(os.getcwd(), 'test_labels.csv')
    test_labels = pd.read_csv(test_labels_filename)
    return train, test, test_labels
