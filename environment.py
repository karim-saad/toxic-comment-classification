import os
from zipfile import ZipFile
import pandas as pd


cwd = os.getcwd()


def environment():
    '''
    Set up the current environment, including unzipping zip files as .csv files

    Returns:
    - train (pd.DataFrame): training dataset
    - test (pd.DataFrame): testing dataset
    - test_labels (pd.DataFrame): testing labels
    '''

    unzip_files()
    return csv_to_dataframe()


def unzip_files():
    '''
    Unzip all datasets if it hasn't already been done
    '''

    if not os.path.isfile('data/train.csv'):
        ZipFile('data/train.csv.zip', 'r').extractall(os.path.join(cwd, 'data'))

    if not os.path.isfile('data/test.csv'):
        ZipFile('data/test.csv.zip', 'r').extractall(os.path.join(cwd, 'data'))

    if not os.path.isfile('data/test_labels.csv'):
        ZipFile('data/test_labels.csv.zip',
                'r').extractall(os.path.join(cwd, 'data'))


def csv_to_dataframe():
    '''
    Converts all .csv files to pandas DataFrame

    Returns:
    - train (pd.DataFrame): training dataset
    - test (pd.DataFrame): testing dataset
    - test_labels (pd.DataFrame): testing labels
    '''

    train_filename = os.path.join(cwd, 'data', 'train.csv')
    train = pd.read_csv(train_filename)

    test_filename = os.path.join(cwd, 'data', 'test.csv')
    test = pd.read_csv(test_filename)

    test_labels_filename = os.path.join(cwd, 'data', 'test_labels.csv')
    test_labels = pd.read_csv(test_labels_filename)

    return train, test, test_labels
