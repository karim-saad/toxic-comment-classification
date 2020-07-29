import os
from zipfile import ZipFile


def unzip_files():
    if not os.path.isfile('train.csv'):
        ZipFile('train.csv.zip', 'r').extractall(os.getcwd())

    if not os.path.isfile('test.csv'):
        ZipFile('test.csv.zip', 'r').extractall(os.getcwd())

    if not os.path.isfile('test_labels.csv'):
        ZipFile('test_labels.csv.zip', 'r').extractall(os.getcwd())
