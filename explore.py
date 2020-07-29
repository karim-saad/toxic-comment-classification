import pandas as pd
import os


def explore(train, test, test_labels):
    print(train.head())
    print(test.head())
    print(test_labels.head())
