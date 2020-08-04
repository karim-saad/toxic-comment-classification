import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def explore(train, test, test_labels):
    """Conducts an exploratory data analysis of the provided data

    Parameters:
    - train (pd.DataFrame): training dataset
    - test (pd.DataFrame): testing dataset
    - test_labels (pd.DataFrame): labels for testing dataset
    """

    # visualise the train, test and test_label data
    print(train.head(), '\n')
    print(test.head(), '\n')
    print(test_labels.head(), '\n')

    # analyse the train test proportion
    train_size = train.shape[0]
    test_size = test.shape[0]
    sum_sizes = train_size + test_size
    print(f'Train size: {train_size}')
    print(f'Test size: {test_size}')
    print(
        f'Train to Test Ratio: {round(train_size/sum_sizes * 100, 2)} : {round(test_size/sum_sizes * 100, 2)}')
    print()

    # Check if there are any missing values in the datasets
    # TO IMPLEMENT
    # if there are null values:
    #     remove those values
    print(
        f'There are {train.isnull().sum().sum()} missing values in the train dataset')
    print(
        f'There are {test.isnull().sum().sum()} missing values in the test dataset')
    print()

    # add 'non_toxic' tag to any comments that don't contain any toxicity
    print('NOTE: This step may take a while...')
    train['non_toxic'] = train.apply(
        lambda row: 1 if row[2:].sum() == 0 else 0, axis=1)
    print('{} comments do not contain any toxicity'.format(
        train['non_toxic'].sum()))
