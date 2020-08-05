import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from PIL import Image


def explore(train, test):
    '''Conducts an exploratory data analysis of the provided data

    Parameters:
    - train (pd.DataFrame): training dataset
    - test (pd.DataFrame): testing dataset
    '''

    # visualise the train, test and test_label data
    print(train.head(), '\n')
    print(test.head(), '\n')

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
    print('Applying \'non-toxic\' tag to all clean comments')
    print('NOTE: This step may take a while...')
    train['non_toxic'] = train.apply(
        lambda row: 1 if row[2:].sum() == 0 else 0, axis=1)
    print('{} comments do not contain any toxicity\n'.format(
        train['non_toxic'].sum()))

    # plotting the number of comments per classification
    x = train.iloc[:, 2:].sum()
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x.index, x.values)
    plt.title('Number of Comments per Class')
    plt.xlabel('Classification of Comment', fontsize=12)
    plt.ylabel('Number of Occurences', fontsize=12)
    rects = ax.patches
    labels = x.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2, height +
                5, label, ha='center', va='bottom')
    plt.savefig('appendix/comments_per_class.png')
    plt.close()
    print('Bar graph showing the number of comments per class is now found at appendix/comments_per_class.png')

    # plotting the number of comments with multiple tags
    x = train.iloc[:, 2:].sum(axis=1).value_counts()
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x.index, x.values, color='m')
    plt.title('Number of Comments with Multiple Tags')
    plt.xlabel('Number of Tags')
    plt.ylabel('Number of Occurences')
    rects = ax.patches
    labels = x.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2, height +
                5, label, ha='center', va='bottom')
    plt.savefig('appendix/comments_with_multiple_tags.png')
    plt.close()
    print('Bar graph showing the number of tags per comment is now found at appendix/comments_with_multiple_tags.png')

    # heatmap showing correlation between toxicity tags
    # includes only toxic comments (no clean comments)
    toxic_df = train.iloc[:, 2:-1]
    toxic_corr = toxic_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(toxic_corr, xticklabels=toxic_corr.columns.values,
                yticklabels=toxic_corr.columns.values, annot=True, )
    plt.savefig('appendix/toxic_correlation_heatmap.png')
    plt.close()
    print('Heatmap showing the correlation between toxicity tags is now found at appendix/toxic_correlation_heatmap.png\n')

    # word cloud for different types of comments
    # https://www.datacamp.com/community/tutorials/wordcloud-python

    # toxic comments
    toxic = train[train.toxic == 1]
    text = toxic.comment_text.values
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(background_color='black',
                          max_words=2000, stopwords=stopwords)
    wordcloud.generate(' '.join(text))
    wordcloud.to_file('appendix/toxic_wordcloud.png')
    print('Word cloud of toxic comments is now found at appendix/toxic_wordcloud.png')

    # severe toxic comments
    severe_toxic = train[train.severe_toxic == 1]
    text = severe_toxic.comment_text.values
    wordcloud = WordCloud(background_color='black',
                          max_words=2000, stopwords=stopwords)
    wordcloud.generate(' '.join(text))
    wordcloud.to_file('appendix/severe_toxic_wordcloud.png')
    print('Word cloud of toxic comments is now found at appendix/severe_toxic_wordcloud.png')

    # obscene comments
    obscene = train[train.obscene == 1]
    text = obscene.comment_text.values
    wordcloud = WordCloud(background_color='black',
                          max_words=2000, stopwords=stopwords)
    wordcloud.generate(' '.join(text))
    wordcloud.to_file('appendix/obscene_wordcloud.png')
    print('Word cloud of toxic comments is now found at appendix/obscene_wordcloud.png')

    # threat comments
    threat = train[train.threat == 1]
    text = threat.comment_text.values
    wordcloud = WordCloud(background_color='black',
                          max_words=2000, stopwords=stopwords)
    wordcloud.generate(' '.join(text))
    wordcloud.to_file('appendix/threat_wordcloud.png')
    print('Word cloud of toxic comments is now found at appendix/threat_wordcloud.png')

    # insult comments
    insult = train[train.insult == 1]
    text = insult.comment_text.values
    wordcloud = WordCloud(background_color='black',
                          max_words=2000, stopwords=stopwords)
    wordcloud.generate(' '.join(text))
    wordcloud.to_file('appendix/insult_wordcloud.png')
    print('Word cloud of toxic comments is now found at appendix/insult_wordcloud.png')

    # identity hate comments
    identity_hate = train[train.identity_hate == 1]
    text = identity_hate.comment_text.values
    wordcloud = WordCloud(background_color='black',
                          max_words=2000, stopwords=stopwords)
    wordcloud.generate(' '.join(text))
    wordcloud.to_file('appendix/identity_hate_wordcloud.png')
    print('Word cloud of toxic comments is now found at appendix/identity_hate_wordcloud.png')
