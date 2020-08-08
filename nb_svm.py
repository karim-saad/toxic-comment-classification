import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score
import re
import string


def nb_svm(train, test):
    length = train.comment_text.str.len()
    print(f'Mean length = {length.mean()}')
    print(f'Standard deviation = {length.std()}')
    print(f'Max length = {length.max()}')

    label_cols = list(train.columns)[2:]
    train['none'] = 1 - train[label_cols].max(axis=1)

    train['comment_text'].fillna('unknown', inplace=True)
    test['comment_text'].fillna('unknown', inplace=True)

    vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize, min_df=3,
                          max_df=0.9, strip_accents='unicode', use_idf=1, smooth_idf=1, sublinear_tf=1)
    x = vec.fit_transform(train['comment_text'])
    x_test = vec.transform(test['comment_text'])

    preds = np.zeros((len(test), len(label_cols)))

    for count, class_name in enumerate(label_cols):
        print(f'Fit {class_name}')
        #m, r = get_mdl(train[class_name], x)
        #print(np.mean(cross_val_score(m, x, train[class_name])))
        #preds[:, count] = m.predict_proba(x_test.multiply(r))[:, 1]
        model, log_count_ratio = nblr(x, train[class_name])
        print(np.mean(cross_val_score(model, x, train[class_name])))
        preds[:, count] = model.predict_proba(
            x_test.multiply(log_count_ratio))[:, 1]


def tokenize(s):
    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
    return re_tok.sub(r' \1 ', s).split()


def pr(x, y, y_i, alpha=1):
    ''' Returns probability

    Parameters
    ---------
    x : something something
        Yeet i wanna try this

    y : another random data type
        Hey hey ``sexy stuff``
    '''

    p = x[y == y_i].sum(axis=0) + alpha
    p_norm = (y == y_i).sum() + alpha
    return p/p_norm


def nblr(x, y):
    y = y.values
    log_count_ratio = np.log(pr(x, y, 1) / pr(x, y, 0))
    model = LogisticRegression(C=4, max_iter=500)
    x_nb = x.multiply(log_count_ratio)
    return model.fit(x_nb, y), log_count_ratio
