import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack


def basic_logistic(train, test, test_labels):
    train_comments = train['comment_text']
    test_comments = test['comment_text']
    all_comments = pd.concat([train_comments, test_comments])

    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 1),
        max_features=10000)
    word_vectorizer.fit(all_comments)
    train_word_features = word_vectorizer.transform(train_comments)
    test_word_features = word_vectorizer.transform(test_comments)

    character_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        stop_words='english',
        ngram_range=(2, 6),
        max_features=50000
    )
    character_vectorizer.fit(all_comments)
    train_char_features = character_vectorizer.transform(train_comments)
    test_char_features = character_vectorizer.transform(test_comments)

    train_features = hstack([train_word_features, train_char_features])
    test_features = hstack([test_word_features, test_char_features])

    scores = []
    sub = pd.DataFrame.from_dict({'id': test['id']})
    for class_name in list(train.columns)[2:]:
        train_target = train[class_name]
        classifier = LogisticRegression(C=0.1, solver='sag')

        # implement gridsearch
        cv_score = np.mean(cross_val_score(
            classifier, train_features, train_target))
        scores.append(cv_score)
        print(f'CV score for class {class_name}: {cv_score}')

        classifier.fit(train_features, train_target)
        sub[class_name] = classifier.predict_proba(test_features)[:, -1]

    print(f'Total CV score: {np.mean(scores)}')
    sub.to_csv('submissions/logistic_sub.csv')
