import pandas as pd
import os
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GlobalMaxPool1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers


def keras_batch_higher(train, test):
    '''Implements an LSTM model with a higher batch size. May attempt to use a pretrained model.

    ``batch_size=64``

    Parameters
    ----------
    train: pd.DataFrame containing train data

    test: pd.DataFrame containing test data
    '''

    print('\nLSTM Model with a higher batch size')

    saved = input('Would you like to use a pretrained model? ')
    print('NOTE: If a model does not exist, a new model will be trained.')

    classes = list(train.columns)[2:]
    y = train[classes].values

    # tokenise all the comment text to individual words
    max_features = 2000
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train['comment_text']))
    tokenized_train = tokenizer.texts_to_sequences(train['comment_text'])
    tokenized_test = tokenizer.texts_to_sequences(test['comment_text'])

    # concatenate / pad all comments to a uniform length
    max_length = 200
    x_train = pad_sequences(tokenized_train, maxlen=max_length)
    x_test = pad_sequences(tokenized_test, maxlen=max_length)

    if ((saved == 'yes' or saved == 'Yes' or saved == 'y' or saved == 'YES') and os.path.isdir('models/batch_higher_model')):
        # attempt running a pretrained model
        model = load_model('models/batch_higher_model')
    else:
        # train a new model (implementation expanded upon in report)
        input_layer = Input(shape=(max_length, ))
        x = Embedding(max_features, 128)(input_layer)
        x = LSTM(60, return_sequences=True, name='lstm_layer')(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(0.1)(x)
        x = Dense(50, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(6, activation='sigmoid')(x)

        model = Model(inputs=input_layer, outputs=x)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics='accuracy')
        model.fit(x_train, y, batch_size=64, epochs=2, validation_split=0.1)
        model.save('models/batch_higher_model')

    # return accuracy and loss parameters
    loss, accuracy = model.evaluate(
        x_train, y, batch_size=64, verbose=1)
    print(f'Loss is {loss}')
    print(f'Accuracy is {accuracy}')

    # predict classifications of test data, and save to a new submissions file
    y_test = model.predict(x_test, batch_size=1024, verbose=1)
    sub = pd.DataFrame.from_dict({'id': test['id']})
    for count, class_name in enumerate(classes):
        sub[class_name] = y_test[:, count]
    sub.to_csv('submissions/keras_submission_batch_higher.csv')
