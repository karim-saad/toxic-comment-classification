import pandas as pd
import os
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GlobalMaxPool1D, Bidirectional
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers


def keras_bidirectional(train, test):
    print('\nBidirectional LSTM Model')

    saved = input('Would you like to use a pretrained model? ')
    print('NOTE: If a model does not exist, a new model will be trained.')

    classes = list(train.columns)[2:]
    y = train[classes].values

    max_features = 2000
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train['comment_text']))
    tokenized_train = tokenizer.texts_to_sequences(train['comment_text'])
    tokenized_test = tokenizer.texts_to_sequences(test['comment_text'])

    max_length = 200
    x_train = pad_sequences(tokenized_train, maxlen=max_length)
    x_test = pad_sequences(tokenized_test, maxlen=max_length)

    if ((saved == 'yes' or saved == 'Yes' or saved == 'y' or saved == 'YES') and os.path.isdir('models/bidirectional_model')):
        model = load_model('models/bidirectional_model')
    else:
        input_layer = Input(shape=(max_length, ))
        x = Embedding(max_features, embedding_size)(input_layer)
        x = Bidirectional(
            LSTM(60, return_sequences=True, name='lstm_layer'))(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(0.1)(x)
        x = Dense(50, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(6, activation='sigmoid')(x)

        model = Model(inputs=input_layer, outputs=x)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics='accuracy')
        model.fit(x_train, y, batch_size=32, epochs=2, validation_split=0.1)
        model.save('models/bidirectional_model')

    loss, accuracy = model.evaluate(
        x_train, y, batch_size=32, verbose=1)
    print(f'Loss is {loss}')
    print(f'Accuracy is {accuracy}')

    y_test = model.predict(x_test, batch_size=1024, verbose=1)
    sub = pd.DataFrame.from_dict({'id': test['id']})
    for count, class_name in enumerate(classes):
        sub[class_name] = y_test[:, count]
    sub.to_csv('submissions/keras_submission_bidirectional.csv')
