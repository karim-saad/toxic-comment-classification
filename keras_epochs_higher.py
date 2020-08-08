import pandas as pd
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


def keras_epochs_higher(train, test):
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
    input_layer = Input(shape=(max_length, ))

    embedding_size = 128
    x = Embedding(max_features, embedding_size)(input_layer)
    x = LSTM(60, return_sequences=True, name='lstm_layer')(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics='accuracy')

    batch_size = 32
    epochs = 4
    model.fit(x_train, y, batch_size=batch_size,
              epochs=epochs, validation_split=0.1)

    loss, accuracy = model.evaluate(
        x_train, y, batch_size=batch_size, verbose=1)
    model.save('models/epochs_higher_model')
    print(f'Loss is {loss}')
    print(f'Accuracy is {accuracy}')

    y_test = model.predict(x_test, batch_size=1024, verbose=1)
    sub = pd.DataFrame.from_dict({'id': test['id']})
    for count, class_name in enumerate(classes):
        sub[class_name] = y_test[:, count]
    sub.to_csv('submissions/keras_submission_epochs_higher.csv')
