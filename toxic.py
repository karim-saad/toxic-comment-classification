from environment import environment
from explore import explore
from basic_keras import basic_keras
from nb_svm import nb_svm
from keras_batch_higher import keras_batch_higher
from keras_batch_lower import keras_batch_lower
from keras_epochs_higher import keras_epochs_higher
from keras_epochs_lower import keras_epochs_lower
from keras_validation_higher import keras_validation_higher
from keras_validation_lower import keras_validation_lower
from keras_bidirectional import keras_bidirectional

if __name__ == '__main__':
    train, test, test_labels = environment()

    while(option is not 'end'):
        print('\nOPTIONS')
        print('Explore, LSTM, NB-SVM, LSTM_tune')
        option = input('Choose your option: ')

        if (option is 'Explore' or option is 'explore'):
            explore(train, test)
        elif (option == 'lstm' or option is 'LSTM'):
            basic_keras(train, test)
        elif (option is 'nb-svm' or option is 'NB-SVM'):
            nb_svm(train, test)
        elif (option is 'lstm_tune' or option is 'LSTM_tune' or options is 'LSTM_TUNE'):
            keras_batch_higher(train, test)
            keras_batch_lower(train, test)
            keras_epochs_higher(train, test)
            keras_epochs_lower(train, test)
            keras_validation_higher(train, test)
            keras_validation_lower(train, test)
            keras_bidirectional(train, test)
        else:
            print('Invalid option')
