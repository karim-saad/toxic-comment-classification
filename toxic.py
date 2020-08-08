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

    option = 'mimi'
    while(option is not 'end'):
        print('\nOPTIONS')
        print('Explore, LSTM, NB-SVM, LSTM_tune')
        option = input('Choose your option: ')

        if (option == 'Explore' or option == 'explore'):
            explore(train, test)
        elif (option == 'lstm' or option == 'LSTM'):
            basic_keras(train, test)
        elif (option == 'nb-svm' or option == 'NB-SVM'):
            nb_svm(train, test)
        elif (option == 'lstm_tune' or option == 'LSTM_tune' or option == 'LSTM_TUNE' or option == 'lstm-tune'):
            keras_batch_lower(train, test)
            keras_epochs_higher(train, test)
            keras_epochs_lower(train, test)
            keras_validation_higher(train, test)
            keras_validation_lower(train, test)
            keras_bidirectional(train, test)
            keras_batch_higher(train, test)
        else:
            print('Invalid option')
