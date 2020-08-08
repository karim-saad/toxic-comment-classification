from environment import environment
from explore import explore
from basic_keras import basic_keras
from nb_svm import nb_svm
from keras_batch_higher import keras_batch_higher
from keras_batch_lower import keras_batch_lower
from keras_epochs_higher import keras_epochs_higher
from keras_epochs_lower import keras_epochs_lower

if __name__ == '__main__':
    train, test, test_labels = environment()

    while(1):
        print('\nOPTIONS')
        print('Explore, LSTM, NB-SVM')
        option = input('Choose your option: ')

        if (option == 'Explore' or option == 'explore'):
            explore(train, test)
        elif (option == 'lstm' or option == 'LSTM'):
            basic_keras(train, test)
        elif (option == 'nb-svm' or option == 'NB-SVM'):
            nb_svm(train, test)
        elif (option == 'lstm_tune'):
            keras_batch_higher(train, test)
            keras_batch_lower(train, test)
            keras_epochs_higher(train, test)
            keras_epochs_lower(train, test)
        elif (option == 'end'):
            break
        else:
            print('Invalid option')
