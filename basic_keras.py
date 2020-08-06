#from keras.preprocessing.text import Tokenizer
from keras_preprocessing.text import Tokenizer


def basic_keras(train, test, test_labels):
    max_features = 2000
    tokenizer = Tokenizer(num_words=max_features)
