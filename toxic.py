from environment import environment
from explore import explore
from basic_keras import basic_keras
#from basic_logistic import basic_logistic

if __name__ == '__main__':
    train, test, test_labels = environment()
    #explore(train, test)
    basic_keras(train, test)
    #basic_logistic(train, test, test_labels)
