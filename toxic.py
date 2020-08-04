from environment import environment
from explore import explore

if __name__ == '__main__':
    train, test, test_labels = environment()
    explore(train, test, test_labels)
