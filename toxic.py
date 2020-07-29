from environment import environment
from explore import explore

if __name__ == '__main__':
    train, test, test_labels = environment()
    print('''
     _________________
    < My name is Mimi! >
     -----------------
            \   ^__^
             \  (oo)\_______
                (__)\       )\/\\
                    ||----w |
                    ||     ||
    ''')
    explore(train, test, test_labels)
