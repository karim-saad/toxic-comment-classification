import os
from zipfile import ZipFile


def unzip_files():
    if not os.path.isdir('train.csv'):
        ZipFile('train.csv.zip', 'r').extractall(os.getcwd())

    if not os.path.isdir('test.csv'):
        ZipFile('test.csv.zip', 'r').extractall(os.getcwd())

    if not os.path.isdir('test_labels.csv'):
        ZipFile('test_labels.csv.zip', 'r').extractall(os.getcwd())


unzip_files()
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
