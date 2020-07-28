import os
from zipfile import ZipFile


def unzip_files():
    if not os.path.isdir('train'):
        new_dir = os.path.join(os.getcwd(), 'train')
        os.mkdir(new_dir)
        ZipFile('train.csv.zip', 'r').extractall(new_dir)

    if not os.path.isdir('test'):
        new_dir = os.path.join(os.getcwd(), 'test')
        os.mkdir(new_dir)
        ZipFile('test.csv.zip', 'r').extractall(new_dir)

    if not os.path.isdir('test_labels'):
        new_dir = os.path.join(os.getcwd(), 'test_labels')
        os.mkdir(new_dir)
        ZipFile('test_labels.csv.zip', 'r').extractall(new_dir)


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
