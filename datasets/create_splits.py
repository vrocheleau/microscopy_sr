from pathlib import Path
from sklearn.model_selection import train_test_split
import csv
import os
from utils import *
import getpass

def write_splits_csv(name, files):
    with open(name, 'w') as out:
        filewriter = csv.writer(out, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for file in files:
            filewriter.writerow([file])

if __name__ == "__main__":

    hold_size = 0.30
    val_size = 0.5

    scale_factors = [4, 8, 16]

    username = getpass.getuser()
    print(username)

    ext = 'czi'
    if username == "AM90950":
        data_dir = '/home/ens/{}/datasets/microscopy_luke'.format(username)
    else:
        data_dir = '/home/{}/datasets/microscopy_luke'.format(username)

    paths = get_paths(data_dir, ext)

    train_files, hold_files = train_test_split(paths, test_size=hold_size)
    test_files, val_files = train_test_split(hold_files, test_size=val_size)

    save_dir = './splits/czi'
    base = save_dir + '/{}'
    os.makedirs(save_dir, exist_ok=True)

    write_splits_csv(base.format('train.csv'), train_files)
    write_splits_csv(base.format('test.csv'), test_files)
    write_splits_csv(base.format('val.csv'), val_files)

    print("Splits created in {}".format(save_dir))