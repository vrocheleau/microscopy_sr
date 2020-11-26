from pathlib import Path
from sklearn.model_selection import train_test_split
import csv
import os
from utils import *
import getpass
from sklearn.model_selection import KFold
import numpy as np

def write_splits_csv(name, files):
    with open(name, 'w') as out:
        filewriter = csv.writer(out, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for file in files:
            filewriter.writerow([file])

if __name__ == "__main__":

    n_splits = 3

    test_size = 0.25

    scale_factors = [4, 8]

    username = getpass.getuser()
    print(username)

    ext = 'czi'
    if username == "AM90950":
        data_dir = '/home/ens/{}/datasets/microscopy_luke'.format(username)
    else:
        data_dir = '/home/{}/datasets/microscopy_luke'.format(username)

    paths = np.asarray(get_paths(data_dir, ext))

    kf = KFold(n_splits=n_splits)
    folds_files, test_files = train_test_split(paths, test_size=test_size)

    save_dir = './splits/'
    os.makedirs(save_dir, exist_ok=True)

    for i, (train_idx, val_idx) in enumerate(kf.split(folds_files)):
        fold_dir = os.path.join(save_dir, 'fold_{}'.format(i))
        os.makedirs(fold_dir, exist_ok=True)
        fold_dir += '/{}'

        train_files = folds_files[train_idx]
        val_files = folds_files[val_idx]

        write_splits_csv(fold_dir.format('train.csv'), train_files)
        write_splits_csv(fold_dir.format('test.csv'), test_files)
        write_splits_csv(fold_dir.format('val.csv'), val_files)

    # Simple holdout
    # train_files, hold_files = train_test_split(paths, test_size=hold_size)
    # test_files, val_files = train_test_split(hold_files, test_size=val_size)
    #
    # save_dir = './splits/czi'
    # base = save_dir + '/{}'
    # os.makedirs(save_dir, exist_ok=True)
    #
    # write_splits_csv(base.format('train.csv'), train_files)
    # write_splits_csv(base.format('test.csv'), test_files)
    # write_splits_csv(base.format('val.csv'), val_files)

    print("Splits created")
