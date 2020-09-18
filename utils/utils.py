from pathlib import Path
import csv
from torch.utils.data.sampler import Sampler
import torch

def get_files(dir_path, ext):
    return [str(path.name) for path in Path(dir_path).rglob('*.{}'.format(ext))]


def get_paths(dir_path, ext):
    return [str(path) for path in Path(dir_path).rglob('*.{}'.format(ext))]


def parse_split_csv(csv_path):
    rows = []
    with open(csv_path, 'rb') as file:
        spamreader = csv.reader(file, delimiter=', ', quotechar='|')
        for row in spamreader:
            rows.append(row)
    return rows

def csv_reader(fname):
    with open(fname, 'r') as f:
        out = list(csv.reader(f))
    return out

class ExpandedRandomSampler(Sampler):
    """Iterate multiple times over the same dataset instead of once.
    Args:
        length (int): initial length of the dataset to sample from
        multiplier (float): desired multiplier for the length of the dataset
    """

    def __init__(self, length, multiplier):
        self.length = length
        self.indices = [i for i in range(length)]
        self.total = round(self.length * multiplier)

    def __iter__(self):
        return (self.indices[i % self.length] for i in torch.randperm(self.total))

    def __len__(self):
        return self.total