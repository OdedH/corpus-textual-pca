import torch
import pandas as pd
from typing import List, Tuple
from sklearn.datasets import fetch_20newsgroups
import os


class moviesDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None, genres: List[str] = None):
        self.data = pd.read_csv(csv_file, delimiter=" ::: ", header=None, index_col=0, engine='python')
        self.data.columns = ['Title', 'Genre', 'Synopsis']
        if genres:
            self.data = self.data[self.data['Genre'].isin(genres)]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        if self.transform:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def get_synopsis_from_sample(sample):
        return sample['Synopsis']


class customized_20newsgroups(torch.utils.data.Dataset):
    def __init__(self, categories: List[str]=None, transform=None):
        self.data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
        self.transform = transform

    def __len__(self):
        return len(self.data.data)

    def __getitem__(self, idx):
        sample = self.data.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class emailsDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, categories: List[str] = None, transform=None):
        self.transform = transform
        folders = os.listdir(path)
        data = []
        for folder in folders:
            files = os.listdir(os.path.join(path, folder))
            for file in files:
                try:
                    with open(os.path.join(path, folder, file)) as f:
                        contents = " ".join(f.readlines())
                        data.append([file.split(".")[0], folder, contents])
                        f.close()
                except Exception as e:
                    pass

        self.data = pd.DataFrame(data, columns=['ID', 'Category', 'Content'])

        if categories:
            self.data = self.data[self.data['Category'].isin(categories)]

    def __getitem__(self, item):
        sample = self.data.iloc[item, :]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_content_from_sample(sample):
        return sample['Content']


