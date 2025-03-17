import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch


class IrisDataset(Dataset):
    def __init__(self, dataframe):
        dataframe = self.label_change(dataframe)
        self.features = dataframe.drop('variety', axis=1)
        self.labels = dataframe['variety']
        self.tensor_transform()

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        labels = self.labels[idx]
        return features, labels
    
    def label_change(self, dataframe):
        dataframe['variety'] = dataframe['variety'].replace('Setosa', 0)
        dataframe['variety'] = dataframe['variety'].replace('Versicolor', 1)
        dataframe['variety'] = dataframe['variety'].replace('Virginica', 2)
        return dataframe

    def tensor_transform(self):
        self.features = torch.FloatTensor(self.features.values)
        self.labels = torch.LongTensor(self.labels.values)

def create_dataloader(dataframe, batch_size):
    dataset = IrisDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def download_data(url):
    my_df = pd.read_csv(url)
    return my_df

def split_data(dataframe):
    train_validate, test = train_test_split(dataframe, test_size=0.2, random_state=41)
    train, validate = train_test_split(train_validate, test_size=0.2, random_state=41)
    return train, validate, test