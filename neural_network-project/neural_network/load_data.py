import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch


# Creates a custom dataset class that reads in a dataframe
class IrisDataset(Dataset):                                 
    def __init__(self, dataframe):
        dataframe = self.label_change(dataframe)                                # encodes categorical words to numbers
        self.features = dataframe.drop(['variety', 'variety_encoded'], axis=1)  # drops label data and saves numerical features
        self.labels = dataframe['variety_encoded']                              # assigns labels to encodings
        self.tensor_transform()                                                 # changes numerical data to tensors

    def __len__(self):
        return len(self.labels)                                                 # returns length of labels
    
    def __getitem__(self, idx):                                                 # defines how data is pulled from dataset 
        features = self.features[idx]                                                  
        labels = self.labels[idx]
        return features, labels
    
    def label_change(self, dataframe):                                          # changes labels to encodings
        label_encoder = LabelEncoder()
        dataframe['variety_encoded'] = label_encoder.fit_transform(dataframe['variety'])
        return dataframe

    def tensor_transform(self):                                                 # changes data to tensors
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