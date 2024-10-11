import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


class DeepFakeDataset(Dataset):
    def __init__(self, data_dir, metadata_file, transform=None):
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.metadata.iloc[idx, 0])
        image = Image.open(img_name)
        v = self.metadata.iloc[idx, 1]
        label = torch.tensor(float(1)) if v == 'fake' else torch.tensor(float(0))

        if self.transform:
            image = self.transform(image)
        else: 
            image = np.array(image)
        return image, label

class DeepFakeDatasetTest(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = os.listdir(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.data[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)
        else: 
            image = np.array(image)
        return image, self.data[idx].replace('_0.png', '.mp4')
