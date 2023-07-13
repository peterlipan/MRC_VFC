import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(ISICDataset, self).__init__()
        file = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.images = file['image'].values
        self.labels = np.argmax(file.iloc[:, 1:].values.astype(int), 1)
        self.transform = transform
        self.n_class = len(np.unique(self.labels))
        self.class_names = file.columns[1:]

        print('Total # images:{}, labels:{}, number of classes'.format(len(self.images),len(self.labels), self.n_class))

    def __getitem__(self, index):
        try:
            image_name = os.path.join(self.root_dir, self.images[index]+'.jpg')
            image = Image.open(image_name).convert('RGB')
        # the file extension of ISIC Archive dataset is JPG
        except FileNotFoundError:
            image_name = os.path.join(self.root_dir, self.images[index]+'.JPG')
            image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.images)