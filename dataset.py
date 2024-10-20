import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

class CustomDataset(Dataset):  # Renamed to follow CamelCase

    @staticmethod
    def getFiles(path):
        paths = []
        for root, _, files in os.walk(path):
            for file in files:
                paths.append(os.path.join(root, file))
        return paths

    @staticmethod
    def readFile(path):
        with h5py.File(path, "r") as dataFrame:
            inp = np.array(dataFrame['inp'], dtype="float32")  / 16384
            out = np.array(dataFrame['out'], dtype="float32")

        return inp, out

    def __init__(self, path):
        super(CustomDataset, self).__init__()
        self.filePaths = self.getFiles(path)

    def __len__(self):
        return len(self.filePaths)

    def __getitem__(self, index):
        if index < len(self.filePaths):
            path = self.filePaths[index]
            inp, out = self.readFile(path)

            inp = torch.from_numpy(inp).float()  # Use torch.from_numpy for better performance

            out = torch.from_numpy(out).float()

            return {
                'input': inp,
                'output': out
            }