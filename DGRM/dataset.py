import torch
from torch.utils.data import Dataset, DataLoader


class MyData(Dataset):
    def __init__(self, nb_user):
        self.idxs = torch.tensor(range(nb_user), dtype=torch.long)

    def __getitem__(self, idx):
        return self.idxs[idx]

    def __len__(self):
        return len(self.idxs)