import torch
from torch import Tensor
from torch.utils.data import Dataset


class RecsysDataset(Dataset):
    def __init__(self, dataframe):
        self.users = dataframe["user"].values
        self.items = dataframe["item"].values
        self.ratings = dataframe["rating"].values
        self.labels = dataframe["labels"].values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor]:
        user = self.users[idx]
        item = self.items[idx]
        rating = self.ratings[idx]
        label = self.labels[idx]
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float),
            torch.tensor(label, dtype=torch.float),
        )
