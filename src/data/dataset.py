import torch
from torch import Tensor
from torch.utils.data import Dataset


class RecsysDataset(Dataset):
    def __init__(self, dataframe):
        self.users = dataframe["user_id"].values
        self.items = dataframe["item_id"].values
        self.ratings = dataframe["rating"].values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor]:
        user = self.users[idx]
        item = self.items[idx]
        rating = self.ratings[idx]
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float),
        )
