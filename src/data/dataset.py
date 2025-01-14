import torch
from torch.utils.data import Dataset


class RecsysDataset(Dataset):
    def __init__(self, user_item_matrix) -> None:
        self.user_item_matrix = user_item_matrix

    def __len__(self) -> int:
        return len(self.user_item_matrix)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user, item, rating = self.user_item_matrix[idx]
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float),
        )
