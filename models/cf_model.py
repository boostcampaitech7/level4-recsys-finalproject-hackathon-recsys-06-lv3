import torch
import torch.nn as nn


class CFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(CFModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        return (user_emb * item_emb).sum(1)
