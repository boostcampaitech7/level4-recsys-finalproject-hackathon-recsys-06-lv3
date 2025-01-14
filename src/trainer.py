import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import RecsysDataset
from src.models.cf_model import CFModel
from src.utils import load_mlflow_tracking_uri


class Trainer:
    def __init__(
        self,
        user_item_matrix,
        num_users,
        num_items,
        embedding_dim=50,
        batch_size=32,
        learning_rate=0.001,
    ):
        self.user_item_matrix = user_item_matrix
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = CFModel(num_users, num_items, embedding_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.tracking_uri = load_mlflow_tracking_uri()
        mlflow.set_tracking_uri(self.tracking_uri)

    def recall_at_top_k(self, predictions, targets, k=10):
        _, top_k_indices = torch.topk(predictions, k)
        hits = (targets.gather(1, top_k_indices) > 0).float().sum()
        return hits / targets.size(0)

    def _run_epoch(self, dataloader, training=True):
        total_loss = 0
        total_recall = 0
        if training:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(training):
            for step, (user, item, rating) in enumerate(dataloader):
                prediction = self.model(user, item)
                loss = self.criterion(prediction, rating)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
                if not training:
                    recall = self.recall_at_top_k(prediction, rating)
                    total_recall += recall.item()
                    mlflow.log_metric("recall_at_top10", recall.item(), step=step)

        avg_loss = total_loss / len(dataloader)
        if not training:
            avg_recall = total_recall / len(dataloader)
            return avg_loss, avg_recall
        return avg_loss

    def train(self, epochs=10):
        dataset = RecsysDataset(self.user_item_matrix)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        mlflow.start_run()
        mlflow.log_param("num_users", self.num_users)
        mlflow.log_param("num_items", self.num_items)
        mlflow.log_param("embedding_dim", self.embedding_dim)
        mlflow.log_param("learning_rate", self.learning_rate)

        for epoch in range(epochs):
            avg_loss = self._run_epoch(dataloader, training=True)
            mlflow.log_metric("mse", avg_loss, step=epoch)
            print(f"Epoch {epoch}, Loss: {avg_loss}")

        torch.save(self.model.state_dict(), "model.pth")
        mlflow.log_artifact("model.pth")
        mlflow.end_run()

    def validate(self):
        dataset = RecsysDataset(self.user_item_matrix)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.load_state_dict(torch.load("model.pth"))
        avg_loss, avg_recall = self._run_epoch(dataloader, training=False)
        print(f"Validation Loss: {avg_loss}, Recall@Top10: {avg_recall}")

    def infer(self, user_id, item_id):
        self.model.load_state_dict(torch.load("model.pth"))
        self.model.eval()
        user = torch.tensor([user_id], dtype=torch.long)
        item = torch.tensor([item_id], dtype=torch.long)
        with torch.no_grad():
            prediction = self.model(user, item)
        return prediction.item()
