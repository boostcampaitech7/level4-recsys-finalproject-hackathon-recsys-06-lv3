from typing import Any
import mlflow
import torch
from torch.utils.data import DataLoader

from src.data.dataset import RecsysDataset


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_df,
        test_df,
        num_users,
        num_items,
        config,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_df = train_df
        self.test_df = test_df
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = config["embedding_dim"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        # MLFlow Tracking URI 로드
        self.tracking_uri = config["mlflow"]["tracking_uri"]
        mlflow.set_tracking_uri(self.tracking_uri)

    def recall_at_top_k(self, predictions, targets, k=10):
        _, top_k_indices = torch.topk(predictions, k)
        hits = (targets.gather(1, top_k_indices) > 0).float().sum()
        return hits / targets.size(0)

    def _run_epoch(
        self, dataloader, training=True
    ) -> tuple[Any | float, Any | float] | Any | float:
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

    def train(self, epochs=10) -> None:
        dataset = RecsysDataset(self.train_df)
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

    def validate(self) -> None:
        dataset = RecsysDataset(self.test_df)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        avg_loss, avg_recall = self._run_epoch(dataloader, training=False)
        mlflow.log_metric("val_loss", avg_loss)
        mlflow.log_metric("recall_at_top10", avg_recall)
        print(f"Validation Loss: {avg_loss}, Recall@Top10: {avg_recall}")

    def infer(self, user_id, item_id) -> float:
        self.model.eval()
        user = torch.tensor([user_id], dtype=torch.long)
        item = torch.tensor([item_id], dtype=torch.long)
        with torch.no_grad():
            prediction = self.model(user, item)
        return prediction.item()
