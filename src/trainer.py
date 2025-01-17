from datetime import datetime
import mlflow.data.dataset_registry
import mlflow.data.dataset_source
from mlflow.data.pandas_dataset import from_pandas
import mlflow
import mlflow.data.pandas_dataset
import numpy as np
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
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        # MLFlow Tracking URI 로드
        self.tracking_uri = config["mlflow"]["tracking_uri"]
        self._mlflow_init(config)

    def _mlflow_init(self, config):
        mlflow.set_tracking_uri(self.tracking_uri)
        run_name = f"{config['model']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        mlflow.start_run(run_name=run_name, log_system_metrics=True)
        mlflow.log_param("num_users", self.num_users)
        mlflow.log_param("num_items", self.num_items)
        mlflow.log_params(config)
        mlflow.data.pandas_dataset.from_pandas(self.train_df, name="train_df")
        mlflow.data.pandas_dataset.from_pandas(self.test_df, name="test_df")

    def recall_at_top_k(self, predictions, targets, k=10):
        top_k_indices = np.argsort(predictions)[-k:]
        relevant_items = np.sum(np.isin(top_k_indices, targets))
        recall = relevant_items / len(targets)
        return recall

    def _run_epoch(self, dataloader, training=True):
        total_loss = 0
        total_recall = 0
        if training:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(training):
            for step, (user, item, rating) in enumerate(dataloader):
                prediction: torch.Tensor = self.model(user, item, rating)
                loss = self.criterion(prediction, rating)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                recall = self.recall_at_top_k(prediction.detach().numpy(), rating)
                total_recall += recall.item()
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        avg_recall = total_recall / len(dataloader)
        return avg_loss, avg_recall

    def train(self, epochs=10) -> None:
        dataset = RecsysDataset(self.train_df)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            avg_loss = self._run_epoch(dataloader, training=True)
            avg_loss, avg_recall = self._run_epoch(dataloader, training=False)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("recall_at_top10", avg_recall, step=epoch)
            print(f"Epoch {epoch}, Loss: {avg_loss}, Recall@Top10: {avg_recall}")

        torch.save(self.model.state_dict(), "model.pth")
        mlflow.log_artifact("model.pth")

    def validate(self) -> None:
        dataset = RecsysDataset(self.test_df)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        avg_loss, avg_recall = self._run_epoch(dataloader, training=False)
        mlflow.log_metric("val_loss", avg_loss)
        mlflow.log_metric("valid_recall_at_top10", avg_recall)
        print(f"Validation Loss: {avg_loss}, Recall@Top10: {avg_recall}")

    def infer(self, user_id, item_id, rating) -> float:
        self.model.eval()
        user = torch.tensor([user_id], dtype=torch.long)
        item = torch.tensor([item_id], dtype=torch.long)
        rating = torch.tensor([rating], dtype=torch.float)
        with torch.no_grad():
            prediction = self.model(user, item, rating)
        return prediction.item()
