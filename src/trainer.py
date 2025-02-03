import os
from datetime import datetime

import mlflow
import mlflow.data.dataset_registry
import mlflow.data.dataset_source
import mlflow.data.pandas_dataset
import numpy as np
import torch
from mlflow.data.pandas_dataset import from_pandas
from torch.utils.data import DataLoader

from src.data.dataset import RecsysDataset


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_df,
        val_df,
        num_users,
        num_items,
        config,
    ) -> None:
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_df = train_df
        self.val_df = val_df
        self.num_users = num_users
        self.num_items = num_items
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_name = config["model"]
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
        mlflow.data.pandas_dataset.from_pandas(self.val_df, name="val_df")

    def _save_model(self):
        save_dir = "./save/models"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_filename = f"{self.model_name}-{timestamp}.pth"
        model_path = os.path.join(save_dir, model_filename)
        torch.save(self.model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        return model_path

    def recall_at_k(self, predictions, targets, k=10):
        """Calculate Recall@K metric

        Args:
            predictions (np.array): Predicted scores for each item
            targets (np.array): Ground truth labels
            k (int): Number of items to consider

        Returns:
            float: Recall@K score
        """
        top_k_indices = np.argsort(predictions)[-k:]
        relevant_items = np.sum(np.isin(top_k_indices, targets))
        recall = relevant_items / len(targets)
        return recall

    def ndcg_at_k(self, predictions, targets, k=10):
        """Calculate NDCG@K metric

        Args:
            predictions (np.array): Predicted scores for each item
            targets (np.array): Ground truth labels
            k (int): Number of items to consider

        Returns:
            float: NDCG@K score
        """
        top_k_indices = np.argsort(predictions)[-k:]

        # Calculate DCG
        dcg = 0
        for i, idx in enumerate(reversed(top_k_indices)):
            if idx in targets:
                dcg += 1 / np.log2(i + 2)  # i + 2 because i starts from 0

        # Calculate IDCG
        idcg = 0
        for i in range(min(len(targets), k)):
            idcg += 1 / np.log2(i + 2)

        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        return ndcg

    def _run_epoch(self, dataloader, training=True):
        total_loss = 0
        total_recall = 0
        total_ndcg = 0

        if training:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(training):
            for step, (user, item, rating, label) in enumerate(dataloader):
                user = user.to(self.device)
                item = item.to(self.device)
                rating = rating.to(self.device)
                label = label.to(self.device)
                prediction = self.model(user, item, rating).to(self.device)

                loss = self.criterion(prediction, label)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Calculate metrics
                pred_np = prediction.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                recall = self.recall_at_k(pred_np, label)
                ndcg = self.ndcg_at_k(pred_np, label)

                total_recall += recall
                total_ndcg += ndcg
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_recall = total_recall / len(dataloader)
        avg_ndcg = total_ndcg / len(dataloader)

        return avg_loss, avg_recall, avg_ndcg

    def train(self, epochs=10) -> None:
        train_dataset = RecsysDataset(self.train_df)
        val_dataset = RecsysDataset(self.val_df)

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        for epoch in range(epochs):
            # 학습
            train_loss, train_recall, train_ndcg = self._run_epoch(
                train_dataloader, training=True
            )

            # 검증
            val_loss, val_recall, val_ndcg = self._run_epoch(
                val_dataloader, training=False
            )
            # Log metrics
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_recall_at_10": train_recall,
                    "train_ndcg_at_10": train_ndcg,
                    "val_loss": val_loss,
                    "val_recall_at_10": val_recall,
                    "val_ndcg_at_10": val_ndcg,
                },
                step=epoch,
            )

            print(f"Epoch {epoch}")
            print(
                f"Train - Loss: {train_loss:.4f}, Recall@10: {train_recall:.4f}, NDCG@10: {train_ndcg:.4f}"
            )
            print(
                f"Val - Loss: {val_loss:.4f}, Recall@10: {val_recall:.4f}, NDCG@10: {val_ndcg:.4f}"
            )

        # 모델 저장
        model_path = self._save_model()
        print(f"\nModel saved to: {model_path}")
        mlflow.log_artifact(model_path)

    def validate(self) -> None:
        dataset = RecsysDataset(self.val_df)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        avg_loss, avg_recall, avg_ndcg = self._run_epoch(dataloader, training=False)

        mlflow.log_metrics(
            {
                "val_loss": avg_loss,
                "val_recall_at_10": avg_recall,
                "val_ndcg_at_10": avg_ndcg,
            }
        )

        print(f"Validation Results:")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Recall_at_10: {avg_recall:.4f}")
        print(f"NDCG_at_10: {avg_ndcg:.4f}")

    def infer(self, user_id, item_id, rating) -> float:
        self.model.eval()
        user = torch.tensor([user_id], dtype=torch.long)
        item = torch.tensor([item_id], dtype=torch.long)
        rating = torch.tensor([rating], dtype=torch.float)
        with torch.no_grad():
            prediction = self.model(user, item, rating)
        return prediction.item()
