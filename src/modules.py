"""
Pytorch Lightning Modules.
"""

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F


class SeqRecBase(pl.LightningModule):
    def __init__(
        self, model, lr=1e-3, padding_idx=0, predict_top_k=10, filter_seen=True
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.padding_idx = padding_idx
        self.predict_top_k = predict_top_k
        self.filter_seen = filter_seen
        self.validation_step_outputs = {"ndcg": [], "hit_rate": [], "mrr": []}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch, batch_idx):
        preds, scores = self.make_prediction(batch)

        scores = scores.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        user_ids = batch["user_id"].detach().cpu().numpy()

        return {"preds": preds, "scores": scores, "user_ids": user_ids}

    def validation_step(self, batch, batch_idx):
        preds, scores = self.make_prediction(batch)
        metrics = self.compute_val_metrics(batch["target"], preds)

        self.log("val_ndcg", metrics["ndcg"], prog_bar=True)
        self.log("val_hit_rate", metrics["hit_rate"], prog_bar=True)
        self.log("val_mrr", metrics["mrr"], prog_bar=True)

        self.validation_step_outputs["ndcg"].append(metrics["ndcg"])
        self.validation_step_outputs["hit_rate"].append(metrics["hit_rate"])
        self.validation_step_outputs["mrr"].append(metrics["mrr"])

    def make_prediction(self, batch):
        outputs = self.prediction_output(batch)

        input_ids = batch["input_ids"]
        rows_ids = torch.arange(
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )
        last_item_idx = (input_ids != self.padding_idx).sum(axis=1) - 1

        preds = outputs[rows_ids, last_item_idx, :]

        scores, preds = torch.sort(preds, descending=True)

        if self.filter_seen:
            seen_items = batch["full_history"]
            preds, scores = self.filter_seen_items(preds, scores, seen_items)
        else:
            scores = scores[:, : self.predict_top_k]
            preds = preds[:, : self.predict_top_k]

        return preds, scores

    def filter_seen_items(self, preds, scores, seen_items):
        max_len = seen_items.size(1)
        scores = scores[:, : self.predict_top_k + max_len]
        preds = preds[:, : self.predict_top_k + max_len]

        final_preds, final_scores = [], []
        for i in range(preds.size(0)):
            not_seen_indexes = torch.isin(preds[i], seen_items[i], invert=True)
            pred = preds[i, not_seen_indexes][: self.predict_top_k]
            score = scores[i, not_seen_indexes][: self.predict_top_k]
            final_preds.append(pred)
            final_scores.append(score)

        final_preds = torch.vstack(final_preds)
        final_scores = torch.vstack(final_scores)

        return final_preds, final_scores

    def compute_val_metrics(self, targets, preds):
        ndcg, hit_rate, mrr = 0, 0, 0

        for i, pred in enumerate(preds):
            if torch.isin(targets[i], pred).item():
                hit_rate += 1
                rank = torch.where(pred == targets[i])[0].item() + 1
                ndcg += 1 / np.log2(rank + 1)
                mrr += 1 / rank

        hit_rate = hit_rate / len(targets)
        ndcg = ndcg / len(targets)
        mrr = mrr / len(targets)

        return {"ndcg": ndcg, "hit_rate": hit_rate, "mrr": mrr}

    def on_validation_epoch_end(self):
        avg_metrics = {k: np.mean(v) for k, v in self.validation_step_outputs.items()}
        mlflow.log_metrics(avg_metrics, step=self.current_epoch)
        self.validation_step_outputs = {"ndcg": [], "hit_rate": [], "mrr": []}


class SeqRec(SeqRecBase):
    def __init__(
        self,
        model,
        lr=1e-3,
        padding_idx=0,
        predict_top_k=10,
        filter_seen=True,
        loss="cross_entropy",
        lambda_value=0.5,
        temperature=1,
        similarity_threshold=0.9,
        similarity_indicies=None,
        similarity_value=None,
    ):
        super().__init__(model, lr, padding_idx, predict_top_k, filter_seen)
        self.loss = loss
        if self.loss == "sim_rec":
            self.lambda_value = lambda_value
            self.temperature = temperature
            self.similarity_threshold = similarity_threshold
            self.sim_matrix = torch.load(
                similarity_indicies, map_location="cuda"
            )  # Indicies
            self.sim_score = torch.load(similarity_value, map_location="cuda")  # Values
            self._init_sim_rec()
        self.training_step_outputs = []

    def _init_sim_rec(self):
        if self.similarity_threshold < 1:
            self.sim_score[self.sim_score <= self.similarity_threshold] = -float("inf")
        else:
            # make the self similarity maximal
            self.sim_matrix = torch.arange(
                self.sim_matrix.shape[0], device="cuda"
            ).reshape(-1, 1)
            self.sim_score = torch.ones_like(self.sim_matrix)
        self.sim_matrix += 1
        self.sim_matrix = torch.concat(
            [
                torch.arange((self.sim_matrix.shape[1]), device="cuda").unsqueeze(
                    dim=0
                ),
                self.sim_matrix,
            ],
            dim=0,
        )
        self.sim_score = torch.concat(
            [
                torch.full(
                    (1, self.sim_score.shape[1]),
                    fill_value=-float("inf"),
                    device="cuda",
                ),
                self.sim_score,
            ],
            dim=0,
        )

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch["input_ids"])
        loss = self.compute_loss(outputs, batch)

        self.training_step_outputs.append(loss.item())

        return loss

    def compute_loss(self, outputs, batch):
        if self.loss == "sim_rec":
            target = batch["labels"].clone()
            target[target == -100] = 0
            target = F.one_hot(target.view(-1), num_classes=outputs.shape[2]).float()
            bce_fct = nn.BCEWithLogitsLoss(reduction="none")
            bce_loss = bce_fct(outputs.view(-1, outputs.size(-1)), target)
            bce_loss = bce_loss.mean()
            sim_loss = -torch.sum(
                self.sim_matrix * torch.log(self.sim_score + 1e-10)
            )  # Prevent log(0)
            loss = (1 - self.lambda_value) * bce_loss + self.lambda_value * sim_loss
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                outputs.view(-1, outputs.size(-1)), batch["labels"].view(-1)
            )

        return loss

    def prediction_output(self, batch):
        return self.model(batch["input_ids"])

    def on_train_epoch_end(self):
        avg_loss = np.mean(self.training_step_outputs)
        mlflow.log_metric("train_loss", avg_loss, step=self.current_epoch)
        self.training_step_outputs.clear()


class SeqRecWithSampling(SeqRec):
    def __init__(
        self,
        model,
        lr=1e-3,
        loss="cross_entropy",
        padding_idx=0,
        predict_top_k=10,
        filter_seen=True,
        lambda_value=0.5,
        temperature=1,
        similarity_threshold=0.9,
        similarity_indicies=None,
        similarity_value=None,
    ):
        super().__init__(model, lr, padding_idx, predict_top_k, filter_seen)

        self.loss = loss
        if self.loss == "sim_rec":
            self.lambda_value = lambda_value
            self.temperature = temperature
            self.similarity_threshold = similarity_threshold
            self.sim_matrix = torch.load(
                similarity_indicies, map_location="cuda"
            )  # Indicies
            self.sim_score = torch.load(similarity_value, map_location="cuda")  # Values
            self._init_sim_rec()
        if hasattr(self.model, "item_emb"):  # for SASRec
            self.embed_layer = self.model.item_emb
        elif hasattr(self.model, "embed_layer"):  # for other models
            self.embed_layer = self.model.embed_layer

    def _init_sim_rec(self):
        if self.similarity_threshold < 1:
            self.sim_score[self.sim_score <= self.similarity_threshold] = -float("inf")
        else:
            # make the self similarity maximal
            self.sim_matrix = torch.arange(
                self.sim_matrix.shape[0], device="cuda"
            ).reshape(-1, 1)
            self.sim_score = torch.ones_like(self.sim_matrix)
        self.sim_matrix += 1
        self.sim_matrix = torch.concat(
            [
                torch.arange((self.sim_matrix.shape[1]), device="cuda").unsqueeze(
                    dim=0
                ),
                self.sim_matrix,
            ],
            dim=0,
        )
        self.sim_score = torch.concat(
            [
                torch.full(
                    (1, self.sim_score.shape[1]),
                    fill_value=-float("inf"),
                    device="cuda",
                ),
                self.sim_score,
            ],
            dim=0,
        )

    def compute_loss(self, outputs, batch):
        # embed  and compute logits for negatives
        if batch["negatives"].ndim == 2:  # for full_negative_sampling=False
            # [N, M, D]
            embeds_negatives = self.embed_layer(batch["negatives"].to(torch.int32))
            # [N, T, D] * [N, D, M] -> [N, T, M]
            logits_negatives = torch.matmul(outputs, embeds_negatives.transpose(1, 2))
        elif batch["negatives"].ndim == 3:  # for full_negative_sampling=True
            # [N, T, M, D]
            embeds_negatives = self.embed_layer(batch["negatives"].to(torch.int32))
            # [N, T, 1, D] * [N, T, D, M] -> [N, T, 1, M] -> -> [N, T, M]
            logits_negatives = torch.matmul(
                outputs.unsqueeze(2), embeds_negatives.transpose(2, 3)
            ).squeeze()
            if logits_negatives.ndim == 2:
                logits_negatives = logits_negatives.unsqueeze(2)

        # embed  and compute logits for positives
        # [N, T]
        labels = batch["labels"].clone()
        labels[labels == -100] = self.padding_idx
        # [N, T, D]
        embeds_labels = self.embed_layer(labels)
        # [N, T, 1, D] * [N, T, D, 1] -> [N, T, 1, 1] -> [N, T]
        logits_labels = torch.matmul(
            outputs.unsqueeze(2), embeds_labels.unsqueeze(3)
        ).squeeze()

        # concat positives and negatives
        # [N, T, M + 1]
        logits = torch.cat([logits_labels.unsqueeze(2), logits_negatives], dim=-1)

        if self.loss == "cross_entropy":
            # [N, T]
            targets = batch["labels"].clone()
            targets[targets != -100] = 0
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
        elif self.loss == "bce":
            # [N, T, M + 1]
            targets = torch.zeros_like(logits)
            targets[:, :, 0] = 1
            loss_fct = nn.BCEWithLogitsLoss(reduction="none")
            loss = loss_fct(logits, targets)
            loss = loss[batch["labels"] != -100]
            loss = loss.mean()

        if self.loss == "sim_rec":
            targets = torch.zeros_like(logits)
            targets[:, :, 0] = 1
            bce_fct = nn.BCEWithLogitsLoss(reduction="none")
            bce_loss = bce_fct(logits, targets)
            bce_loss = bce_loss[batch["labels"] != -100]
            # bce_loss = bce_loss.mean()

            sim_loss = -torch.sum(
                self.sim_matrix * torch.log(self.sim_score + 1e-10)
            )  # Prevent log(0)
            loss = (1 - self.lambda_value) * bce_loss + self.lambda_value * sim_loss
        return loss

    def prediction_output(self, batch):
        outputs = self.model(batch["input_ids"])
        outputs = torch.matmul(outputs, self.embed_layer.weight.T)

        return outputs

    def _create_similarity_distribution(self, positive_indices):

        num_items = self.sim_matrix.shape[0]
        num_positives = positive_indices.shape[0]
        # (num_positives, top_k_similar)
        pos_similarity_indices = torch.index_select(
            self.sim_matrix, index=positive_indices, dim=0
        )
        pos_similarity_values = torch.index_select(
            self.sim_score, index=positive_indices, dim=0
        )

        # (num_positives, num_items)
        similarities = torch.full(
            (num_positives, num_items),
            fill_value=-float("inf"),
            device=self.sim_matrix.device,
        )
        similarities.scatter_(
            dim=1, index=pos_similarity_indices, src=pos_similarity_values
        )

        similarities /= self.temperature

        distribution = torch.nn.functional.softmax(similarities, dim=-1)
        return distribution
