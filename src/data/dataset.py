"""
Torch datasets and collate function.
"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pandas as pd


class LMDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        max_length=128,
        num_negatives=None,
        negative_sample={"type": "full"},
        user_col="user_id",
        item_col="item_id",
        time_col="time_idx",
    ):
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.negative_sample = negative_sample["type"]
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.data = (
            df.sort_values(time_col).groupby(user_col)[item_col].agg(list).to_dict()
        )
        self.user_ids = list(self.data.keys())

        if num_negatives:
            self.all_items = df[item_col].unique()
        if num_negatives and self.negative_sample == "popularity":
            self.pop_type = negative_sample["pop_type"]
            self.weight_type = negative_sample["weight_type"]
            self._calculate_popularity(df)

    def _calculate_popularity(self, df):
        popularity = df["item_id"].value_counts().reset_index()["item_id"]
        if self.weight_type == "bot":
            popularity /= popularity + 1
        if self.weight_type == "mid":
            popularity = popularity.mean() / (popularity + 1)
        if self.pop_type == "rank" or self.pop_type == "rank-prob":
            popularity = popularity.rank(method="max", ascending=True)
        self.popularity = popularity.to_numpy()

        self.num_items = len(self.popularity)
        self.prob_distribution = self.popularity / np.sum(self.popularity)

    def __len__(self):
        return len(self.data)

    def sample_negatives(self, item_sequence):
        negatives = self.all_items[~np.isin(self.all_items, item_sequence)]
        # pos_item = self.all_items[item_sequence]
        if self.negative_sample == "full":
            negatives = np.random.choice(
                negatives,
                size=self.num_negatives * (len(item_sequence) - 1),
                replace=True,
            )
            negatives = negatives.reshape(len(item_sequence) - 1, self.num_negatives)
        elif self.negative_sample == "popularity":
            assert (
                self.num_negatives != None
            ), "config.yaml에 num_negatives를 입력해주세요."
            negatives = self._popularity_negatives(item_sequence)
        else:
            # Default Random Choice
            negatives = np.random.choice(
                negatives, size=self.num_negatives, replace=False
            )

        return negatives

    def _popularity_negatives(self, item_sequence):
        neg_tf_list = ~np.isin(self.all_items, item_sequence)
        negatives = self.all_items[neg_tf_list]
        if self.pop_type == "rank":
            # negatives = self.popularity[neg_tf_list]
            negatives = negatives[np.argsort(a=self.popularity[neg_tf_list])]
            return negatives[: self.num_negatives]
        # Negative Sampling 수행 (인기도 기반 확률 분포를 사용)
        prob_dist = self.prob_distribution[~np.isin(self.all_items, item_sequence)]
        # item sequence 제외 probability 재 계산
        prob_dist = prob_dist / np.sum(prob_dist)
        # item sequence 제외 추천
        negatives = np.random.choice(
            negatives, self.num_negatives, p=prob_dist, replace=False
        )
        return negatives


class CausalLMDataset(LMDataset):
    def __init__(
        self,
        df,
        max_length=128,
        num_negatives=None,
        negative_sample={"type": "full"},
        user_col="user_id",
        item_col="item_id",
        time_col="time_idx",
        label_masking_probability=0,
    ):
        super().__init__(
            df,
            max_length,
            num_negatives,
            negative_sample,
            user_col,
            item_col,
            time_col,
        )

        self.label_masking_probability = label_masking_probability

    def __getitem__(self, idx):
        item_sequence = self.data[self.user_ids[idx]]

        if len(item_sequence) > self.max_length + 1:
            item_sequence = item_sequence[-self.max_length - 1 :]

        input_ids = np.array(item_sequence[:-1], dtype=np.int64)
        labels = np.array(item_sequence[1:], dtype=np.int64)

        if self.label_masking_probability > 0:
            mask = np.random.rand(len(labels)) < self.label_masking_probability
            labels[mask] = -100

        if self.num_negatives:
            negatives = self.sample_negatives(item_sequence)
            return {"input_ids": input_ids, "labels": labels, "negatives": negatives}

        return {"input_ids": input_ids, "labels": labels}


class CausalLMPredictionDataset(LMDataset):
    def __init__(
        self,
        df,
        max_length=128,
        valid_mode=False,
        user_col="user_id",
        item_col="item_id",
        time_col="time_idx",
    ):
        super().__init__(
            df,
            max_length=max_length,
            num_negatives=None,
            user_col=user_col,
            item_col=item_col,
            time_col=time_col,
        )

        self.valid_mode = valid_mode

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_sequence = self.data[user_id]

        if self.valid_mode:
            target = item_sequence[-1]
            input_ids = item_sequence[-self.max_length - 1 : -1]
            item_sequence = item_sequence[:-1]

            return {
                "input_ids": input_ids,
                "user_id": user_id,
                "full_history": item_sequence,
                "target": target,
            }
        else:
            input_ids = item_sequence[-self.max_length :]

            return {
                "input_ids": input_ids,
                "user_id": user_id,
                "full_history": item_sequence,
            }


class PaddingCollateFn:
    def __init__(self, padding_value=0, labels_padding_value=-100):
        self.padding_value = padding_value
        self.labels_padding_value = labels_padding_value

    def __call__(self, batch):
        collated_batch = {}

        for key in batch[0].keys():
            if np.isscalar(batch[0][key]):
                collated_batch[key] = torch.tensor([example[key] for example in batch])
                continue

            if key == "labels":
                padding_value = self.labels_padding_value
            else:
                padding_value = self.padding_value
            values = [torch.tensor(example[key]) for example in batch]
            collated_batch[key] = pad_sequence(
                values, batch_first=True, padding_value=padding_value
            )

        return collated_batch
