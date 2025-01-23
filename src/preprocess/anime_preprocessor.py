import os
import re

import numpy as np
import pandas as pd

from .abstract_preprocessor import AbstractPreProcessor


class AnimePreProcessor(AbstractPreProcessor):
    def __init__(self, dataset: str, data_path, export_path, sample_size: int = 10):
        super().__init__(dataset, data_path, export_path)
        self.data = {
            "anime_with_synopsis": pd.read_csv(os.path.join(data_path, "anime_with_synopsis.csv")),
            "anime": pd.read_csv(os.path.join(data_path, "anime.csv")),
            "animelist": pd.read_csv(os.path.join(data_path, "animelist.csv")),
            "user_detail": pd.read_csv(os.path.join(data_path, "user_detail.csv")),
        }
        self.sample_size = sample_size

    def pre_process(self) -> None:        
        item_synopsis: pd.DataFrame = self.data["anime_with_synopsis"]
        items: pd.DataFrame = self.data["anime"]
        ratings: pd.DataFrame = self.data["animelist"]
        users: pd.DataFrame = self.data["user_detail"]

        # items 전처리
        items = self._preprocess_genres(items)
        items = self._process_item_features(items)

        # item_synopsis 전처리
        item_synopsis = self._preprocess_synopsis(item_synopsis)

        # ratings 전처리
        ratings.rename(
            columns={
                "anime_id": "item_id", "user_id": "user_id", "rating": "rating", "watching_status": "watching_status", "watched_episodes": "watched_episodes"
            }, inplace=True
        )
        ratings.columns = ratings.columns.str.lower()

        # users 전처리
        users.rename(columns={"Mal ID": "user_id"}, inplace=True)
        users.columns = users.columns.str.lower()

        # 유저 리뷰 수 기준 이상치 제거 및 train-test split
        ratings = self._filter_users_with_interactions(ratings)
        train, test = self._split_train_test(ratings)

        # export_dfs에 전처리된 데이터 저장
        self.export_dfs = {
            "items": items,
            "item_synopsis": item_synopsis,
            "train": train,
            "test": test,
            "users": users
        }

    def _preprocess_genres(self, items: pd.DataFrame) -> pd.DataFrame:
        genres_dummies = items["Genres"].str.get_dummies(sep=", ")
        genres_dummies = genres_dummies.reindex(sorted(genres_dummies.columns), axis=1)
        items["Genres"] = genres_dummies.apply(
            lambda row: ",".join(genres_dummies.columns[row == 1]), axis=1
        )
        return items
    
    def _process_item_features(self, items) -> pd.DataFrame:
        replace_features = ["Score", "Episodes", "Ranked"]
        items[replace_features] = items[replace_features].replace("Unknown", np.nan)
        items[replace_features] = items[replace_features].astype(float)

        items["Aired"] = (items["Aired"].apply(
            lambda x: (re.search(r"\d{4}", x).group(0) if re.search(r"\d{4}", x) else np.nan)
            ).astype(float)
        )

        items["Duration"] = items["Duration"].replace("Unknown", np.nan)
        items["Duration"] = items["Duration"].apply(
            lambda x: int(x.split()[0]) if isinstance(x, str) else x
        )
        
        items.rename(columns={"MAL_ID": "item_id"}, inplace=True)
        items.columns = items.columns.str.lower()

        return items
    
    def _preprocess_synopsis(self, item_synopsis) -> pd.DataFrame:
        item_synopsis["Score"] = item_synopsis["Score"].replace("Unknown", np.nan).astype(float)
        
        item_synopsis.rename(columns={"MAL_ID": "item_id", "sypnopsis": "synopsis"}, inplace=True)
        item_synopsis.columns = item_synopsis.columns.str.lower()

        return item_synopsis
    
    def _filter_users_with_interactions(self, ratings) -> pd.DataFrame:
        user_rating_counts = ratings["user_id"].value_counts()
        Q1 = user_rating_counts.quantile(0.25)
        Q3 = user_rating_counts.quantile(0.75)
        IQR = Q3 - Q1
        upper_fence = Q3 + 1.5 * IQR

        filtered_users = user_rating_counts[user_rating_counts <= upper_fence].index
        filtered_ratings = ratings[ratings["user_id"].isin(filtered_users)]

        filtered_ratings.loc[:, 'interaction'] = np.where(
            ((filtered_ratings['rating'] == 0) | (filtered_ratings['rating'] >= 6)) & 
            (filtered_ratings['watching_status'] != 4),
            1, 0
        )

        user_interaction_counts = (
            filtered_ratings[filtered_ratings['interaction'] == 1]['user_id'].value_counts()
        )
        filtered_users_with_interactions = user_interaction_counts[user_interaction_counts >= 11].index

        return filtered_ratings[
            filtered_ratings["user_id"].isin(filtered_users_with_interactions)
        ]
    
    def _split_train_test(self, ratings) -> tuple:
        test = (
            ratings[ratings["interaction"] == 1]
            .groupby("user_id")
            .apply(lambda x: x.sample(n=self.sample_size, random_state=42)).reset_index(drop=True)
        )

        train = ratings[
            ~ratings.set_index(["user_id", "item_id"]).index.isin(
                test.set_index(["user_id", "item_id"]).index
            )
        ].reset_index(drop=True)

        return train, test

    def save_data(self) -> None:
        os.makedirs(self.export_path, exist_ok=True)
        os.makedirs(os.path.join(self.export_path, self.dataset), exist_ok=True)

        for key, df in self.export_dfs.items():
            file_path = os.path.join(self.export_path, self.dataset, f"{key}.csv")
            print(f"Saving {key} to {file_path}")
            print(f"{key} column list is {df.columns} - shape({df.shape})")

            df.to_csv(file_path, index=False)