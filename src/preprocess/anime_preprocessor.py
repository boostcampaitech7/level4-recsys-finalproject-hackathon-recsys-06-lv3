import os
import re

import numpy as np
import pandas as pd

from .abstract_preprocessor import AbstractPreProcessor


class AnimePreProcessor(AbstractPreProcessor):
    def __init__(self, dataset: str, data_path, export_path):
        super().__init__(dataset, data_path, export_path)

    def pre_process(self) -> None:
        item_synopsis: pd.DataFrame = self.data["anime_with_synopsis"]
        items: pd.DataFrame = self.data["anime"]
        ratings: pd.DataFrame = self.data["animelist"]
        users: pd.DataFrame = self.data["user_detail"]

        # items에서 Genres 전처리
        genres_dummies = items["Genres"].str.get_dummies(sep=", ")
        genres_dummies = genres_dummies.reindex(sorted(genres_dummies.columns), axis=1)
        items["Genres"] = genres_dummies.apply(
            lambda row: ",".join(genres_dummies.columns[row == 1]), axis=1
        )

        # 데이터 중 "Unknown" 값을 포함하는 일부 피처들에 대해 NaN으로 대체
        # Genres 피처에 "Unknown"값 존재에 따라 해당 피처 제외 및 안정적 처리를 위함
        features_list = ["Score", "Episodes", "Ranked"]
        items[features_list] = items[features_list].replace("Unknown", np.nan)
        items[features_list] = items[features_list].astype(float)

        items["Aired"] = (
            items["Aired"]
            .apply(
                lambda x: (
                    re.search(r"\d{4}", x).group(0)
                    if re.search(r"\d{4}", x)
                    else np.nan
                )
            )
            .astype(float)
        )
        items["Duration"] = items["Duration"].replace("Unknown", np.nan)
        items["Duration"] = items["Duration"].apply(
            lambda x: int(x.split()[0]) if isinstance(x, str) else x
        )

        item_synopsis["Score"] = (
            item_synopsis["Score"].replace("Unknown", np.nan).astype(float)
        )

        # 각 df에서 피처명 조정
        item_synopsis.rename(
            columns={"MAL_ID": "item_id", "sypnopsis": "synopsis"}, inplace=True
        )
        items.rename(
            columns={"MAL_ID": "item_id"}, 
            inplace=True)
        ratings.rename(
            columns={
                "anime_id": "item_id",
                "user_id": "user_id",
                "rating": "rating",
                "watching_status": "watching_status",
                "watched_episodes": "watched_episodes",
            },
            inplace=True,
        )
        users.rename(columns={"Mal ID": "user_id"}, inplace=True)

        # 칼럼명 소문자로 통일
        item_synopsis.columns = item_synopsis.columns.str.lower()
        items.columns = items.columns.str.lower()
        ratings.columns = ratings.columns.str.lower()
        users.columns = users.columns.str.lower()

        # 이상치 처리
        # 이상치 제거를 위해 user_id에 따른 rating 갯수 확인
        user_rating_counts = ratings["user_id"].value_counts()

        Q1 = user_rating_counts.quantile(0.25)
        Q3 = user_rating_counts.quantile(0.75)
        IQR = Q3 - Q1
        upper_fence = Q3 + 1.5 * IQR

        filtered_users = user_rating_counts[user_rating_counts <= upper_fence].index
        ratings = ratings[ratings["user_id"].isin(filtered_users)]

        # interaction 칼럼 추가
        ratings['interaction'] = np.where(
            ((ratings['rating'] == 0) | (ratings['rating'] >= 6)) & (ratings['watching_status'] != 4),
            1, 0
        )
        user_interection_counts = ratings[ratings['interaction'] == 1]['user_id'].value_counts()
        filtered_users_with_interactions = user_interection_counts[user_interection_counts >= 11].index
        ratings = ratings[ratings['user_id'].isin(filtered_users_with_interactions)]

        # export_dfs에 전처리된 데이터 저장
        self.export_dfs = {
            "anime": items,
            "anime_with_synopsis": item_synopsis,
            "animelist": ratings,
            "user_detail": users
        }

    def save_data(self) -> None:
        os.makedirs(self.export_path, exist_ok=True)
        os.makedirs(os.path.join(self.export_path, self.dataset), exist_ok=True)

        for key in self.export_dfs:
            print(f"{key} is export file in {self.export_path}")
            print(
                f"{key} column list is {self.export_dfs[key].columns} - shape({self.export_dfs[key].shape})"
            )
            self.export_dfs[key].to_csv(
                os.path.join(self.export_path, self.dataset, key), index=False
            )
