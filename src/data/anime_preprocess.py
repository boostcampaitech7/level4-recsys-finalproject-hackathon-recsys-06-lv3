import os
import re

import numpy as np
import pandas as pd

from src.data.preprocess import Preprocess


class AnimePreprocess(Preprocess):
    def __init__(self, dataset: str, data_path, export_path):
        super(AnimePreprocess).__init__(dataset, data_path, export_path)

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

        # ratings에서 rating 피처의 값이 0인 데이터 제외
        ratings = ratings[ratings["rating"] != 0]

        # 각 df에서 피처명 조정
        item_synopsis.rename(
            columns={"MAL_ID": "Item_id", "sypnopsis": "Synopsis"}, inplace=True
        )
        items.rename(columns={"MAL_ID": "Item_id"}, inplace=True)
        ratings.rename(
            columns={
                "anime_id": "Item_id",
                "user_id": "User_id",
                "rating": "Rating",
                "watching_status": "Watching_status",
                "watched_episodes": "Watched_episodes",
            },
            inplace=True,
        )
        users.rename(columns={"Mal ID": "User_id"}, inplace=True)

        self.export_dfs = {
            "anime_with_synopsis": item_synopsis,
            "anime": items,
            "animelist": ratings,
            "user_detail": users,
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
