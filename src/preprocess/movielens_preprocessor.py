import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from .abstract_preprocessor import AbstractPreProcessor


class MovieLensPreProcessor(AbstractPreProcessor):
    def __init__(
        self,
        dataset: str,
        data_path: str,
        export_path: str,
    ):
        super().__init__(dataset, data_path, export_path)
        self.data = {
            "items": pd.read_csv(os.path.join(data_path, "movies.csv")),
            "ratings": pd.read_csv(os.path.join(data_path, "ratings.csv")),
            # "tags": pd.read_csv(os.path.join(data_path, "tags.csv")),
        }

    def get_user_interactions(self) -> defaultdict:
        """
        유저별 상호작용 아이템을 반환하는 함수
        """
        ratings = self.data["ratings"]
        user_interactions = defaultdict(list)
        for user_id, item_id in zip(ratings["userId"], ratings["movieId"]):
            user_interactions[user_id].append(item_id)

        return user_interactions

    def _pre_process(self) -> None:
        items = self.data["items"]
        ratings = self.data["ratings"]

        # items 전처리
        items = self._clean_movie_titles_and_years(items)
        # items = self._preprocess_genres(items)

        # ratings 전처리
        ratings.rename(
            columns={
                "movieId": "item_id",
                "userId": "user_id",
                "rating": "rating",
                "timestamp": "timestamp",
            },
            inplace=True,
        )

        # 유저 리뷰 수 기준 이상치 제거
        ratings = self._filter_users_with_interactions(ratings)

        # user_id와 item_id 매핑 생성
        user2id = {id: idx for idx, id in enumerate(ratings["user_id"].unique())}
        item2id = {id: idx for idx, id in enumerate(ratings["item_id"].unique())}

        # 매핑된 값으로 user_id와 item_id 변경
        ratings["user_id"] = ratings["user_id"].map(user2id)
        ratings["item_id"] = ratings["item_id"].map(item2id)

        self.user2id = user2id
        self.item2id = item2id
        self.id2user = {v: k for k, v in user2id.items()}
        self.id2item = {v: k for k, v in item2id.items()}

        train, valid, valid_full, test, item_count = self._split_train_valid_test(
            ratings
        )

        self.item_count = item_count

        # export_dfs에 전처리된 데이터 저장
        self.export_dfs = {
            "items": items,
            "train": train,
            "valid": valid,
            "valid_full": valid_full,
            "test": test,
        }

    def _preprocess_genres(self, items: pd.DataFrame) -> pd.DataFrame:
        """
        추후 장르 원핫인코딩 or 멀티 인코딩 코드 상의 후 작성
        """
        return items

    def _clean_movie_titles_and_years(self, items) -> pd.DataFrame:
        """
        items에서 year 생성 및 결측치 처리
        title에서 year 제거
        movieId를 item_id로 변경
        """

        # items의 title에서 연도 제거
        items["year"] = items["title"].str.extract(r"\((\d{4})\)")
        # 제목에서 연도 부분 제거
        items["title"] = items["title"].str.replace(r" \(\d{4}\)", "", regex=True)
        # year 컬럼을 정수형으로 변환
        items["year"] = items["year"].fillna(0).astype(int)

        year_json = {
            "40697": 1993,
            "79607": 1970,
            "87442": 2010,
            "107434": 2009,
            "108548": 2007,
            "108583": 1975,
            "112406": 2019,
            "113190": 2021,
            "115133": 1996,
            "115685": 2011,
            "125571": 1990,
            "125632": 2002,
            "125958": 2008,
            "126438": 2013,
            "126929": 2014,
            "127005": 1991,
            "128612": 2015,
            "128734": 2014,
            "129651": 2010,
            "129705": 2014,
            "129887": 2003,
            "130454": 1993,
        }

        # year_json의 key와 value를 items 데이터프레임에 반영
        for key, value in year_json.items():
            key = int(key)
            items.loc[items["movieId"] == int(key), "year"] = value

        items.rename(columns={"movieId": "item_id"}, inplace=True)
        return items

    def _filter_users_with_interactions(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        사용자별 상호작용 데이터를 필터링하고 interaction 컬럼을 생성
        """
        # 2000년 이후 마지막 상호작용이 있는 유저 필터링
        users_to_keep = ratings[ratings["timestamp"] >= 946684800]["user_id"].unique()
        ratings = ratings[ratings["user_id"].isin(users_to_keep)]

        # 평점이 4.0 이상인 데이터만 사용
        ratings = ratings[ratings["rating"] >= 4.0]

        # 최소 3개 이상의 상호작용이 있는 유저 필터링
        user_counts = ratings.groupby("user_id").size().reset_index(name="count")
        ratings = ratings[
            ratings["user_id"].isin(user_counts[user_counts["count"] > 2]["user_id"])
        ]
        return ratings

    def _split_train_valid_test(
        self, ratings
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
        # ratings 정렬 & time_idx 추가
        # ratings = ratings.sort_values(["user_id", "timestamp"])

        ratings["time_idx"] = ratings.groupby("user_id").cumcount()
        ratings["time_idx_reversed"] = ratings.groupby("user_id").cumcount(
            ascending=False
        )

        # train, valid, test 데이터 분리
        train = ratings[ratings.time_idx_reversed >= 2]
        valid = ratings[ratings.time_idx_reversed == 1]
        valid_full = ratings[ratings.time_idx_reversed >= 1]
        test = ratings[ratings.time_idx_reversed == 0]

        item_count = ratings["item_id"].max()

        return train, valid, valid_full, test, item_count
