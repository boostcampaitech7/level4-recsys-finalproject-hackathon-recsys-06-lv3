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
        items = self._process_item_features(items)
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

        # 유저 리뷰 수 기준 이상치 제거 및 train-test split
        ratings = self._filter_users_with_interactions(ratings)
        train, valid, test = self._split_train_valid_test(ratings)

        # 영화 일정 리뷰 수 이하 제거 (추후 구현 필요)

        # export_dfs에 전처리된 데이터 저장
        self.export_dfs = {
            "items": items,
            "train": train,
            "valid": valid,
            "test": test,
        }

    def _preprocess_genres(self, items: pd.DataFrame) -> pd.DataFrame:
        """
        추후 장르 원핫인코딩 or 멀티 인코딩 코드 상의 후 작성
        """
        return items

    def _process_item_features(self, items) -> pd.DataFrame:
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
        # IQR 우선 계산
        user_rating_counts = (
            ratings.groupby("user_id").size().reset_index(name="rating_count")
        )

        Q1 = user_rating_counts["rating_count"].quantile(0.25)
        Q3 = user_rating_counts["rating_count"].quantile(0.75)
        IQR = Q3 - Q1
        upper_fence = Q3 + 1.5 * IQR

        # 4.0 이상 평점이 2개 이하인 유저 필터링 (valid와 test, train에 하나씩 필요)
        user_high_ratings = (
            ratings[ratings["rating"] >= 4.0]
            .groupby("user_id")
            .size()
            .reset_index(name="high_rating_count")
        )
        valid_users = user_high_ratings[user_high_ratings["high_rating_count"] > 2][
            "user_id"
        ]
        filtered_ratings = ratings[ratings["user_id"].isin(valid_users)].copy()

        # IQR 기반 이상치 제거
        valid_users = user_rating_counts[
            user_rating_counts["rating_count"] <= upper_fence
        ]["user_id"]
        filtered_ratings = filtered_ratings[
            filtered_ratings["user_id"].isin(valid_users)
        ]

        # interaction 컬럼 생성
        filtered_ratings.loc[:, "interaction"] = (
            filtered_ratings["rating"] >= 4.0
        ).astype(int)

        return filtered_ratings

    def _split_train_valid_test(
        self, ratings
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_list, valid_list, test_list = [], [], []

        # user_id 기준 그룹화
        grouped = ratings.groupby("user_id", group_keys=False)

        for _, group in tqdm(grouped, desc="Splitting train/valid/test"):
            # positive,negative 상호작용 분리
            pos_mask = group["interaction"] == 1
            pos_interactions = group[pos_mask]

            # 상위 2개 timestamp 추출
            top2 = pos_interactions.nlargest(2, "timestamp")

            # Test/Valid 분리
            test_data = top2.iloc[[0]] if len(top2) >= 1 else None  # 최신 1개
            valid_data = top2.iloc[[1]] if len(top2) >= 2 else None  # 차순위 1개

            # Train 데이터 구성
            ## 남은 positive: 전체 positive에서 top2 제외
            train_data = pos_interactions.drop(top2.index, errors="ignore")

            # 데이터 추가
            if test_data is not None:
                test_list.append(test_data)
            if valid_data is not None:
                valid_list.append(valid_data)
            if not train_data.empty:
                train_list.append(train_data)

        # DataFrame 병합
        train = (
            pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
        )
        valid = (
            pd.concat(valid_list, ignore_index=True) if valid_list else pd.DataFrame()
        )
        test = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

        return train, valid, test
