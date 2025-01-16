import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import *
from src.models.recommender import (AnimeSimilarityRecommender,
                                    UserSimilarityRecommender)
from src.train import extract_embeddings, save_embeddings
from src.utils.metrics import RecommenderMetrics, calculate_metrics_for_users


class RecommenderEvaluator:
    def __init__(self, anime_df: pd.DataFrame, rating_df: pd.DataFrame):
        """추천 시스템 평가를 위한 클래스 초기화"""
        self.anime_df = anime_df
        self.rating_df = rating_df
        self.metrics = RecommenderMetrics()

        # 설정에서 평가 기준 가져오기
        self.k = EvaluationConfig.EVALUATION_K
        self.n_test_users = EvaluationConfig.N_TEST_USERS
        self.rating_threshold = EvaluationConfig.MIN_RATING_THRESHOLD
        self.metrics_to_track = EvaluationConfig.RECOMMENDATION_METRICS

        # 아이템 특성 정보 생성
        self.item_features = self._create_item_features()
        # 아이템 인기도 정보 생성
        self.item_popularity = self._create_item_popularity()

    def _create_item_features(self) -> Dict[int, List[str]]:
        """애니메이션의 장르 정보를 딕셔너리로 변환"""
        features = {}
        for _, row in self.anime_df.iterrows():
            if pd.notna(row["Genres"]):
                features[row["anime_id"]] = row["Genres"].split(", ")
        return features

    def _create_item_popularity(self) -> Dict[int, int]:
        """각 애니메이션의 평가 횟수를 계산"""
        return dict(self.rating_df["anime_id"].value_counts())

    def _save_evaluation_results(
        self, metrics: Dict[str, float], recommendation_type: str
    ):
        """평가 결과를 파일로 저장"""
        if not LoggingConfig.SAVE_METRICS:
            return

        os.makedirs(METRIC_SAVE_PATH, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            METRIC_SAVE_PATH, f"evaluation_{recommendation_type}_{timestamp}.json"
        )

        # 평가 결과 저장
        with open(filename, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "recommendation_type": recommendation_type,
                    "metrics": metrics,
                    "config": {
                        "k": self.k,
                        "n_test_users": self.n_test_users,
                        "rating_threshold": self.rating_threshold,
                    },
                },
                f,
                indent=2,
            )

    def evaluate_recommendations(
        self,
        recommendations: Dict[int, List[int]],
        recommendation_type: str = "general",
    ) -> Dict[str, float]:
        """추천 결과에 대한 종합적인 평가 수행"""
        # 평점이 많은 상위 사용자들을 테스트 대상으로 선택
        test_users = (
            self.rating_df["user_id"]
            .value_counts()
            .head(self.n_test_users)
            .index.tolist()
        )

        # 테스트 데이터 생성 (긍정적 평가만 고려)
        test_data = {
            user: self.rating_df[
                (self.rating_df["user_id"] == user)
                & (self.rating_df["rating"] >= self.rating_threshold)
            ]["anime_id"].tolist()
            for user in test_users
        }

        # 전체 메트릭 계산
        metrics = calculate_metrics_for_users(
            test_data=test_data,
            predictions=recommendations,
            item_features=self.item_features,
            item_popularity=self.item_popularity,
            k=self.k,
        )

        # 설정된 메트릭만 선택
        metrics = {k: v for k, v in metrics.items() if k in self.metrics_to_track}

        # 결과 저장
        self._save_evaluation_results(metrics, recommendation_type)

        return metrics


def get_item_based_recommendations(
    recommender: AnimeSimilarityRecommender,
    rating_df: pd.DataFrame,
    n_users: int,
    k: int,
    rating_threshold: float,
) -> Dict[int, List[int]]:
    """아이템 기반 추천 생성"""
    recommendations = {}
    test_users = rating_df["user_id"].value_counts().head(n_users).index.tolist()

    for user_id in test_users:
        # 사용자가 높은 평점을 준 아이템 기반으로 추천
        user_ratings = rating_df[
            (rating_df["user_id"] == user_id)
            & (rating_df["rating"] >= rating_threshold)
        ]
        if not user_ratings.empty:
            seed_anime = user_ratings.iloc[0]["anime_id"]
            similar_animes = recommender.find_similar_animes(
                seed_anime, n=k, return_dist=True
            )
            recommendations[user_id] = [anime_id for anime_id, _ in similar_animes[1]]

    return recommendations


def get_user_based_recommendations(
    recommender: UserSimilarityRecommender, n_users: int, k: int
) -> Dict[int, List[int]]:
    """사용자 기반 추천 생성"""
    recommendations = {}
    test_users = (
        recommender.rating_df["user_id"].value_counts().head(n_users).index.tolist()
    )

    for user_id in test_users:
        recs = recommender.get_recommendations(
            user_id=user_id, n_similar_users=20, n_recommendations=k
        )
        if not recs.empty:
            recommendations[user_id] = recs["anime_id"].tolist()

    return recommendations


def test_recommendations(model, rating_df, anime_df, synopsis_df, mappings):
    """추천 시스템 테스트 및 평가"""
    print("\n임베딩 추출 중...")
    user_weights, anime_weights = extract_embeddings(model)
    save_embeddings(user_weights, anime_weights, EMBEDDING_SAVE_PATH)

    # 평가기 초기화
    evaluator = RecommenderEvaluator(anime_df, rating_df)

    # 1. 아이템 기반 추천 테스트
    print("\n아이템 기반 추천 테스트...")
    item_recommender = AnimeSimilarityRecommender(
        anime_weights, mappings[2], mappings[3], anime_df, synopsis_df
    )

    item_based_metrics = evaluator.evaluate_recommendations(
        get_item_based_recommendations(
            item_recommender,
            rating_df,
            evaluator.n_test_users,
            evaluator.k,
            evaluator.rating_threshold,
        ),
        "item_based",
    )

    # 2. 사용자 기반 추천 테스트
    print("\n사용자 기반 추천 테스트...")
    user_recommender = UserSimilarityRecommender(
        user_weights, mappings[0], mappings[1], rating_df, anime_df
    )

    user_based_metrics = evaluator.evaluate_recommendations(
        get_user_based_recommendations(
            user_recommender, evaluator.n_test_users, evaluator.k
        ),
        "user_based",
    )

    # 3. 결과 출력
    print("\n=== 평가 결과 ===")
    print("\n아이템 기반 추천:")
    for metric, value in item_based_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n사용자 기반 추천:")
    for metric, value in user_based_metrics.items():
        print(f"{metric}: {value:.4f}")

    return {"item_based": item_based_metrics, "user_based": user_based_metrics}
