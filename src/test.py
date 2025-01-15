from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import *
from src.models.recommender import (AnimeSimilarityRecommender,
                                    UserSimilarityRecommender)
from src.train import extract_embeddings, save_embeddings


def ranking_based_recommendations(
    model: tf.keras.Model,
    random_user: int,
    rating_df: pd.DataFrame,
    df: pd.DataFrame,
    mappings: Tuple,
    synopsis_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    특정 사용자를 위한 애니메이션 추천 예측

    Args:
        model: 학습된 모델
        random_user: 사용자 ID
        rating_df: 평점 데이터프레임
        df: 애니메이션 정보 데이터프레임
        mappings: (user2user_encoded, user_encoded2user, anime2anime_encoded, anime_encoded2anime) 매핑
        synopsis_df: 시놉시스 데이터프레임

    Returns:
        추천 결과 데이터프레임
    """
    user2user_encoded, user_encoded2user, anime2anime_encoded, anime_encoded2anime = (
        mappings
    )

    print(f"사용자 {random_user}를 위한 추천:")
    print("===" * 25)

    # 사용자가 시청한 애니메이션 찾기
    animes_watched_by_user = rating_df[rating_df.user_id == random_user]
    anime_not_watched_df = df[
        ~df["anime_id"].isin(animes_watched_by_user.anime_id.values)
    ]

    # 인코딩된 ID로 변환
    anime_not_watched = list(
        set(anime_not_watched_df["anime_id"]).intersection(
            set(anime2anime_encoded.keys())
        )
    )
    anime_not_watched = [[anime2anime_encoded.get(x)] for x in anime_not_watched]

    # 사용자 ID 인코딩
    user_encoder = user2user_encoded.get(random_user)

    # 예측을 위한 입력 배열 생성
    user_anime_array = np.hstack(
        ([[user_encoder]] * len(anime_not_watched), anime_not_watched)
    )
    user_anime_array = [user_anime_array[:, 0], user_anime_array[:, 1]]

    # 예측
    ratings = model.predict(user_anime_array).flatten()

    # 상위 10개 추천 선택
    top_ratings_indices = (-ratings).argsort()[:10]
    recommended_anime_ids = [
        anime_encoded2anime.get(anime_not_watched[x][0]) for x in top_ratings_indices
    ]

    # 결과 생성
    Results = []
    for index, anime_id in enumerate(anime_not_watched):
        rating = ratings[index]
        id_ = anime_encoded2anime.get(anime_id[0])

        if id_ in recommended_anime_ids:
            try:
                condition = df.anime_id == id_
                name = df[condition]["Name"].values[0]
                genre = df[condition].Genres.values[0]
                sypnopsis = synopsis_df[synopsis_df.MAL_ID == id_].sypnopsis.values[0]
            except:
                continue

            Results.append(
                {
                    "name": name,
                    "pred_rating": rating,
                    "genre": genre,
                    "synopsis": sypnopsis,
                }
            )

    print("---" * 25)
    print("> Top 10 애니메이션 추천")
    print("---" * 25)

    Results = pd.DataFrame(Results).sort_values(by="pred_rating", ascending=False)
    print(Results)

    return Results


def test_item_recommendations(
    anime_weights: np.ndarray,
    anime2anime_encoded: Dict[int, int],
    anime_encoded2anime: Dict[int, int],
    anime_df: pd.DataFrame,
    synopsis_df: pd.DataFrame,
    test_anime: str = "Death Note",
    n_recommendations: int = 5,
) -> pd.DataFrame:
    """
    아이템 기반 추천 시스템 테스트

    Args:
        anime_weights: 애니메이션 임베딩 가중치
        anime2anime_encoded: 애니메이션 ID -> 인코딩된 ID 매핑
        anime_encoded2anime: 인코딩된 ID -> 애니메이션 ID 매핑
        anime_df: 애니메이션 정보 데이터프레임
        synopsis_df: 시놉시스 데이터프레임
        test_anime: 테스트할 애니메이션 이름
        n_recommendations: 추천 수

    Returns:
        추천 결과 데이터프레임
    """
    item_recommender = AnimeSimilarityRecommender(
        anime_weights, anime2anime_encoded, anime_encoded2anime, anime_df, synopsis_df
    )

    similar_animes = item_recommender.find_similar_animes(
        test_anime, n=n_recommendations
    )
    print(f"\n{test_anime}와 유사한 애니메이션:")
    print(similar_animes)

    return similar_animes


def test_user_recommendations(
    user_weights: np.ndarray,
    user2user_encoded: Dict[int, int],
    user_encoded2user: Dict[int, int],
    rating_df: pd.DataFrame,
    anime_df: pd.DataFrame,
    n_similar_users: int = 20,
    n_recommendations: int = 5,
    min_rating: float = 3.0,
) -> Tuple[int, pd.DataFrame]:
    """
    사용자 기반 추천 시스템 테스트

    Args:
        user_weights: 사용자 임베딩 가중치
        user2user_encoded: 사용자 ID -> 인코딩된 ID 매핑
        user_encoded2user: 인코딩된 ID -> 사용자 ID 매핑
        rating_df: 평점 데이터프레임
        anime_df: 애니메이션 정보 데이터프레임
        n_similar_users: 유사 사용자 수
        n_recommendations: 추천 수
        min_rating: 최소 평점 기준

    Returns:
        (테스트 사용자 ID, 추천 결과 데이터프레임) 튜플
    """
    user_recommender = UserSimilarityRecommender(
        user_weights, user2user_encoded, user_encoded2user, rating_df, anime_df
    )

    # 평점이 많은 사용자 선택
    print("데이터프레임 컬럼:", rating_df.columns)
    user_ratings_count = rating_df["user_id"].value_counts()
    test_user = user_ratings_count.index[0]

    print(f"\n테스트 사용자 ID: {test_user}")
    print(f"테스트 사용자의 평점 수: {len(rating_df[rating_df.user_id == test_user])}")

    recommendations = user_recommender.get_recommendations(
        user_id=test_user,
        n_similar_users=n_similar_users,
        n_recommendations=n_recommendations,
    )

    print(f"\n사용자 {test_user}를 위한 추천:")
    print(recommendations)

    return test_user, recommendations


def evaluate_recommendations(
    recommendations: pd.DataFrame,
    anime_df: pd.DataFrame,
    rating_df: pd.DataFrame,
    test_user: int,
) -> Dict[str, float]:
    """
    추천 결과 평가

    Args:
        recommendations: 추천 결과 데이터프레임
        anime_df: 애니메이션 정보 데이터프레임
        rating_df: 평점 데이터프레임
        test_user: 테스트 사용자 ID

    Returns:
        평가 지표 딕셔너리
    """
    if recommendations.empty:
        return {"precision": 0.0, "recall": 0.0, "diversity": 0.0, "novelty": 0.0}

    # 사용자의 실제 시청 기록
    user_ratings = rating_df[rating_df["user_id"] == test_user]
    liked_animes = set(user_ratings[user_ratings["rating"] >= 7]["anime_id"])

    # 추천된 애니메이션
    recommended_animes = set(recommendations["anime_id"])

    # Precision & Recall
    relevant_recommended = len(liked_animes.intersection(recommended_animes))
    precision = (
        relevant_recommended / len(recommended_animes) if recommended_animes else 0
    )
    recall = relevant_recommended / len(liked_animes) if liked_animes else 0

    # Diversity (장르 다양성)
    genres = []
    for _, row in recommendations.iterrows():
        if pd.notna(row["genre"]):
            genres.extend(row["genre"].split(", "))
    genre_diversity = len(set(genres)) / len(genres) if genres else 0

    # Novelty (인기도 기반)
    anime_popularity = rating_df["anime_id"].value_counts()
    mean_popularity = np.mean(
        [np.log1p(anime_popularity.get(aid, 0)) for aid in recommended_animes]
    )
    novelty = 1 - (mean_popularity / np.log1p(anime_popularity.max()))

    return {
        "precision": precision,
        "recall": recall,
        "diversity": genre_diversity,
        "novelty": novelty,
    }


def test_recommendations(model, rating_df, anime_df, synopsis_df, mappings):
    """추천 시스템 테스트"""
    print("\n임베딩 추출 중...")
    user_weights, anime_weights = extract_embeddings(model)
    save_embeddings(user_weights, anime_weights, EMBEDDING_SAVE_PATH)

    # 아이템 기반 추천 테스트
    print("\n아이템 기반 추천 테스트...")
    similar_animes = test_item_recommendations(
        anime_weights, mappings[2], mappings[3], anime_df, synopsis_df
    )

    # 사용자 기반 추천 테스트
    print("\n사용자 기반 추천 테스트...")
    test_user, recommendations = test_user_recommendations(
        user_weights, mappings[0], mappings[1], rating_df, anime_df
    )

    # # 추천 결과 평가
    # if not recommendations.empty:
    #     print("\n추천 결과 평가:")
    #     metrics = evaluate_recommendations(
    #         recommendations,
    #         anime_df,
    #         rating_df,
    #         test_user
    #     )
    #     for metric, value in metrics.items():
    #         print(f"{metric}: {value:.4f}")

    # 특정 사용자를 위한 예측 기반 추천
    print("\n예측 기반 추천 테스트...")
    # 평점이 많은 사용자 선택
    user_ratings_count = rating_df["user_id"].value_counts()
    random_user = user_ratings_count.index[0]

    predicted_recommendations = ranking_based_recommendations(
        model, random_user, rating_df, anime_df, mappings, synopsis_df
    )
