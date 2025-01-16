import numpy as np
import pandas as pd

from src.config import *
from src.data.preprocessing import preprocess_data


def load_rating_data(input_dir, min_ratings=100):
    file_path = f"{input_dir}/animelist.csv"

    rating_df = pd.read_csv(file_path, usecols=["user_id", "anime_id", "rating"])
    rating_df = rating_df[rating_df["rating"] > 0].reset_index()

    # 최소 평가 수 필터링
    n_ratings = rating_df["user_id"].value_counts()
    valid_users = n_ratings[n_ratings >= min_ratings].index
    rating_df = rating_df[rating_df["user_id"].isin(valid_users)].copy()

    return rating_df


def load_anime_data(input_dir):
    df = pd.read_csv(f"{input_dir}/anime.csv", low_memory=True)
    df = df.replace("Unknown", np.nan)

    df["anime_id"] = df["MAL_ID"]

    return df


def load_synopsis_data(input_dir):
    cols = ["MAL_ID", "Name", "Genres", "sypnopsis"]
    return pd.read_csv(f"{input_dir}/anime_with_synopsis.csv", usecols=cols)


def load_data():
    """데이터 로드 및 전처리"""
    print("데이터 로딩 중...")
    try:
        rating_df = load_rating_data(
            INPUT_DIR,
            # min_ratings=100,
        )

        if rating_df.empty:
            raise ValueError("평점 데이터를 로드할 수 없습니다.")

        anime_df = load_anime_data(INPUT_DIR)
        synopsis_df = load_synopsis_data(INPUT_DIR)

        # 데이터 전처리
        print("\n데이터 전처리 중...")
        (X_train, X_test, y_train, y_test), mappings = preprocess_data(rating_df)

        return (
            rating_df,
            anime_df,
            synopsis_df,
            (X_train, X_test, y_train, y_test),
            mappings,
        )

    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {str(e)}")
        raise
