from typing import Tuple

import pandas as pd


def preprocess_data(
    rating_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[Tuple, Tuple]:
    """
    평점 데이터 전처리 및 학습/테스트 데이터 분할

    Args:
        rating_df: 사용자-아이템 평점 데이터프레임
        test_size: 테스트 데이터 비율
        random_state: 랜덤 시드

    Returns:
        ((X_train, X_test, y_train, y_test),
         (user2user_encoded, user_encoded2user, anime2anime_encoded, anime_encoded2anime))
    """
    # 데이터 유효성 검사
    if rating_df.empty:
        raise ValueError("데이터프레임이 비어 있습니다.")

    print(f"전처리 전 데이터 크기: {len(rating_df):,}행")
    print(f"유니크 사용자 수: {rating_df['user_id'].nunique():,}")
    print(f"유니크 애니메이션 수: {rating_df['anime_id'].nunique():,}")

    # 사용자 ID 인코딩
    user_ids = rating_df["user_id"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    user_encoded2user = {i: x for i, x in enumerate(user_ids)}

    # 애니메이션 ID 인코딩
    anime_ids = rating_df["anime_id"].unique().tolist()
    anime2anime_encoded = {x: i for i, x in enumerate(anime_ids)}
    anime_encoded2anime = {i: x for i, x in enumerate(anime_ids)}

    # ID를 인코딩된 값으로 변환
    rating_df["user"] = rating_df["user_id"].map(user2user_encoded)
    rating_df["anime"] = rating_df["anime_id"].map(anime2anime_encoded)

    # 평점 정규화 (0-1 사이로)
    rating_df["rating"] = rating_df["rating"] / 10.0

    # 사용자별 train-test split
    train_data = []
    test_data = []

    for user, group in rating_df.groupby("user"):
        group = group.sample(frac=1, random_state=random_state)  # 데이터 섞기
        n_test = max(1, int(len(group) * test_size))  # 최소 1개는 test에 포함

        test_data.append(group.iloc[:n_test])
        train_data.append(group.iloc[n_test:])

    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)

    # 학습/테스트 데이터 준비
    X_train = train_df[["user", "anime"]].values
    y_train = train_df["rating"].values
    X_test = test_df[["user", "anime"]].values
    y_test = test_df["rating"].values

    # 인코딩 매핑 딕셔너리들을 튜플로 반환
    id_mappings = (
        user2user_encoded,
        user_encoded2user,
        anime2anime_encoded,
        anime_encoded2anime,
    )

    return (X_train, X_test, y_train, y_test), id_mappings
