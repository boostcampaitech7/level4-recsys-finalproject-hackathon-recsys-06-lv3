import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import src.models as models
from src.trainer import Trainer
from src.utils import get_config


def load_data(data_path, val_size: float = 0.2, random_state: int = 42):
    # random seed 설정
    np.random.seed(random_state)
    df = pd.read_csv(data_path)
    # 사용자 ID 인코딩
    user_ids = df["user_id"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    user_encoded2user = {i: x for i, x in enumerate(user_ids)}

    # 애니메이션 ID 인코딩
    anime_ids = df["anime_id"].unique().tolist()
    anime2anime_encoded = {x: i for i, x in enumerate(anime_ids)}
    anime_encoded2anime = {i: x for i, x in enumerate(anime_ids)}

    # ID를 인코딩된 값으로 변환
    df["user"] = df["user_id"].map(user2user_encoded)
    df["item"] = df["anime_id"].map(anime2anime_encoded)

    # 평점 정규화 (0-1 사이로)
    df["rating"] = df["rating"] / 10.0
    df["labels"] = 0.9  # Label 값 Float으로 처리
    # 사용자별 인덱스 분할을 한 번에 처리하는 방식
    train_idx = []
    val_idx = []

    for _, group in df.groupby("user"):
        n = len(group)
        n_test = max(1, int(n * val_size))

        # 인덱스를 섞고 분할
        shuffled_idx = np.random.permutation(group.index)
        val_idx.extend(shuffled_idx[:n_test])
        train_idx.extend(shuffled_idx[n_test:])

    # 인덱스로 한 번에 분할
    train_df = df.loc[train_idx]
    val_df = df.loc[val_idx]
    # 인코딩 매핑 딕셔너리들을 튜플로 반환
    id_mappings = (
        user2user_encoded,
        user_encoded2user,
        anime2anime_encoded,
        anime_encoded2anime,
    )

    return (train_df, val_df), id_mappings


if __name__ == "__main__":
    config = get_config()
    data_path = config["data_path"]
    (train_df, val_df), mappings = load_data(data_path)
    # 모델 크기 계산
    num_users = len(mappings[0])
    num_items = len(mappings[2])
    model_name = config["model"]
    model = getattr(models, model_name)(num_users, num_items, config[model_name])
    criterion = getattr(nn, config["loss"])()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    trainer = Trainer(
        model, criterion, optimizer, train_df, val_df, num_users, num_items, config
    )
    trainer.train(config["epochs"])
    # trainer.validate()
    mlflow.end_run()
