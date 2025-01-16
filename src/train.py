import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
)

from src.config import *
from src.models.recommender import RecommenderNet


def create_callbacks(model_path: str) -> list:
    """모델 학습을 위한 콜백 함수들을 생성"""

    def lr_schedule(epoch):
        if epoch < RAMPUP_EPOCHS:
            return (MAX_LR - START_LR) / RAMPUP_EPOCHS * epoch + START_LR
        elif epoch < RAMPUP_EPOCHS + SUSTAIN_EPOCHS:
            return MAX_LR
        else:
            return (MAX_LR - MIN_LR) * EXP_DECAY ** (
                epoch - RAMPUP_EPOCHS - SUSTAIN_EPOCHS
            ) + MIN_LR

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(model_path, "model_{epoch:02d}.h5"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
        LearningRateScheduler(lr_schedule, verbose=1),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    return callbacks


def extract_embeddings(model: tf.keras.Model) -> Tuple[np.ndarray, np.ndarray]:
    """학습된 모델에서 사용자와 아이템 임베딩을 추출"""
    user_layer = model.get_layer("user_embedding")
    anime_layer = model.get_layer("anime_embedding")

    user_weights = user_layer.get_weights()[0]
    anime_weights = anime_layer.get_weights()[0]

    user_weights = user_weights / np.linalg.norm(user_weights, axis=1).reshape((-1, 1))
    anime_weights = anime_weights / np.linalg.norm(anime_weights, axis=1).reshape(
        (-1, 1)
    )

    return user_weights, anime_weights


def save_embeddings(
    user_weights: np.ndarray, anime_weights: np.ndarray, save_path: str
):
    """임베딩을 파일로 저장"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, "user_embeddings.npy"), user_weights)
    np.save(os.path.join(save_path, "anime_embeddings.npy"), anime_weights)


def create_and_train_model(strategy, train_data, test_data, n_users, n_animes):
    """모델 생성 및 학습"""
    print("\n모델 생성 중...")
    if TPU_INIT:
        with strategy.scope():
            model = RecommenderNet(n_users, n_animes, EMBEDDING_SIZE).build()
    else:
        model = RecommenderNet(n_users, n_animes, EMBEDDING_SIZE).build()

    # 콜백 설정
    callbacks = create_callbacks(MODEL_SAVE_PATH)

    # 모델 학습
    print("\n모델 학습 시작...")
    X_train, y_train = train_data
    X_test, y_test = test_data

    history = model.fit(
        [X_train[:, 0], X_train[:, 1]],
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
        callbacks=callbacks,
    )

    return model, history
