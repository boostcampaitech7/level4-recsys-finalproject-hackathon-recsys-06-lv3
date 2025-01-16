import json
import os
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (Callback, EarlyStopping,
                                        LearningRateScheduler, ModelCheckpoint)

from src.config import *
from src.models.recommender import RecommenderNet
from src.utils.metrics import TrainingMetrics


class MetricsCallback(Callback):
    """사용자 정의 메트릭을 계산하고 기록하는 콜백"""

    def __init__(self, validation_data, batch_size=MemoryConfig.METRIC_BATCH_SIZE):
        super().__init__()
        self.validation_data = validation_data # 검증 데이터 저장
        self.batch_size = batch_size # 배치 크기 설정
        self.metrics = TrainingMetrics() # 메트릭 계산 클래스 초기화
        self.metrics_to_track = EvaluationConfig.TRAINING_METRICS # 추적할 메트릭
        self.history = {metric: [] for metric in self.metrics_to_track} # 메트릭 기록
        self.best_loss = float('inf')
        self.best_metrics = {}

        # 메모리 최적화를 위한 추가 설정
        self.gc_frequency = MemoryConfig.GC_FREQUENCY
        self.use_mixed_precision = MemoryConfig.USE_MIXED_PRECISION

        # 혼합 정밀도 설정
        if self.use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        
    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """설정된 메트릭들을 배치 방식으로 계산"""
        metric_values = {}
        for metric_name in self.metrics_to_track:
            if hasattr(self.metrics, metric_name):
                metric_func = getattr(self.metrics, metric_name)
                if metric_name in ['rmse', 'mae', 'binary_crossentropy']:
                    # 배치 처리가 필요한 메트릭
                    value = metric_func(y_true, y_pred, self.batch_size)
                else:
                    # AUC나 AP같은 전체 데이터가 필요한 메트릭
                    value = metric_func(y_true, y_pred)
                metric_values[metric_name] = float(value)
        return metric_values

    def _save_metrics(self, epoch: int, metrics: Dict[str, float]):
        """메트릭 결과를 파일로 저장"""
        if not LoggingConfig.SAVE_METRICS:
            return

        os.makedirs(METRIC_SAVE_PATH, exist_ok=True)

        # 현재 시간을 파일명에 포함
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            METRIC_SAVE_PATH, f"metrics_epoch_{epoch}_{timestamp}.json"
        )

        # 메트릭 저장
        with open(filename, "w") as f:
            json.dump(
                {"epoch": epoch, "timestamp": timestamp, "metrics": metrics},
                f,
                indent=2,
            )

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % LoggingConfig.LOG_INTERVAL != 0:
            return

        # 메모리 정리
        if (epoch + 1) % self.gc_frequency == 0:
            import gc
            gc.collect()

        X_val, y_val = self.validation_data
        y_pred = self.model.predict([X_val[:, 0], X_val[:, 1]])

        # 메트릭 계산
        current_metrics = self._calculate_metrics(y_val, y_pred)

        # 히스토리 업데이트
        for metric_name, value in current_metrics.items():
            self.history[metric_name].append(value)

        # 현재 손실값
        current_loss = logs.get("val_loss", float("inf"))

        # 베스트 모델 체크
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_metrics = current_metrics

            # 베스트 메트릭 저장
            if LoggingConfig.SAVE_BEST_ONLY:
                self._save_metrics(epoch, current_metrics)
        elif not LoggingConfig.SAVE_BEST_ONLY:
            # 모든 에포크의 메트릭 저장
            self._save_metrics(epoch, current_metrics)

        # 로그 출력
        print(f"\nEpoch {epoch + 1} Validation Metrics:")
        for metric_name, value in current_metrics.items():
            print(f"{metric_name}: {value:.4f}")

    def get_best_metrics(self) -> Dict[str, float]:
        """최상의 성능을 보인 메트릭들을 반환"""
        return self.best_metrics


def create_callbacks(model_path: str, validation_data) -> list:
    """모델 학습을 위한 콜백 함수들을 생성"""

    def lr_schedule(epoch):
        """학습률 스케줄링 함수
        1. Warmup (RAMPUP_EPOCHS): 학습률을 천천히 증가
        2. Hold (SUSTAIN_EPOCHS): 최대 학습률 유지
        3. Decay: 지수적으로 감소
        """
        if epoch < RAMPUP_EPOCHS:
            # Warm-up 기간: 선형적으로 증가
            return START_LR + (MAX_LR - START_LR) * (epoch / RAMPUP_EPOCHS)
        elif epoch < RAMPUP_EPOCHS + SUSTAIN_EPOCHS:
            # 최대 학습률 유지 기간
            return MAX_LR
        else:
            # 지수적 감소 기간
            decay_epoch = epoch - RAMPUP_EPOCHS - SUSTAIN_EPOCHS
            return MAX_LR * (EXP_DECAY ** decay_epoch) + MIN_LR

    # 모델 저장 경로 생성
    os.makedirs(model_path, exist_ok=True)
    
    callbacks = [
        # 모델 체크포인트 저장
        ModelCheckpoint(
            filepath=os.path.join(model_path, "model_{epoch:02d}_{val_loss:.4f}.h5"),
            save_best_only=LoggingConfig.SAVE_BEST_ONLY,
            monitor="val_loss",
            mode="min",
        ),
        
        # 학습률 조정
        LearningRateScheduler(
            lr_schedule, 
            verbose=1
        ),
        
        # 조기 종료
        EarlyStopping(
            monitor="val_loss",
            patience=EvaluationConfig.EARLY_STOPPING_PATIENCE,
            min_delta=EvaluationConfig.EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),
        
        # 메모리 정리를 위한 콜백
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: gc.collect() 
            if (epoch + 1) % MemoryConfig.GC_FREQUENCY == 0 else None
        ),
        
        # 커스텀 메트릭 콜백
        MetricsCallback(
            validation_data,
            batch_size=MemoryConfig.METRIC_BATCH_SIZE
        ),
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
    X_train, y_train = train_data
    X_test, y_test = test_data
    callbacks = create_callbacks(MODEL_SAVE_PATH, (X_test, y_test))

    # 모델 학습
    print("\n모델 학습 시작...")
    history = model.fit(
        [X_train[:, 0], X_train[:, 1]],
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
        callbacks=callbacks,
    )

    # 메트릭 기록 가져오기
    metrics_history = None
    for callback in callbacks:
        if isinstance(callback, MetricsCallback):
            metrics_history = callback.history
            break

    return model, history, metrics_history
