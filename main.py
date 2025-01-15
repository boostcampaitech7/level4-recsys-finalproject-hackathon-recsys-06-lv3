import tensorflow as tf

from src.config import *
from src.data.data_loader import load_data
from src.test import test_recommendations
from src.train import create_and_train_model


def init_tpu():
    """TPU 초기화"""
    if not TPU_INIT:
        return tf.distribute.get_strategy()

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("TPU 초기화 완료")
        return strategy
    except:
        strategy = tf.distribute.get_strategy()
        print("TPU를 찾을 수 없습니다. CPU/GPU를 사용합니다.")
        return strategy


def main():
    """메인 실행 함수"""
    try:
        # TPU/GPU 전략 초기화
        strategy = init_tpu()

        # 데이터 로드 및 전처리
        (
            rating_df,
            anime_df,
            synopsis_df,
            (X_train, X_test, y_train, y_test),
            mappings,
        ) = load_data()

        # 모델 크기 계산
        n_users = len(mappings[0])
        n_animes = len(mappings[2])
        print(f"\n총 사용자 수: {n_users:,}, 총 애니메이션 수: {n_animes:,}")

        if n_users == 0 or n_animes == 0:
            raise ValueError("사용자 또는 애니메이션 데이터가 없습니다.")

        # 모델 학습
        model, history = create_and_train_model(
            strategy, (X_train, y_train), (X_test, y_test), n_users, n_animes
        )

        # 추천 시스템 테스트
        test_recommendations(model, rating_df, anime_df, synopsis_df, mappings)

    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        return


if __name__ == "__main__":
    main()
