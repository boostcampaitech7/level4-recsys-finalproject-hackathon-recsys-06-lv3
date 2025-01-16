import json
import os
import platform
from datetime import datetime

import tensorflow as tf

from src.config import *
from src.data.data_loader import load_data
from src.test import test_recommendations
from src.train import create_and_train_model


def device_init():
    """사용 가능한 디바이스 확인 및 초기화"""
    print("\n=== 디바이스 초기화 ===")

    # TPU 설정이 명시적으로 요청된 경우
    if TPU_INIT:
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            print("TPU 디바이스가 설정되었습니다.")
            return strategy
        except:
            print("TPU를 찾을 수 없습니다. 다른 디바이스를 확인합니다...")

    # GPU 확인
    if len(tf.config.list_physical_devices("GPU")) > 0:
        strategy = tf.distribute.MirroredStrategy()
        print(
            f"GPU 디바이스가 설정되었습니다: {len(tf.config.list_physical_devices('GPU'))}개 발견"
        )
        return strategy

    # Apple Silicon MPS 확인
    if platform.processor() == "arm" and platform.system() == "Darwin":
        try:
            # MPS 디바이스 활성화
            physical_devices = tf.config.list_physical_devices("MPS")
            if len(physical_devices) > 0:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print("Apple Silicon MPS 디바이스가 설정되었습니다.")
                return tf.distribute.get_strategy()
        except:
            print("MPS 디바이스를 사용할 수 없습니다. CPU를 사용합니다...")

    # CPU 사용
    print("CPU 디바이스가 설정되었습니다.")
    return tf.distribute.get_strategy()


def get_device_info():
    """현재 사용 중인 디바이스 정보를 반환"""
    device_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
    }

    # GPU 정보 추가
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        device_info["gpu_count"] = len(gpus)
        device_info["gpu_devices"] = [gpu.name for gpu in gpus]
    else:
        device_info["gpu_count"] = 0
        device_info["gpu_devices"] = []

    return device_info


def save_experiment_results(
    metrics_history, evaluation_results, model_config, device_info
):
    """실험 결과를 파일로 저장"""
    if not LoggingConfig.SAVE_METRICS:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(METRIC_SAVE_PATH, f"experiment_results_{timestamp}.json")

    results = {
        "timestamp": timestamp,
        "device_info": device_info,
        "model_config": model_config,
        "training_metrics": metrics_history,
        "evaluation_results": evaluation_results,
        "config": {
            "evaluation": {
                "k": EvaluationConfig.EVALUATION_K,
                "n_test_users": EvaluationConfig.N_TEST_USERS,
                "rating_threshold": EvaluationConfig.MIN_RATING_THRESHOLD,
                "training_metrics": EvaluationConfig.TRAINING_METRICS,
                "recommendation_metrics": EvaluationConfig.RECOMMENDATION_METRICS,
            }
        },
    }

    os.makedirs(METRIC_SAVE_PATH, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n실험 결과가 저장되었습니다: {filename}")


def run_experiment():
    """전체 실험 프로세스 실행"""
    try:
        # 디바이스 초기화
        strategy = device_init()
        device_info = get_device_info()

        # GPU 메모리 증가 설정
        if device_info["gpu_count"] > 0:
            for device in tf.config.list_physical_devices("GPU"):
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except:
                    continue

        # 데이터 로드 및 전처리
        print("\n데이터 로드 및 전처리 중...")
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
        print("\n모델 학습 시작...")
        model, history, metrics_history = create_and_train_model(
            strategy, (X_train, y_train), (X_test, y_test), n_users, n_animes
        )

        # 추천 시스템 테스트 및 평가
        print("\n추천 시스템 평가 시작...")
        evaluation_results = test_recommendations(
            model, rating_df, anime_df, synopsis_df, mappings
        )

        # 결과 저장
        print("\n실험 결과 저장 중...")
        save_experiment_results(
            metrics_history, evaluation_results, model.get_config(), device_info
        )

    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        raise


def load_experiment_results(filename: str) -> dict:
    """저장된 실험 결과를 불러오기"""
    filepath = os.path.join(METRIC_SAVE_PATH, filename)
    with open(filepath, "r") as f:
        results = json.load(f)
    return results


def compare_experiments(experiment_files: list):
    """여러 실험 결과를 비교"""
    results = []
    for filename in experiment_files:
        result = load_experiment_results(filename)
        results.append(
            {
                "timestamp": result["timestamp"],
                "device_info": result["device_info"],
                "training_metrics": result["training_metrics"],
                "evaluation_results": result["evaluation_results"],
            }
        )

    print("\n=== 실험 결과 비교 ===")
    for i, result in enumerate(results, 1):
        print(f"\n실험 {i} ({result['timestamp']}):")
        print(f"사용 디바이스: {result['device_info'].get('gpu_devices', ['CPU'])}")

        # 트레이닝 메트릭 출력
        print("\n트레이닝 메트릭:")
        for metric, value in result["training_metrics"].items():
            if isinstance(value, list):
                print(f"{metric}: {value[-1]:.4f} (최종)")
            else:
                print(f"{metric}: {value:.4f}")

        # 추천 시스템 평가 결과 출력
        print("\n추천 시스템 평가:")
        for rec_type, metrics in result["evaluation_results"].items():
            print(f"\n{rec_type}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    run_experiment()

    # 실험 결과 비교 (Python 스크립트 내에서) 예시
    # experiment_files = [
    #     'experiment_results_20240116_120000.json',
    #     'experiment_results_20240116_130000.json'
    # ]
    # compare_experiments(experiment_files)
