# 경로 설정
INPUT_DIR = "./data/anime-2020"
MODEL_SAVE_PATH = "./save/models"
EMBEDDING_SAVE_PATH = "./save/embeddings"
METRIC_SAVE_PATH = "./save/metrics"  # 메트릭 저장 경로 추가

# 데이터 로드 파라미터
SAMPLE_SIZE = 10000

# 모델 파라미터
EMBEDDING_SIZE = 128
BATCH_SIZE = 4096
EPOCHS = 2

# 학습률 스케줄링 파라미터
START_LR = 0.00001
MIN_LR = 0.000001    # 더 작은 값으로 조정
MAX_LR = 0.0001      # 약간 증가
RAMPUP_EPOCHS = 3    # 더 짧게 조정
SUSTAIN_EPOCHS = 5   # 추가
EXP_DECAY = 0.85     # 약간 증가

# TPU 설정
TPU_INIT = False

# Memory 최적화 설정
class MemoryConfig:
    METRIC_BATCH_SIZE = 4096  # 메트릭 계산을 위한 배치 크기
    USE_MIXED_PRECISION = True  # 혼합 정밀도 사용
    SHUFFLE_BUFFER_SIZE = 10000  # 데이터 셔플링을 위한 버퍼 크기
    GC_FREQUENCY = 5  # 가비지 컬렉션 빈도 (에포크 단위)

# 평가 관련 설정
class EvaluationConfig:
    # 일반적인 평가 설정
    EVALUATION_K = 10  # Top-K 추천 수
    N_TEST_USERS = 50  # 테스트할 사용자 수
    MIN_RATING_THRESHOLD = 7  # 긍정적 평가 기준

    # 데이터 로딩 설정
    MIN_RATINGS = 50  # 사용자당 최소 평가 수
    TEST_SIZE = 0.2  # 테스트 데이터 비율
    RANDOM_STATE = 42  # 랜덤 시드

    # 메트릭 설정
    TRAINING_METRICS = [
        "rmse",
        "mae",
        "binary_crossentropy",
        "auc",
        "average_precision",
    ]

    RECOMMENDATION_METRICS = [
        "precision",
        "recall",
        "ndcg",
        "map",
        "diversity",
        "novelty",
        "personalization",
    ]

    # 조기 종료 설정
    EARLY_STOPPING_PATIENCE = 5  # 3에서 5로 증가 (20 에포크에 맞춰 조정)
    EARLY_STOPPING_MIN_DELTA = 0.0005  # 더 작은 값으로 조정


# 로깅 설정
class LoggingConfig:
    SAVE_METRICS = True  # 메트릭 저장 여부
    LOG_INTERVAL = 2  # 에포크마다 로깅
    SAVE_BEST_ONLY = True  # 최상의 모델만 저장
    CHECKPOINT_FREQ = 5  # 체크포인트 에포크 저장 주기
