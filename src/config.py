# 경로 설정
INPUT_DIR = "./anime-2020"
MODEL_SAVE_PATH = "./save/models"
EMBEDDING_SAVE_PATH = "./save/embeddings"

# 데이터 로드 파라미터
SAMPLE_SIZE = 1000000

# 모델 파라미터
EMBEDDING_SIZE = 128
BATCH_SIZE = 10000
EPOCHS = 3

# 학습률 스케줄링 파라미터
START_LR = 0.00001
MIN_LR = 0.00001
MAX_LR = 0.00005
RAMPUP_EPOCHS = 5
SUSTAIN_EPOCHS = 0
EXP_DECAY = 0.8

# TPU 설정
TPU_INIT = False
