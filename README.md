## Dataset

- [anime dataset 2023](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)
- [anime dataset 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020?select=anime.csv)


<details><summary> Server Structure </summary>

현재 서버는 총 5개로 구성되어 있습니다:
1. **Naver Cloud 서버** - ML Flow 서버
2. **AI 학습 서버 1**
3. **AI 학습 서버 2**
4. **AI 학습 서버 3**
5. **AI 학습 서버 4**

### 서버 및 스토리지 연결성 시각화

```plaintext
+---------------------+       +---------------------+
|                     |       |                     |
|  AI 학습 서버 1     +------>+                     |
|                     |       |                     |
+---------------------+       |                     |
                              |                     |
+---------------------+       |                     |
|                     |       |                     |
|  AI 학습 서버 2     +------>+                     |
|                     |       |                     |
+---------------------+       |                     |
                              |                     |
+---------------------+       |                     |
|                     |       |                     |
|  AI 학습 서버 3     +------>+ Naver Cloud 서버    |
|                     |       |  (MLFlow 서버)      +------> Naver Object Storage
+---------------------+       |                     |
                              |                     |
+---------------------+       |                     |
|                     |       |                     |
|  AI 학습 서버 4     +------>+                     |
|                     |       |                     |
+---------------------+       |                     |
                              |                     |
                              +---------------------+
````



### MLFlow Tracking 설정

각각의 AI 학습 서버들은 Naver Cloud 서버로 MLFlow Tracking을 전송합니다. 이 과정을 통해 모든 학습 과정과 결과를 중앙에서 관리할 수 있습니다.

#### 설정 방법

1. **Naver Cloud 서버**:
   - MLFlow 서버를 설정하고 실행합니다.
   - Naver Cloud의 MLFlow 서버는 자체적으로 Naver Object Storage와 연결되어 있어, 모델 및 실험 데이터를 안전하게 저장할 수 있습니다.

2. **AI 학습 서버들**:
   - 각 서버에서 MLFlow를 설치하고, `config.yaml` 파일에 Naver Cloud 서버의 `tracking_uri`를 설정합니다.
   - 학습 스크립트에서 MLFlow를 사용하여 학습 과정을 기록합니다.
</details>