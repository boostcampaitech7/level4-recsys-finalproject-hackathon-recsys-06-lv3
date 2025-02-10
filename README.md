# TEAM .fillna Hackathon Wrap-Up

## TVING : Cold-Start problem on SASRec

### 프로젝트 구조

<details>
    <summary> 프로젝트 코드 구조</summary>

```bash
├── src # AI 모델 학습을 위한 부분
│   ├── config # config.yaml 값 가져 오는 함수 제공
│   ├── model # AI 모델 src ex) Light GBM, XGBoost
│   └── pre_process # 모델 학습전 전처리
│   └── custom_wandb
│   └── plot 
│   └── server 
├── data #.gitignore
│   └── .csv #.gitignore
│     └── processed # 기타 csv 저장을 위한 저장소
|     └── raw # 원본 csv 저장을 위한 저장소
├── EDA # 개인 EDA 폴더
│   └── {팀원 명} 
│        ├──*.ipynb
├── app.py # 모델 학습을 위한 python 파일
├── config-sample.yaml # 하이퍼 파라미터 및 모델 & 서버 선택을 위한 설정 값
├── .gitignore
├── Readme.md
└── requirements.txt
```

</details>
<details>
    <summary> 라이브러리 버전</summary>

**Python 버전 : 3.12.5**

**Library 정보** - (requirements.txt)

```txt
mlflow
torch
torchvision
pandas
numpy
scikit-learn
pyyaml
python-dotenv
black
isort
flake8
jupyter
pynvml
tqdm
boto3
transformers
sentence-transformers
clearml
hydra-core
omegaconf
pytorch-lightning
#conda install -c conda-forge scikit-surprise
recommenders
```
</details> 
<details>
<summary> Server Structure </summary>

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



<aside>

## 목차

1. 프로젝트 소개
2. Dataset
3. Model
4. Train/Valid/Test Split & Metrics
5. Preprocessing
6. Cold Start 문제 정의
7. Cold Start 개선 전략
8. 결론
9. Appendix

---

# 1. 프로젝트 소개

- 프로젝트 기간 :  2025/01/10 ~ 2025/02/07
- 프로젝트 주제 :  SASRec에서 발생하는 Cold Start 문제 정의 및 개선
- 데이터셋 : [MovieLens-20M](https://grouplens.org/datasets/movielens/20m/)
- [프로젝트 GitHub Link](https://github.com/boostcampaitech7/level4-recsys-finalproject-hackathon-recsys-06-lv3)
- 프로젝트 개요

> NAVER Connect Foundation Boostcamp AI Tech에서 진행한 **TVING** 기업 해커톤 </br>
MovieLens-20M 데이터셋으로 SASRec 기반 추천시스템을 개발 후 </br>
모델에서 발생한 Cold-Start 문제 정의, 개선 전략 고안 및 실험 결과를 담은 프로젝트

---

## 프로젝트 진행 과정

1. 프로젝트 주제에 맞는 Public Dataset 선정
2. Dataset에 맞는 Baseline Model 탐색
3. 프로젝트 기본 구조 수립 및 코드 작성
4. 파트 별 역할 분담 진행
5. Cold-Start  실험 설계 및 문제 정의
6. Cold-Start  문제 개선 시도
7. 결과 정리 및 문서화

---
# 2. Dataset
## 2-1. 데이터셋 선정 조건

>- User-Item interaction 존재하는 추천 데이터
>- User 수 10만, Item 수 1만 이상
>- 저작권 관련 Public한 데이터
>- Side Information이 있거나, 없더라도 크롤링해서 구할 수 있어야 함
>- Sparsity가 지나치게 높거나 데이터의 크기가 너무 큰 데이터 셋은 제외


- 조건을 만족하는 데이터셋 중 **MovieLens-20M** 선정 ✅
- 데이터셋 링크 : https://grouplens.org/datasets/movielens/20m/

## 2-2. 데이터셋 구성

### 데이터셋 파일 구성

| 파일 명 | 행 수 | 주요 내용 |
| --- | --- | --- |
| ratings.csv | 20,000,263 | 사용자 시청 이력 (userId, movieId, rating, timestamp) |
| movies.csv | 27,278 | 영화 정보 (movieId, title, genres) |
| tags.csv | 465,548 | 영화에 대한 태그 정보 (userId, movieId, tag, timestamp) |
| links.csv | 5,905 | 영화 id와 매칭되는 tmdb, imdb의 id(movieId, imdbId, tmdbId) |
| genome-score.csv | 11,307 | 영화-태그 관련성 데이터 (movieId, tagId, relevance) |
| genome-tags.csv | 15,934 | genome tag id에 대한 tag 내용(tagId, tag) |

## 2-3. 기본 EDA

### ratings.csv
- Interaction: 20,000,263개 / User: 138,493명 / Item: 26,744편  / Data Sparsity = 99.46%
- rating : 0.5~5.0까지의 Explicit 데이터/ timestamp : Unix timestamp 형식
- 기간 : 1995-01-09 11:46:44 (789652004) ~ 2015-03-31 06:40:02 (1427784002)
- 전체 평점 분포

![image.png](https://rigorous-shoemaker-76b.notion.site/image/attachment%3A041938f9-29bf-4a20-a55a-613f5223b23c%3Aimage.png?table=block&id=a772327c-5e7a-439c-b563-5aee81f40953&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=2000&userId=&cache=v2)

- User별 평균 144개, 중앙값 68개, 최대 9,254개, 최소 20개 평점 등록

![image.png](https://rigorous-shoemaker-76b.notion.site/image/attachment%3A6beded44-d0f0-4fb7-971f-674ba3643f27%3Aimage.png?table=block&id=6b09f458-4e31-4b87-9ccf-8a31e300b0d2&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=2000&userId=&cache=v2)

- Item별 평점 개수는 평균 747개, 최대 67,310개(펄프 픽션), 최소 1개(영화 3,972편)

### movies.csv

- 27,278개 Item의 제목과 장르 정보
- Year(연도) 정보가 Title에 포함되어 있어 전처리시 Year 피처 생성
- Genre는 다중 레이블 형태로, 단일 Item이 여러 장르 보유
- Genre 개수는 20개
- Genres에 ‘(no genres listed)’ 존재해 ‘Unknown’으로 처리
- Drama, Documentary가 양의 상관관계가 보이고, Horror는 음의 상관관계가 보이나 
전반적으로 장르와 평점의 상관관계는 크지 않음

![image.png](https://rigorous-shoemaker-76b.notion.site/image/attachment%3A92b87bde-181a-46c7-adfa-1305a191220f%3Aimage.png?table=block&id=008bcb89-cdc7-485c-9cc3-a7c33ca6b8c3&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=1060&userId=&cache=v2)

---

# 3. Model

## 3-1. Baseline 모델 계열 선정 기준

- Baseline 선정을 위해 ML-20m 데이터 셋을 여러 계열의 RecSys 모델로 실험

| **모델** | **특징** | **장점** | **단점** |
| --- | --- | --- | --- |
| **SASRec** | Self-Attention Based | 병렬 처리 가능, 장/단기 패턴 동시 모델링, Item 간 관계 학습 효과적 | 높은 메모리 사용량, 긴 sequence에서 계산 복잡도 증가 |
| **BERT4Rec** | Bidirectional Transformer | 양방향 문맥 활용, 복잡한 패턴 포착 우수, 마스킹 기반 효과적 학습 | 높은 계산 비용, 긴 학습 시간, 추천 상황과 학습 목적 불일치 |
| **GRU4Rec** | RNN Based | 구조가 단순하여 구현 용이, 메모리 효율적, 순차 패턴 학습 특화 | 느린 학습 속도, 장기 패턴 포착 한계, 병렬화 어려움 |
| **NCF** | CF Based | 구조가 단순, 정적 협업 필터링에 효과적, 계산 비용이 상대적으로 낮음 | 부가 정보를 활용하지 못해, 추가 정보를 활용하는 문제에서 제한적 |
| **KGAT** | GNN Based | 전체적인 Item의 특성 반영으로 인해, 타 모델 대비 우수한 성능 | 높은 계산 비용 및 지나치게 긴 학습 및 추론 시간 |
- Movie Lens의 timestamp를 활용해 User의 순차적인 상호작용을 모델링하기 위해 
**Sequential 계열의 SASRec을 선택**

## 3-2. Baseline 선택

- Baseline : `SASRec`
    - 데이터셋에서 병렬 처리가 가능
    - 장·단기 패턴을 동시에 효과적으로 모델링 가능
    - **효율성과 성능** 측면에서 **가장 유리**
    - Next-Item prediction task와 학습 목적이 일치하여 **실시간 추천 시스템 구축에 적합**

### SASRec 핵심 구조

- **Self-Attention 기반 아키텍처**: SASRec는 Transformer의 self-attention 매커니즘을 활용하여, 사용자 행동 Sequence 내의 각 Item 간 관계를 동적으로 파악
- **순차적 패턴 캡처:** 사용자의 과거 interaction들을 분석하여, 순차적 패턴과 장기/단기 의존성을 효과적으로 학습
- **Positional Embedding**: 입력 Sequence의 순서 정보를 보존하기 위해 positional embedding을 사용, 각 행동의 상대적 위치 정보를 모델에 전달
- **병렬 처리 효율성:** RNN과 달리, self-attention 구조 덕분에 Sequence 데이터를 병렬 처리할 수 있어 training 및 inference 속도가 향상
- **Next-Item Prediction**: 학습된 사용자 행동 패턴을 기반으로, 다음에 소비될 가능성이 높은 Item을 예측하여 개인화된 추천 리스트를 생성

### SASRec 대비 타 모델의 한계

- **BERT4Rec**
    - BERT4Rec은 양방향 인코더 구조로 인해 **더 많은 파라미터와 연산**이 필요
    - 대용량 데이터 셋에서는 `BERT4Rec`성능이 좋지만,  ML-20m에는 **적합하지 않음**
    - 현업에서는 마지막 Item 이후의 다음 Item을 예측하는 것이 목적
    → **단방향 인코더**인 `SASRec`이 **더 효율적**이라 판단
- **KGAT**
    - GNN 기반 모델로 우수한 성능을 가졌으나, 타 모델 대비 **매우 많은 리소스 필요**
    - 서버의 단일 GPU에서 사용하기 위해선 과도한 Dataset Slicing 필요
    - Baseline으로 사용하기에는 **실험이 매우 제한적**

---

# 4. Train/Valid/Test Split & Metrics

## 4-1. Leave-One-Out Split (LOO)

- Timestamp가 존재하는 sequential 문제이므로 data leakage가 발생하면 안됨
- User-independent 모델이 아니기 때문에 User Split 방식 불가
- 각 User sequence 별로 가장 최근 상호작용 Item을 Test, 그 이전 Item을 Validation으로 지정하는 
Leave-One-Out(이후 LOO) 방식 채택

## 4-2. Metrics

- LOO 상황에서는 User당 정답이 1개이기 때문에 
Hitrate@k = Recall@k = 10 x Precision@k의 관계가 성립
그래서, 의미에 가장 부합하는 Hitrate@k를 사용
- 정답이 모델의 추천 리스트 어디에 있는 지도 고려해야 하므로 NDCG@k와 MRR@k도 사용
단, LOO에서는 정답이 하나이므로 기존 방식과는 다르게 동작

| 정답 순위 | NDCG@10 | NDCG@20 | MRR@10 |
| --- | --- | --- | --- |
| 1 | 1.000000 | 1.000000 | 1.000000 |
| 2 | 0.630930 | 0.630930 | 0.500000 |
| 3 | 0.500000 | 0.500977 | 0.333333 |
| 4 | 0.430677 | 0.420821 | 0.250000 |
| 5 | 0.386853 | 0.365927 | 0.200000 |
- 실험 환경에서는 추천 size(k) 10, 20, 50, 100  지표 확인
- 실험 결과에서 지표 비교는 Metric@10을 사용

---

# 5. Preprocessing

## 5-1. Explicit Feedback을 Implicit Feedback으로 변환

- 사용자의 **모든 상호작용**을 동등한 **implicit interaction**으로 변환
- **일정 rating 이상만 긍정적 상호작용**으로 처리해서 사용하는 방식은 설정 rating 미만의 행동 순서 패턴이 사라지기 때문에 User Seqeunce가 중요한 Sequential 모델에는 **부적합하다고 판단**

## 5-2. 최근 15년 동안 상호작용이 있었던 User로 한정

- 마지막 상호작용이 2000년 이전인 User들의 분포와 성능 편차가 커서 균일한 실험환경 조성을 위해 최근 15년 동안 상호작용이 있었던 User로 Dataset을 한정

![image.png](https://rigorous-shoemaker-76b.notion.site/image/attachment%3A12f743be-8df9-4f47-900d-a733fe46af41%3Aimage.png?table=block&id=1956ac11-5bd2-808b-af07-e2c4d8320c92&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=1420&userId=&cache=v2)

## 5-3. User의 최근 상호작용을 일부 제거해 Cold Start User 생성

- 원본 데이터는 User의 최소 상호작용 개수가 20개 → **Cold Start User 부재**
- User 중 일부를 Random Sampling해 상호작용 일부를 제거하면 아래의 그래프처럼 random seed에 따라 실험의 편차가 발생해 부적합하다고 판단

![image.png](https://rigorous-shoemaker-76b.notion.site/image/attachment%3Ac34b400e-9034-49d1-ade8-98f2b0ac49a6%3Aimage.png?table=block&id=86d6ba58-bffb-4d32-9d19-e8cd1bf902a5&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=2000&userId=&cache=v2)

- **적용:** 균일한 실험 결과를 위해 전체 User에 대해 최근 상호작용을 일부 제거해 상호작용 ****20개 미만의 Cold Start User 생성

## 5-4. 상호작용이 5개 이하인 Item 제거

- 본 프로젝트에서는 User Cold Start 문제에 집중하기 위해,  Item Cold Start 문제로 인한 상황 배제
- 상호작용이 5개 이하인 Item(영화) 9,251개 / 상호작용 20,460행 제거
Data Sparsity 99.46 → 99.09%

---

# 6. Cold Start 문제 정의

## 6-1. 기본 조건

<aside>

- Cold Start Item을 배제한 Cold Start User
- train 상호작용이 아예 없는 Strict Cold Start 상황 배제
- User당 최소 상호작용은 3개(train/valid/test에 최소 1개씩 필요)
</aside>

## 6-2. Cold Start 정의

![image.png](https://rigorous-shoemaker-76b.notion.site/image/attachment%3Aba1df151-48e6-49e2-a7df-1f9bf0b5251e%3Aimage.png?table=block&id=909ef489-67a8-4393-a0f6-617fbc9b10dc&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=2000&userId=&cache=v2)

- Count  = User당 상호작용 수(Sequence 길이)
- 여기서 Count는 User의 전체 상호작용 수로 train+valid+test 데이터 모두 포함
- Count를 X축으로 봤을 때 지표상 성능이 급격히 향상되는 지점 확인

<aside>

> **Cold User** = Count가 16개 이하인 User </br>
**Warm User** = Count가 17개 이상인 User
> 
</aside>

- 지표

|  | Warm User Metrics | Cold User Metrics | Metrics Drop Rate (%) |
| --- | --- | --- | --- |
| HitRate@10 | 0.280211 | 0.181410 | -35.26% |
| nDCG@10 | 0.152828 | 0.092474 | -39.49% |
| MRR@10 | 0.114237 | 0.065764 | -42.43% |

---

# 7. Cold Start 개선 전략

## 7-1. Loss Function 변경(BCE → CE)

- 가설 : SASRec의 Loss Function을 Binary Cross Entropy에서 Cross Entropy로 변경
    - 현재 Task는 User가 특정 Item 소비 예측인 Binary Classification이 아닌, 다수의 Item 중 어떤 Item의 소비 가능성이 가장 클 것이냐를 예측하는 Multi-Class Classification Problem
    - 따라서 모델이 전체 Item에 대해 Positive Item의 상대적 우선순위를 학습하는 것이 전반적인 모델의 성능을 향상시킬 것이라 가정

$$
\mathcal{L}_{BCE} = -\sum_{u \in U} \sum_{t=1}^{n_u} \log(\sigma (r_{t,i_t}^{(u)})) + \log(1-\sigma(r_{t,-}^{(u)}))
\\
\mathcal{L}_{CE} = -\sum_{u \in U} \sum_{t \in T_u} \log \frac{\exp(r_{t,i_t}^{(u)})}{\sum_{i \in I} \exp(r_{t,i}^{(u)})}
\\
$$

- 방법 : Baseline인 BCE를 적용한 SASRec과 CE를 적용한 SASRec을 같은 실험 환경에서 진행
- 결론

![image.png](https://rigorous-shoemaker-76b.notion.site/image/attachment%3A6f5808f3-2314-4cfc-9103-7fd01cac0648%3Aimage.png?table=block&id=1956ac11-5bd2-8052-9062-ff8e6e3635fe&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=2000&userId=&cache=v2)

![image.png](https://rigorous-shoemaker-76b.notion.site/image/attachment%3Afe69fbb7-1902-4ffd-a341-6fb7e00534ba%3Aimage.png?table=block&id=1946ac11-5bd2-80bd-b5d5-c64f6fa57d08&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=2000&userId=&cache=v2)

- 전반적인 성능지표 향상 및 hitrate@10에서 
Warm User 성능 향상(35.7%) 대비 Cold User에서의 더 큰 성능(80.6%) 향상
- 그러나 Epoch당 train time이 42s에서 77s로 증가하는 trade-off 발생

## 7-2. Negative Sampling Pool 생성

- 가설 : CE를 모든 item에 대해 계산하지 않아도, Negative Sampling Pool이 충분히 크면 그 안에서 CE를 계산하는 것이 Performance-Cost 사이의 trade-off를 줄일 수 있을 것이라 가정

$$
\mathcal{L}_{CE-sampled} = -\sum_{u \in U} \sum_{t=1}^{n_u} \log \frac{\exp(r_{t,i_t}^{(u)})}{\exp(r_{t,i_t}^{(u)}) + \sum_{i \in I^{-(u)}_N} \exp(r_{t,i}^{(u)})}
$$

- CE SASRec:
    - Non-User Interaction Item에 대해, 1000개 단위의 Uniform Random Sampling을 통한 N.S Pool 생성

![image.png](https://rigorous-shoemaker-76b.notion.site/image/attachment%3A7f09e3c1-32f1-4e4b-8a4f-c4c1023d0847%3Aimage.png?table=block&id=1946ac11-5bd2-803d-8542-d6ba23c4353e&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=2000&userId=&cache=v2)

| **hitrate@10** | **CE** | **CE 2000** | **CE 3000** | **CE 5000** |
| --- | --- | --- | --- | --- |
| **Cold User** | 0.3268 | 0.3170 | **0.3344** | 0.3298 |
| **Warm User** | 0.3802 | 0.3939 | **0.4043** | 0.3989 |
| Time/Epoch | 77 | 55 | 59 | 69 |
- 결론
    - N.S Pool 3000에서 성능이 가장 높게 나왔고, Epoch 당 학습 시간도 CE대비 23% 감소
    - N.S Pool을 이용해서 CE를 계산하는 방식이 Cost & Performance 측면에서 효율적

## 7-3. Negative Sampling 방식 변경

- 가설 : N.S Pool을 Uniform Random Sapling에서 Popularity based로 바꾸면 성능에 긍정적 영향
- 기반 : Top Rank Negative Sample, Popularity Based Negative Sample
- 방법 : 총 9가지 방법**(Rank / Rank-Probability / Count-Probability) X (Top / Mid / Bot)**
    - **Rank** : 인기도 랭킹 기반 위치(Top, Mid, Bot 중 선택된)로 고정해 순서대로 뽑는 방법
    - **Rank-Prob** : 인기도 랭킹 기반 확률 분포를 활용하여 R.S
    - **Count-Prob** : Item Interaction Count 기반 확률 분포 활용하여 R.S
    - **[Top, Mid, Bot]** : Top - 상위값 우선순위 , Mid - 중간값 우선순위, Bot- 하위값 우선순위
- 실험 결과 (환경 - SASRec CE N.S size 3000)

| NDCG@10 | R.S 3000 | R-Mid | C-P-Mid | R-P-Mid | R-P-Top | R-P-Bot |
| --- | --- | --- | --- | --- | --- | --- |
| Cold User | 0.1874 | 0.0048 | 0.1808 | **0.1924** | 0.0880 | 0.0892 |
| Warm User | 0.2460 | 0.0058 | 0.2356 | **0.2487** | 0.1412 | 0.1393 |

![image.png](https://rigorous-shoemaker-76b.notion.site/image/attachment%3A1ede7d61-2ffd-4e01-a495-5c0db86e9a5c%3Aimage.png?table=block&id=1956ac11-5bd2-80a5-9bd7-e060734b55c9&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=1020&userId=&cache=v2)

- Rank : Top, Mid, Bot 모두에서 성능 대폭 하락
- Rank-Prob : **Mid에서 2.7%의 성능 향상**
- Count-Prob : Mid에서 성능 소폭 하락
- N.S size에 따라 최대 5.8%(size 2000)까지 성능 향상이 확인되지만,  Size가 커지면 R.S와 성능에 차이가 없어짐
- 결론
    - Popularity Based NS에서도 Distance가 Mid인 Sampling 기법이 성능에 **긍정적 영향**을 끼침
    - 다만 Sample size가 커지면 R.S를 통한 N.S와 큰 차이가 없어 성능 향상이 관측 되지 않음(N>5000)

## 7-4. Positive Augmentation

- 가설 : Cold Start User의 Sequence Augmentation으로 성능 개선
    - N.S처럼 Positive Sample도 생성해서 User의 Interaction Sequence를 늘리면 성능에 긍정적 영향
- 방법 : Contents Based(C.B) 방식으로 유사한 Item 증강
    - side info(title, genres, year, tags)를 활용하여 C.B 유사도 계산
    - Cold Start User 기존 Item마다 n개씩 증강 후 Sequence 구성
    - Data leakage를  방지하기위해  train 데이터만 증강 수행
- 실험 : item 증강 위치 수정, n값 변경
    - 증강 Item들을 앞에 두고 기존 Item을 뒤에 배치
    - 기존 Item들의 증강 Item들을 각 Item의 앞쪽에 배치
    - 기존 Item들의 증강 Item들을 각 Item의 뒤쪽에 배치
    - n 값 변경 (1, 3, 5, 10)
- 실험 결과 : 모든 실험에서 Baseline 성능에 비해 하락

![image.png](https://rigorous-shoemaker-76b.notion.site/image/attachment%3A7f7539ff-ec29-474d-92a6-5959662643a5%3Aimage.png?table=block&id=1946ac11-5bd2-80bb-8c83-fe8c3943e98b&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=1420&userId=&cache=v2)

- 결론
    - Sequential 모델에서는 User가 상호작용한 Item의 순서가 중요한데, 임의적인 증강 item 배치가 Cold Start User 학습에 Noise로 작용해 성능이 하락한 것으로 추론
    - Positive한 상호작용만 한 것이 아니라 모든 상호작용을 대상으로 했기 때문에 부정적인 상호작용 item(rating 3점 이하)를 증강할 경우 Noise가 될 여지가 있음

## 7-5. SIMRec

- 가설 : Loss function에 Item간의 Similarity가 반영되면 성능 개선이 있을 것이다.
- 기반 : Amazon-Science의 SIMRec
- 방법 : SIMRec 논문에 따라, Title Textual 기반의 Similarity 계산(by Sentence Transformer)
- 실험 : SASRec Baseline + Rank Prob(Mid) 환경에서 SIMRec 적용

![image.png](https://rigorous-shoemaker-76b.notion.site/image/attachment%3Aa2695816-f9df-481c-aa23-204eb3364d12%3Aimage.png?table=block&id=1956ac11-5bd2-8022-a926-fd8bd59acdde&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=2000&userId=&cache=v2)

- 결론 : **Drop** - 훈련 시간이 매우 길어지면서, 성능도 기존 Loss Function에 비해 개선 없음
- 제언 :
    - SIMRec을 Loss Function이 아닌 N.S 방식으로 도입
    - Title Textual 대신 Other Item Similarity Based(Scenario, Director, Actor, Multimodal,  …) 도입

---

# 8. 결론

## 8-1. 실험 결과 요약

<aside>

본 프로젝트에서는 Cold Start 문제를 해결하기 위한 다양한 접근법을 실험하였으며, 그 결과 주요 성과를 다음과 같이 정리할 수 있다.

1. **Loss Function 변경(BCE → CE)**
    - 기존 BCE Loss 대신 CE Loss를 적용함으로써 Cold Start User의 성능을 80.6% 향상시킴
    - Warm User 대비 Cold User에서의 성능 향상이 더욱 두드러짐
    - 단, CE Loss 적용 시 학습 시간이 증가(42s → 77s per epoch)하는 trade-off 발생
2. **Negative Sampling Pool 활용**
    - CE를 전체 아이템에 대해 계산하는 대신 적절한 Negative Sampling Pool을 구성
    - 성능의 하락 없이 비용적 측면에서 더욱 효율적인 학습이 가능해짐
    - N.S Pool 크기를 3000으로 설정했을 때, 성능과 학습 속도의 균형이 가장 적절함
3. **Negative Sampling 방식 개선**
    - N.S Pool을 Random Sampling 보단 Rank-Prob-Mid으로 구성하는 게 성능이 1~6% 더 높게 측정 됨
    - N.S Pool의 Size가 커질수록 R.S와의 차이는 미미함
</aside>

## 8-2. 한계점 및 추가 연구 방향

- **MovieLens Dataset의 특성에 따른 Online 적용 어려움**
    - Rating 기록은 User가 리뷰를 남긴 시점이지, 실제로 소비한 시점이 아님. 
    즉, online 상황과 매우 다르며, rating 기록 자체도 Bulk 형식으로 짧은 시간 혹은 동일 시점에 다량의 평가가 기록되어 있음
    - User별 최소 상호작용 수가 20으로 보장되어 있어 Cold Start User가 존재하지 않음. 따라서, 임의로 만든 Cold Start User는 실제와 다를 수 있음
- **데이터 및 모델의 일반화 가능성**
    - MovieLens-20M 데이터셋을 활용하여 실험을 진행했으나, 다른 데이터셋에서도 동일한 개선 효과를 보장할 수 있는지 추가 검증 필요
    - 다양한 사용자 행동 패턴을 반영할 수 있도록 도메인별 데이터셋을 활용한 추가 실험이 필요함
- **Item Popularity 기반 Filtering의 효과**
    - 상호작용이 적은 아이템을 거르는 Threshold를 높이면 Metric 지표가 향상
    → 인기없는 아이템을 배제할수록 정확도는 증가
    - 하지만 다양성(coverage, serendipity) 측면에서 trade-off 발생하므로
    추천 시스템의 목적에 따라 Threshold 적용 수치를 신중히 고려해야 함

![image.png](https://rigorous-shoemaker-76b.notion.site/image/attachment%3A73d4a982-50dd-41f4-a37d-7a94e2e0ebe4%3Aimage.png?table=block&id=1956ac11-5bd2-807f-8322-fbe7983cf8ad&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=2000&userId=&cache=v2)

---

# 9. Appendix

## 9-1. 팀 구성 및 역할

| 이름 | 역할 |
| --- | --- |
| [김건율](https://github.com/ChoonB) | 팀장, 데이터 분석, 전처리, 프로젝트 실험 및 코드 관리, N.S Pool 생성 |
| [백우성](https://github.com/13aek) |  |
| [유대선](https://github.com/xenx96) | GNN 계열 모델 & 샘플링 기법 개발, 실험 관리 및 추적, 프로젝트 구조 및 코드 컨벤션 관리 |
| [이제준](https://github.com/passi3) | EDA, 전처리, 기본 추천 모델링 및 테스트, 가설 검토 및 판단 |
| [김성윤](https://github.com/Timeisfast) |  |
| [박승우](https://github.com/zip-sa) |  |

## 9-2. 협업 방식

- Slack : 팀내 실시간 커뮤니케이션, Github 연동 이슈 공유, 허들에서 실시간 소통 진행
- Zoom : 정기 회의 진행 시 사용
- Notion : 토론 및 합의 내용, 팀 활동 관련 기록
- GitHub : 버전 관리와 코드 협업을 목표로 사용 
각 팀원은 기능 단위로 이슈와 관련 브랜치를 생성해 작업 후, Pull Request로 코드 리뷰 후 병합
- GitHub Projects : 팀원 간 작업 분배 및 스케쥴링
- MLflow : 실험 로그 관리 및 모니터링

## 9-3. References
<details>
    <summary> References</summary>

### 9-3-1. Model

[Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)

[BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690)

[is BERT4Rec really better than SASRec?](https://arxiv.org/pdf/2309.07602)

[Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)

[KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/abs/1905.07854)

### 9-3-2. Metric/Evaluation

[Bridging Offline-Online Evaluation with a Time Dependent and Popularity Bias-Free Offline Metric for Recommenders](https://arxiv.org/abs/2308.06885)

[Synthetic Data-Based Simulators for Recommender System A Survey](https://arxiv.org/abs/2206.11338)

[The MovieLens Beliefs Dataset- Collecting Pre-Choice Data for Online Recsys](https://arxiv.org/abs/2405.11053)

[AIE  Auction Information Enhanced Framework for CTR](https://arxiv.org/abs/2408.07907)

[Do Offline Metrics Predict Online Performance in RS](https://arxiv.org/abs/2011.07931)

[Offline Evaluation of Reward-Optimizing Recommender Systems- The Case of Simulation](https://arxiv.org/abs/2209.08642)

[Towards Unified Metrics for Accuracy and Diversity for RecSys](https://dl.acm.org/doi/10.1145/3460231.3474234)

[What We Evaluate When We Evaluate Recommender Systems- Understanding Recommender Systems’ Performance using Item Response Theory](https://dl.acm.org/doi/10.1145/3604915.3608809)

### 9-3-3. Negative Sample

[Cold-Start Recommendation based on Knowledge Graph and Meta-Learning under Positive and Negative sampling](https://dl.acm.org/doi/10.1145/3654804)

[Negative Sampling in Next-POI Recommendations Observation, Approach, and Evaluation](https://dl.acm.org/doi/10.1145/3589334.3645681)

[On the Theories Behind Hard Negative Sampling for Recommendation](https://arxiv.org/abs/2302.03472)

[Revisiting Negative Sampling vs. Non-sampling in Implicit Data](https://dl.acm.org/doi/10.1145/3522672)

[Region or Global? A Principle for Negative Sampling in Graph-based Recommendation](https://ieeexplore.ieee.org/document/9723516)

[A Case Study on Sampling Strategies for Evaluating Neural Sequential Item Recommendation Models](https://arxiv.org/pdf/2107.13045)

### 9-3-4. Cold Start

[Alleviating Cold-Start Problems in Recommendationthrough Pseudo-Labelling over Knowledge Graph](https://arxiv.org/abs/2011.05061)

[Cold-Start Recommendation based on Knowledge Graph and Meta-Learning under Positive and Negative sampling](https://dl.acm.org/doi/10.1145/3654804)

</details>
