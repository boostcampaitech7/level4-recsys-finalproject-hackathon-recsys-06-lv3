from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Dense, Dot, Dropout,
                                     Embedding, Flatten, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class RecommenderNet:
    def __init__(
        self,
        n_users: int,
        n_animes: int,
        embedding_size: int = 128,
        dropout_rate: float = 0.1,
        l2_reg: float = 0.01,
        learning_rate: float = 0.001,
        architecture: str = "dot_product",
    ):
        """
        추천 시스템 모델 초기화

        Args:
            n_users: 총 사용자 수
            n_animes: 총 애니메이션 수
            embedding_size: 임베딩 차원
            dropout_rate: 드롭아웃 비율
            l2_reg: L2 정규화 계수
            learning_rate: 학습률
            architecture: 모델 아키텍처 ('dot_product' 또는 'mlp')
        """
        self.n_users = n_users
        self.n_animes = n_animes
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.architecture = architecture

    def _create_base_layers(self) -> tuple:
        """
        기본 입력층과 임베딩 층 생성

        Returns:
            (user_input, anime_input, user_embedding, anime_embedding) 튜플
        """
        # 입력층
        user_input = Input(shape=(1,), name="user_input")
        anime_input = Input(shape=(1,), name="anime_input")

        # 임베딩 층
        user_embedding = Embedding(
            input_dim=self.n_users,
            output_dim=self.embedding_size,
            embeddings_regularizer=l2(self.l2_reg),
            embeddings_initializer="he_normal",
            name="user_embedding",
        )(user_input)

        anime_embedding = Embedding(
            input_dim=self.n_animes,
            output_dim=self.embedding_size,
            embeddings_regularizer=l2(self.l2_reg),
            embeddings_initializer="he_normal",
            name="anime_embedding",
        )(anime_input)

        return user_input, anime_input, user_embedding, anime_embedding

    def _build_dot_product_model(self) -> Model:
        """
        내적 기반 추천 모델 생성

        Returns:
            컴파일된 케라스 모델
        """
        (
            user_input,
            anime_input,
            user_embedding,
            anime_embedding,
        ) = self._create_base_layers()

        # 내적 계산
        dot_product = Dot(axes=2, normalize=True, name="dot_product")(
            [user_embedding, anime_embedding]
        )

        # Flatten
        x = Flatten()(dot_product)

        # Dense layers
        x = Dense(
            64, kernel_initializer="he_normal", kernel_regularizer=l2(self.l2_reg)
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(
            32, kernel_initializer="he_normal", kernel_regularizer=l2(self.l2_reg)
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(self.dropout_rate)(x)

        # 출력층
        output = Dense(1, activation="sigmoid", name="output")(x)

        return Model(inputs=[user_input, anime_input], outputs=output)

    def _build_mlp_model(self) -> Model:
        """
        MLP 기반 추천 모델 생성

        Returns:
            컴파일된 케라스 모델
        """
        (
            user_input,
            anime_input,
            user_embedding,
            anime_embedding,
        ) = self._create_base_layers()

        # Flatten embeddings
        user_flat = Flatten()(user_embedding)
        anime_flat = Flatten()(anime_embedding)

        # Concatenate embeddings
        concat = Concatenate()([user_flat, anime_flat])

        # Dense layers
        x = Dense(
            256, kernel_initializer="he_normal", kernel_regularizer=l2(self.l2_reg)
        )(concat)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(
            128, kernel_initializer="he_normal", kernel_regularizer=l2(self.l2_reg)
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(
            64, kernel_initializer="he_normal", kernel_regularizer=l2(self.l2_reg)
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(self.dropout_rate)(x)

        # 출력층
        output = Dense(1, activation="sigmoid", name="output")(x)

        return Model(inputs=[user_input, anime_input], outputs=output)

    def build(
        self,
        metrics: Optional[List[str]] = None,
        loss: str = "binary_crossentropy",
        k: int = 10,
    ) -> Model:
        """
        추천 시스템 모델 생성 및 컴파일

        Args:
            metrics: 평가 지표 리스트
            loss: 손실 함수

        Returns:
            컴파일된 케라스 모델
        """
        if metrics is None:
            metrics = [
                "mae",
                "mse",
                tf.keras.metrics.Precision(top_k=k),
                tf.keras.metrics.Recall(top_k=k),
                tf.keras.metrics.NDCGMetric(k=k),
            ]

        if self.architecture == "dot_product":
            model = self._build_dot_product_model()
        elif self.architecture == "mlp":
            model = self._build_mlp_model()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss=loss, metrics=metrics
        )

        return model

    def get_config(self) -> dict:
        """
        모델 설정 반환

        Returns:
            설정 딕셔너리
        """
        return {
            "n_users": self.n_users,
            "n_animes": self.n_animes,
            "embedding_size": self.embedding_size,
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg,
            "learning_rate": self.learning_rate,
            "architecture": self.architecture,
        }

    @classmethod
    def from_config(cls, config: dict) -> "RecommenderNet":
        """
        설정으로부터 모델 인스턴스 생성

        Args:
            config: 설정 딕셔너리

        Returns:
            RecommenderNet 인스턴스
        """
        return cls(**config)


class UserSimilarityRecommender:
    def __init__(
        self,
        user_weights: np.ndarray,
        user2user_encoded: Dict[int, int],
        user_encoded2user: Dict[int, int],
        rating_df: pd.DataFrame,
        anime_df: pd.DataFrame,
    ):
        """
        사용자 유사도 기반 추천 시스템

        Args:
            user_weights: 학습된 사용자 임베딩 가중치
            user2user_encoded: 사용자 ID -> 인코딩된 ID 매핑
            user_encoded2user: 인코딩된 ID -> 사용자 ID 매핑
            rating_df: 평점 데이터프레임
            anime_df: 애니메이션 정보 데이터프레임
        """
        self.user_weights = user_weights
        self.user2user_encoded = user2user_encoded
        self.user_encoded2user = user_encoded2user
        self.rating_df = rating_df
        self.anime_df = anime_df

    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        """
        특정 사용자의 모든 평점 정보를 조회

        Args:
            user_id: 사용자 ID

        Returns:
            해당 사용자의 평점 정보를 담은 데이터프레임
        """
        return self.rating_df[self.rating_df.user_id == user_id]

    def find_similar_users(
        self, user_id: int, n: int = 10, return_dist: bool = False
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """
        주어진 사용자와 유사한 사용자를 찾음

        Args:
            user_id: 사용자 ID
            n: 반환할 유사 사용자 수
            return_dist: 거리 값 반환 여부

        Returns:
            유사한 사용자 정보를 담은 데이터프레임 또는 (거리, 인덱스) 튜플
        """
        try:
            # 입력 사용자의 인코딩된 인덱스 찾기
            encoded_index = self.user2user_encoded.get(user_id)
            if encoded_index is None:
                raise ValueError(f"사용자 ID {user_id}를 찾을 수 없습니다.")
            weights = self.user_weights

            # 코사인 유사도 계산
            dists = np.dot(weights, weights[encoded_index])
            sorted_dists = np.argsort(dists)[-n - 1 :]

            if return_dist:
                return dists, sorted_dists

            # 결과 데이터프레임 생성
            similarity_arr = []
            for idx in sorted_dists:
                decoded_id = self.user_encoded2user.get(idx)
                user_ratings = self.get_user_ratings(decoded_id)

                similarity = dists[idx]
                avg_rating = user_ratings.rating.mean()
                n_ratings = len(user_ratings)

                similarity_arr.append(
                    {
                        "user_id": decoded_id,
                        "similarity": similarity,
                        "avg_rating": avg_rating,
                        "n_ratings": n_ratings,
                    }
                )

            result_df = pd.DataFrame(similarity_arr).sort_values(
                by="similarity", ascending=False
            )
            return result_df[result_df.user_id != user_id]

        except Exception as e:
            print(f"유사 사용자 찾기 중 오류 발생: {str(e)}")
            return pd.DataFrame()

    def get_recommendations(
        self, user_id: int, n_similar_users: int = 10, n_recommendations: int = 10
    ) -> pd.DataFrame:
        """
        유사한 사용자들이 높게 평가한 애니메이션을 추천

        Args:
            user_id: 사용자 ID
            n_similar_users: 참고할 유사 사용자 수
            n_recommendations: 추천할 애니메이션 수

        Returns:
            추천 애니메이션 정보를 담은 데이터프레임
        """
        # 유사 사용자 찾기
        similar_users = self.find_similar_users(user_id, n_similar_users)
        print(f"\n유사 사용자 수: {len(similar_users)}")

        # 현재 사용자가 시청한 애니메이션 목록
        user_watched = set(self.get_user_ratings(user_id).anime_id.values)
        print(f"현재 사용자가 시청한 애니메이션 수: {len(user_watched)}")

        # 유사 사용자들의 높은 평점 애니메이션 수집
        recommended_animes = []
        for _, row in similar_users.iterrows():
            similar_user_id = row["user_id"]
            similar_user_ratings = self.get_user_ratings(similar_user_id)

            # 높은 평점(4점 이상)을 준 애니메이션 중 아직 보지 않은 것들을 추가
            high_rated = similar_user_ratings[similar_user_ratings.rating >= 1]

            for _, rating in high_rated.iterrows():
                if rating.anime_id not in user_watched:
                    anime_info = self.anime_df[
                        self.anime_df.anime_id == rating.anime_id
                    ].iloc[0]
                    recommended_animes.append(
                        {
                            "anime_id": rating.anime_id,
                            "name": anime_info.Name,
                            "genre": anime_info.Genres,
                            "rating": rating.rating,
                            "recommender_similarity": row["similarity"],
                        }
                    )

        # 결과 정리
        if not recommended_animes:
            return pd.DataFrame()

        result_df = pd.DataFrame(recommended_animes)
        result_df = (
            result_df.groupby("anime_id")
            .agg(
                {
                    "name": "first",
                    "genre": "first",
                    "rating": "mean",
                    "recommender_similarity": "mean",
                }
            )
            .reset_index()
        )

        # 최종 점수 계산 (평점과 추천자 유사도의 가중 평균)
        result_df["score"] = (
            result_df.rating * 0.7 + result_df.recommender_similarity * 0.3
        )

        return result_df.sort_values("score", ascending=False).head(n_recommendations)


class AnimeSimilarityRecommender:
    def __init__(
        self,
        anime_weights: np.ndarray,
        anime2anime_encoded: Dict[int, int],
        anime_encoded2anime: Dict[int, int],
        anime_df: pd.DataFrame,
        synopsis_df: pd.DataFrame,
    ):
        """
        애니메이션 유사도 기반 추천 시스템

        Args:
            anime_weights: 학습된 애니메이션 임베딩 가중치
            anime2anime_encoded: 애니메이션 ID -> 인코딩된 ID 매핑
            anime_encoded2anime: 인코딩된 ID -> 애니메이션 ID 매핑
            anime_df: 애니메이션 정보 데이터프레임
            synopsis_df: 애니메이션 시놉시스 데이터프레임
        """
        self.anime_weights = anime_weights
        self.anime2anime_encoded = anime2anime_encoded
        self.anime_encoded2anime = anime_encoded2anime
        self.df = anime_df
        self.synopsis_df = synopsis_df

    def get_anime_frame(self, anime: Union[int, str]) -> pd.DataFrame:
        """
        애니메이션 ID나 이름으로 해당 애니메이션 정보를 조회

        Args:
            anime: 애니메이션 ID 또는 제목

        Returns:
            해당 애니메이션의 정보를 담은 데이터프레임
        """
        if isinstance(anime, int):
            return self.df[self.df.anime_id == anime]
        if isinstance(anime, str):
            return self.df[self.df.Name == anime]

    def find_similar_animes(
        self,
        name: Union[int, str],
        n: int = 10,
        return_dist: bool = False,
        neg: bool = False,
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """
        주어진 애니메이션과 유사한 애니메이션을 찾음

        Args:
            name: 애니메이션 ID 또는 영문 제목
            n: 반환할 유사 애니메이션 수
            return_dist: 거리 값 반환 여부
            neg: 가장 유사하지 않은 애니메이션 반환 여부

        Returns:
            유사한 애니메이션 정보를 담은 데이터프레임 또는 (거리, 인덱스) 튜플
        """
        try:
            # 입력 애니메이션의 인코딩된 인덱스 찾기
            index = self.get_anime_frame(name).anime_id.values[0]
            encoded_index = self.anime2anime_encoded.get(index)
            weights = self.anime_weights

            # 코사인 유사도 계산
            dists = np.dot(weights, weights[encoded_index])
            sorted_dists = np.argsort(dists)

            n = n + 1

            # 유사/비유사 애니메이션 선택
            closest = sorted_dists[:n] if neg else sorted_dists[-n:]

            if return_dist:
                return dists, closest

            # 결과 데이터프레임 생성
            similarity_arr = []
            for close in closest:
                decoded_id = self.anime_encoded2anime.get(close)
                synopsis = self.synopsis_df[self.synopsis_df["MAL_ID"] == decoded_id][
                    "sypnopsis"
                ]
                anime_frame = self.get_anime_frame(decoded_id)

                anime_name = anime_frame.Name.values[0]
                genre = anime_frame.Genres.values[0]
                similarity = dists[close]
                similarity_arr.append(
                    {
                        "anime_id": decoded_id,
                        "name": anime_name,
                        "similarity": similarity,
                        "genre": genre,
                        "synopsis": synopsis.iloc[0] if not synopsis.empty else "",
                    }
                )

            result_df = pd.DataFrame(similarity_arr).sort_values(
                by="similarity", ascending=False
            )
            return result_df[result_df.anime_id != index].drop(["anime_id"], axis=1)

        except Exception as e:
            print(f"Error finding similar animes for {name}: {str(e)}")
            return pd.DataFrame()
