from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


class RecommenderMetrics:
    @staticmethod
    def precision_at_k(actual: List[int], predicted: List[int], k: int) -> float:
        """
        k개의 추천 결과에 대한 정밀도를 계산

        Args:
            actual: 실제 시청/평가한 아이템 리스트
            predicted: 추천된 아이템 리스트
            k: 상위 k개 고려
        """
        if len(predicted) > k:
            predicted = predicted[:k]

        num_hits = len(set(actual) & set(predicted))
        return num_hits / min(k, len(predicted)) if predicted else 0.0

    @staticmethod
    def recall_at_k(actual: List[int], predicted: List[int], k: int) -> float:
        """
        k개의 추천 결과에 대한 재현율을 계산

        Args:
            actual: 실제 시청/평가한 아이템 리스트
            predicted: 추천된 아이템 리스트
            k: 상위 k개 고려
        """
        if len(predicted) > k:
            predicted = predicted[:k]

        num_hits = len(set(actual) & set(predicted))
        return num_hits / len(actual) if actual else 0.0

    @staticmethod
    def ndcg_at_k(actual: List[int], predicted: List[int], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain at k를 계산

        Args:
            actual: 실제 시청/평가한 아이템 리스트
            predicted: 추천된 아이템 리스트
            k: 상위 k개 고려
        """
        if len(predicted) > k:
            predicted = predicted[:k]

        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(actual), k))])
        dcg = sum(
            [1.0 / np.log2(i + 2) for i, item in enumerate(predicted) if item in actual]
        )

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def map_at_k(actual: List[int], predicted: List[int], k: int) -> float:
        """
        Mean Average Precision at k를 계산

        Args:
            actual: 실제 시청/평가한 아이템 리스트
            predicted: 추천된 아이템 리스트
            k: 상위 k개 고려
        """
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0

        for i, p in enumerate(predicted):
            if p in actual:
                num_hits += 1
                score += num_hits / (i + 1)

        return score / min(len(actual), k) if actual else 0.0

    @staticmethod
    def diversity_score(
        predicted_items: List[List[int]], item_features: Dict[int, List[str]]
    ) -> float:
        """
        추천 결과의 다양성을 계산

        Args:
            predicted_items: 사용자별 추천된 아이템 리스트
            item_features: 아이템별 특성(장르 등) 딕셔너리
        """
        if not predicted_items:
            return 0.0

        unique_features = set()
        total_features = []

        for items in predicted_items:
            for item in items:
                if item in item_features:
                    features = item_features[item]
                    unique_features.update(features)
                    total_features.extend(features)

        return len(unique_features) / len(total_features) if total_features else 0.0

    @staticmethod
    def novelty_score(
        predicted_items: List[int], item_popularity: Dict[int, int]
    ) -> float:
        """
        추천된 아이템의 참신성을 계산

        Args:
            predicted_items: 추천된 아이템 리스트
            item_popularity: 아이템별 인기도(조회수/평가수 등) 딕셔너리
        """
        if not predicted_items or not item_popularity:
            return 0.0

        max_popularity = max(item_popularity.values())
        novelty_scores = [
            1 - (item_popularity.get(item, 0) / max_popularity)
            for item in predicted_items
        ]

        return np.mean(novelty_scores)

    @staticmethod
    def personalization_score(predicted_items: List[List[int]]) -> float:
        """
        사용자 간 추천 결과의 다양성을 계산

        Args:
            predicted_items: 사용자별 추천된 아이템 리스트
        """
        if len(predicted_items) < 2:
            return 0.0

        similarity_sum = 0
        num_pairs = 0

        for i in range(len(predicted_items)):
            for j in range(i + 1, len(predicted_items)):
                set_i = set(predicted_items[i])
                set_j = set(predicted_items[j])

                if set_i and set_j:
                    similarity = len(set_i & set_j) / len(set_i | set_j)
                    similarity_sum += similarity
                    num_pairs += 1

        return 1 - (similarity_sum / num_pairs) if num_pairs > 0 else 0.0


class TrainingMetrics:
    @staticmethod
    def _batch_process(y_true: np.ndarray, y_pred: np.ndarray, batch_size: int = 10000):
        """배치 단위로 데이터 처리"""
        total = len(y_true)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            yield y_true[start:end], y_pred[start:end]

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray, batch_size: int = 10000) -> float:
        """Root Mean Square Error를 배치 방식으로 계산"""
        squared_sum = 0
        count = 0
        for batch_true, batch_pred in TrainingMetrics._batch_process(y_true, y_pred, batch_size):
            squared_sum += np.sum((batch_true - batch_pred) ** 2)
            count += len(batch_true)
        return np.sqrt(squared_sum / count)

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray, batch_size: int = 10000) -> float:
        """Mean Absolute Error를 배치 방식으로 계산"""
        abs_sum = 0
        count = 0
        for batch_true, batch_pred in TrainingMetrics._batch_process(y_true, y_pred, batch_size):
            abs_sum += np.sum(np.abs(batch_true - batch_pred))
            count += len(batch_true)
        return abs_sum / count

    @staticmethod
    def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray, batch_size: int = 10000) -> float:
        """Binary Cross Entropy를 배치 방식으로 계산"""
        epsilon = 1e-15
        cross_entropy_sum = 0
        count = 0
        for batch_true, batch_pred in TrainingMetrics._batch_process(y_true, y_pred, batch_size):
            batch_pred = np.clip(batch_pred, epsilon, 1 - epsilon)
            cross_entropy_sum += np.sum(
                -batch_true * np.log(batch_pred) - 
                (1 - batch_true) * np.log(1 - batch_pred)
            )
            count += len(batch_true)
        return cross_entropy_sum / count

    @staticmethod
    def auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Area Under the Curve 점수를 계산"""
        try:
            return roc_auc_score(y_true, y_pred)
        except:
            return 0.0

    @staticmethod
    def average_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Average Precision을 계산"""
        try:
            return average_precision_score(y_true, y_pred)
        except:
            return 0.0


def calculate_metrics_for_users(
    test_data: Dict[int, List[int]],
    predictions: Dict[int, List[int]],
    item_features: Dict[int, List[str]],
    item_popularity: Dict[int, int],
    k: int = 10,
) -> Dict[str, float]:
    """
    전체 테스트 세트에 대한 평가 지표를 계산

    Args:
        test_data: 실제 데이터 {user_id: [item_ids]}
        predictions: 예측 데이터 {user_id: [item_ids]}
        item_features: 아이템 특성 정보
        item_popularity: 아이템 인기도 정보
        k: 추천 개수

    Returns:
        평가 지표 딕셔너리
    """
    metrics = defaultdict(list)

    for user_id, actual_items in test_data.items():
        if user_id not in predictions:
            continue

        pred_items = predictions[user_id]

        # Accuracy metrics
        metrics["precision"].append(
            RecommenderMetrics.precision_at_k(actual_items, pred_items, k)
        )
        metrics["recall"].append(
            RecommenderMetrics.recall_at_k(actual_items, pred_items, k)
        )
        metrics["ndcg"].append(
            RecommenderMetrics.ndcg_at_k(actual_items, pred_items, k)
        )
        metrics["map"].append(RecommenderMetrics.map_at_k(actual_items, pred_items, k))

    # Beyond-accuracy metrics
    metrics["diversity"] = RecommenderMetrics.diversity_score(
        list(predictions.values()), item_features
    )
    metrics["novelty"] = np.mean(
        [
            RecommenderMetrics.novelty_score(items, item_popularity)
            for items in predictions.values()
        ]
    )
    metrics["personalization"] = RecommenderMetrics.personalization_score(
        list(predictions.values())
    )

    # Average all metrics
    return {k: np.mean(v) if isinstance(v, list) else v for k, v in metrics.items()}
