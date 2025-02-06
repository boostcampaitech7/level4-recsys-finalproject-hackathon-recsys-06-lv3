import numpy as np
import pandas as pd
import torch
import yaml
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, recall_at_k
from tqdm.auto import tqdm

def get_config():
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        return config


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def compute_metrics(ground_truth, preds, k=10):
    if not hasattr(ground_truth, "rating"):
        ground_truth = ground_truth.assign(rating=1)

    # when we have 1 true positive, HitRate == Recall and MRR == MAP
    metrics = {
        f"ndcg_at_{k}": ndcg_at_k(
            ground_truth,
            preds,
            col_user="user_id",
            col_item="item_id",
            col_prediction="prediction",
            col_rating="rating",
            k=k,
        ),
        f"hit_rate_at_{k}": recall_at_k(
            ground_truth,
            preds,
            col_user="user_id",
            col_item="item_id",
            col_prediction="prediction",
            col_rating="rating",
            k=k,
        ),
        f"mrr_at_{k}": map_at_k(
            ground_truth,
            preds,
            col_user="user_id",
            col_item="item_id",
            col_prediction="prediction",
            col_rating="rating",
            k=k,
        ),
    }

    return metrics


def compute_sampled_metrics(
    seqrec_module,
    predict_dataset,
    test,
    item_counts,
    popularity_sampling=True,
    num_negatives=100,
    k=10,
    device="cuda",
):
    test = test.set_index("user_id")["item_id"].to_dict()
    all_items = item_counts.index.values
    item_weights = item_counts.values
    # probabilities = item_weights/item_weights.sum()

    seqrec_module = seqrec_module.eval().to(device)

    ndcg, hit_rate, mrr = 0.0, 0.0, 0.0
    user_count = 0

    for user in tqdm(predict_dataset):
        if user["user_id"] not in test:
            continue

        positive = test[user["user_id"]]
        indices = ~np.isin(all_items, user["full_history"])
        negatives = all_items[indices]
        if popularity_sampling:
            probabilities = item_weights[indices]
            probabilities = probabilities / probabilities.sum()
        else:
            probabilities = None
        negatives = np.random.choice(
            negatives, size=num_negatives, replace=False, p=probabilities
        )
        items = np.concatenate([np.array([positive]), negatives])

        batch = {
            "input_ids": torch.tensor(user["input_ids"]).unsqueeze(0).to(device),
            "attention_mask": torch.tensor([1] * len(user["input_ids"]))
            .unsqueeze(0)
            .to(device),
        }
        pred = seqrec_module.prediction_output(batch)
        pred = pred[0, -1, items]

        rank = (-pred).argsort().argsort()[0].item() + 1
        if rank <= k:
            ndcg += 1 / np.log2(rank + 1)
            hit_rate += 1
            mrr += 1 / rank
        user_count += 1

    ndcg = ndcg / user_count
    hit_rate = hit_rate / user_count
    mrr = mrr / user_count

    return {"ndcg": ndcg, "hit_rate": hit_rate, "mrr": mrr}


def preds2recs(preds, item_mapping=None):
    user_ids = np.hstack([pred["user_ids"] for pred in preds])
    scores = np.vstack([pred["scores"] for pred in preds])
    preds = np.vstack([pred["preds"] for pred in preds])

    user_ids = np.repeat(user_ids[:, None], repeats=scores.shape[1], axis=1)

    recs = pd.DataFrame(
        {
            "user_id": user_ids.flatten(),
            "item_id": preds.flatten(),
            "prediction": scores.flatten(),
        }
    )

    if item_mapping is not None:
        recs.item_id = recs.item_id.map(item_mapping)

    return recs
