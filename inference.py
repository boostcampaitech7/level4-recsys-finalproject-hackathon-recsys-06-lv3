import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.models import NCF
from src.utils import get_config


def parse_args():
    parser = argparse.ArgumentParser(description="Generate recommendations for users")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "specific"],
        default="all",
        help="Mode to run inference: all users or specific users",
    )
    parser.add_argument(
        "--users",
        type=str,
        default=None,
        help="Comma-separated list of user IDs (for specific mode)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of recommendations per user"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file name (without path)"
    )
    return parser.parse_args()


def prepare_output_path(args):
    # Create save/results directory if it doesn't exist
    results_dir = os.path.join("save", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Generate default filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        mode_str = "all" if args.mode == "all" else "specific"
        filename = f"recommendations_{mode_str}_{timestamp}.csv"
    else:
        filename = args.output if args.output.endswith(".csv") else f"{args.output}.csv"

    return os.path.join(results_dir, filename)


def load_model_and_mappings(model_path, config):
    # Load data and get mappings
    df = pd.read_csv(config["data_path"])

    # Create user mappings
    user_ids = df["user_id"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    user_encoded2user = {i: x for i, x in enumerate(user_ids)}

    # Create anime mappings
    anime_ids = df["anime_id"].unique().tolist()
    anime2anime_encoded = {x: i for i, x in enumerate(anime_ids)}
    anime_encoded2anime = {i: x for i, x in enumerate(anime_ids)}

    # Initialize model
    num_users = len(user2user_encoded)
    num_items = len(anime2anime_encoded)
    model = NCF(num_users, num_items, config["NCF"])

    # Load trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return (
        model,
        (
            user2user_encoded,
            user_encoded2user,
            anime2anime_encoded,
            anime_encoded2anime,
        ),
        df,
    )


def get_top_k_recommendations(
    model, user_id, encoded_user_id, anime_encoded2anime, k=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create tensors for all items for this user
    num_items = len(anime_encoded2anime)
    user_tensor = torch.full((num_items,), encoded_user_id, dtype=torch.long).to(device)
    item_tensor = torch.arange(num_items, dtype=torch.long).to(device)
    rating_tensor = torch.full((num_items,), 0.5, dtype=torch.float).to(
        device
    )  # 중간값으로 설정

    # Get predictions
    with torch.no_grad():
        predictions = model(user_tensor, item_tensor, rating_tensor)

    # Get top k items
    top_k_scores, top_k_indices = torch.topk(predictions, k)

    results = []
    for idx, score in zip(top_k_indices.cpu().numpy(), top_k_scores.cpu().numpy()):
        original_item_id = anime_encoded2anime[idx]
        results.append(
            {"user_id": user_id, "item_id": original_item_id, "score": float(score)}
        )

    return results


def validate_user_ids(user_ids, df):
    """사용자 ID의 유효성을 검증하고 존재하지 않는 ID를 보고"""
    existing_users = set(df["user_id"].unique())
    valid_users = []
    invalid_users = []

    for user_id in user_ids:
        if user_id in existing_users:
            valid_users.append(user_id)
        else:
            invalid_users.append(user_id)

    if invalid_users:
        print(
            f"Warning: Following user IDs were not found in the dataset: {invalid_users}"
        )

    return valid_users


def main():
    args = parse_args()
    output_path = prepare_output_path(args)

    # Load configuration
    config = get_config()

    # Load the latest model from saved/models
    model_dir = "./save/models"
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    latest_model = max(
        model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x))
    )
    model_path = os.path.join(model_dir, latest_model)

    print(f"Loading model from: {model_path}")

    # Load model and mappings
    (
        model,
        (
            user2user_encoded,
            user_encoded2user,
            anime2anime_encoded,
            anime_encoded2anime,
        ),
        df,
    ) = load_model_and_mappings(model_path, config)

    # Determine which users to process
    if args.mode == "specific" and args.users:
        user_ids = [int(uid.strip()) for uid in args.users.split(",")]
        user_ids = validate_user_ids(user_ids, df)
        if not user_ids:
            print("No valid user IDs provided. Exiting...")
            return
        print(f"Generating recommendations for {len(user_ids)} specific users...")
    else:
        user_ids = list(user2user_encoded.keys())
        print(f"Generating recommendations for all {len(user_ids)} users...")

    # Get recommendations for selected users
    all_recommendations = []
    for user_id in tqdm(user_ids):
        if user_id in user2user_encoded:
            encoded_user_id = user2user_encoded[user_id]
            user_recommendations = get_top_k_recommendations(
                model, user_id, encoded_user_id, anime_encoded2anime, k=args.top_k
            )
            all_recommendations.extend(user_recommendations)

    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(all_recommendations)
    results_df.to_csv(output_path, index=False)
    print(f"\nRecommendations saved to: {output_path}")

    # Print sample of recommendations
    print("\nSample of recommendations:")
    print(results_df.head(min(10, len(results_df))))

    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total number of users processed: {len(user_ids)}")
    print(f"Total recommendations generated: {len(results_df)}")
    print(f"Average prediction score: {results_df['score'].mean():.4f}")


if __name__ == "__main__":
    main()
