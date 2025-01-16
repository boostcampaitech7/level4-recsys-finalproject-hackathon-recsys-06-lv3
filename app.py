from src.trainer import Trainer
import src.models as models
import torch.nn as nn
import torch

from utils import get_config

if __name__ == "__main__":
    user_item_matrix = ...  # 사용자-아이템 행렬 로드
    num_users = ...
    num_items = ...
    config = get_config()
    model_name = config["model"]
    model = getattr(models, model_name)(num_users, num_items, config[model_name])
    criterion = getattr(nn, config["loss"])()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    trainer = Trainer(
        model, criterion, optimizer, user_item_matrix, num_users, num_items, config
    )
    trainer.train(config["epochs"])
    trainer.validate()
    # Inference
    user_id = 1
    item_id = 1
    prediction = trainer.infer(user_id, item_id)
    print(f"Predicted rating for user {user_id} and item {item_id}: {prediction}")
