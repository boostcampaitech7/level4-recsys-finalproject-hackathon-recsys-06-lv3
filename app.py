from trainer import Trainer

if __name__ == "__main__":
    user_item_matrix = ...  # 사용자-아이템 행렬 로드
    num_users = ...
    num_items = ...

    trainer = Trainer(user_item_matrix, num_users, num_items)
    trainer.train()
    trainer.validate()
    user_id = 1
    item_id = 1
    prediction = trainer.infer(user_id, item_id)
    print(f"Predicted rating for user {user_id} and item {item_id}: {prediction}")
