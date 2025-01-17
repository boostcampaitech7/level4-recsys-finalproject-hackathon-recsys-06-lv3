import torch
import yaml


def get_config():
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        return config


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model
