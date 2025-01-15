import torch
import yaml


def load_mlflow_tracking_uri():

    # config.yaml 파일에서 tracking_uri 읽기
    with open("../config.yaml", "r") as file:
        config = yaml.safe_load(file)
        return config["mlflow"]["tracking_uri"]


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model
