"""
Run experiment.
"""

import os
import time
from datetime import datetime

import hydra
import mlflow
import mlflow.data.dataset_registry
import mlflow.data.dataset_source
import mlflow.data.pandas_dataset
import mlflow.entities
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from clearml import Task
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from torch.utils.data import DataLoader

from src.data.dataset import (
    CausalLMDataset,
    CausalLMPredictionDataset,
    PaddingCollateFn,
)
from src.models import SASRec
from src.modules import SeqRec, SeqRecWithSampling
from src.preprocess import MovieLensPreProcessor
from src.utils import compute_metrics, compute_sampled_metrics, preds2recs


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))

    if hasattr(config, "cuda_visible_devices"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_visible_devices)

    if hasattr(config, "project_name"):
        task = Task.init(
            project_name=config.project_name,
            task_name=config.task_name,
            reuse_last_task_id=False,
        )
        task.connect(OmegaConf.to_container(config))
    else:
        task = None

    print("전처리 중...")
    np.random.seed(42)
    preprocessor = MovieLensPreProcessor(
        "MovieLens_20m", config.data_path, config.export_path
    )
    items, train, valid, valid_full, test = preprocessor.get_data()
    item_count = preprocessor.item_count
    print("전처리 완료")

    mlflow_init(config, train, valid, test)

    train_loader, eval_loader = create_dataloaders(train, valid_full, config)
    model = create_model(config, item_count=item_count)
    start_time = time.time()
    trainer, seqrec_module = training(model, train_loader, eval_loader, config)
    training_time = time.time() - start_time
    mlflow.log_param("Training-Time", training_time)
    print("training_time", training_time)

    recs_valid, valid_dataset = predict(trainer, seqrec_module, train, config)
    evaluate(
        recs_valid,
        valid,
        train,
        seqrec_module,
        valid_dataset,
        task,
        config,
        prefix="val",
    )

    recs_test, test_dataset = predict(trainer, seqrec_module, valid_full, config)
    evaluate(
        recs_test, test, train, seqrec_module, test_dataset, task, config, prefix="test"
    )
    save_recommendations_to_csv(recs_test, "test_recommendations.csv")

    if task is not None:
        task.get_logger().report_single_value("training_time", training_time)
        task.close()

    mlflow.end_run()


def mlflow_init(config, train, valid, test):
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    if hasattr(config["mlflow"], "run_name"):
        run_name = config["mlflow"]["run_name"]
    else:
        run_name = f"{config['model']}-{config['mlflow']['user']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    exp = mlflow.get_experiment_by_name(config["mlflow"]["user"])
    mlflow.start_run(
        run_name=run_name,
        log_system_metrics=True,
        description=config["mlflow"]["description"],
        experiment_id=exp.experiment_id,
    )
    mlflow.log_params(config)
    mf_train = mlflow.data.pandas_dataset.from_pandas(train, name="train_df")
    mf_valid = mlflow.data.pandas_dataset.from_pandas(valid, name="valid_df")
    mf_test = mlflow.data.pandas_dataset.from_pandas(test, name="test_df")
    mlflow.log_input(mf_train, "train_df")
    mlflow.log_input(mf_valid, "valid_df")
    mlflow.log_input(mf_test, "test_df")


def create_dataloaders(train, valid, config):
    valid_size = config.dataloader.valid_size
    valid_users = valid.user_id.unique()
    if valid_size and (valid_size < len(valid_users)):
        valid_users = np.random.choice(valid_users, size=valid_size, replace=False)
        valid = valid[valid.user_id.isin(valid_users)]

    train_dataset = CausalLMDataset(train, **config["dataset"])
    eval_dataset = CausalLMPredictionDataset(
        valid, max_length=config.dataset.max_length, valid_mode=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
        collate_fn=PaddingCollateFn(),
        persistent_workers=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.dataloader.test_batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        collate_fn=PaddingCollateFn(),
        persistent_workers=True,
    )
    return train_loader, eval_loader


def create_model(config, item_count):
    if hasattr(config.dataset, "num_negatives") and config.dataset.num_negatives:
        add_head = False
    else:
        add_head = True

    if config.model == "SASRec":
        model = SASRec(item_num=item_count, add_head=add_head, **config.SASRec)

    return model


def training(model, train_loader, eval_loader, config):
    if hasattr(config.dataset, "num_negatives") and config.dataset.num_negatives:
        seqrec_module = SeqRecWithSampling(model, **config["seqrec_module"])
    else:
        seqrec_module = SeqRec(model, **config["seqrec_module"])

    early_stopping = EarlyStopping(
        monitor="val_ndcg", mode="max", patience=config.patience, verbose=False
    )
    model_summary = ModelSummary(max_depth=4)
    checkpoint = ModelCheckpoint(
        save_top_k=1, monitor="val_ndcg", mode="max", save_weights_only=True
    )
    progress_bar = TQDMProgressBar(refresh_rate=100)
    callbacks = [early_stopping, model_summary, checkpoint, progress_bar]

    trainer = pl.Trainer(
        callbacks=callbacks,
        devices=1,
        accelerator="gpu",
        enable_checkpointing=True,
        max_epochs=config.epochs,
        num_sanity_val_steps=0,
    )

    trainer.fit(
        model=seqrec_module, train_dataloaders=train_loader, val_dataloaders=eval_loader
    )

    seqrec_module.load_state_dict(torch.load(checkpoint.best_model_path)["state_dict"])

    return trainer, seqrec_module


def predict(trainer, seqrec_module, data, config):
    if config.model in ["SASRec", "GPT4Rec", "RNN"]:
        predict_dataset = CausalLMPredictionDataset(
            data, max_length=config.dataset.max_length
        )

    predict_loader = DataLoader(
        predict_dataset,
        shuffle=False,
        collate_fn=PaddingCollateFn(),
        batch_size=config.dataloader.test_batch_size,
        num_workers=config.dataloader.num_workers,
        persistent_workers=True,
    )

    seqrec_module.predict_top_k = max(config.top_k_metrics)
    preds = trainer.predict(model=seqrec_module, dataloaders=predict_loader)

    recs = preds2recs(preds)
    print("recs shape", recs.shape)

    return recs, predict_dataset


def evaluate(recs, test, train, seqrec_module, dataset, task, config, prefix="test"):

    all_metrics = {}
    for k in config.top_k_metrics:
        metrics = compute_metrics(test, recs, k=k)
        metrics = {prefix + "_" + key: value for key, value in metrics.items()}
        print(metrics)
        all_metrics.update(metrics)
        mlflow.log_metrics(metrics)

    if config.sampled_metrics:
        item_counts = train.item_id.value_counts()

        uniform_metrics = compute_sampled_metrics(
            seqrec_module,
            dataset,
            test,
            item_counts,
            popularity_sampling=False,
            num_negatives=100,
            k=10,
        )
        uniform_metrics = {
            prefix + "_" + key + "_uniform": value
            for key, value in uniform_metrics.items()
        }
        print(uniform_metrics)

        popularity_metrics = compute_sampled_metrics(
            seqrec_module, dataset, test, item_counts, num_negatives=100, k=10
        )
        popularity_metrics = {
            prefix + "_" + key + "_popularity": value
            for key, value in popularity_metrics.items()
        }
        print(popularity_metrics)

    if task:
        clearml_logger = task.get_logger()

        for key, value in all_metrics.items():
            clearml_logger.report_single_value(key, value)
        if config.sampled_metrics:
            for key, value in uniform_metrics.items():
                clearml_logger.report_single_value(key, value)
            for key, value in popularity_metrics.items():
                clearml_logger.report_single_value(key, value)

        if config.sampled_metrics:
            all_metrics.update(uniform_metrics)
            all_metrics.update(popularity_metrics)
        all_metrics = pd.Series(all_metrics).to_frame().reset_index()
        all_metrics.columns = ["metric_name", "metric_value"]

        clearml_logger.report_table(
            title=f"{prefix}_metrics", series="dataframe", table_plot=all_metrics
        )
        task.upload_artifact(f"{prefix}_metrics", all_metrics)


def save_recommendations_to_csv(recs, file_name):
    """
    추천 결과를 CSV 파일로 저장하는 함수.
    Args:
        recs (pd.DataFrame): 추천 결과를 담은 DataFrame.
        file_name (str): 저장할 CSV 파일 이름.
    """
    recs.to_csv(file_name, index=False)
    print(f"추천 결과가 {file_name} 파일로 저장되었습니다.")


if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     config = get_config()
#     data_path = config["data_path"]
#     (train_df, val_df), mappings = load_data(data_path)
#     # 모델 크기 계산
#     num_users = len(mappings[0])
#     num_items = len(mappings[2])
#     model_name = config["model"]
#     model = getattr(models, model_name)(num_users, num_items, config[model_name])
#     criterion = getattr(nn, config["loss"])()
#     optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

#     trainer = Trainer(
#         model, criterion, optimizer, train_df, val_df, num_users, num_items, config
#     )
#     trainer.train(config["epochs"])
#     # trainer.validate()
#     mlflow.end_run()
