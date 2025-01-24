import os
from abc import ABC, abstractmethod

import pandas as pd


class AbstractPreProcessor(ABC):
    def __init__(self, dataset: str, data_path: str, export_path: str):
        self.dataset = dataset
        self.data_path = data_path
        self.export_path = export_path
        self.export_dfs: dict[str, pd.DataFrame] = {}
        
    @abstractmethod
    def pre_process(self) -> None:
        raise NotImplementedError("Not implemented pre_process method")

    def save_data(self) -> None:
        """
        전처리된 데이터를 파일로 저장하는 메서드.
        """
        os.makedirs(self.export_path, exist_ok=True)
        os.makedirs(os.path.join(self.export_path, self.dataset), exist_ok=True)

        for key, df in self.export_dfs.items():
            file_path = os.path.join(self.export_path, self.dataset, f"{key}.csv")
            print(f"Saving {key} to {file_path}")
            print(f"{key} column list is {df.columns} - shape({df.shape})")
            df.to_csv(file_path, index=False)

    def load_or_process(self):
        """
        데이터가 이미 저장되어 있으면 파일에서 로드하고,
        그렇지 않으면 전처리 후 저장함.
        """
        train_file = os.path.join(self.export_path, self.dataset, "train.csv")
        test_file = os.path.join(self.export_path, self.dataset, "test.csv")

        if os.path.exists(train_file) and os.path.exists(test_file):
            print("Loading train and test datasets from saved files...")
            train = pd.read_csv(train_file)
            test = pd.read_csv(test_file)
        else:
            print("Processed data not found. Running pre_process...")
            self.pre_process()
            self.save_data()
            train = self.export_dfs.get("train")
            test = self.export_dfs.get("test")

        return train, test