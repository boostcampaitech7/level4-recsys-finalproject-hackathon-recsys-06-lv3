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
    def _pre_process(self) -> None:
        raise NotImplementedError("Not implemented pre_process method")

    def load_or_process(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        데이터가 이미 저장되어 있으면 파일에서 로드하고,
        그렇지 않으면 전처리 후 저장함.
        """
        required_files = {
            "items": os.path.join(self.export_path, self.dataset, "items.csv"),
            "train": os.path.join(self.export_path, self.dataset, "train.csv"),
            "valid": os.path.join(self.export_path, self.dataset, "valid.csv"),
            "test": os.path.join(self.export_path, self.dataset, "test.csv"),
        }

        if all(os.path.exists(f) for f in required_files.values()):
            print("Loading items, train, valid, test datasets from saved files...")
            try:
                items = pd.read_csv(required_files["items"])
                train = pd.read_csv(required_files["train"])
                valid = pd.read_csv(required_files["valid"])
                test = pd.read_csv(required_files["test"])
            except Exception as e:
                print(f"Error loading files: {e}")
                raise
        else:
            print("Processed data not found. Running pre_process...")
            self._pre_process()
            self._save_data()
            items = self.export_dfs.get("items")
            train = self.export_dfs.get("train")
            valid = self.export_dfs.get("valid")
            test = self.export_dfs.get("test")

            if any(df is None for df in [items, train, valid, test]):
                raise ValueError("Missing required DataFrame after preprocessing")

        return items, train, valid, test

    def _save_data(self) -> None:
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
