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

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        전처리된 데이터를 로드하거나 전처리 후 저장하는 메서드.
        """
        required_files = {
            "items": os.path.join(self.export_path, self.dataset, "items.csv"),
            "train": os.path.join(self.export_path, self.dataset, "train.csv"),
            "valid": os.path.join(self.export_path, self.dataset, "valid.csv"),
            "test": os.path.join(self.export_path, self.dataset, "test.csv"),
        }

        if all(os.path.exists(f) for f in required_files.values()):
            return self._load_data(required_files)
        return self._process_data()

    def _load_data(
        self, required_files: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        저장된 데이터 파일들을 로드하는 메서드
        """
        print("Loading items, train, valid, test datasets from saved files...")
        try:
            items = pd.read_csv(required_files["items"])
            train = pd.read_csv(required_files["train"])
            valid = pd.read_csv(required_files["valid"])
            test = pd.read_csv(required_files["test"])
            return items, train, valid, test
        except Exception as e:
            print(f"Error loading files: {e}")
            raise

    def _process_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        데이터를 전처리하고 저장하는 메서드
        """
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
