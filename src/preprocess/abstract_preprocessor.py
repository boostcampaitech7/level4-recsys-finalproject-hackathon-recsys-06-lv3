import os
from abc import ABC, abstractmethod

import pandas as pd


class AbstractPreProcessor(ABC):
    def __init__(self, dataset, data_path: str, export_path: str):
        self.dataset = dataset
        self.data_path = data_path
        self.export_path = export_path
        self.export_dfs: dict[str, pd.DataFrame] = {}
        self.data: dict[str, pd.DataFrame] = {}
        self._load_data()
        self.pre_process()
        self.save_data()

    def _load_data(self) -> None:
        """data path에 있는 csv 파일을 불러옵니다."""
        file_list = os.listdir(self.data_path)
        for file in file_list:
            if file.endswith(".csv"):
                print(f"Loading files: {file}")
                df = pd.read_csv(os.path.join(self.data_path, file), sep=",")
                self.data.update({file.replace(".csv", ""): df})

    @abstractmethod
    def pre_process(self) -> None:
        raise NotImplementedError("Not implemented pre_process method")

    @abstractmethod
    def save_data(self) -> None:
        raise NotImplementedError("Not implemented save_data method")