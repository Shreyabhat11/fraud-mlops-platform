from pathlib import Path
from typing import Dict

import pandas as pd

class DataValidator:

    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        self.datasets = datasets
    
    def validate_loaded_data(self):

        for name, df in self.datasets.items():

            if df.empty:
                raise ValueError(f"{name} is empty!")

        print("All datasets loaded successfully.")
    
    def validate_transaction_ids(self):

        train_transaction = self.datasets["train_transaction"]
        train_identity = self.datasets["train_identity"]

        test_transaction = self.datasets["test_transaction"]
        test_identity = self.datasets["test_identity"]

        assert train_transaction["TransactionID"].is_unique
        assert train_identity["TransactionID"].is_unique
        assert test_transaction["TransactionID"].is_unique
        assert test_identity["TransactionID"].is_unique
        print("Transaction IDs are unique.")

    def validate_target(self):
        train = self.datasets["train_transaction"]
        if "isFraud" not in train.columns:
            raise ValueError("Target column missing!")

        print("Target column found.")

    def check_duplicates(self):
        for name, df in self.datasets.items():
            duplicates = df.duplicated().sum()
            print(f"{name}: {duplicates} duplicate rows")

    def missing_value_report(self):
        report = {}

        for name, df in self.datasets.items():
            missing = df.isnull().mean() * 100
            report[name] = missing.sort_values(
                ascending=False
            )

        return report
    
    def merge_data(self):

        train = pd.merge(
            self.datasets["train_transaction"],
            self.datasets["train_identity"],
            on="TransactionID",
            how="left"
        )

        test = pd.merge(
            self.datasets["test_transaction"],
            self.datasets["test_identity"],
            on="TransactionID",
            how="left"
        )

        print("Merge successful.")

        print(train.shape)

        print(test.shape)

        return train, test
    
    def run(self):

        self.validate_loaded_data()
        self.validate_transaction_ids()
        self.validate_target()
        self.check_duplicates()
        train, test = self.merge_data()
        missing = self.missing_value_report()

        return train, test, missing