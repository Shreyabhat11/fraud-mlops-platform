from pathlib import Path

import pandas as pd


class DataIngestion:

    def __init__(self, data_dir: str):

        self.data_dir = Path(data_dir)

    def load_data(self):

        train_transaction = pd.read_csv(
            self.data_dir / "train_transaction.csv"
        )

        train_identity = pd.read_csv(
            self.data_dir / "train_identity.csv"
        )

        test_transaction = pd.read_csv(
            self.data_dir / "test_transaction.csv"
        )

        test_identity = pd.read_csv(
            self.data_dir / "test_identity.csv"
        )

        return {
            "train_transaction": train_transaction,
            "train_identity": train_identity,
            "test_transaction": test_transaction,
            "test_identity": test_identity
        }