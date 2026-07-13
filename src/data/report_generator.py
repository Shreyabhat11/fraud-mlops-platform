import json
from pathlib import Path

import pandas as pd
class ReportGenerator:

    def __init__(self, report_dir="reports"):

        self.report_dir = Path(report_dir)

        self.report_dir.mkdir(parents=True, exist_ok=True)

    def dataset_summary(self, df):

        summary = pd.DataFrame({
            "Column": df.columns,
            "DataType": df.dtypes.astype(str),
            "MissingValues": df.isnull().sum(),
            "MissingPercent": (df.isnull().mean()*100),
            "UniqueValues": df.nunique()
        })

        return summary  
    
    def save_summary(self, df, filename):

        summary = self.dataset_summary(df)

        summary.to_csv(
            self.report_dir / filename,
            index=False
            )
    def save_missing_values(self, df):

            missing = (
                df.isnull().mean().sort_values(ascending=False)*100
            )

            missing.to_csv(
                self.report_dir / "missing_values.csv"
            )
    
    def save_schema(self, df):

        schema = {
            column: str(dtype)
            for column, dtype
            in df.dtypes.items()
        }

        with open(
            self.report_dir/"schema.json",
            "w"
        ) as f:
            json.dump(
                schema,
                f,
                indent=4
            )

    def save_validation_report(self, df):

        report = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "duplicates": int(df.duplicated().sum()),
            "missing_cells": int(df.isnull().sum().sum()),
            "memory_mb": round(
                df.memory_usage(deep=True).sum()/1024**2,2
            )
        }

        with open(
            self.report_dir/"validation_report.json",
            "w"
        ) as f:

            json.dump(
                report,
                f,
                indent=4
            )
    
    def save_sample(self, df):

        df.head(1000).to_csv(
            self.report_dir/"merged_train_sample.csv",
            index=False
        )

    def generate(self, df):

            self.save_summary(
                df,
                "data_summary.csv"
            )

            self.save_missing_values(df)
            self.save_schema(df)
            self.save_validation_report(df)
            self.save_sample(df)

            print("Reports Generated Successfully.")