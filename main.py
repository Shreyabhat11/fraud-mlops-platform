from src.data.ingestion import DataIngestion
from src.data.validation import DataValidator
from src.data.report_generator import ReportGenerator

ingestion = DataIngestion("data/raw")
datasets = ingestion.load_data()

validator = DataValidator(datasets)

train_df, test_df, missing = validator.run()

report = ReportGenerator()
report.generate(train_df)

# Save processed datasets
train_df.to_csv(
    "data/processed/train_merged.csv",
    index=False
)

test_df.to_csv(
    "data/processed/test_merged.csv",
    index=False
)

print("Pipeline Completed Successfully.")