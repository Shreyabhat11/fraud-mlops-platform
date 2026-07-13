import pandas as pd

from src.features.build_features import FeatureBuilder
from src.features.preprocessor import DataPreprocessor

train = pd.read_csv("data/processed/train_merged.csv")

X = train.drop(columns=["isFraud"])

builder = FeatureBuilder()

builder.fit(X)

X = builder.transform(X)

preprocessor = DataPreprocessor(builder)

pipeline = preprocessor.build()

X_processed = pipeline.fit_transform(X)

print(X_processed.head())

print(X_processed.shape)