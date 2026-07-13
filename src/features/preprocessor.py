from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class DataPreprocessor:
    """
    Builds the preprocessing pipeline.

    Responsibilities
    ----------------
    1. Impute missing numerical values.
    2. Impute missing categorical values.
    3. Apply transformations consistently to train and inference data.
    """

    def __init__(self, feature_builder):

        self.feature_builder = feature_builder

    def build(self):
        """
        Build the sklearn ColumnTransformer.
        """

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    self.feature_builder.numeric_columns
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    self.feature_builder.categorical_columns
                )
            ],
            remainder="drop"
        )

        preprocessor.set_output(transform="pandas")

        return preprocessor