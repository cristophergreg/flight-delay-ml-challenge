import numpy as np
import pandas as pd
from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression

class DelayModel:
    FEATURES_COLS = [
        "OPERA_Latin American Wings",
        "OPERA_Grupo LATAM",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
        "TIPOVUELO_I",
        "MES_4",
        "MES_7",
        "MES_10",
        "MES_11",
        "MES_12",
    ]

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    @staticmethod
    def _build_delay_if_missing(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Ensure the binary target column exists; build it if missing.

        If `target_col` is absent, it is computed from 'Fecha-I' and 'Fecha-O' using
        the 15-minute rule (`1` if (Fecha-O - Fecha-I) > 15 minutes, else `0`).

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_col (str): Target column name to ensure (e.g., "delay").

        Returns:
            pd.DataFrame: Same DataFrame with `target_col` present (dtype=int).

        Raises:
            ValueError: If `target_col` is not present and date columns
                ('Fecha-I', 'Fecha-O') are missing to derive it.
        """
        if target_col in df.columns:
            df[target_col] = df[target_col].astype(int)
            return df

        required = {"Fecha-I", "Fecha-O"}
        if not required.issubset(df.columns):
            raise ValueError(f"To build '{target_col}', the following columns are required: {required}.")

        fi = pd.to_datetime(df["Fecha-I"], errors="coerce")
        fo = pd.to_datetime(df["Fecha-O"], errors="coerce")
        min_diff = (fo - fi).dt.total_seconds() / 60.0
        df[target_col] = (min_diff > 15).astype(int)
        return df

    @classmethod
    def _encode_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode OPERA, TIPOVUELO and MES and enforce the feature contract.

        The result is reindexed to `FEATURES_COLS`, filling missing columns with 0.
        Only these 10 columns are returned (dtype=int).

        Args:
            df (pd.DataFrame): Input with 'OPERA', 'TIPOVUELO' and 'MES'.

        Returns:
            pd.DataFrame: Feature matrix with the exact columns in `FEATURES_COLS`.

        Raises:
            ValueError: If any required source column is missing.
        """
        needed = {"OPERA", "TIPOVUELO", "MES"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for features: {missing}")

        X = df.copy()
        X["OPERA"] = X["OPERA"].astype(str)
        X["TIPOVUELO"] = X["TIPOVUELO"].astype(str)
        X["MES"] = pd.to_numeric(X["MES"], errors="coerce").fillna(-1).astype(int)

        X = pd.get_dummies(
            X[["OPERA", "TIPOVUELO", "MES"]],
            columns=["OPERA", "TIPOVUELO", "MES"],
            prefix=["OPERA", "TIPOVUELO", "MES"],
        )

        X = X.reindex(columns=cls.FEATURES_COLS, fill_value=0).astype(int)
        return X

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        df = data.copy()

        y = None
        if target_column:
            df = self._build_delay_if_missing(df, target_column)
            y = df[[target_column]].astype(int)

        X = self._encode_features(df)

        if y is not None:
            return X, y
        return X

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        y = target.iloc[:, 0].astype(int).values
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        model.fit(features, y)
        self._model = model

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            # Defensive fallback in case predict is called before fit.
            return [0 for _ in range(len(features))]
        preds = self._model.predict(features)
        return [int(v) for v in preds.tolist()]