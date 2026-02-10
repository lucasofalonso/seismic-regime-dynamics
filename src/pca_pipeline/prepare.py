from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def prepare_X(
    features_df: pd.DataFrame,
    *,
    cols_to_remove: Iterable[str],
    impute_strategy: str = "median",
) -> Tuple[np.ndarray, pd.Index, SimpleImputer, StandardScaler]:
    """
    Prepara a matriz X para PCA:
      - seleciona colunas numéricas
      - remove colunas indesejadas
      - imputa NaNs
      - padroniza

    Retorna:
      X_scaled, feature_names, imputer, scaler
    """

    numeric = features_df.select_dtypes(include=[np.number]).copy()

    X_df = numeric.drop(columns=list(cols_to_remove), errors="ignore")
    feature_names = X_df.columns

    imputer = SimpleImputer(strategy=impute_strategy)
    X_imp = imputer.fit_transform(X_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, feature_names, imputer, scaler
