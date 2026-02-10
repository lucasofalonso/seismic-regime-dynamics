from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def fit_pca_full(
    X_scaled: np.ndarray,
    *,
    random_state: int = 42,
) -> PCA:
    """
    PCA sem definir n_components (para análise de variância).
    """
    pca = PCA(random_state=random_state)
    pca.fit(X_scaled)
    return pca


def cumulative_variance(pca: PCA) -> np.ndarray:
    """
    Retorna a variância explicada acumulada.
    """
    return np.cumsum(pca.explained_variance_ratio_)


def fit_pca(
    X_scaled: np.ndarray,
    *,
    n_components: int,
    random_state: int = 42,
) -> tuple[PCA, np.ndarray]:
    """
    PCA final com n_components fixo.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(X_scaled)
    return pca, Z


def pca_loadings(
    pca: PCA,
    feature_names: pd.Index,
    *,
    sort_by: str = "PC1",
) -> pd.DataFrame:
    """
    Retorna DataFrame de loadings (features × componentes).
    """
    cols = [f"PC{i+1}" for i in range(pca.n_components_)]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=cols,
    )

    if sort_by in loadings.columns:
        loadings = loadings.sort_values(sort_by, key=lambda s: s.abs(), ascending=False)

    return loadings
