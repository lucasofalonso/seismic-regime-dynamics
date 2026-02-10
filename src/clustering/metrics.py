from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples


def silhouette_by_cluster(Z: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    Retorna stats do silhouette por cluster: mean, median, std, count.
    """
    sil = silhouette_samples(Z, labels)
    sil_df = pd.DataFrame({"cluster": labels, "silhouette": sil})

    out = sil_df.groupby("cluster")["silhouette"].agg(["mean", "median", "std", "count"])
    return out
