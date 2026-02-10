from __future__ import annotations

from typing import Iterable, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def plot_inertia_silhouette_vs_k(
    Z: np.ndarray,
    *,
    ks: Iterable[int] = range(2, 15),
    n_init: int = 20,
    random_state: int = 42,
    figsize: Tuple[int, int] = (7, 4),
):
    """
    Plota Inertia (elbow) e Silhouette score vs k.
    Replica exatamente a lógica do notebook.
    """
    inertias: List[float] = []
    silhouettes: List[float] = []

    for k in ks:
        km = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=random_state,
        )
        labels = km.fit_predict(Z)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(Z, labels))

    fig, ax1 = plt.subplots(figsize=figsize)

    # Inertia
    ax1.plot(ks, inertias, marker="o", color="tab:blue")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Silhouette
    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, linestyle="--", color="tab:orange")
    ax2.set_ylabel("Silhouette score", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title("Elbow (Inertia) + Silhouette vs k")
    fig.tight_layout()

    return fig, (ax1, ax2)


def plot_clusters_pca(
    Z: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = 0.2,
    cmap: str = "tab10",
    figsize: Tuple[int, int] = (6, 5),
    show_centroids: bool = True,
):
    """
    Scatter PC1 vs PC2 colorido por cluster, com centróides.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=labels,
        cmap=cmap,
        alpha=alpha,
    )

    if show_centroids:
        unique_labels = np.unique(labels)
        centroids = np.vstack([
            Z[labels == k].mean(axis=0) for k in unique_labels
        ])
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            c="black",
            marker="X",
            s=200,
            label="Centroids",
        )
        ax.legend()

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Cluster centroids in PCA space")

    return fig, ax
