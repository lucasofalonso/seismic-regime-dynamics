import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_explained_variance(pca: PCA, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    evr = pca.explained_variance_ratio_
    ax.bar(range(len(evr)), evr)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    return ax


def plot_cumulative_variance(pca: PCA, thresholds=(0.8, 0.9), ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    cum = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(cum, marker="o")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    for t in thresholds:
        ax.axhline(t, ls="--")
    return ax
