# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: conformal_fdr (3.12.8)
#     language: python
#     name: python3
# ---

# %%
# Schematic 2D "chemical space" scatter for the method figure.
# No real molecules/embeddings — synthetic points arranged in a few
# UMAP/t-SNE-like blobs, colored by a synthetic pIC50 value.
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

OUT_DIR = os.path.join("..", "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

N_POINTS = 1000
N_CLUSTERS = 8
THRESHOLD = 7.0
SEED = 0

# %%
rng = np.random.default_rng(SEED)

# points arranged in blobs, like a typical UMAP/t-SNE embedding
coords, cluster_id = make_blobs(
    n_samples=N_POINTS,
    centers=N_CLUSTERS,
    cluster_std=1.1,
    random_state=SEED,
)

# each cluster gets its own "activity level" so pIC50 is spatially
# coherent (as in real chemical-space plots), plus per-point noise
cluster_means = rng.normal(5.0, 3, size=N_CLUSTERS)
pic50 = cluster_means[cluster_id] + rng.normal(0, 0.5, size=N_POINTS)
pic50 = np.clip(pic50, 0, 100.0)

p_ic50_preds = pic50 + rng.normal(0, 0.5, size=N_POINTS)

label = (pic50 >= THRESHOLD).astype(int)

# %%
def style_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")


# %%
# continuous pIC50, greyscale (light = low, dark = high)
fig, ax = plt.subplots(figsize=(5, 5))
sc = ax.scatter(
    coords[:, 0], coords[:, 1],
    c=pic50, cmap="Greys", vmin=pic50.min(), vmax=pic50.max(),
    s=70, edgecolors="black", linewidths=0.5,
)
style_axes(ax)
cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label("pIC50")
fig.tight_layout()

# %%
# continuous pIC50, greyscale (light = low, dark = high)
fig, ax = plt.subplots(figsize=(5, 5))
sc = ax.scatter(
    coords[:, 0], coords[:, 1],
    c=p_ic50_preds, cmap="Greys", vmin=pic50.min(), vmax=pic50.max(),
    s=70, edgecolors="black", linewidths=0.5,
)
style_axes(ax)
cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label("pIC50 predictions")
fig.tight_layout()

# %%
# thresholded pIC50: 0 (< 7) light, 1 (>= 7) dark
fig, ax = plt.subplots(figsize=(5, 5))
colors = np.where(label == 1, "black", "white")
ax.scatter(
    coords[:, 0], coords[:, 1],
    c=colors, s=70, edgecolors="black", linewidths=0.5,
)
style_axes(ax)
fig.tight_layout()

# %%
# now distribtuion of pIC50 values of the 0 and 1 classes, for the figure caption
import seaborn as sns
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(5, 3))  

sns.histplot(
    p_ic50_preds[label == 0], color="lightgray", label="Inactive (measured pIC50 < 7)", kde=True,  ax=ax
)
ax.set_xlabel("Predicted pIC50")
ax.set_ylabel("Density")

#position legend
ax.legend(loc="upper left")
fig.tight_layout()


# %%
fig, ax = plt.subplots(figsize=(5, 3))  

sns.histplot(
    p_ic50_preds[label == 0], color="lightgray", label="Inactive (measured pIC50 < 7)", kde=True,  ax=ax
)
ax.set_xlabel("Predicted pIC50")
ax.set_ylabel("Density")

ax.axvline(7.1, color="red", linestyle="--", label="Test compound (pred. pIC50 = 7.1)")
#highlight the area under the curve for the test compound
x_fill = np.linspace(7.1, p_ic50_preds[label == 0].max(), 100)
y_fill = sns.kdeplot(p_ic50_preds[label == 0], bw_adjust=0.5).get_lines()[0].get_data()[1]
y_fill = np.interp(x_fill, sns.kdeplot(p_ic50_preds[label == 0], bw_adjust=0.5).get_lines()[0].get_data()[0], y_fill)
ax.fill_between(x_fill, y_fill, color="red", alpha=0.3, label="p-value for test compound")

#position legend
ax.legend(loc="upper left")
fig.tight_layout()

# %%
