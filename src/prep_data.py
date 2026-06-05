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
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator, Descriptors
import umap
from sklearn.cluster import KMeans

import utils

# %%
#read the data
input_file = "../data/expansion_data_train.csv"
output_file = "../data/expansion_data_prep_with_splits_KSOL.csv"

df = pd.read_csv(input_file)
df.head()

num_bits = 2048
radius_fp = 2
feature_cols = [f"FP_{i}" for i in range(num_bits)] + [nm for nm,fn in Descriptors._descList]
target_col = "KSOL"

num_clusters = 10

#get features
fp_array = utils.calculate_fingerprints(df["SMILES"].tolist(), num_bits=num_bits, radius=radius_fp)
df[[f"FP_{i}" for i in range(fp_array.shape[1])]] = fp_array

desc_array = utils.get_rdkit_descriptors(df["SMILES"].tolist())
df[[nm for nm,fn in Descriptors._descList]] = desc_array

df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
print(f"Data shape after dropping NaNs: {df.shape}")

# %%
#now calclate umap embeddings based on the fingerprints
def calculate_umap_embeddings(fp_array, n_components=2):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(fp_array)
    return embedding

df[["UMAP_1", "UMAP_2"]] = calculate_umap_embeddings(df[[f"FP_{i}" for i in range(num_bits)]].values, n_components=2)

#now cluster the umap embeddings using k-means
def cluster_umap_embeddings(embedding, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embedding)
    return cluster_labels
df["Cluster"] = cluster_umap_embeddings(df[["UMAP_1", "UMAP_2"]].values, n_clusters=num_clusters)

# %%
#now set up data splits

#scaffold split
train_idx_scaffold, val_idx_scaffold, test_idx_scaffold = utils.balanced_scaffold_split(df["SMILES"].tolist(), frac_train=0.8, frac_val=0.1, seed=42)
print(f"Scaffold split: Train: {len(train_idx_scaffold)}, Val: {len(val_idx_scaffold)}, Test: {len(test_idx_scaffold)}")
df["scaffold_split"] = "test"
df.loc[train_idx_scaffold, "scaffold_split"] = "train"
df.loc[val_idx_scaffold, "scaffold_split"] = "val"

#cluster-based split
train_idx_cluster, val_idx_cluster, test_idx_cluster = utils.cluster_based_split(df["SMILES"].tolist(), frac_train=0.8, frac_val=0.1, random_seed=42, distance_threshold=0.3)
print(f"Cluster-based split: Train: {len(train_idx_cluster)}, Val: {len(val_idx_cluster)}, Test: {len(test_idx_cluster)}")
df["cluster_split"] = "test"
df.loc[train_idx_cluster, "cluster_split"] = "train"
df.loc[val_idx_cluster, "cluster_split"] = "val"

#random split
train_idx_random, val_idx_random, test_idx_random = np.split(np.random.permutation(np.arange(len(df))), [int(0.8*len(df)), int(0.9*len(df))])
print(f"Random split: Train: {len(train_idx_random)}, Val: {len(val_idx_random)}, Test: {len(test_idx_random)}")
df["random_split"] = "test"
df.loc[train_idx_random, "random_split"] = "train"
df.loc[val_idx_random, "random_split"] = "val"

#umap based cluster split
umap_cluster_col = "Cluster"
val_cluster = 0
test_cluster = 1

df["umap_cluster_split"] = "train"
df.loc[df[umap_cluster_col] == val_cluster, "umap_cluster_split"] = "val"
df.loc[df[umap_cluster_col] == test_cluster, "umap_cluster_split"] = "test"
print(f"UMAP cluster-based split: Train: {(df['umap_cluster_split'] == 'train').sum()}, Val: {(df['umap_cluster_split'] == 'val').sum()}, Test: {(df['umap_cluster_split'] == 'test').sum()}")


# %%
#now plot the umap embeddings colored by the different splits
import matplotlib.pyplot as plt
import seaborn as sns

f, axs = plt.subplots(1, 4, figsize=(20, 5))

for i, tmp_split in enumerate(["scaffold_split", "cluster_split", "random_split", "umap_cluster_split"]):
    sns.scatterplot(data=df, x="UMAP_1", y="UMAP_2", hue=tmp_split, ax=axs[i], hue_order=["train", "val", "test"], s=50)
    axs[i].set_title(tmp_split.replace("_", " ").title())

plt.tight_layout()
plt.show()

# %%
df.to_csv(output_file, index=False)

# %%




# %%
