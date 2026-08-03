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
"""
Synthetic scaffold R-group enumeration experiment.

Two scaffolds:
  A —  (majority in train + calibration)
  B —  (minority in calibration, 100% of test)

Binary label: MolLogP >= LOGP_THRESHOLD → active (lipophilic).

Goal: verify that conformal KS importance weights correctly identify scaffold-B
compounds in the calibration set, reducing the NN-distance distribution shift
and improving FDR control on the all-B test set.
"""
import itertools
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import umap
from rdkit import Chem
from rdkit.Chem import Crippen, rdFingerprintGenerator, Descriptors, Draw
from rdkit.Chem.rdmolops import molzip
from IPython.display import display
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

import utils
from conformal_predictor import ConformalPredictor

# %%
# Scaffold SMILES with three labeled attachment points [*:1], [*:2], [*:3].
# molzip pairs dummy atoms by map number — no ring-bond label conflicts.
SCAFFOLD_A = "c1cc(-c2c([*:1])nn3nc([*:2])ccc23)nc(N(c2cccc([*:3])c2))n1"      # 1,3,5-trisubstituted benzene
SCAFFOLD_B = "C1NC([*:1])CN([*:3])C1[*:2]"       # piperidine: C4,C2 + N1 substituted

LOGP_THRESHOLD = 3  # label=1 if MolLogP >= threshold
NUM_BITS = 2048
RADIUS   = 2
SEED     = 42
Noise_A  = 0.5
Noise_B  = 0.5
# Single R-group list — each fragment SMILES starts with the dummy atom [*].
# molzip is called twice: once with [*:1]-tagged mols and once with [*:2]-tagged
# mols, so every fragment can appear at either position without orientation issues.
R_GROUPS = [
    "[*]C",           # methyl
    "[*]CC",          # ethyl
    "[*]CCC",         # propyl
    "[*]OC",          # methoxy
    "[*]O",           # hydroxy
    "[*]N",           # amino
    "[*]F",           # fluoro
    "[*]Cl",          # chloro
    "[*]Br",          # bromo
    "[*]C(F)(F)F",    # trifluoromethyl
    "[*]C(C)(C)C",    # tert-butyl
    "[*]C#N",         # cyano
    "[*]c1ccccc1",    # phenyl
    "[*]c1ccncc1",    # 4-pyridyl
    "[*]c1ccco1",     # 2-furanyl
    "[*]c1cccs1",     # 2-thienyl
    "[*]C1CC1",       # cyclopropyl
    "[*]C1CCCCC1",    # cyclohexyl
]

# %%
# Show the two scaffold structures.
_scaffold_mols = [Chem.MolFromSmiles(SCAFFOLD_A), Chem.MolFromSmiles(SCAFFOLD_B)]
display(Draw.MolsToGridImage(
    _scaffold_mols, molsPerRow=2, subImgSize=(450, 300),
    legends=["Scaffold A", "Scaffold B"],
))


# %%
def _tag(frag_smi: str, map_num: int) -> Chem.Mol:
    """Replace [*] in a fragment SMILES with [*:map_num] and return the Mol."""
    tagged = frag_smi.replace("[*]", f"[*:{map_num}]", 1)
    return Chem.MolFromSmiles(tagged)


def enumerate_compounds(scaffold_smi: str, r_groups: list[str]) -> pd.DataFrame:
    """
    Enumerate all R1×R2×R3 combinations for a scaffold using RDKit molzip.

    The scaffold must contain [*:1], [*:2], and [*:3] as labeled attachment
    points. Each fragment is tried at every position independently.
    Invalid combinations (valence errors, sanitisation failures) are silently
    dropped.
    """
    scaffold = Chem.MolFromSmiles(scaffold_smi)
    records, seen = [], set()

    for frag1, frag2, frag3 in itertools.product(r_groups, r_groups, r_groups):
        r1 = _tag(frag1, 1)
        r2 = _tag(frag2, 2)
        r3 = _tag(frag3, 3)
        if r1 is None or r2 is None or r3 is None:
            continue
        try:
            mol = molzip(scaffold, r1)
            mol = molzip(mol, r2)
            mol = molzip(mol, r3)
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen:
            continue
        seen.add(can)
        records.append({"smiles": can, "r1": frag1, "r2": frag2, "r3": frag3})
    return pd.DataFrame(records)


def _extract(df, fp_cols, desc_cols):
    """Pull (X_full, fp_only, y) from a pre-computed feature DataFrame."""
    X    = df[fp_cols + desc_cols].to_numpy()
    fp   = df[fp_cols].to_numpy().astype(np.uint8)
    y    = df["label"].to_numpy()
    return X, fp, y


# %%
def compute_fdr_curve(p_adj, y_true):
    estim, true = [], []
    for t in np.sort(p_adj):
        sel = p_adj <= t
        n   = sel.sum()
        fp  = (y_true[sel] == 0).sum()
        estim.append(t)
        true.append(0.0 if n == 0 else fp / n)
    return np.array(estim), np.array(true)


def fdr_at_level(p_adj, y_true, level=0.2):
    sel = p_adj <= level
    n   = sel.sum()
    if n == 0:
        return {"n": 0, "true_fdr": float("nan"), "actives": 0}
    fp = (y_true[sel] == 0).sum()
    return {"n": int(n), "true_fdr": float(fp / n), "actives": int((y_true[sel] == 1).sum())}



# %%
df_a = enumerate_compounds(SCAFFOLD_A, R_GROUPS)
df_a["scaffold"] = "A"
df_b = enumerate_compounds(SCAFFOLD_B, R_GROUPS)
df_b["scaffold"] = "B"
df_all = pd.concat([df_a, df_b], ignore_index=True)

# Compute fingerprints + RDKit descriptors for all compounds at once.
# This makes MolLogP (and all other descriptors) available before any split.
smiles_all = df_all["smiles"].tolist()

fp_array   = utils.calculate_fingerprints(smiles_all, num_bits=NUM_BITS, radius=RADIUS)
desc_array = utils.get_rdkit_descriptors(smiles_all)
desc_array[np.absolute(desc_array) > 1e20] = np.inf

fp_cols   = [f"FP_{i}" for i in range(fp_array.shape[1])]
desc_cols = [nm for nm, _ in Descriptors._descList]

feat_df = pd.concat([
    df_all[["smiles", "scaffold", "r1", "r2", "r3"]].reset_index(drop=True),
    pd.DataFrame(fp_array,   columns=fp_cols),
    pd.DataFrame(desc_array, columns=desc_cols),
], axis=1).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


#add noise based on defined noise distributions
rng = np.random.default_rng(SEED)

s_a = rng.normal(0, Noise_A, df_a.shape[0])
s_b = rng.normal(0, Noise_B, df_b.shape[0])
s_combined = np.concatenate((s_a, s_b))

feat_df["MolLogP"] = feat_df["MolLogP"] + s_combined

# Label from MolLogP — now available from the descriptor set
feat_df["label"] = (feat_df["MolLogP"] >= LOGP_THRESHOLD).astype(int)

# %%
# UMAP embedding of all compounds coloured by scaffold.
# Jaccard metric is appropriate for binary Morgan fingerprints.
_fp_umap = feat_df[fp_cols].to_numpy().astype(np.uint8)
_reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                     metric="jaccard", random_state=SEED)
_emb = _reducer.fit_transform(_fp_umap)

_palette = {"A": "steelblue", "B": "tomato"}
fig, ax = plt.subplots(figsize=(8, 6))
for sc, col in _palette.items():
    _m = feat_df["scaffold"] == sc
    ax.scatter(_emb[_m, 0], _emb[_m, 1], s=6, alpha=0.5, color=col, label=f"Scaffold {sc}")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("UMAP of Morgan fingerprints — coloured by scaffold")
ax.legend(markerscale=3)
plt.tight_layout()
plt.show()
plt.close(fig)

df_a = feat_df[feat_df["scaffold"] == "A"].reset_index(drop=True)
df_b = feat_df[feat_df["scaffold"] == "B"].reset_index(drop=True)

print(f"Scaffold A (benzene):  {len(df_a)} compounds")
print(f"Scaffold B (pyridine): {len(df_b)} compounds")
print("\nMolLogP by scaffold:")
print(feat_df.groupby("scaffold")["MolLogP"].describe().round(2))
print(f"\nActive rate — A: {df_a['label'].mean():.1%}  B: {df_b['label'].mean():.1%}")


# %%
# Scaffold A: 90% train, 10% calib, 0% test
# Scaffold B: 50% train, 10% calib, 40% test

idx_a  = rng.permutation(len(df_a))
cut_a  = int(0.90 * len(df_a))
train_a = df_a.iloc[idx_a[:cut_a]].reset_index(drop=True)
calib_a = df_a.iloc[idx_a[cut_a:]].reset_index(drop=True)

idx_b  = rng.permutation(len(df_b))
cut_b1 = int(0.50 * len(df_b))
cut_b2 = int(0.60 * len(df_b))
train_b = df_b.iloc[idx_b[:cut_b1]].reset_index(drop=True)
calib_b = df_b.iloc[idx_b[cut_b1:cut_b2]].reset_index(drop=True)
test_b  = df_b.iloc[idx_b[cut_b2:]].reset_index(drop=True)

df_train = pd.concat([train_a, train_b], ignore_index=True)
df_calib = pd.concat([calib_a, calib_b], ignore_index=True)
df_test  = test_b.copy()

print("Split:")
print(f"  Train: {len(df_train)}  ({len(train_a)} A + {len(train_b)} B)")
print(f"  Calib: {len(df_calib)}  ({len(calib_a)} A + {len(calib_b)} B)")
print(f"  Test:  {len(df_test)}  (all B)")

X_train, fp_train, y_train = _extract(df_train, fp_cols, [])
X_calib, fp_calib, y_calib = _extract(df_calib, fp_cols, [])
X_test,  fp_test,  y_test  = _extract(df_test,  fp_cols, [])

calib_scaffold = df_calib["scaffold"].to_numpy()

# %%
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {
    "objective":        "binary:logistic",
    "max_depth":        4,
    "eta":              0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "seed":             SEED,
}
model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)

y_pred_calib = model.predict(xgb.DMatrix(X_calib))
y_pred_test  = model.predict(xgb.DMatrix(X_test))

#plot the roc curve
plt.figure(figsize=(6, 6))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
plt.plot(fpr, tpr, color='blue', label='ROC curve')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()

print(f"Test AUROC: {roc_auc_score(y_test, y_pred_test):.4f}")
print(f"Test AUPRC: {average_precision_score(y_test, y_pred_test):.4f}")
print(f"Test actives: {y_test.sum()} / {len(y_test)}")


# %%
y_pred_test

# %%
neg_mask = y_calib == 0

# NC scores: label - model_score (all calib compounds, not just negatives).
# For test compounds the label is set to conf_threshold (0.5), not the true label —
# conformal prediction tests each compound under the null (score at boundary).
nc_calib  = ConformalPredictor.get_nonconformity_scores(y_pred_calib, y_calib, 0.5)
nc_target = ConformalPredictor.get_nonconformity_scores(
    y_pred_test, np.full(len(y_pred_test), 0.5), 0.5)

# --- neg-only methods (score-based) ---
knn_uw = ConformalPredictor(method="knn", num_nn=-1, weighted=False)
knn_uw.fit(fp_calib[neg_mask], y_pred_calib[neg_mask])
p_knn_uw = knn_uw.p_adjust(knn_uw.predict_pvalues(fp_test, y_pred_test), method="BH")

knn_w = ConformalPredictor(method="knn", num_nn=-1, weighted=True)
knn_w.fit(fp_calib[neg_mask], y_pred_calib[neg_mask], fp_target=fp_test, n_eff=100)
p_knn_w = knn_w.p_adjust(knn_w.predict_pvalues(fp_test, y_pred_test), method="BH")

ks_w = ConformalPredictor(method="ks", weighted=True)
ks_w.fit(fp_calib[neg_mask], y_pred_calib[neg_mask], fp_target=fp_test, n_eff=100)
p_ks = ks_w.p_adjust(ks_w.predict_pvalues(fp_test, y_pred_test), method="BH")

# --- all-calib methods (nonconformity-score based) ---
knn_nc_uw = ConformalPredictor(method="knn", num_nn=-1, weighted=False)
knn_nc_uw.fit(fp_calib, nc_calib)
p_knn_nc_uw = knn_nc_uw.p_adjust(
    knn_nc_uw.predict_pvalues(fp_test, nc_target, nonconformities=True), method="BH")

knn_nc_w = ConformalPredictor(method="knn", num_nn=-1, weighted=True)
knn_nc_w.fit(fp_calib, nc_calib, fp_target=fp_test, n_eff=100)
p_knn_nc_w = knn_nc_w.p_adjust(
    knn_nc_w.predict_pvalues(fp_test, nc_target, nonconformities=True), method="BH")

ks2_w = ConformalPredictor(method="ks", weighted=True)
ks2_w.fit(fp_calib, nc_calib, fp_target=fp_test, n_eff=100)
p_ks_nc = ks2_w.p_adjust(
    ks2_w.predict_pvalues(fp_test, nc_target, nonconformities=True), method="BH")

print(f"\nKS neg-only:  initial={ks_w.initial_ks_distance:.4f} → final={ks_w.final_ks_distance:.4f}  ESS={ks_w.effective_sample_size:.1f}/{neg_mask.sum()}")
print(f"KS NC (all):  initial={ks2_w.initial_ks_distance:.4f} → final={ks2_w.final_ks_distance:.4f}  ESS={ks2_w.effective_sample_size:.1f}/{len(y_calib)}")


# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df_a["MolLogP"], bins=25, alpha=0.6, color="steelblue", label="Scaffold A (benzene)")
ax.hist(df_b["MolLogP"], bins=25, alpha=0.6, color="tomato",    label="Scaffold B (pyridine)")
ax.axvline(LOGP_THRESHOLD, color="black", linestyle="--", label=f"Threshold = {LOGP_THRESHOLD}")
ax.set_xlabel("MolLogP")
ax.set_ylabel("Count")
ax.set_title("LogP distribution by scaffold")
ax.legend()
plt.tight_layout()
plt.show()
plt.close(fig)

# %%
X_ctrl_ks,  X_tgt_ks  = ks_w.get_nn_distances(fp_test, nns=5)
X_ctrl_knn, X_tgt_knn = knn_w.get_nn_distances(fp_test, nns=5)
w_ks  = ks_w.get_calibration_weights()
w_knn = knn_w.get_calibration_weights(fp_target=fp_test)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (X_ctrl, X_tgt, w, title) in zip(axes, [
    (X_ctrl_ks,  X_tgt_ks,  w_ks,  "KS weighted"),
    (X_ctrl_knn, X_tgt_knn, w_knn, "kNN weighted"),
]):
    sort_idx = np.argsort(X_ctrl)
    x_c  = X_ctrl[sort_idx]
    y_uw = np.arange(1, len(x_c) + 1) / len(x_c)
    cw   = np.cumsum(w[sort_idx])
    y_w  = cw / cw[-1]
    x_t  = np.sort(X_tgt)
    y_t  = np.arange(1, len(x_t) + 1) / len(x_t)

    ax.plot(x_c, y_uw, color="steelblue",  label="Calib (unweighted)")
    ax.plot(x_c, y_w,  color="darkorange", label="Calib (weighted)")
    ax.plot(x_t, y_t,  color="green", linestyle="--", label="Test (scaffold B)")
    ax.set_xlabel("Mean 5-NN Tanimoto distance")
    ax.set_ylabel("ECDF")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("NN-distance ECDF: calibration before/after weighting vs test", fontsize=12)
plt.tight_layout()
plt.show()
plt.close(fig)


# %%
calib_scaffold_neg = calib_scaffold[neg_mask]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (w, title) in zip(axes, [(w_ks, "KS weights"), (w_knn, "kNN avg weights")]):
    mask_a = calib_scaffold_neg == "A"
    mask_b = calib_scaffold_neg == "B"
    ax.boxplot(
        [w[mask_a], w[mask_b]],
        labels=[f"Scaffold A\n(benzene, n={mask_a.sum()})",
                f"Scaffold B\n(pyridine, n={mask_b.sum()})"],
        showfliers=0
    )
    ax.axhline(1.0, linestyle="--", color="red", alpha=0.5, label="uniform weight")
    ax.set_ylabel("Importance weight")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("Do KS weights correctly up-weight scaffold-B calibration compounds?", fontsize=12)
plt.tight_layout()
plt.show()
plt.close(fig)

# %%
fig, ax = plt.subplots(figsize=(7, 6))
for label, p_adj in [
    ("kNN weighted",      p_knn_w),
    ("kNN unweighted",    p_knn_uw),
    ("kNN NC weighted",   p_knn_nc_w),
    ("kNN NC unweighted", p_knn_nc_uw),
    ("KS weighted",       p_ks),
    ("KS NC weighted",    p_ks_nc),
]:
    e, t = compute_fdr_curve(p_adj, y_test)
    ax.plot(e, t, marker="o", markersize=3, label=label)
ax.plot([0, 1], [0, 1], linestyle="--", color="red", label="y = x")
ax.set_xlabel("FDR Threshold")
ax.set_ylabel("True FDR")
ax.set_title("FDR control — test on scaffold B only\n(calibrated on mixed A+B set)")
ax.legend(fontsize=8)
ax.grid(True)
plt.tight_layout()
plt.show()
plt.close(fig)


# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (label, p_adj) in zip(axes.flatten(), [
    ("kNN weighted",      p_knn_w),
    ("kNN unweighted",    p_knn_uw),
    ("kNN NC weighted",   p_knn_nc_w),
    ("kNN NC unweighted", p_knn_nc_uw),
    ("KS weighted",       p_ks),
    ("KS NC weighted",    p_ks_nc),
]):
    active = y_test == 1
    ax.scatter(y_pred_test[~active], p_adj[~active], s=15, alpha=0.4,
               color="steelblue", label="inactive")
    ax.scatter(y_pred_test[active],  p_adj[active],  s=15, alpha=0.6,
               color="tomato",    label="active")
    ax.axhline(0.2, linestyle="--", color="black", linewidth=0.8, label="FDR=0.2")
    ax.set_xlabel("Model score")
    ax.set_ylabel("FDR threshold")
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

fig.suptitle("Model score vs estimated FDR — test compounds (all scaffold B)", fontsize=12)
plt.tight_layout()
plt.show()
plt.close(fig)

# %%

# %%

# %%
