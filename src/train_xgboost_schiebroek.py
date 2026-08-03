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
import os
import sys
import glob
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit.Chem import AllChem, rdFingerprintGenerator, Descriptors


# %%
import utils
from conformal_predictor import ConformalPredictor


# %%
DATA_DIR = os.path.join("..", "data", "data_ schiebroek_2026")
RESULTS_DIR = os.path.join("..", "results")
THRESHOLD = 7       # pchembl_value >= 6.0 → active (IC50 <= 1 µM)
CONF_THRESHOLD = 0.5  # classification probability cutoff
NUM_BITS = 2048
NUM_BOOST_ROUND = 100
MIN_CALIB_NEGATIVES = 10
SPLIT_TYPE = "butina"  # one of: "random", "scaffold", "butina", "umap"


# %%
def parse_pchembl(val):
    """Parse pchembl_value from tensor([x]) or plain float format."""
    s = str(val)
    match = re.search(r"tensor\(\[([0-9.]+)\]\)", s)
    if match:
        return float(match.group(1))
    return float(s)


# %%
def compute_features(df, smiles_col="smiles", target_col="label"):
    """Return (full_feature_array, fp_only_array, labels) for a dataframe of SMILES."""
    fp_array = utils.calculate_fingerprints(df[smiles_col].tolist(), num_bits=NUM_BITS, radius=2)
    fp_cols = [f"FP_{i}" for i in range(fp_array.shape[1])]

    desc_array = utils.get_rdkit_descriptors(df[smiles_col].tolist())
    desc_array[np.absolute(desc_array) > 1e20] = np.inf
    desc_cols = [nm for nm, fn in Descriptors._descList]

    feature_cols = fp_cols + desc_cols
    feat_df = pd.concat(
        [pd.DataFrame(fp_array, columns=fp_cols, index=df.index),
         pd.DataFrame(desc_array, columns=desc_cols, index=df.index),
         df[[target_col]]],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    return feat_df[feature_cols].to_numpy(), feat_df[fp_cols].to_numpy(), feat_df[target_col].to_numpy()


# %%
def load_and_split(prefix, split_type=SPLIT_TYPE):
    """Merge train/val/test CSVs for a prefix and re-split."""
    paths = [
        os.path.join(DATA_DIR, f"{prefix}_train_ach.csv"),
        os.path.join(DATA_DIR, f"{prefix}_val_ach.csv"),
        os.path.join(DATA_DIR, f"{prefix}_test.csv"),
    ]
    df = pd.concat(
        [pd.read_csv(p) for p in paths], ignore_index=True
    ).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    df["target"] = df["pchembl_value"].apply(parse_pchembl)
    n_before = len(df)
    
    df = df[np.isfinite(df["target"])].reset_index(drop=True)
    if len(df) < n_before:
        print(f"  dropped {n_before - len(df)} rows with non-finite target values")
    df["label"] = (df["target"] >= THRESHOLD).astype(int)

    smiles = df["smiles"].tolist()
    if split_type == "random":
        train_idx, val_idx, test_idx = utils.random_split(
            smiles, frac_train=0.5, frac_val=0.25
        )
    elif split_type == "scaffold":
        train_idx, val_idx, test_idx = utils.balanced_scaffold_split(
            smiles, frac_train=0.5, frac_val=0.25
        )
    elif split_type == "butina":
        train_idx, val_idx, test_idx = utils.cluster_based_split(
            smiles, frac_train=0.5, frac_val=0.25
        )
    elif split_type == "umap":
        train_idx, val_idx, test_idx = utils.umap_split(smiles)
    else:
        raise ValueError(f"Unknown split_type: {split_type!r}")

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


# %%
def train_model(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        "objective": "binary:logistic",
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    return xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND, verbose_eval=False)


# %%
def run_conformal_fdr(model, X_calib, y_calib, X_target, y_target, fp_calib, fp_target):
    y_pred_calib = model.predict(xgb.DMatrix(X_calib))
    y_pred_target = model.predict(xgb.DMatrix(X_target))

    neg_mask = y_calib == 0  # label 0 = inactive / calibration negatives

    # --- nonconformity scores (shared across all methods) ---
    nc_calib = ConformalPredictor.get_nonconformity_scores(
        y_pred_calib, y_calib, CONF_THRESHOLD, function="difference"
    )
    nc_target = ConformalPredictor.get_nonconformity_scores(
        y_pred_target, np.full(len(y_pred_target), CONF_THRESHOLD),
        CONF_THRESHOLD, function="difference"
    )

    # --- kNN: neg-only calibration, raw model scores ---
    knn_w = ConformalPredictor(method="knn", num_nn=-1, weighted=True)
    knn_w.fit(fp_calib[neg_mask], y_pred_calib[neg_mask], fp_target=fp_target, n_eff=100)
    knn_uw = ConformalPredictor(method="knn", num_nn=-1, weighted=False)
    knn_uw.fit(fp_calib[neg_mask], y_pred_calib[neg_mask])

    p_knn = knn_w.predict_pvalues(fp_target, y_pred_target)
    p_knn_uw = knn_uw.predict_pvalues(fp_target, y_pred_target)
    p_adjusted_knn = knn_w.p_adjust(p_knn, method="BH")
    p_adjusted_knn_uw = knn_uw.p_adjust(p_knn_uw, method="BH")

    # --- kNN: all calibration, nonconformity scores ---
    knn_nc_w = ConformalPredictor(method="knn", num_nn=-1, weighted=True)
    knn_nc_w.fit(fp_calib, nc_calib, fp_target=fp_target, n_eff=100)
    knn_nc_uw = ConformalPredictor(method="knn", num_nn=-1, weighted=False)
    knn_nc_uw.fit(fp_calib, nc_calib)

    p_knn_nc = knn_nc_w.predict_pvalues(fp_target, nc_target, nonconformities=True)
    p_knn_nc_uw = knn_nc_uw.predict_pvalues(fp_target, nc_target, nonconformities=True)
    p_adjusted_knn_nc = knn_nc_w.p_adjust(p_knn_nc, method="BH")
    p_adjusted_knn_nc_uw = knn_nc_uw.p_adjust(p_knn_nc_uw, method="BH")

    # --- KS: neg-only calibration, raw model scores ---
    ks_w = ConformalPredictor(method="ks", weighted=True)
    ks_w.fit(fp_calib[neg_mask], y_pred_calib[neg_mask], fp_target=fp_target, n_eff=100)

    p_ks = ks_w.predict_pvalues(fp_target, y_pred_target)
    p_adjusted_ks = ks_w.p_adjust(p_ks, method="BH")

    # --- KS: all calibration, nonconformity scores ---
    ks2_w = ConformalPredictor(method="ks", weighted=True)
    ks2_w.fit(fp_calib, nc_calib, fp_target=fp_target, n_eff=100)

    p_ks_nc = ks2_w.predict_pvalues(fp_target, nc_target, nonconformities=True)
    p_adjusted_ks_nc = ks2_w.p_adjust(p_ks_nc, method="BH")

    ks_distances = {
        "ks_neg": (ks_w.initial_ks_distance,  ks_w.final_ks_distance),
        "ks_nc":  (ks2_w.initial_ks_distance, ks2_w.final_ks_distance),
    }

    return {
        "y_target": y_target,
        "y_pred_target": y_pred_target,
        "p_adjusted_knn": p_adjusted_knn,
        "p_adjusted_knn_uw": p_adjusted_knn_uw,
        "p_adjusted_knn_nc": p_adjusted_knn_nc,
        "p_adjusted_knn_nc_uw": p_adjusted_knn_nc_uw,
        "p_adjusted_ks": p_adjusted_ks,
        "p_adjusted_ks_nc": p_adjusted_ks_nc,
        # predictors and score arrays for ECDF diagnostics
        "knn_w": knn_w, "knn_nc_w": knn_nc_w,
        "ks_w": ks_w, "ks2_w": ks2_w,
        "fp_target": fp_target,
        "scores_calib_neg": y_pred_calib[neg_mask],
        "nc_calib": nc_calib,
        "nc_target": nc_target,
        "ks_distances": ks_distances,
    }


# %%
def compute_fdr_curve(p_adjusted, y_target):
    """Sweep over p_adjusted thresholds and record true vs estimated FDR."""
    true_fdr, estim_fdr, num_selected = [], [], []
    for thresh in np.sort(p_adjusted):
        selected = p_adjusted <= thresh
        false_positives = (y_target[selected] == 0).sum()
        n = selected.sum()
        true_fdr.append(0 if n == 0 else false_positives / n)
        estim_fdr.append(thresh)
        num_selected.append(n)
    return estim_fdr, true_fdr, num_selected


# %%
def plot_ecdf_diagnostics(cfdr, prefix=""):
    """ECDF of mean 5-NN distances (calib→target vs target→target) before/after weighting."""
    fp_target = cfdr["fp_target"]
    panels = [
        (cfdr["knn_w"],    "kNN neg-only"),
        (cfdr["knn_nc_w"], "kNN all-calib (NC)"),
        (cfdr["ks_w"],     "KS neg-only"),
        (cfdr["ks2_w"],    "KS all-calib (NC)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (pred, label) in zip(axes.flatten(), panels):
        X_control, X_target = pred.get_nn_distances(fp_target, nns=5)
        w = pred.get_calibration_weights(fp_target=fp_target)

        # unweighted calibration ECDF
        sort_uw = np.argsort(X_control)
        x_uw = X_control[sort_uw]
        y_uw = np.arange(1, len(x_uw) + 1) / len(x_uw)
        ax.plot(x_uw, y_uw, label="Calibration (unweighted)", color="steelblue")

        # weighted calibration ECDF
        cw = np.cumsum(w[sort_uw])
        y_w = cw / cw[-1]
        ax.plot(x_uw, y_w, label="Calibration (weighted)", color="darkorange")

        # target ECDF
        x_tgt = np.sort(X_target)
        y_tgt = np.arange(1, len(x_tgt) + 1) / len(x_tgt)
        ax.plot(x_tgt, y_tgt, label="Target", color="green", linestyle="--")

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Mean 5-NN Tanimoto distance")
        ax.set_ylabel("ECDF")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    title = "ECDF: mean 5-NN distance — calibration before/after weighting vs target"
    if prefix:
        title += f" — {prefix}"
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# %%
def plot_fdr_curves(cfdr, prefix):
    y_target = cfdr["y_target"]

    curves = {
        "kNN weighted": cfdr["p_adjusted_knn"],
        "kNN unweighted": cfdr["p_adjusted_knn_uw"],
        "kNN NC weighted": cfdr["p_adjusted_knn_nc"],
        "kNN NC unweighted": cfdr["p_adjusted_knn_nc_uw"],
        "KS weighted": cfdr["p_adjusted_ks"],
        "KS NC weighted": cfdr["p_adjusted_ks_nc"],
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, p_adj in curves.items():
        estim, true, _ = compute_fdr_curve(p_adj, y_target)
        ax.plot(estim, true, marker="o", markersize=3, label=label)

    ax.plot([0, 1], [0, 1], linestyle="--", color="red", label="y = x")
    ax.set_xlabel("Estimated FDR")
    ax.set_ylabel("True FDR")
    ax.set_title(f"FDR Control — {prefix}")
    ax.grid(True)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# %%
def fdr_metrics_at_level(p_adjusted, y_target, fdr_level=0.2):
    selected = p_adjusted <= fdr_level
    n = selected.sum()
    if n == 0:
        return {"n_selected": 0, "true_fdr": float("nan"), "n_actives_selected": 0}
    fp = (y_target[selected] == 0).sum()
    tp = (y_target[selected] == 1).sum()
    return {"n_selected": int(n), "true_fdr": float(fp / n), "n_actives_selected": int(tp)}



# %%
def process_dataset(prefix):
    print(f"\n{'='*60}\n{prefix}")
    train_df, val_df, test_df = load_and_split(prefix)
    print(f"  sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    X_train, _ , y_train        = compute_features(train_df)
    X_calib, fp_calib, y_calib  = compute_features(val_df)
    X_target, fp_target, y_target = compute_features(test_df)

    model = train_model(X_train, y_train)

    y_pred_test = model.predict(xgb.DMatrix(X_target))
    auroc = roc_auc_score(y_target, y_pred_test)
    auprc = average_precision_score(y_target, y_pred_test)
    print(f"  test — AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")

    n_calib_neg  = int((y_calib == 0).sum())
    n_test_active = int((y_target == 1).sum())
    print(f"  test actives: {n_test_active}/{len(y_target)}, calib negatives: {n_calib_neg}")

    if n_calib_neg < MIN_CALIB_NEGATIVES:
        print(f"  skipping conformal FDR (too few calibration negatives)")
        return {
            "prefix": prefix,
            "n_train": len(train_df), "n_calib": len(val_df), "n_test": len(test_df),
            "n_test_active": n_test_active, "n_calib_neg": n_calib_neg,
            "auroc_test": auroc, "auprc_test": auprc,
            "conformal_ran": False,
        }

    cfdr = run_conformal_fdr(model, X_calib, y_calib, X_target, y_target, fp_calib, fp_target)

    plot_fdr_curves(cfdr, prefix)
    plot_ecdf_diagnostics(cfdr, prefix)

    m = fdr_metrics_at_level(cfdr["p_adjusted_knn"], y_target)
    print(f"  FDR@0.2 — selected: {m['n_selected']}, true FDR: {m['true_fdr']:.4f}, actives found: {m['n_actives_selected']}")

    return {
        "prefix": prefix,
        "n_train": len(train_df), "n_calib": len(val_df), "n_test": len(test_df),
        "n_test_active": n_test_active, "n_calib_neg": n_calib_neg,
        "auroc_test": auroc, "auprc_test": auprc,
        "conformal_ran": True,
        "fdr02_n_selected": m["n_selected"],
        "fdr02_true_fdr": m["true_fdr"],
        "fdr02_n_actives": m["n_actives_selected"],
        "ks_distances": cfdr["ks_distances"],
        "conformal_results": cfdr,
    }


# %%
train_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_train_ach.csv")))
train_files = [f for f in train_files if "_df_" not in os.path.basename(f)]
prefixes = [os.path.basename(f).replace("_train_ach.csv", "") for f in train_files]
print(f"Found {len(prefixes)} datasets")

os.makedirs(RESULTS_DIR, exist_ok=True)

all_rows = []
all_conformal = []
for prefix in prefixes[:100]:
    row = process_dataset(prefix)
    try:
        all_rows.append({k: v for k, v in row.items() if k != "conformal_results"})
        if row.get("conformal_ran") and "conformal_results" in row:
            all_conformal.append({"prefix": prefix, **row["conformal_results"]})
    except Exception as exc:
        print(f"  ERROR: {exc}")
        all_rows.append({"prefix": prefix, "error": str(exc)})

summary = pd.DataFrame(all_rows)
out_path = os.path.join(RESULTS_DIR, "schiebroek_results_" + SPLIT_TYPE + "_" + str(THRESHOLD) + ".csv")
summary.to_csv(out_path, index=False)
print(f"\nSaved summary to {out_path}")
print(summary.describe())

conformal_path = os.path.join(RESULTS_DIR, "schiebroek_conformal_results_" + SPLIT_TYPE + "_" + str(THRESHOLD) + ".pkl")
with open(conformal_path, "wb") as f:
    pickle.dump(all_conformal, f)
print(f"Saved {len(all_conformal)} conformal results → {conformal_path}")


# %%
#read all conformal results
conformal_path = os.path.join(RESULTS_DIR, "schiebroek_conformal_results_" + SPLIT_TYPE + "_" + str(THRESHOLD) + ".pkl")
with open(conformal_path, 'rb') as f:
    all_conformal = pickle.load(f)
    
# Heatmap: one separate figure per method
method_keys = {
    "p_adjusted_knn":      "kNN weighted",
    "p_adjusted_knn_uw":   "kNN unweighted",
    "p_adjusted_knn_nc":   "kNN NC weighted",
    "p_adjusted_knn_nc_uw": "kNN NC unweighted",
    "p_adjusted_ks":       "KS weighted",
    "p_adjusted_ks_nc":    "KS NC weighted",
}

for key, label in method_keys.items():
    estim_all, true_all = [], []
    for cfdr in all_conformal:
        estim, true, _ = compute_fdr_curve(cfdr[key], cfdr["y_target"])
        estim_all.extend(estim)
        true_all.extend(true)

    fig, ax = plt.subplots(figsize=(7, 6))
    h = ax.hist2d(estim_all, true_all, bins=20, range=[[0, 1], [0, 1]], cmap="YlOrRd")
    plt.colorbar(h[3], ax=ax, label="Count")
    ax.plot([0, 1], [0, 1], linestyle="--", color="blue", label="y = x (ideal)")
    ax.set_xlabel("Estimated FDR")
    ax.set_ylabel("True FDR")
    ax.set_title(f"Estimated vs True FDR — {label}")
    ax.legend()
    plt.tight_layout()
    plt.show()

# %%
# KS distance summary: initial vs final across all datasets
ks_method_keys = {
    "ks_neg": "KS neg-only",
    "ks_nc":  "KS all-calib (NC)",
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (key, label) in zip(axes.flatten(), ks_method_keys.items()):
    init_vals, final_vals = [], []
    for cfdr in all_conformal:
        ksd = cfdr.get("ks_distances", {}).get(key)
        if ksd is not None:
            init_vals.append(ksd[0])
            final_vals.append(ksd[1])
    init_vals  = np.array(init_vals)
    final_vals = np.array(final_vals)

    ax.scatter(init_vals, final_vals, alpha=0.5, s=20, color="steelblue")
    lim = max(init_vals.max(), final_vals.max()) * 1.05 if len(init_vals) else 1.0
    ax.plot([0, lim], [0, lim], linestyle="--", color="red", label="no change")
    ax.set_xlabel("Initial KS distance")
    ax.set_ylabel("Final KS distance")
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    delta = init_vals - final_vals
    print(f"{label}: median reduction = {np.median(delta):.4f}  "
          f"(init {np.median(init_vals):.4f} → final {np.median(final_vals):.4f})")

fig.suptitle("KS distance before vs after weighting (one point per dataset)", fontsize=12)
plt.tight_layout()
plt.show()
plt.close(fig)

# %%
