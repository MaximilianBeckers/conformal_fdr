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
from ks_conformal_predictor import KSConformalPredictor


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
def compute_features(df, smiles_col="smiles", target_col = "label"):
    """Return (full_feature_array, fp_only_array) for a list of SMILES."""
    
    #get feature
    fp_array = utils.calculate_fingerprints(df[smiles_col].tolist(), num_bits=NUM_BITS, radius=2)
    fp_cols = [f"FP_{i}" for i in range(fp_array.shape[1])]
    df[fp_cols] = fp_array

    desc_array = utils.get_rdkit_descriptors(df[smiles_col].tolist())
    desc_array[np.absolute(desc_array)>1e20] = np.inf
    desc_cols = [nm for nm,fn in Descriptors._descList]
    df[desc_cols] = desc_array
    
    
    feature_cols = fp_cols + desc_cols
    df[feature_cols + [target_col]] = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    return df[feature_cols].to_numpy(), df[fp_cols].to_numpy(), df[target_col].to_numpy()


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

    # --- entropy balancing: neg-only calibration, raw model scores ---
    ebc_w = ConformalPredictor(method="entropy_balancing", weighted=True, ks_bound=0.2, ks_penalty=0)
    ebc_w.fit(fp_calib[neg_mask], y_pred_calib[neg_mask], fp_target=fp_target)
    ebc_uw = ConformalPredictor(method="entropy_balancing", weighted=False)
    ebc_uw.fit(fp_calib[neg_mask], y_pred_calib[neg_mask])

    print("Initial KS distance: " + str(ebc_w.initial_ks_distance))
    print("Final KS distance:   " + str(ebc_w.final_ks_distance))
    print("Effective Sample Size: " + str(ebc_w.effective_sample_size))

    p_values = ebc_w.predict_pvalues(fp_target, y_pred_target)
    p_values_uw = ebc_uw.predict_pvalues(fp_target, y_pred_target)
    p_adjusted = ebc_w.p_adjust(p_values, method="BH")
    p_adjusted_uw = ebc_uw.p_adjust(p_values_uw, method="BH")

    # --- entropy balancing: all calibration, nonconformity scores ---
    nc_calib = ConformalPredictor.get_nonconformity_scores(
        y_pred_calib, y_calib, CONF_THRESHOLD, function="difference"
    )
    nc_target = ConformalPredictor.get_nonconformity_scores(
        y_pred_target, np.full(len(y_pred_target), CONF_THRESHOLD),
        CONF_THRESHOLD, function="difference"
    )

    ebc2_w = ConformalPredictor(method="entropy_balancing", weighted=True, ks_bound=0.2, ks_penalty=0)
    ebc2_w.fit(fp_calib, nc_calib, fp_target=fp_target)
    ebc2_uw = ConformalPredictor(method="entropy_balancing", weighted=False)
    ebc2_uw.fit(fp_calib, nc_calib)

    p_nc = ebc2_w.predict_pvalues(fp_target, nc_target, nonconformities=True)
    p_nc_uw = ebc2_uw.predict_pvalues(fp_target, nc_target, nonconformities=True)
    p_adjusted_nc = ebc2_w.p_adjust(p_nc, method="BH")
    p_adjusted_nc_uw = ebc2_uw.p_adjust(p_nc_uw, method="BH")

    # --- kNN: neg-only calibration, raw model scores ---
    knn_w = ConformalPredictor(method="knn", num_nn=30, weighted=True)
    knn_w.fit(fp_calib[neg_mask], y_pred_calib[neg_mask])
    knn_uw = ConformalPredictor(method="knn", num_nn=30, weighted=False)
    knn_uw.fit(fp_calib[neg_mask], y_pred_calib[neg_mask])

    p_knn = knn_w.predict_pvalues(fp_target, y_pred_target)
    p_knn_uw = knn_uw.predict_pvalues(fp_target, y_pred_target)
    p_adjusted_knn = knn_w.p_adjust(p_knn, method="BH")
    p_adjusted_knn_uw = knn_uw.p_adjust(p_knn_uw, method="BH")

    # --- kNN: all calibration, nonconformity scores ---
    knn_nc_w = ConformalPredictor(method="knn", num_nn=30, weighted=True)
    knn_nc_w.fit(fp_calib, nc_calib)
    knn_nc_uw = ConformalPredictor(method="knn", num_nn=30, weighted=False)
    knn_nc_uw.fit(fp_calib, nc_calib)

    p_knn_nc = knn_nc_w.predict_pvalues(fp_target, nc_target, nonconformities=True)
    p_knn_nc_uw = knn_nc_uw.predict_pvalues(fp_target, nc_target, nonconformities=True)
    p_adjusted_knn_nc = knn_nc_w.p_adjust(p_knn_nc, method="BH")
    p_adjusted_knn_nc_uw = knn_nc_uw.p_adjust(p_knn_nc_uw, method="BH")

    return {
        "y_target": y_target,
        "y_pred_target": y_pred_target,
        "p_values": p_values,
        "p_adjusted": p_adjusted,
        "p_adjusted_uw": p_adjusted_uw,
        "p_adjusted_nc": p_adjusted_nc,
        "p_adjusted_nc_uw": p_adjusted_nc_uw,
        "p_adjusted_knn": p_adjusted_knn,
        "p_adjusted_knn_uw": p_adjusted_knn_uw,
        "p_adjusted_knn_nc": p_adjusted_knn_nc,
        "p_adjusted_knn_nc_uw": p_adjusted_knn_nc_uw,
        "nn_dist_calib": ebc2_w.X_control,
        "nn_dist_target": ebc2_w.X_target,
        "ebc": ebc2_w,
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
def plot_fdr_curves(cfdr, prefix, save_dir):
    y_target = cfdr["y_target"]

    curves = {
        "Weighted (neg-only)": cfdr["p_adjusted"],
        "Unweighted (neg-only)": cfdr["p_adjusted_uw"],
        "Nonconformity weighted": cfdr["p_adjusted_nc"],
        "Nonconformity unweighted": cfdr["p_adjusted_nc_uw"],
        "kNN weighted": cfdr["p_adjusted_knn"],
        "kNN unweighted": cfdr["p_adjusted_knn_uw"],
        "kNN NC weighted": cfdr["p_adjusted_knn_nc"],
        "kNN NC unweighted": cfdr["p_adjusted_knn_nc_uw"],
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

    out_path = os.path.join(save_dir, f"{prefix}_fdr_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  FDR plot saved → {out_path}")


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

    plot_fdr_curves(cfdr, prefix, RESULTS_DIR)

    m = fdr_metrics_at_level(cfdr["p_adjusted"], y_target)
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
        "conformal_results": cfdr,
    }


# %%
if __name__ == "__main__":
    train_files = sorted(glob.glob(os.path.join(DATA_DIR, "CHEMBL226*_train_ach.csv")))
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
    out_path = os.path.join(RESULTS_DIR, "schiebroek_results.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")
    print(summary.describe())

    conformal_path = os.path.join(RESULTS_DIR, "schiebroek_conformal_results.pkl")
    with open(conformal_path, "wb") as f:
        pickle.dump(all_conformal, f)
    print(f"Saved {len(all_conformal)} conformal results → {conformal_path}")


    # Heatmap: one separate figure per method
    method_keys = {
        "p_adjusted":       "Weighted (neg-only)",
        "p_adjusted_uw":    "Unweighted (neg-only)",
        "p_adjusted_nc":    "Nonconformity weighted",
        "p_adjusted_nc_uw": "Nonconformity unweighted",
        "p_adjusted_knn":      "kNN weighted",
        "p_adjusted_knn_uw":   "kNN unweighted",
        "p_adjusted_knn_nc":   "kNN NC weighted",
        "p_adjusted_knn_nc_uw": "kNN NC unweighted",
    }

    if all_conformal:
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
            heatmap_path = os.path.join(RESULTS_DIR, f"fdr_heatmap_{key}.png")
            fig.savefig(heatmap_path, dpi=150)
            plt.show()
            print(f"Heatmap saved → {heatmap_path}")


# %%

# %%
