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
import pickle
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score

import utils
from conformal_predictor import ConformalPredictor

RESULTS_DIR = os.path.join("..", "results")


# %%
def read_dataset(target_col_orig, task_type="classification", threshold=150, log=False):
    
    df = pd.read_csv("../data/expansion_data_prep_with_splits_" + target_col_orig + ".csv")
    print(f"Size of dataset: {df.shape[0]}")

    df = df.dropna(subset=[target_col_orig])
    df = df[np.isfinite(df[target_col_orig])]

    if log:
        df[target_col_orig] = np.log10(df[target_col_orig])

    df[target_col_orig + "_binary"] = (df[target_col_orig] >= threshold).astype(int)
    target_col_binary = target_col_orig + "_binary"

    if task_type == "regression":
        target_col = target_col_orig
    elif task_type == "classification":
        target_col = target_col_binary

    print(f"Class distribution:\n{df[target_col].value_counts()}")

    num_bits = 2048
    descriptor_cols = [nm for nm, fn in Descriptors._descList]
    fp_cols = [f"FP_{i}" for i in range(num_bits)]
    feature_cols = fp_cols + descriptor_cols

    return df, target_col, feature_cols, fp_cols, threshold


def train_model(df, target_col, feature_cols, split_col="random_split", task_type="classification"):
    
    X_train = df[df[split_col] == "train"][feature_cols]
    y_train = df[df[split_col] == "train"][target_col]

    dtrain = xgb.DMatrix(X_train, label=y_train)

    if task_type == "classification":
        params = {
            "objective": "binary:logistic",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
            "n_estimators": 100,
        }
    else:
        params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
            "n_estimators": 100,
        }

    model = xgb.train(params, dtrain, num_boost_round=100)
    return model


def evaluate_model(model, df, target_col, feature_cols, split_col="random_split", task_type="classification"):
    X_val = df[df[split_col] == "val"][feature_cols]
    y_val = df[df[split_col] == "val"][target_col]
    dval = xgb.DMatrix(X_val, label=y_val)
    y_pred_prob = model.predict(dval)

    X_test = df[df[split_col] == "test"][feature_cols]
    y_test = df[df[split_col] == "test"][target_col]
    print(f"Test set class distribution:\n{y_test.value_counts()}")
    dtest = xgb.DMatrix(X_test, label=y_test)
    y_test_pred_prob = model.predict(dtest)

    metrics = {}
    if task_type == "classification":
        metrics["auroc_val"] = roc_auc_score(y_val, y_pred_prob)
        metrics["auprc_val"] = average_precision_score(y_val, y_pred_prob)
        metrics["auroc_test"] = roc_auc_score(y_test, y_test_pred_prob)
        metrics["auprc_test"] = average_precision_score(y_test, y_test_pred_prob)
        print(f"Validation AUROC: {metrics['auroc_val']:.4f}")
        print(f"Validation AUPRC: {metrics['auprc_val']:.4f}")
        print(f"Test AUROC: {metrics['auroc_test']:.4f}")
        print(f"Test AUPRC: {metrics['auprc_test']:.4f}")

    return y_val, y_pred_prob, y_test, y_test_pred_prob, metrics


def run_conformal_fdr(df, model, target_col, feature_cols, fp_cols, split_col, task_type, threshold):
    fp_calib  = df[df[split_col] == "val"][fp_cols].to_numpy()
    fp_target = df[df[split_col] == "test"][fp_cols].to_numpy()

    y_calib  = df[df[split_col] == "val"][target_col].to_numpy()
    y_target = df[df[split_col] == "test"][target_col].to_numpy()

    y_pred_calib  = model.predict(xgb.DMatrix(df[df[split_col] == "val"][feature_cols]))
    y_pred_target = model.predict(xgb.DMatrix(df[df[split_col] == "test"][feature_cols]))

    conf_threshold = 0.5 if task_type == "classification" else threshold
    neg_mask = y_calib <= conf_threshold

    # --- nonconformity scores (shared across all methods) ---
    nc_calib  = ConformalPredictor.get_nonconformity_scores(
        y_pred_calib, y_calib, conf_threshold, function="difference"
    )
    nc_target = ConformalPredictor.get_nonconformity_scores(
        y_pred_target, np.full(len(y_pred_target), conf_threshold),
        conf_threshold, function="difference"
    )

    # --- kNN: neg-only calibration, raw model scores ---
    knn_w  = ConformalPredictor(method="knn", num_nn=-1, weighted=True)
    knn_w.fit(fp_calib[neg_mask], y_pred_calib[neg_mask], fp_target=fp_target, n_eff=100)
    knn_uw = ConformalPredictor(method="knn", num_nn=-1, weighted=False)
    knn_uw.fit(fp_calib[neg_mask], y_pred_calib[neg_mask])

    p_adjusted_knn    = knn_w.p_adjust(knn_w.predict_pvalues(fp_target, y_pred_target), method="BH")
    p_adjusted_knn_uw = knn_uw.p_adjust(knn_uw.predict_pvalues(fp_target, y_pred_target), method="BH")

    # --- kNN: all calibration, nonconformity scores ---
    knn_nc_w  = ConformalPredictor(method="knn", num_nn=-1, weighted=True)
    knn_nc_w.fit(fp_calib, nc_calib, fp_target=fp_target, n_eff=100)
    knn_nc_uw = ConformalPredictor(method="knn", num_nn=-1, weighted=False)
    knn_nc_uw.fit(fp_calib, nc_calib)

    p_adjusted_knn_nc    = knn_nc_w.p_adjust(
        knn_nc_w.predict_pvalues(fp_target, nc_target, nonconformities=True), method="BH"
    )
    p_adjusted_knn_nc_uw = knn_nc_uw.p_adjust(
        knn_nc_uw.predict_pvalues(fp_target, nc_target, nonconformities=True), method="BH"
    )

    # --- KS: neg-only calibration, raw model scores ---
    ks_w = ConformalPredictor(method="ks", weighted=True)
    ks_w.fit(fp_calib[neg_mask], y_pred_calib[neg_mask], fp_target=fp_target,
             n_eff=100)
    p_adjusted_ks = ks_w.p_adjust(ks_w.predict_pvalues(fp_target, y_pred_target), method="BH")

    # --- KS: all calibration, nonconformity scores ---
    ks2_w = ConformalPredictor(method="ks", weighted=True)
    ks2_w.fit(fp_calib, nc_calib, fp_target=fp_target, n_eff=100)
    p_adjusted_ks_nc = ks2_w.p_adjust(
        ks2_w.predict_pvalues(fp_target, nc_target, nonconformities=True), method="BH"
    )

    return {
        "y_target": y_target,
        "y_pred_target": y_pred_target,
        "p_adjusted_knn": p_adjusted_knn,
        "p_adjusted_knn_uw": p_adjusted_knn_uw,
        "p_adjusted_knn_nc": p_adjusted_knn_nc,
        "p_adjusted_knn_nc_uw": p_adjusted_knn_nc_uw,
        "p_adjusted_ks": p_adjusted_ks,
        "p_adjusted_ks_nc": p_adjusted_ks_nc,
        "threshold": conf_threshold,
        # predictors and score arrays for ECDF diagnostics
        "knn_w": knn_w, "knn_nc_w": knn_nc_w,
        "ks_w": ks_w, "ks2_w": ks2_w,
        "fp_target": fp_target,
        "scores_calib_neg": y_pred_calib[neg_mask],
        "nc_calib": nc_calib,
        "nc_target": nc_target,
    }


def plot_ecdf_diagnostics(cfdr, title=""):
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

    suptitle = "ECDF: mean 5-NN distance — calibration before/after weighting vs target"
    if title:
        suptitle += f" — {title}"
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def compute_fdr_curve(p_adjusted, y_target, threshold):
    true_fdr, estim_fdr, num_selected = [], [], []
    for tmp_fdr in np.sort(p_adjusted):
        selected = p_adjusted <= tmp_fdr
        false_positives = (y_target[selected] <= threshold).sum()
        true_fdr.append(0 if selected.sum() == 0 else false_positives / selected.sum())
        estim_fdr.append(tmp_fdr)
        num_selected.append(selected.sum())
    return estim_fdr, true_fdr, num_selected


def fdr_metrics_at_level(p_adjusted, y_target, conf_threshold, fdr_level=0.2):
    selected = p_adjusted <= fdr_level
    n = selected.sum()
    if n == 0:
        return {"n_selected": 0, "true_fdr": float("nan"), "n_actives_selected": 0}
    fp = (y_target[selected] <= conf_threshold).sum()
    tp = (y_target[selected] > conf_threshold).sum()
    return {"n_selected": int(n), "true_fdr": float(fp / n), "n_actives_selected": int(tp)}


def plot_score_vs_fdr(results, target_col_orig, split_col, fdr_level=0.2):
    """Scatter: model score (x) vs adjusted p-value (y), coloured by true label."""
    y_pred = results["y_pred_target"]
    y_true = results["y_target"]
    conf_threshold = results["threshold"]

    method_keys = {
        "p_adjusted_knn":      "kNN weighted",
        "p_adjusted_knn_uw":   "kNN unweighted",
        "p_adjusted_knn_nc":   "kNN NC weighted",
        "p_adjusted_knn_nc_uw": "kNN NC unweighted",
        "p_adjusted_ks":       "KS weighted",
        "p_adjusted_ks_nc":    "KS NC weighted",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (key, label) in zip(axes.flatten(), method_keys.items()):
        p_adj = results[key]
        active = y_true > conf_threshold
        ax.scatter(y_pred[~active], p_adj[~active], s=15, alpha=0.4,
                   color="steelblue", label="inactive")
        ax.scatter(y_pred[active],  p_adj[active],  s=15, alpha=0.6,
                   color="tomato",   label="active")
        ax.axhline(fdr_level, linestyle="--", color="black", linewidth=0.8,
                   label=f"FDR={fdr_level}")
        ax.set_xlabel("Model score")
        ax.set_ylabel("FDR threshold")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7)
        ax.set_ylim(0,1)
        ax.set_xlim(0,1)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Score vs FDR thresholds — {target_col_orig} ({split_col})", fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_fdr_curves(results, target_col_orig, split_col):
    conf_threshold = results["threshold"]
    y_target = results["y_target"]

    curves = {
        "kNN weighted":      results["p_adjusted_knn"],
        "kNN unweighted":    results["p_adjusted_knn_uw"],
        "kNN NC weighted":   results["p_adjusted_knn_nc"],
        "kNN NC unweighted": results["p_adjusted_knn_nc_uw"],
        "KS weighted":       results["p_adjusted_ks"],
        "KS NC weighted":    results["p_adjusted_ks_nc"],
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, p_adj in curves.items():
        estim, true, _ = compute_fdr_curve(p_adj, y_target, conf_threshold)
        ax.plot(estim, true, marker="o", markersize=3, label=label)

    ax.plot([0, 1], [0, 1], linestyle="--", color="red", label="y = x")
    ax.set_xlabel("FDR Threshold")
    ax.set_ylabel("True FDR")
    ax.set_title(f"FDR Control — {target_col_orig} ({split_col})")
    ax.grid(True)
    ax.legend(fontsize=8)
    plt.tight_layout()

    plt.show()
    plt.close(fig)


# %%
target_col_orig = "KSOL"
split_col = "cluster_split"
task_type = "classification"
threshold = 200
log=False

os.makedirs(RESULTS_DIR, exist_ok=True)

df, target_col, feature_cols, fp_cols, threshold = read_dataset(
    target_col_orig, task_type=task_type, threshold=threshold, log=log
)

sns.histplot(df[target_col_orig], bins=50)
plt.axvline(x=threshold, color="red", linestyle="--")
plt.title(f"Distribution of {target_col_orig} with threshold at {threshold:.2f}")
plt.xlabel(target_col_orig)
plt.show()

model = train_model(df, target_col, feature_cols, split_col=split_col, task_type=task_type)

y_val, y_pred_prob, y_test, y_test_pred_prob, metrics = evaluate_model(
    model, df, target_col, feature_cols, split_col=split_col, task_type=task_type
)

results = run_conformal_fdr(df, model, target_col, feature_cols, fp_cols, split_col, task_type, threshold)

conf_threshold = results["threshold"]
y_target = results["y_target"]

plot_fdr_curves(results, target_col_orig, split_col)
plot_score_vs_fdr(results, target_col_orig, split_col)
plot_ecdf_diagnostics(results, title=f"{target_col_orig} ({split_col})")

method_keys = {
    "p_adjusted_knn":      "kNN weighted",
    "p_adjusted_knn_uw":   "unweighted",
    "p_adjusted_knn_nc":   "kNN NC weighted",
    "p_adjusted_knn_nc_uw": "NC unweighted",
    "p_adjusted_ks":       "KS weighted",
    "p_adjusted_ks_nc":    "KS NC weighted",
}

rows = []
for key, label in method_keys.items():
    m = fdr_metrics_at_level(results[key], y_target, conf_threshold)
    rows.append({"method": label, **m})
    print(f"  {label}: selected={m['n_selected']}, true FDR={m['true_fdr']:.4f}, actives={m['n_actives_selected']}")

summary = pd.DataFrame(rows)
summary_path = os.path.join(RESULTS_DIR, f"{target_col_orig}_{split_col}_fdr_metrics.csv")
summary.to_csv(summary_path, index=False)
print(f"Metrics saved → {summary_path}")

conformal_path = os.path.join(RESULTS_DIR, f"{target_col_orig}_{split_col}_conformal_results.pkl")
with open(conformal_path, "wb") as f:
    pickle.dump({"target_col": target_col_orig, "split_col": split_col,
                    "metrics": metrics, **results}, f)
print(f"Conformal results saved → {conformal_path}")

# %%
target_col_orig = "KSOL"
split_col = "random_split"
task_type = "classification"
threshold = 200
log=False

os.makedirs(RESULTS_DIR, exist_ok=True)

df, target_col, feature_cols, fp_cols, threshold = read_dataset(
    target_col_orig, task_type=task_type, threshold=threshold, log=log
)

sns.histplot(df[target_col_orig], bins=50)
plt.axvline(x=threshold, color="red", linestyle="--")
plt.title(f"Distribution of {target_col_orig} with threshold at {threshold:.2f}")
plt.xlabel(target_col_orig)
plt.show()

model = train_model(df, target_col, feature_cols, split_col=split_col, task_type=task_type)

y_val, y_pred_prob, y_test, y_test_pred_prob, metrics = evaluate_model(
    model, df, target_col, feature_cols, split_col=split_col, task_type=task_type
)

results = run_conformal_fdr(df, model, target_col, feature_cols, fp_cols, split_col, task_type, threshold)

conf_threshold = results["threshold"]
y_target = results["y_target"]

plot_fdr_curves(results, target_col_orig, split_col)
plot_score_vs_fdr(results, target_col_orig, split_col)
plot_ecdf_diagnostics(results, title=f"{target_col_orig} ({split_col})")

method_keys = {
    "p_adjusted_knn":      "kNN weighted",
    "p_adjusted_knn_uw":   "kNN unweighted",
    "p_adjusted_knn_nc":   "kNN NC weighted",
    "p_adjusted_knn_nc_uw": "kNN NC unweighted",
    "p_adjusted_ks":       "KS weighted",
    "p_adjusted_ks_nc":    "KS NC weighted",
}

rows = []
for key, label in method_keys.items():
    m = fdr_metrics_at_level(results[key], y_target, conf_threshold)
    rows.append({"method": label, **m})
    print(f"  {label}: selected={m['n_selected']}, true FDR={m['true_fdr']:.4f}, actives={m['n_actives_selected']}")

summary = pd.DataFrame(rows)
summary_path = os.path.join(RESULTS_DIR, f"{target_col_orig}_{split_col}_fdr_metrics.csv")
summary.to_csv(summary_path, index=False)
print(f"Metrics saved → {summary_path}")

conformal_path = os.path.join(RESULTS_DIR, f"{target_col_orig}_{split_col}_conformal_results.pkl")
with open(conformal_path, "wb") as f:
    pickle.dump({"target_col": target_col_orig, "split_col": split_col,
                    "metrics": metrics, **results}, f)
print(f"Conformal results saved → {conformal_path}")

# %%
target_col_orig = "KSOL"
split_col = "scaffold_split"
task_type = "classification"
threshold = 200
log=False

os.makedirs(RESULTS_DIR, exist_ok=True)

df, target_col, feature_cols, fp_cols, threshold = read_dataset(
    target_col_orig, task_type=task_type, threshold=threshold, log=log
)

sns.histplot(df[target_col_orig], bins=50)
plt.axvline(x=threshold, color="red", linestyle="--")
plt.title(f"Distribution of {target_col_orig} with threshold at {threshold:.2f}")
plt.xlabel(target_col_orig)
plt.show()

model = train_model(df, target_col, feature_cols, split_col=split_col, task_type=task_type)

y_val, y_pred_prob, y_test, y_test_pred_prob, metrics = evaluate_model(
    model, df, target_col, feature_cols, split_col=split_col, task_type=task_type
)

results = run_conformal_fdr(df, model, target_col, feature_cols, fp_cols, split_col, task_type, threshold)

conf_threshold = results["threshold"]
y_target = results["y_target"]

plot_fdr_curves(results, target_col_orig, split_col)
plot_score_vs_fdr(results, target_col_orig, split_col)
plot_ecdf_diagnostics(results, title=f"{target_col_orig} ({split_col})")

method_keys = {
    "p_adjusted_knn":      "kNN weighted",
    "p_adjusted_knn_uw":   "unweighted",
    "p_adjusted_knn_nc":   "kNN NC weighted",
    "p_adjusted_knn_nc_uw": "NC unweighted",
    "p_adjusted_ks":       "KS weighted",
    "p_adjusted_ks_nc":    "KS NC weighted",
}

rows = []
for key, label in method_keys.items():
    m = fdr_metrics_at_level(results[key], y_target, conf_threshold)
    rows.append({"method": label, **m})
    print(f"  {label}: selected={m['n_selected']}, true FDR={m['true_fdr']:.4f}, actives={m['n_actives_selected']}")

summary = pd.DataFrame(rows)
summary_path = os.path.join(RESULTS_DIR, f"{target_col_orig}_{split_col}_fdr_metrics.csv")
summary.to_csv(summary_path, index=False)
print(f"Metrics saved → {summary_path}")

conformal_path = os.path.join(RESULTS_DIR, f"{target_col_orig}_{split_col}_conformal_results.pkl")
with open(conformal_path, "wb") as f:
    pickle.dump({"target_col": target_col_orig, "split_col": split_col,
                    "metrics": metrics, **results}, f)
print(f"Conformal results saved → {conformal_path}")

# %%

# %%

# %%
