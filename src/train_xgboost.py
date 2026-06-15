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
from entropy_balancing_conformal_predictor import EntropyBalancingConformalPredictor
from ks_conformal_predictor import KSConformalPredictor


# %%
def read_dataset(target_col_orig, split_col="random_split", task_type="classification", threshold=150):
    
    df = pd.read_csv("../data/expansion_data_prep_with_splits_" + target_col_orig + ".csv")
    print(f"Size of dataset: {df.shape[0]}")

    df = df.dropna(subset=[target_col_orig])
    df = df[np.isfinite(df[target_col_orig])]

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
    X_calib = df[df[split_col] == "val"][fp_cols].to_numpy()
    X_target = df[df[split_col] == "test"][fp_cols].to_numpy()

    y_calib = df[df[split_col] == "val"][target_col].to_numpy()
    y_target = df[df[split_col] == "test"][target_col].to_numpy()

    dval = xgb.DMatrix(df[df[split_col] == "val"][feature_cols])
    dtest = xgb.DMatrix(df[df[split_col] == "test"][feature_cols])
    y_pred_prob = model.predict(dval)
    y_test_pred_prob = model.predict(dtest)

    conf_threshold = 0.5 if task_type == "classification" else threshold

    nn_distances_calib = utils.get_nearest_neighbor_distances(X_calib[y_calib <= conf_threshold], X_target, nns=5)
    nn_distances_target = utils.get_nearest_neighbor_distances(X_target, X_target, nns=5)

    ebc = EntropyBalancingConformalPredictor(nn_distances_calib, nn_distances_target, ks_bound=0.5, ks_penalty=0.0)
    #ebc = KSConformalPredictor(nn_distances_calib, nn_distances_target, ess_min=int(0.5*nn_distances_calib.shape[0]))

    ebc.fit()

    p_values = ebc.predict_pvalues(y_pred_prob[y_calib <= conf_threshold], y_test_pred_prob)
    p_values_unweighted = ebc.predict_pvalues(y_pred_prob[y_calib <= conf_threshold], y_test_pred_prob, weighted=False)

    p_adjusted = ebc.p_adjust(p_values, method="BH")
    p_adjusted_unweighted = ebc.p_adjust(p_values_unweighted, method="BH")

    nn_distances_calib_all = utils.get_nearest_neighbor_distances(X_calib, X_target, nns=5)
    ebc2 = EntropyBalancingConformalPredictor(nn_distances_calib_all, nn_distances_target, ks_bound=0.5, ks_penalty=0.0)
    #ebc2 = KSConformalPredictor(nn_distances_calib_all, nn_distances_target, ess_min=int(0.5*nn_distances_calib_all.shape[0]))

    ebc2.fit()

    conform_scores_calib = ebc2.get_nonconformity_scores(y_pred_prob, y_calib, conf_threshold, function="weighted")
    conform_scores_target = ebc2.get_nonconformity_scores(y_test_pred_prob, np.ones(y_test_pred_prob.shape[0]) * conf_threshold, conf_threshold, function="weighted")

    p_values_nonconformity = ebc2.predict_pvalues(conform_scores_calib, conform_scores_target, nonconformities=True, weighted=True)
    p_values_nonconformity_unweighted = ebc2.predict_pvalues(conform_scores_calib, conform_scores_target, nonconformities=True, weighted=False)

    p_adjusted_nonconformity = ebc2.p_adjust(p_values_nonconformity, method="BH")
    p_adjusted_nonconformity_unweighted = ebc2.p_adjust(p_values_nonconformity_unweighted, method="BH")

    return {
        "y_target": y_target,
        "y_test_pred_prob": y_test_pred_prob,
        "p_values": p_values,
        "p_adjusted": p_adjusted,
        "p_adjusted_unweighted": p_adjusted_unweighted,
        "p_adjusted_nonconformity": p_adjusted_nonconformity,
        "p_adjusted_nonconformity_unweighted": p_adjusted_nonconformity_unweighted,
        "nn_distances_calib": nn_distances_calib_all,
        "nn_distances_target": nn_distances_target,
        "ebc": ebc2,
        "threshold": conf_threshold,
    }


def compute_fdr_curve(p_adjusted, y_target, threshold):
    true_fdr, estim_fdr, num_selected = [], [], []
    for tmp_fdr in np.sort(p_adjusted):
        selected = p_adjusted <= tmp_fdr
        false_positives = (y_target[selected] <= threshold).sum()
        true_fdr.append(0 if selected.sum() == 0 else false_positives / selected.sum())
        estim_fdr.append(tmp_fdr)
        num_selected.append(selected.sum())
    return estim_fdr, true_fdr, num_selected


# %%
if __name__ == "__main__":
    target_col_orig = "HLM CLint"
    split_col = "scaffold_split"
    task_type = "classification"
    threshold = 200

    df, target_col, feature_cols, fp_cols, threshold = read_dataset(
        target_col_orig, split_col=split_col, task_type=task_type, threshold=threshold
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

    print(np.sum(results["ebc"].weights) / results["ebc"].weights.shape[0])
    print("Initial KS distance:", results["ebc"].initial_ks_distance)
    print("Final KS distance:", results["ebc"].final_ks_distance)
    print("Effective number of samples:", results["ebc"].effective_sample_size)

    conf_threshold = results["threshold"]
    y_target = results["y_target"]

    estim_fdr, true_fdr, _ = compute_fdr_curve(results["p_adjusted"], y_target, conf_threshold)
    estim_fdr_uw, true_fdr_uw, _ = compute_fdr_curve(results["p_adjusted_unweighted"], y_target, conf_threshold)
    estim_fdr_raw, true_fdr_raw, _ = compute_fdr_curve(results["p_values"], y_target, conf_threshold)
    estim_fdr_nc, true_fdr_nc, _ = compute_fdr_curve(results["p_adjusted_nonconformity"], y_target, conf_threshold)
    estim_fdr_nc_uw, true_fdr_nc_uw, _ = compute_fdr_curve(results["p_adjusted_nonconformity_unweighted"], y_target, conf_threshold)

    plt.plot(estim_fdr, true_fdr, marker="o", label="FDR-corrected with Entropy Balancing")
    plt.plot(estim_fdr_uw, true_fdr_uw, marker="o", label="FDR-corrected unweighted")
    plt.plot(estim_fdr_raw, true_fdr_raw, marker="o", label="Uncorrected p-values")
    plt.plot(estim_fdr_nc, true_fdr_nc, marker="o", label="Nonconformity Scores")
    plt.plot(estim_fdr_nc_uw, true_fdr_nc_uw, marker="o", label="Nonconformity Scores Unweighted")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red")
    plt.xlabel("Estimated FDR")
    plt.ylabel("True FDR")
    plt.title("FDR Control with Entropy Balancing Conformal Predictor")
    plt.grid()
    plt.legend()
    plt.show()

# %%

# %%

# %%
