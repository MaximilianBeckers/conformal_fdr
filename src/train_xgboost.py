# %%
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import utils

# %%
target_col = "KSOL"

df = pd.read_csv("../data/expansion_data_prep_with_splits_" + target_col + ".csv")
df[target_col] = -1 * np.log(df[target_col])

sns.histplot(df[target_col], bins=50)

split_col = "scaffold_split" # scaffold_split	cluster_split	random_split	umap_cluster_split

#set target column and binarize it
percentile = 80
threshold = np.percentile(df[target_col], percentile)
df[target_col + "_binary"] = (df[target_col] >= threshold).astype(int)
target_col = target_col + "_binary"

#remove nans and infs
df = df.dropna(subset=[target_col])
df = df[np.isfinite(df[target_col])]

print(f"Threshold for top {percentile}%: {threshold:.2f}")
print(f"Class distribution:\n{df[target_col].value_counts()}")

num_bits = 2048
descriptor_cols = [nm for nm,fn in Descriptors._descList]
fp_cols = [f"FP_{i}" for i in range(num_bits)]
feature_cols = fp_cols + descriptor_cols

# %%
# train XGBoost model
X_train = df[df[split_col] == "train"][feature_cols]
y_train = df[df[split_col] == "train"][target_col]

dtrain = xgb.DMatrix(X_train, label=y_train)

#train a model
params = {
    "objective":"binary:logistic",
    #"objective":"reg:squarederror",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
    "n_estimators": 100
}
    
model = xgb.train(params, dtrain, num_boost_round=100)

# %% [markdown]
# Get predictions for the calibration set

# %%
# now predict the validation set
X_val = df[df[split_col] == "val"][feature_cols]
y_val = df[df[split_col] == "val"][target_col]

dval = xgb.DMatrix(X_val, label=y_val)
y_pred_prob = model.predict(dval)

sns.swarmplot(x=y_val, y=y_pred_prob)
sns.boxplot(x=y_val, y=y_pred_prob, color="lightgray", showfliers=False)

# get auprc and auroc
auroc_val = roc_auc_score(y_val, y_pred_prob)
print(f"Validation AUROC: {auroc_val:.4f}")
auprc_val = average_precision_score(y_val, y_pred_prob)
print(f"Validation AUPRC: {auprc_val:.4f}")

# %% [markdown]
# Now get the predictions for the test set

# %%
# now predict the probabilities for the test set
X_test = df[df[split_col] == "test"][feature_cols]
y_test = df[df[split_col] == "test"][target_col]
print(f"Test set class distribution:\n{y_test.value_counts()}")

dtest = xgb.DMatrix(X_test, label=y_test)
y_test_pred_prob = model.predict(dtest)

sns.swarmplot(x=y_test, y=y_test_pred_prob)
sns.boxplot(x=y_test, y=y_test_pred_prob, color="lightgray", showfliers=False)

# get auprc and auroc
auroc_test = roc_auc_score(y_test, y_test_pred_prob)
print(f"Test AUROC: {auroc_test:.4f}")
auprc_test = average_precision_score(y_test, y_test_pred_prob)
print(f"Test AUPRC: {auprc_test:.4f}")

# %% [markdown]
# Now get False Discovery Rates (FDR)

# %%
from entropy_balancing_conformal_predictor import EntropyBalancingConformalPredictor

X_calib = df[df[split_col] == "val"][fp_cols].to_numpy()
X_target = df[df[split_col] == "test"][fp_cols].to_numpy()

y_calib = df[df[split_col] == "val"][target_col].to_numpy()
y_target = df[df[split_col] == "test"][target_col].to_numpy()


# now get the fingerprints
nn_distances_calib = utils.get_nearest_neighbor_distances(X_calib[y_calib == 0], X_target, nns=1)
nn_distances_target = utils.get_nearest_neighbor_distances(X_target, X_target, nns=1)

print(nn_distances_calib.shape[0], nn_distances_target.shape[0])

ebc = EntropyBalancingConformalPredictor(nn_distances_calib, nn_distances_target, max_order=1, lambda_reg=1000.0, use_weighted_ks=True)
ebc.fit()

p_values = ebc.predict_pvalues(y_pred_prob[y_calib == 0], y_test_pred_prob)
p_values_unweighted = ebc.predict_pvalues(y_pred_prob[y_calib == 0], y_test_pred_prob, weighted=False)

p_adjusted = ebc.p_adjust(p_values, method="BH")
p_adjusted_unweighted = ebc.p_adjust(p_values_unweighted, method="BH")


#**************************************************************
#**************** now the nonconformity scores ****************
#**************************************************************
nn_distances_calib = utils.get_nearest_neighbor_distances(X_calib, X_target, nns=1)
nn_distances_target = utils.get_nearest_neighbor_distances(X_target, X_target, nns=1)

ebc = EntropyBalancingConformalPredictor(nn_distances_calib, nn_distances_target, max_order=1, lambda_reg=1000.0, use_weighted_ks=True)
ebc.fit()

threshold = 0.5
conform_scores_calib = ebc.get_nonconformity_scores(y_pred_prob, y_calib, threshold=threshold)
conform_scores_target = ebc.get_nonconformity_scores(y_test_pred_prob, np.ones(y_test_pred_prob.shape[0])*threshold, threshold=threshold)

p_values_nonconformity = ebc.predict_pvalues(conform_scores_calib, conform_scores_target, nonconformities=True, weighted=True)
p_values_nonconformity_unweighted = ebc.predict_pvalues(conform_scores_calib, conform_scores_target, nonconformities=True, weighted=False)

p_adjusted_nonconformity = ebc.p_adjust(p_values_nonconformity, method="BH")
p_adjusted_nonconformity_unweighted = ebc.p_adjust(p_values_nonconformity_unweighted, method="BH")

# %%
sns.ecdfplot(nn_distances_calib, label="Calibration")
sns.ecdfplot(nn_distances_target, label="Target")
plt.legend()
plt.xlabel("Nearest neighbor distance")
plt.ylabel("ECDF")
plt.title("ECDF of nearest neighbor distances")
plt.grid()
plt.show()

# %%
print(np.sum(ebc.weights)/ebc.weights.shape[0])
print(np.min(ebc.weights))
print(np.max(ebc.weights))
print("Initial KS distance:", ebc.initial_ks_distance)
print("Final KS distance:", ebc.final_ks_distance)
print("Effective number of samples:", ebc.effective_sample_size)

# %%
true_fdr = []
estim_fdr = []
for tmp_fdr in np.linspace(0.01, 1, 100):
    selected = p_adjusted <= tmp_fdr
    true_positives = (y_target[selected] == 1).sum()
    false_positives = (y_target[selected] == 0).sum()
    if selected.sum() == 0:
        true_fdr.append(0.0)
    else:
        true_fdr.append(false_positives / selected.sum())
    estim_fdr.append(tmp_fdr)

#now unweighted
true_fdr_unweighted = []
estim_fdr_unweighted = []
for tmp_fdr in np.linspace(0.01, 1, 100):
    selected = p_adjusted_unweighted <= tmp_fdr
    true_positives = (y_target[selected] == 1).sum()
    false_positives = (y_target[selected] == 0).sum()
    if selected.sum() == 0:
        true_fdr_unweighted.append(0.0)
    else:
        true_fdr_unweighted.append(false_positives / selected.sum())
    estim_fdr_unweighted.append(tmp_fdr)

#now fdr values for uncorrected p-values
true_fdr_uncorrected = []
estim_fdr_uncorrected = []
for tmp_fdr in np.linspace(0.01, 1, 100):
    selected = p_values <= tmp_fdr
    true_positives = (y_target[selected] == 1).sum()
    false_positives = (y_target[selected] == 0).sum()
    if selected.sum() == 0:
        true_fdr_uncorrected.append(0.0)
    else:
        true_fdr_uncorrected.append(false_positives / selected.sum())
    estim_fdr_uncorrected.append(tmp_fdr)

# now fdr values for nonconformity scores
true_fdr_nonconformity = []
estim_fdr_nonconformity = []
for tmp_fdr in np.linspace(0.01, 1, 100):
    selected = p_adjusted_nonconformity <= tmp_fdr
    true_positives = (y_target[selected] == 1).sum()
    false_positives = (y_target[selected] == 0).sum()
    if selected.sum() == 0:
        true_fdr_nonconformity.append(0.0)
    else:
        true_fdr_nonconformity.append(false_positives / selected.sum())
    estim_fdr_nonconformity.append(tmp_fdr)

# now fdr values for nonconformity scores unweighted
true_fdr_nonconformity_unweighted = []
estim_fdr_nonconformity_unweighted = []
for tmp_fdr in np.linspace(0.01, 1, 100):
    selected = p_adjusted_nonconformity_unweighted <= tmp_fdr
    true_positives = (y_target[selected] == 1).sum()
    false_positives = (y_target[selected] == 0).sum()
    if selected.sum() == 0:
        true_fdr_nonconformity_unweighted.append(0.0)
    else:
        true_fdr_nonconformity_unweighted.append(false_positives / selected.sum())
    estim_fdr_nonconformity_unweighted.append(tmp_fdr)



plt.plot(estim_fdr, true_fdr, marker="o", label="FDR-corrected with Entropy Balancing")
plt.plot(estim_fdr_unweighted, true_fdr_unweighted, marker="o", label="FDR-corrected unweighted")
plt.plot(estim_fdr_uncorrected, true_fdr_uncorrected, marker="o", label="Uncorrected p-values")
plt.plot(estim_fdr_nonconformity, true_fdr_nonconformity, marker="o", label="Nonconformity Scores")
plt.plot(estim_fdr_nonconformity_unweighted, true_fdr_nonconformity_unweighted, marker="o", label="Nonconformity Scores Unweighted")
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



