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
df = pd.read_csv("../data/expansion_data_prep_with_splits_KSOL.csv")

split_col = "umap_cluster_split" # scaffold_split	cluster_split	random_split	umap_cluster_split

#set target column and binarize it
target_col = "KSOL"
threshold = 200
df[target_col + "_binary"] = (df[target_col] >= threshold).astype(int)
target_col = target_col + "_binary"

num_bits = 2048
descriptor_cols = [nm for nm,fn in Descriptors._descList]
fp_cols = [f"FP_{i}" for i in range(num_bits)]
feature_cols = fp_cols + descriptor_cols

# %%
# train XGBoost model
X_train = df[df[split_col] == "train"][feature_cols]
y_train = df[df[split_col] == "train"][target_col]

dtrain = xgb.DMatrix(X_train, label=y_train)

#train
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
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


#now only subset in the null distribution
X_calib = X_calib[y_calib == 0]

# now get the fingerprints
nn_distances_calib = utils.get_nearest_neighbor_distances(X_calib, X_target, nns=1)
nn_distances_target = utils.get_nearest_neighbor_distances(X_target, X_target, nns=1)

print(nn_distances_calib.shape[0], nn_distances_target.shape[0])

ebc = EntropyBalancingConformalPredictor(nn_distances_calib, nn_distances_target, max_order=1, lambda_reg=5, use_weighted_ks=True)
ebc.fit()

f_target = y_test_pred_prob
f_calib = y_pred_prob[y_calib == 0]

p_values = ebc.predict_pvalues(f_calib, f_target)
p_adjusted = ebc.p_adjust(p_values, method="BH")

#now unweighted conformal predictor
p_values_unweighted = ebc.predict_pvalues(f_calib, f_target, weighted=False)
p_adjusted_unweighted = ebc.p_adjust(p_values_unweighted, method="BH")

# %%
sns.histplot(nn_distances_calib)
sns.histplot(nn_distances_target)

# %%
print(np.sum(ebc.weights_))
print(np.min(ebc.weights_))
print(np.max(ebc.weights_))

# %%
true_fdr = []
estim_fdr = []
for tmp_fdr in np.linspace(0.01, 1, 100):
    selected = p_adjusted <= tmp_fdr
    true_positives = (y_target[selected] == 1).sum()
    false_positives = (y_target[selected] == 0).sum()
    true_fdr.append(false_positives / max(1, selected.sum()))
    estim_fdr.append(tmp_fdr)

true_fdr_unweighted = []
estim_fdr_unweighted = []
for tmp_fdr in np.linspace(0.01, 1, 100):
    selected = p_adjusted_unweighted <= tmp_fdr
    true_positives = (y_target[selected] == 1).sum()
    false_positives = (y_target[selected] == 0).sum()
    true_fdr_unweighted.append(false_positives / max(1, selected.sum()))
    estim_fdr_unweighted.append(tmp_fdr)

plt.plot(estim_fdr, true_fdr, marker="o", label="Entropy Balancing")
plt.plot(estim_fdr_unweighted, true_fdr_unweighted, marker="o", label="Unweighted")
plt.plot([0, 1], [0, 1], linestyle="--", color="red")
plt.xlabel("Estimated FDR")
plt.ylabel("True FDR")
plt.title("FDR Control with Entropy Balancing Conformal Predictor")
plt.grid()
plt.legend()
plt.show()

# %%


# %%



