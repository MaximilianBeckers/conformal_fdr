import math
import numpy as np
from utils import get_dists_between_two_sets


class KNNConformalPredictor:
    """
    Conformal predictor that computes p-values using only the k nearest
    neighbours in the calibration set for each test compound.

    No global optimisation is needed. Covariate shift is handled locally:
    each test compound is compared against the calibration samples that are
    most similar to it in fingerprint space.

    Two variants are supported:
      weighted=True  — neighbours are weighted by Tanimoto similarity
                       (closer neighbours contribute more)
      weighted=False — standard unweighted k-NN conformal p-value

    Weighted p-value for test compound j with score s_j:

        p_j = (∑_{i ∈ kNN(j)} sim(i,j) · 1[score_i > s_j] + 1) /
              (∑_{i ∈ kNN(j)} sim(i,j) + 1)

    The +1 term gives the test compound weight 1 (its Tanimoto similarity
    to itself), consistent with Tibshirani (2019).

    Parameters
    ----------
    k : int, default=30
        Number of nearest calibration neighbours to use per test compound.
        Set k=-1 to use all calibration samples (equivalent to global weighting).
    weighted : bool, default=True
        Whether to weight neighbours by Tanimoto similarity.
    """

    VALID_ADJUST_METHODS = ("BH", "BY", "Holm", "Hochberg")

    def __init__(self, k=30, weighted=True):
        self.k = k
        self.weighted = weighted
        self.fp_calib = None
        self.scores_calib = None
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, fp_calib, scores_calib):
        """
        Store calibration fingerprints and scores.

        Parameters
        ----------
        fp_calib : np.ndarray of shape (n_calib, n_bits)
            Binary Morgan fingerprints for the calibration set.
        scores_calib : array-like of shape (n_calib,)
            Model scores (or nonconformity scores) for the calibration set.

        Returns
        -------
        self
        """
        self.fp_calib = np.asarray(fp_calib)
        self.scores_calib = np.asarray(scores_calib, dtype=np.float64)
        self.is_fitted = True
        return self

    def get_nonconformity_scores(self, scores, labels, threshold,
                                function="difference"):
        """
        Compute nonconformity scores (same interface as EntropyBalancingConformalPredictor).

        Parameters
        ----------
        scores : array-like — model predictions.
        labels : array-like — true labels (or CONF_THRESHOLD for target compounds).
        threshold : float — classification probability cutoff.
        function : str — 'difference' computes labels - scores.
        """
        if function == "difference":
            return np.array(labels - scores)
        M = 100
        V = []
        for i in range(len(scores)):
            V.append(M if labels[i] > threshold else threshold)
            V[i] = V[i] - scores[i]
        return np.array(V)

    def predict_pvalues(self, fp_target, scores_target, nonconformities=False):
        """
        Compute conformal p-values for test compounds.

        Parameters
        ----------
        fp_target : np.ndarray of shape (n_target, n_bits)
            Binary Morgan fingerprints for the test set.
        scores_target : array-like of shape (n_target,)
            Model scores (or nonconformity scores) for the test set.
        nonconformities : bool, default=False
            If True, counts calibration samples with score <= test score
            (higher nonconformity score = more non-conforming).

        Returns
        -------
        np.ndarray of shape (n_target,)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_pvalues().")

        fp_target = np.asarray(fp_target)
        scores_target = np.asarray(scores_target, dtype=np.float64)

        # dist_matrix shape: (n_calib, n_target)
        dist_matrix = get_dists_between_two_sets(
            self.fp_calib.astype(np.uint8),
            fp_target.astype(np.uint8),
        )

        k_eff = len(self.scores_calib) if self.k == -1 else min(self.k, len(self.scores_calib))
        p_values = np.zeros(len(scores_target))

        for j in range(len(scores_target)):
            dists_j = dist_matrix[:, j]
            nn_idx = np.argsort(dists_j)[:k_eff]
            nn_scores = self.scores_calib[nn_idx]
            s_j = scores_target[j]

            if self.weighted:
                weights = np.maximum(1.0 - dists_j[nn_idx], 0.0)
                w_test = 1.0
            else:
                weights = np.ones(k_eff)
                w_test = 1.0

            denom = weights.sum() + w_test
            if nonconformities:
                numer = weights[nn_scores <= s_j].sum() + w_test
            else:
                numer = weights[nn_scores > s_j].sum() + w_test
            p_values[j] = numer / denom

        return p_values

    @staticmethod
    def p_adjust(p_values, method="BH"):
        """
        Adjust p-values for multiple testing.

        Parameters
        ----------
        p_values : array-like of shape (n,)
        method : str — one of 'BH', 'BY', 'Holm', 'Hochberg'.

        Returns
        -------
        np.ndarray of shape (n,) — adjusted p-values in original order.
        """
        p_values = np.asarray(p_values)
        n = len(p_values)

        sort_idx = np.argsort(p_values)
        p_sorted = p_values[sort_idx]
        adjusted = np.zeros(n)
        prev = 1.0

        Hn = (math.log(n) + 0.5772 + 0.5 / n
              - 1.0 / (12 * n ** 2) + 1.0 / (120 * n ** 4))

        if method == "BH":
            for i in range(n - 1, -1, -1):
                adjusted[i] = min(prev, p_sorted[i] * n / (i + 1.0))
                prev = adjusted[i]

        elif method == "BY":
            for i in range(n - 1, -1, -1):
                adjusted[i] = min(prev, p_sorted[i] * (n / (i + 1.0)) * Hn)
                prev = adjusted[i]

        elif method == "Holm":
            prev = 0.0
            for i in range(n):
                adjusted[i] = max(prev, (n - i) * p_sorted[i])
                prev = adjusted[i]
            adjusted = np.minimum(adjusted, 1.0)

        elif method == "Hochberg":
            for i in range(n - 1, -1, -1):
                adjusted[i] = min(prev, p_sorted[i] * (n - i))
                prev = adjusted[i]

        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose from {KNNConformalPredictor.VALID_ADJUST_METHODS}."
            )

        return adjusted[np.argsort(sort_idx)]
