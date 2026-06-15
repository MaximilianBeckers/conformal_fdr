import math
import numpy as np
from scipy.optimize import minimize, Bounds
from utils import get_dists_between_two_sets, get_nearest_neighbor_distances


class ConformalPredictor:
    """
    Unified conformal predictor.

    method="entropy_balancing":
        Maximises weight entropy over calibration samples subject to
        KS(reweighted control, target) <= ks_bound (Tibshirani 2019).
        Covariate shift is corrected globally via one weight per calibration
        compound, estimated from NN-distance distributions.

    method="knn":
        Builds a per-test-compound null distribution from the k nearest
        calibration neighbours, weighted by Tanimoto similarity.
        No global optimisation; covariate shift is handled locally.

    Parameters
    ----------
    method : {"entropy_balancing", "knn"}, default "knn"
    num_nn : int, default 30
        Number of nearest calibration neighbours (knn only; -1 = all).
    weighted : bool, default True
        Use importance/similarity weights in the p-value formula.
    ks_bound : float, default 0.5
        KS distance upper bound (entropy_balancing only).
    ks_penalty : float, default 0.0
        Coefficient on KS distance added to the entropy objective
        (entropy_balancing only).
    """

    VALID_METHODS = ("entropy_balancing", "knn")
    VALID_ADJUST_METHODS = ("BH", "BY", "Holm", "Hochberg")

    def __init__(self, method="knn", num_nn=30, weighted=True,
                 ks_bound=0.5, ks_penalty=0.0):
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}, got {method!r}")
        self.method = method
        self.num_nn = num_nn
        self.weighted = weighted
        self.ks_bound = ks_bound
        self.ks_penalty = ks_penalty

        self.fp_calib = None
        self.scores_calib = None
        self.weights = None
        self.is_fitted = False

        # EB diagnostics — None for knn
        self.initial_ks_distance = None
        self.final_ks_distance = None
        self.effective_sample_size = None
        self.X_control = None
        self.X_target = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, fp_calib, scores_calib, fp_target=None):
        """
        Store calibration data and, for entropy_balancing, run optimisation.

        Parameters
        ----------
        fp_calib : array of shape (n_calib, n_bits)
            Morgan fingerprints for the calibration set.
        scores_calib : array of shape (n_calib,)
            Model or nonconformity scores for the calibration set.
        fp_target : array of shape (n_target, n_bits), optional
            Required when method="entropy_balancing" and weighted=True.
        """
        self.fp_calib = np.asarray(fp_calib, dtype=np.uint8)
        self.scores_calib = np.asarray(scores_calib, dtype=np.float64)

        if self.method == "entropy_balancing" and self.weighted:
            if fp_target is None:
                raise ValueError(
                    "fp_target is required for method='entropy_balancing' with weighted=True"
                )
            fp_target_u8 = np.asarray(fp_target, dtype=np.uint8)
            X_control = get_nearest_neighbor_distances(self.fp_calib, fp_target_u8, nns=5)
            X_target = get_nearest_neighbor_distances(fp_target_u8, fp_target_u8, nns=5)
            self._fit_entropy_balancing(X_control, X_target)
            self.X_control = X_control
            self.X_target = X_target

        self.is_fitted = True
        return self

    def predict_pvalues(self, fp_target, scores_target,
                        nonconformities=False, w_test=None):
        """
        Compute conformal p-values for test compounds.

        Parameters
        ----------
        fp_target : array of shape (n_target, n_bits)
        scores_target : array of shape (n_target,)
        nonconformities : bool, default False
            True  → count calibration samples with score <= test score
                     (for nonconformity scores where lower = more conforming).
            False → count calibration samples with score > test score
                     (for raw model scores where higher = more active).
        w_test : array of shape (n_target,) or None
            Per-test importance weights; entropy_balancing only.
            Defaults to 1.0 for every test compound.

        Returns
        -------
        np.ndarray of shape (n_target,)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_pvalues().")

        fp_target = np.asarray(fp_target)
        scores_target = np.asarray(scores_target, dtype=np.float64)

        if self.method == "knn":
            return self._predict_knn(fp_target, scores_target, nonconformities)
        return self._predict_eb(scores_target, nonconformities, w_test)

    @staticmethod
    def get_nonconformity_scores(scores, labels, threshold, function="difference"):
        """
        Compute nonconformity scores.

        function="difference": returns labels - scores.
        """
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        if function == "difference":
            return labels - scores
        M = 100.0
        return np.where(labels > threshold, M, threshold) - scores

    def estimate_test_weights(self, nn_dist_test, k=5):
        """
        Estimate per-test importance weights via k-NN interpolation over
        the fitted calibration weights (entropy_balancing only).
        """
        if self.method != "entropy_balancing" or self.weights is None or self.X_control is None:
            raise RuntimeError(
                "estimate_test_weights() requires method='entropy_balancing' "
                "with weighted=True after fit()."
            )
        nn_dist_test = np.asarray(nn_dist_test, dtype=np.float64)
        k_eff = min(k, len(self.X_control))
        diffs = np.abs(nn_dist_test[:, None] - self.X_control[None, :])
        k_nearest = np.argsort(diffs, axis=1)[:, :k_eff]
        return self.weights[k_nearest].mean(axis=1)

    @staticmethod
    def p_adjust(p_values, method="BH"):
        """Adjust p-values for multiple testing (BH / BY / Holm / Hochberg)."""
        p_values = np.asarray(p_values)
        n = len(p_values)
        sort_idx = np.argsort(p_values)
        p_sorted = p_values[sort_idx]
        adjusted = np.zeros(n)
        prev = 1.0
        Hn = math.log(n) + 0.5772 + 0.5/n - 1.0/(12*n**2) + 1.0/(120*n**4)

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
                f"Choose from {ConformalPredictor.VALID_ADJUST_METHODS}."
            )
        return adjusted[np.argsort(sort_idx)]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_entropy_balancing(self, X_control, X_target):
        n = len(X_control)
        w0 = np.ones(n)

        init_ks = self._weighted_ks_distance(X_control, X_target, weights_1=w0)
        self.initial_ks_distance = init_ks

        if init_ks <= self.ks_bound:
            self.weights = w0
            self.final_ks_distance = init_ks
            self.effective_sample_size = float(n)
            return

        bounds = Bounds(np.zeros(n), np.full(n, np.inf))
        constraints = [
            {'type': 'eq',
             'fun': lambda w: np.sum(w) / n - 1.0},
            {'type': 'ineq',
             'fun': lambda w: self.ks_bound - self._weighted_ks_distance(
                 X_control, X_target, weights_1=w)},
        ]

        def objective(w):
            return (np.sum(w * np.log(w + 1e-10))
                    + self.ks_penalty * self._weighted_ks_distance(
                        X_control, X_target, weights_1=w))

        result = minimize(
            objective, w0, method='SLSQP', bounds=bounds,
            constraints=constraints, options={'ftol': 1e-9, 'maxiter': 10000},
        )

        if not result.success:
            print(f"Warning: entropy balancing failed ({result.message}); "
                  "falling back to uniform weights.")
            self.weights = w0
        else:
            self.weights = result.x

        self.final_ks_distance = self._weighted_ks_distance(
            X_control, X_target, weights_1=self.weights)
        self.effective_sample_size = (
            np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)
        )

    def _predict_knn(self, fp_target, scores_target, nonconformities):
        dist_matrix = get_dists_between_two_sets(
            self.fp_calib.astype(np.uint8), fp_target.astype(np.uint8)
        )
        k_eff = (len(self.scores_calib) if self.num_nn == -1
                 else min(self.num_nn, len(self.scores_calib)))
        p_values = np.zeros(len(scores_target))

        for j in range(len(scores_target)):
            dists_j = dist_matrix[:, j]
            nn_idx = np.argsort(dists_j)[:k_eff]
            nn_scores = self.scores_calib[nn_idx]
            s_j = scores_target[j]

            weights = (np.maximum(1.0 - dists_j[nn_idx], 0.0)
                       if self.weighted else np.ones(k_eff))
            w_test = 1.0
            denom = weights.sum() + w_test
            numer = (weights[nn_scores <= s_j].sum() if nonconformities
                     else weights[nn_scores > s_j].sum()) + w_test
            p_values[j] = numer / denom

        return p_values

    def _predict_eb(self, scores_target, nonconformities, w_test=None):
        cal = self.scores_calib
        weights = (self.weights if (self.weighted and self.weights is not None)
                   else np.ones(len(cal)))
        w_test_arr = (np.ones(len(scores_target)) if w_test is None
                      else np.asarray(w_test, dtype=np.float64))
        total_cal = np.sum(weights)

        if not nonconformities:
            return np.array([
                (np.sum(weights[cal > s]) + wt) / (total_cal + wt)
                for s, wt in zip(scores_target, w_test_arr)
            ])
        return np.array([
            (np.sum(weights[cal <= s]) + wt) / (total_cal + wt)
            for s, wt in zip(scores_target, w_test_arr)
        ])

    @staticmethod
    def _weighted_ks_distance(sample_1, sample_2, weights_1=None, weights_2=None):
        """Weighted KS distance between two 1-D distributions."""
        sample_1 = np.asarray(sample_1)
        sample_2 = np.asarray(sample_2)
        idx1 = np.argsort(sample_1)
        idx2 = np.argsort(sample_2)
        data1_sorted = sample_1[idx1]
        data2_sorted = sample_2[idx2]
        w1 = np.ones(len(sample_1)) if weights_1 is None else np.asarray(weights_1)[idx1]
        w2 = np.ones(len(sample_2)) if weights_2 is None else np.asarray(weights_2)[idx2]
        cum_w1 = np.cumsum(w1 / w1.sum())
        cum_w2 = np.cumsum(w2 / w2.sum())
        all_points = np.concatenate([data1_sorted, data2_sorted])
        i1 = np.searchsorted(data1_sorted, all_points, side='right')
        i2 = np.searchsorted(data2_sorted, all_points, side='right')
        cdf1 = np.where(i1 > 0, cum_w1[np.minimum(i1 - 1, len(cum_w1) - 1)], 0.0)
        cdf2 = np.where(i2 > 0, cum_w2[np.minimum(i2 - 1, len(cum_w2) - 1)], 0.0)
        return float(np.max(np.abs(cdf1 - cdf2)))
