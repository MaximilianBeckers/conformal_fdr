import math
import numpy as np
from scipy.optimize import minimize, Bounds


class KSConformalPredictor:
    """
    Covariate-shift corrected conformal predictor that directly minimises the
    KS distance between the reweighted control and target NN-distance
    distributions, subject to a minimum effective sample size (ESS).

    Unlike entropy balancing (which maximises weight entropy subject to a KS
    bound), this formulation prioritises tight distribution matching and uses
    ESS as the regularising constraint to prevent degenerate weight solutions.

    Solve:  minimise  KS(weighted_control, target)
            s.t.      mean(w) = 1   (i.e. sum(w) = n)
                      w >= 0
                      ESS(w) >= ess_min
                        where ESS = (sum w)^2 / sum(w^2) = n^2 / sum(w^2)

    The ESS constraint is equivalent to sum(w^2) <= n^2 / ess_min, a simple
    quadratic inequality that SLSQP handles natively.

    Parameters
    ----------
    X_control : np.ndarray of shape (n_control,)
        NN-distance vector for the calibration set.
    X_target : np.ndarray of shape (n_target,)
        NN-distance vector for the target set.
    ess_min : float, default=30.0
        Minimum effective sample size. Prevents the solution from concentrating
        all weight on very few calibration samples.
    """

    VALID_ADJUST_METHODS = ("BH", "BY", "Holm", "Hochberg")

    def __init__(self, X_control, X_target, ess_min=30.0):
        self.X_control = np.asarray(X_control, dtype=np.float64)
        self.X_target = np.asarray(X_target, dtype=np.float64)
        self.ess_min = ess_min

        self.weights = None
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self):
        """
        Minimise KS(weighted_control, target) subject to ESS >= ess_min.

        Returns
        -------
        self
        """
        n = self.X_control.shape[0]
        w0 = np.ones(n)

        self.initial_ks_distance = self._weighted_ks_distance(
            self.X_control, self.X_target, weights_1=w0
        )

        ess_min = min(self.ess_min, float(n))

        bounds = Bounds(np.zeros(n), np.full(n, np.inf))
        constraints = [
            {
                'type': 'eq',
                'fun': lambda w: np.sum(w) / n - 1.0,
            },
            {
                # ESS >= ess_min  <=>  sum(w^2) <= n^2 / ess_min
                # fun(w) >= 0 form: n^2 / ess_min - sum(w^2) >= 0
                'type': 'ineq',
                'fun': lambda w: n ** 2 / ess_min - np.sum(w ** 2),
            },
        ]

        result = minimize(
            lambda w: self._weighted_ks_distance(
                self.X_control, self.X_target, weights_1=w
            ),
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 10000},
        )

        if not result.success:
            print(f"Warning: KS minimisation failed ({result.message}); falling back to uniform weights.")
            self.weights = w0
        else:
            self.weights = result.x

        self.is_fitted = True
        self.final_ks_distance = self._weighted_ks_distance(
            self.X_control, self.X_target, weights_1=self.weights
        )
        self.effective_sample_size = (
            np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)
        )
        return self

    def get_nonconformity_scores(self, scores, labels, threshold,
                                 function="difference"):
        if function == "difference":
            return np.array(labels - scores)
        M = 100
        V = []
        for i in range(len(scores)):
            V.append(M if labels[i] > threshold else threshold)
            V[i] = V[i] - scores[i]
        return np.array(V)

    def estimate_test_weights(self, nn_dist_test, k=5):
        """
        Estimate importance weights for test compounds via k-NN interpolation
        from the fitted calibration weights.

        Parameters
        ----------
        nn_dist_test : array-like of shape (n_test,)
        k : int

        Returns
        -------
        np.ndarray of shape (n_test,)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before estimate_test_weights().")
        nn_dist_test = np.asarray(nn_dist_test, dtype=np.float64)
        k_eff = min(k, len(self.X_control))
        diffs = np.abs(nn_dist_test[:, None] - self.X_control[None, :])
        k_nearest = np.argsort(diffs, axis=1)[:, :k_eff]
        return self.weights[k_nearest].mean(axis=1)

    def predict_pvalues(self, calibration_scores, observed_scores,
                        nonconformities=False, weighted=True, w_test=None):
        """
        Compute conformal p-values.

        Parameters
        ----------
        calibration_scores : array-like of shape (n_control,)
        observed_scores : array-like of shape (n_observed,)
        nonconformities : bool, default=False
        weighted : bool, default=True
        w_test : array-like of shape (n_observed,) or None

        Returns
        -------
        np.ndarray of shape (n_observed,)
        """
        if weighted and not self.is_fitted:
            raise RuntimeError(
                "Call fit() before predict_pvalues() with weighted=True."
            )

        cal = np.asarray(calibration_scores)
        obs = np.asarray(observed_scores)

        if weighted:
            weights = self.weights
            if cal.shape[0] != weights.shape[0]:
                raise ValueError(
                    f"calibration_scores length ({cal.shape[0]}) must match "
                    f"number of control samples ({weights.shape[0]})."
                )
        else:
            weights = np.ones(cal.shape[0])

        w_test_arr = np.ones(len(obs)) if w_test is None else np.asarray(w_test, dtype=np.float64)

        total_cal = np.sum(weights)
        if not nonconformities:
            return np.array([
                (np.sum(weights[cal > s]) + wt) / (total_cal + wt)
                for s, wt in zip(obs, w_test_arr)
            ])
        return np.array([
            (np.sum(weights[cal <= s]) + wt) / (total_cal + wt)
            for s, wt in zip(obs, w_test_arr)
        ])

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
                f"Choose from {KSConformalPredictor.VALID_ADJUST_METHODS}."
            )

        return adjusted[np.argsort(sort_idx)]

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