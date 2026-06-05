import math
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds


class EntropyBalancingConformalPredictor:
    """
    Covariate-shift corrected conformal predictor using entropy balancing weights.

    Workflow
    --------
    1. Instantiate with X_control and X_target feature matrices.
    2. Call fit() to compute entropy balancing weights.
    3. Call predict_pvalues() with a nonconformity score vector to get p-values.
    4. Optionally call p_adjust() to correct for multiple testing.

    Parameters
    ----------
    X_control : np.ndarray of shape (n_control, n_features)
        Feature matrix for the calibration / control set.
    X_target : np.ndarray of shape (n_target, n_features)
        Feature matrix for the target distribution.
    max_order : int, default=1
        Moment order for balancing constraints.
        1 → match feature means only (recommended for binary fingerprints).
    lambda_reg : float, default=0.0
        Regularisation strength for soft moment matching. When > 0, the hard
        moment constraints are dropped and replaced with a penalty term:
        ``- lambda_reg * ||Phi_c.T @ w - target_moments||^2``
        added to the entropy objective. This always yields a solution and is
        more numerically stable than hard constraints for high-dimensional
        fingerprints. Larger values enforce tighter moment matching at the
        cost of weight entropy.
    """

    VALID_ADJUST_METHODS = ("BH", "BY", "Holm", "Hochberg")

    def __init__(self, X_control, X_target, max_order=1, lambda_reg=1.0, use_weighted_ks=True):
        self.X_control = np.asarray(X_control, dtype=np.float64)
        self.X_target = np.asarray(X_target, dtype=np.float64)
        self.max_order = max_order  # number of moments to balance (1 = means only)
        self.lambda_reg = lambda_reg  # soft penalty strength for moment matching
        self.use_weighted_ks = use_weighted_ks  # whether to use weighted Kolmogorov-Smirnov test

        self.weights_ = None          # set after fit()
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_features(X, max_order):
        feats = [X]
        for k in range(2, max_order + 1):
            feats.append(X ** k)
        return np.concatenate(feats, axis=1)
    

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self):
        """
        Solve the entropy balancing optimisation problem and store weights.

        Returns
        -------
        self
        """

        if not self.use_weighted_ks:
            Phi_c = self._build_features(self.X_control, self.max_order)
            Phi_t = self._build_features(self.X_target, self.max_order)
            target_moments = Phi_t.mean(axis=0)

        n = self.X_control.shape[0]

        # Objective: maximize entropy - lambda * penalty
        # We minimize the negative of this
        def objective(w):
            entropy = -np.sum(w * np.log(w + 1e-10))
            
            if self.use_weighted_ks:
                # Weighted KS distance penalty
                penalty = self._weighted_ks_distance(
                    self.X_control, self.X_target, weights_1=w
                )
            else:
                # Moment matching penalty
                residual = Phi_c.T @ w - target_moments
                penalty = np.sum(residual ** 2)
            
            return -(entropy - self.lambda_reg * penalty)

        # Constraints: w >= 0 and sum(w) == 1
        bounds = Bounds(np.zeros(n), np.full(n, np.inf))
        linear_constraint = LinearConstraint(
            np.ones((1, n)),
            lb=[1.0],
            ub=[1.0]
        )

        # Initial guess: uniform weights
        w0 = np.ones(n) / n

        print("Starting entropy balancing optimization...")
        print(f"Sum of initial weights: {w0.sum()}, min: {w0.min()}, max: {w0.max()}")

        # Solve
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=linear_constraint,
            options={'ftol': 1e-9, 'maxiter': 10000}
        )

        if not result.success:
            raise RuntimeError(
                f"Entropy balancing optimisation failed: "
                f"{result.message}"
            )

        self.weights_ = result.x
        self._is_fitted = True
        return self

    def predict_pvalues(self, calibration_scores, observed_scores, weighted=True):
        """
        Compute conformal p-values.

        For each observed score s*, the p-value is:
            p = sum of w_i for all calibration points i where s_i >= s*

        Parameters
        ----------
        calibration_scores : array-like of shape (n_control,)
            Scalar nonconformity score for each control/calibration molecule.
        observed_scores : array-like of shape (n_observed,)
            Scalar nonconformity scores for the molecules to test.
        weighted : bool, default=True
            If True, use entropy balancing weights (requires fit() to have been
            called). If False, use uniform weights (standard conformal p-values)
            — fit() does not need to be called in this case.

        Returns
        -------
        np.ndarray of shape (n_observed,)
        """
        if weighted and not self._is_fitted:
            raise RuntimeError("Call fit() before predict_pvalues() with weighted=True.")

        cal = np.asarray(calibration_scores)
        obs = np.asarray(observed_scores)

        if weighted:
            weights = self.weights_
            if cal.shape[0] != weights.shape[0]:
                raise ValueError(
                    f"calibration_scores length ({cal.shape[0]}) must match "
                    f"number of control samples ({weights.shape[0]})."
                )
        else:
            weights = np.full(cal.shape[0], 1.0 / cal.shape[0])

        p_values = np.array([
            np.sum(weights[cal > s]) for s in obs
        ])
        return p_values

    @staticmethod
    def p_adjust(p_values, method="BH"):
        """
        Adjust p-values for multiple testing.

        Parameters
        ----------
        p_values : array-like of shape (n,)
        method : str
            One of 'BH' (Benjamini-Hochberg), 'BY' (Benjamini-Yekutieli),
            'Holm', or 'Hochberg'.

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

        # Harmonic-series approximation (used by BY)
        Hn = math.log(n) + 0.5772 + 0.5 / n - 1.0 / (12 * n ** 2) + 1.0 / (120 * n ** 4)

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
                f"Choose from {EntropyBalancingConformalPredictor.VALID_ADJUST_METHODS}."
            )

        # Restore original order
        return adjusted[np.argsort(sort_idx)]
    
    @staticmethod
    def _weighted_ks_distance(sample_1, sample_2, weights_1=None,
                              weights_2=None):
        """
        Get weighted Kolmogorov-Smirnov distance.

        Parameters
        ----------
        sample_1 : np.ndarray of shape (n1,)
        sample_2 : np.ndarray of shape (n2,)
        weights_1 : np.ndarray of shape (n1,), optional
            Weights for samples in sample_1. If None, uniform weights.
        weights_2 : np.ndarray of shape (n2,), optional
            Weights for samples in sample_2. If None, uniform weights.

        Returns
        -------
        float
            Weighted KS distance between the two distributions.
        """
        # Convert to numpy if needed
        sample_1 = np.asarray(sample_1)
        sample_2 = np.asarray(sample_2)

        # Pre-compute sorted indices
        idx1 = np.argsort(sample_1)
        idx2 = np.argsort(sample_2)

        data1_sorted = sample_1[idx1]
        data2_sorted = sample_2[idx2]

        # Handle weights
        if weights_1 is None:
            weights_1 = np.ones(len(sample_1))
        if weights_2 is None:
            weights_2 = np.ones(len(sample_2))

        # Re-order weights according to sorted indices
        weights_1_sorted = np.asarray(weights_1)[idx1]
        weights_2_sorted = np.asarray(weights_2)[idx2]

        # Normalize weights
        weights_1_norm = weights_1_sorted / np.sum(weights_1_sorted)
        weights_2_norm = weights_2_sorted / np.sum(weights_2_sorted)

        # Cumulative sums
        cum_w1 = np.cumsum(weights_1_norm)
        cum_w2 = np.cumsum(weights_2_norm)

        # Compute max difference at first dataset's points
        idx1_eval = np.searchsorted(data1_sorted, data1_sorted,
                                    side='right')
        idx2_eval = np.searchsorted(data2_sorted, data1_sorted,
                                    side='right')

        cdf1_vals = cum_w1[idx1_eval - 1]
        cdf2_vals = cum_w2[np.maximum(idx2_eval - 1, 0)]

        ks_dist = np.max(np.abs(cdf1_vals - cdf2_vals))

        return ks_dist


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":

    # Create synthetic binary fingerprint data for control and target sets.
    # NOTE: exact moment matching (tol=0) becomes infeasible for high-dimensional
    # binary fingerprints. Use tol > 0 for approximate balancing in that regime.
    np.random.seed(42)
    n_control = 1000
    n_target = 200
    n_features = 512

    X_control = np.random.randint(0, 2, size=(n_control, n_features)).astype(np.float32)
    X_target = np.random.randint(0, 2, size=(n_target, n_features)).astype(np.float32)

    predictor = EntropyBalancingConformalPredictor(
        X_control, X_target, max_order=1, solver="SCS", lambda_reg=1.0
    )
    predictor.fit()

    print("Weights:", predictor.weights_)
    print("Sum:    ", predictor.weights_.sum())

    print("\nTarget bit frequencies (first 10):")
    print(X_target.mean(axis=0)[:10])
    print("\nWeighted control bit frequencies (first 10):")
    print((predictor.weights_[:, None] * X_control).sum(axis=0)[:10])

    # Use mean bit value per molecule as a simple scalar nonconformity score
    calib_scores = X_control.mean(axis=1)
    obs_scores = X_target.mean(axis=1)

    p_values = predictor.predict_pvalues(calib_scores, obs_scores)
    print("\nP-values:", p_values)

    adjusted = predictor.p_adjust(p_values, method="BH")
    print("Adjusted p-values (BH):", adjusted)
