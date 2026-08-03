import math
import numpy as np
from scipy.optimize import minimize, Bounds, brentq
from utils import get_dists_between_two_sets, get_nearest_neighbor_distances


class ConformalPredictor:
    """
    Unified conformal predictor with three covariate-shift correction methods.

    All three methods produce weighted p-values. They differ in how the calibration weights are derived.

    Methods
    -------
    "entropy_balancing":
        Global importance weights estimated by maximising weight entropy
        subject to KS(reweighted_control, target) <= ks_bound.  One weight
        per calibration compound, derived from nearest-neighbour distance
        distributions (NN-distances of calibration to target vs. target to
        target).  Requires fp_target in fit().

    "ks":
        Global importance weights estimated by directly minimising
        KS(reweighted_control, target) subject to ESS >= n_eff.  No entropy
        regularisation; pure distribution matching.  More aggressive than
        entropy_balancing but can be noisier.  Requires fp_target and n_eff
        in fit() (default n_eff=30).

    "knn":
        Local, per-test-compound weights based on Tanimoto fingerprint
        similarity.  The num_nn nearest calibration neighbours of each test
        compound receive weight exp(-gamma * Tanimoto_distance); all others
        are ignored.  No global optimisation.

        gamma controls locality:
          - gamma = 0   → all num_nn neighbours weighted equally (ESS = num_nn)
          - gamma → ∞   → weight concentrates on the single nearest neighbour
                          (ESS → 1)
        Instead of setting gamma manually, pass n_eff to fit() to
        automatically find the gamma that gives mean ESS = n_eff across all
        target compounds (via Brent root-finding).

    Parameters
    ----------
    method : {"entropy_balancing", "ks", "knn"}, default "knn"
    num_nn : int, default 30
        Number of nearest calibration neighbours (knn only; -1 = all).
    weighted : bool, default True
        Use importance/similarity weights in the p-value formula.
        If False, uniform weights are used (unweighted conformal baseline).
    gamma : float, default 1.0
        Bandwidth for the exponential similarity kernel (knn only).
        Overridden automatically when n_eff is passed to fit().
    ks_bound : float, default 0.5
        KS distance upper bound (entropy_balancing only).
    ks_penalty : float, default 0.0
        Coefficient on KS term added to the entropy objective
        (entropy_balancing only).

    fit() parameters
    ----------------
    fp_target : array of shape (n_target, n_bits)
        Required for entropy_balancing (weighted=True), ks, and knn with n_eff.
    n_eff : float or None
        knn   — target mean ESS; gamma is solved automatically.
        ks    — minimum ESS floor for the weight optimisation (default 30).
        entropy_balancing — ignored.
    """

    VALID_METHODS = ("entropy_balancing", "knn", "ks")
    VALID_ADJUST_METHODS = ("BH", "BY", "Holm", "Hochberg")

    def __init__(self, method="knn", num_nn=-1, weighted=True, gamma=1.0,
                 ks_bound=0.5, ks_penalty=0.0):
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}, got {method!r}")
        self.method = method
        self.num_nn = num_nn
        self.weighted = weighted
        self.gamma = gamma
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

    def fit(self, fp_calib, scores_calib, fp_target=None, n_eff=None,
            max_calib_for_opt=5000, max_target_for_fit=5000):
        """
        Store calibration data and run method-specific fitting.

        Parameters
        ----------
        fp_calib : array of shape (n_calib, n_bits)
            Binary Morgan fingerprints for the calibration set.
        scores_calib : array of shape (n_calib,)
            Model scores or nonconformity scores for the calibration set.
            Pass raw model scores for the standard conformal approach;
            pass nonconformity scores (labels - scores) when using
            nonconformities=True in predict_pvalues().
        fp_target : array of shape (n_target, n_bits), optional
            Binary Morgan fingerprints for the target (test) set.
            Required when weighted=True for:
              - method="entropy_balancing"
              - method="ks"
              - method="knn" with n_eff is not None
            Ignored when weighted=False (no optimisation runs).
        n_eff : float or None
            Effective sample size target/floor (only used when weighted=True):
              - knn: target mean ESS — gamma is solved so that the mean ESS
                across all target compounds equals n_eff.  Must be in (1, num_nn].
              - ks:  minimum ESS floor for the weight optimisation constraint
                ESS >= n_eff.  Defaults to 30 if None.
              - entropy_balancing: ignored.

        Returns
        -------
        self
        """
        self.fp_calib = np.asarray(fp_calib, dtype=np.uint8)
        self.scores_calib = np.asarray(scores_calib, dtype=np.float64)

        def _subsample_target(fp):
            fp = np.asarray(fp, dtype=np.uint8)
            if max_target_for_fit is not None and len(fp) > max_target_for_fit:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(fp), max_target_for_fit, replace=False)
                print(f"fit: subsampling fp_target {len(fp)} → {max_target_for_fit} "
                      f"for distribution-shift estimation")
                return fp[idx]
            return fp

        if self.method == "entropy_balancing" and self.weighted:
            if fp_target is None:
                raise ValueError(
                    "fp_target is required for method='entropy_balancing' with weighted=True"
                )
            fp_target_u8 = _subsample_target(fp_target)
            X_control = get_nearest_neighbor_distances(self.fp_calib, fp_target_u8, nns=5)
            X_target = get_nearest_neighbor_distances(fp_target_u8, fp_target_u8, nns=5)
            self._fit_entropy_balancing(X_control, X_target, max_calib_for_opt=max_calib_for_opt)
            self.X_control = X_control
            self.X_target = X_target

        if self.method == "knn" and self.weighted and n_eff is not None:
            if fp_target is None:
                raise ValueError("fp_target is required when weighted=True and n_eff is specified")
            self._set_gamma_by_ess(_subsample_target(fp_target), n_eff)

        if self.method == "ks" and self.weighted:
            if fp_target is None:
                raise ValueError("fp_target is required for method='ks' with weighted=True")
            fp_target_u8 = _subsample_target(fp_target)
            X_control = get_nearest_neighbor_distances(self.fp_calib, fp_target_u8, nns=5)
            X_target = get_nearest_neighbor_distances(fp_target_u8, fp_target_u8, nns=5)
            self._fit_ks(X_control, X_target,
                         ess_min=float(n_eff) if n_eff is not None else 30.0,
                         max_calib_for_opt=max_calib_for_opt)
            self.X_control = X_control
            self.X_target = X_target

        self.is_fitted = True
        return self

    def predict_pvalues(self, fp_target, scores_target,
                        nonconformities=False, w_test=None):
        """
        Compute conformal p-values for test compounds.

        The p-value for test compound j is the weighted fraction of calibration
        samples at least as extreme as j:

            p_j = (Σ_i w_i · 1[α_i ≥ α_j] + w_test_j) / (Σ_i w_i + w_test_j)

        where α denotes the score (or nonconformity score) and the indicator
        direction flips with the nonconformities flag.

        Parameters
        ----------
        fp_target : array of shape (n_target, n_bits)
            Binary Morgan fingerprints for the test set.
            Used for Tanimoto distance computation (knn); ignored by eb/ks.
        scores_target : array of shape (n_target,)
            Model scores or nonconformity scores for the test set.
            Must match the type passed to fit() as scores_calib.
        nonconformities : bool, default False
            False → higher score = more active; counts calibration samples
                    with score > test score (standard model-score p-value).
            True  → higher nonconformity score = less conforming; counts
                    calibration samples with score <= test score.
        w_test : array of shape (n_target,) or None
            Per-test-compound importance weights (entropy_balancing / ks only).
            Obtain via estimate_test_weights().  Defaults to 1.0 for all test
            compounds when None.

        Returns
        -------
        np.ndarray of shape (n_target,) — raw (unadjusted) p-values in [0, 1].
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_pvalues().")

        fp_target = np.asarray(fp_target)
        scores_target = np.asarray(scores_target, dtype=np.float64)

        if self.method == "knn":
            return self._predict_knn(fp_target, scores_target, nonconformities)
        return self._predict_eb(scores_target, nonconformities, w_test)

    
    def get_nonconformity_scores(scores, labels, threshold, function="res"):
        """
        Compute nonconformity scores.

        function="res": returns labels - scores.
        """
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        if function == "res":
            return labels - scores
        elif function == "clip":
            M = 100.0
            return np.where(labels > threshold, M, threshold) - scores
        else:
            raise ValueError(f"Unknown function '{function}'. Use 'res' or 'clip'.")

    def get_nn_distances(self, fp_target, nns=5):
        """
        Return (X_control, X_target) mean nns-NN distance vectors.

        For ks / entropy_balancing: returns the stored arrays from fit().
        For knn: computes them on the fly from self.fp_calib and fp_target.

        Parameters
        ----------
        fp_target : array of shape (n_target, n_bits)
        nns : int, default 5

        Returns
        -------
        X_control : np.ndarray of shape (n_calib,)   — mean nns-NN dist, calib → target
        X_target  : np.ndarray of shape (n_target,)  — mean nns-NN dist, target → target
        """
        fp_target_u8 = np.asarray(fp_target, dtype=np.uint8)
        if self.method in ("ks", "entropy_balancing") and self.X_control is not None:
            return self.X_control, self.X_target
        X_control = get_nearest_neighbor_distances(self.fp_calib, fp_target_u8, nns=nns)
        X_target  = get_nearest_neighbor_distances(fp_target_u8, fp_target_u8, nns=nns)
        return X_control, X_target

    def get_calibration_weights(self, fp_target=None):
        """
        Return per-calibration-compound importance weights, normalized to sum to n_calib.

        For ks / entropy_balancing: returns self.weights (global).
        For knn: averages the local exp(-gamma * dist) weights over all fp_target compounds,
                 so each calibration compound gets a summary importance score.

        Parameters
        ----------
        fp_target : array of shape (n_target, n_bits)
            Required for method="knn".

        Returns
        -------
        np.ndarray of shape (n_calib,)
        """
        n_calib = len(self.scores_calib)
        if not self.weighted:
            return np.ones(n_calib)
        if self.method in ("entropy_balancing", "ks"):
            return self.weights if self.weights is not None else np.ones(n_calib)

        # knn: average local weights across all target compounds
        if fp_target is None:
            raise ValueError("fp_target is required for method='knn'")
        fp_target_u8 = np.asarray(fp_target, dtype=np.uint8)
        dist_matrix = get_dists_between_two_sets(self.fp_calib, fp_target_u8)  # (n_calib, n_target)
        k_eff = n_calib if self.num_nn == -1 else min(self.num_nn, n_calib)
        n_target = fp_target_u8.shape[0]
        avg_weights = np.zeros(n_calib)
        for j in range(n_target):
            dists_j = dist_matrix[:, j]
            nn_idx = np.argsort(dists_j)[:k_eff]
            raw_w = np.exp(-self.gamma * dists_j[nn_idx])
            w_sum = raw_w.sum()
            w_normed = raw_w * (k_eff / w_sum) if w_sum > 0 else np.ones(k_eff)
            avg_weights[nn_idx] += w_normed
        avg_weights /= n_target
        return avg_weights

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

    def _set_gamma_by_ess(self, fp_target, n_eff, _chunk=512):
        """Find and set gamma so that mean ESS over fp_target equals n_eff."""
        k_eff = (len(self.scores_calib) if self.num_nn == -1
                 else min(self.num_nn, len(self.scores_calib)))

        # Build nn_dists (n_target, k_eff) in chunks to cap memory.
        fp_cal = self.fp_calib.astype(np.uint8)
        chunks = []
        for start in range(0, len(fp_target), _chunk):
            dist_chunk = get_dists_between_two_sets(fp_cal, fp_target[start:start + _chunk])
            chunks.append(np.sort(dist_chunk, axis=0)[:k_eff].T)
        nn_dists = np.concatenate(chunks, axis=0)  # (n_target, k_eff)

        if n_eff >= k_eff:
            self.gamma = 0.0
            print(f"kNN fit: gamma=0.0 (n_eff={n_eff} >= k_eff={k_eff}, uniform weights)")
            return self
        if n_eff <= 1.0:
            raise ValueError(f"n_eff must be > 1, got {n_eff}")

        def mean_ess(gamma):
            w = np.exp(-gamma * nn_dists)           # (n_target, k_eff)
            w_sum = w.sum(axis=1)
            w2_sum = (w ** 2).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                ess = np.where(w2_sum > 0, w_sum ** 2 / w2_sum, 1.0)
            return float(np.nanmean(ess))

        # Find an upper bracket where mean ESS < n_eff
        gamma_high = 1.0
        while mean_ess(gamma_high) > n_eff and gamma_high < 1e6:
            gamma_high *= 2.0

        if mean_ess(gamma_high) > n_eff:
            # Target ESS not achievable (e.g. all distances identical); use gamma_high
            self.gamma = gamma_high
            print(f"kNN fit: gamma={self.gamma:.6f} (ESS target {n_eff} not achievable, using max gamma)")
            return self

        self.gamma = brentq(lambda g: mean_ess(g) - n_eff, 0.0, gamma_high, xtol=1e-6)
        achieved_ess = mean_ess(self.gamma)
        print(f"kNN fit: gamma={self.gamma:.6f}, mean n_eff={achieved_ess:.2f} (target={n_eff}, k_eff={k_eff})")
        return self

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

    def _fit_ks(self, X_control, X_target, ess_min, max_calib_for_opt=5000):
        """Minimise KS(reweighted control, target) s.t. ESS >= ess_min."""
        n = len(X_control)
        ess_min = min(ess_min, float(n))

        self.initial_ks_distance = self._weighted_ks_distance(
            X_control, X_target, weights_1=np.ones(n))

        if n > max_calib_for_opt:
            rng = np.random.default_rng(0)
            sub_idx = rng.choice(n, max_calib_for_opt, replace=False)
            X_opt = X_control[sub_idx]
            print(f"KS fit: subsampling {max_calib_for_opt}/{n} calibration points for optimisation")
        else:
            sub_idx = None
            X_opt = X_control

        n_opt = len(X_opt)
        ess_floor = min(ess_min, float(n_opt))
        w0_opt = np.ones(n_opt)

        bounds = Bounds(np.zeros(n_opt), np.full(n_opt, np.inf))
        constraints = [
            {'type': 'eq',   'fun': lambda w: np.sum(w) / n_opt - 1.0},
            {'type': 'ineq', 'fun': lambda w: n_opt ** 2 / ess_floor - np.sum(w ** 2)},
        ]

        result = minimize(
            lambda w: self._weighted_ks_distance(X_opt, X_target, weights_1=w),
            w0_opt, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 10000},
        )

        if not result.success:
            print(f"Warning: KS minimisation failed ({result.message}); falling back to uniform weights.")
            w_opt = w0_opt
        else:
            w_opt = result.x

        if sub_idx is not None:
            sort_order = np.argsort(X_control[sub_idx])
            full_weights = np.interp(
                X_control, X_control[sub_idx[sort_order]], w_opt[sort_order])
            full_weights = full_weights / full_weights.mean()
        else:
            full_weights = w_opt

        self.weights = full_weights
        self.final_ks_distance = self._weighted_ks_distance(
            X_control, X_target, weights_1=self.weights)
        self.effective_sample_size = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)

    def _fit_entropy_balancing(self, X_control, X_target, max_calib_for_opt=5000):
        n = len(X_control)

        init_ks = self._weighted_ks_distance(X_control, X_target, weights_1=np.ones(n))
        self.initial_ks_distance = init_ks

        if init_ks <= self.ks_bound:
            self.weights = np.ones(n)
            self.final_ks_distance = init_ks
            self.effective_sample_size = float(n)
            return

        if n > max_calib_for_opt:
            rng = np.random.default_rng(0)
            sub_idx = rng.choice(n, max_calib_for_opt, replace=False)
            X_opt = X_control[sub_idx]
            print(f"EB fit: subsampling {max_calib_for_opt}/{n} calibration points for optimisation")
        else:
            sub_idx = None
            X_opt = X_control

        n_opt = len(X_opt)
        w0_opt = np.ones(n_opt)

        bounds = Bounds(np.zeros(n_opt), np.full(n_opt, np.inf))
        constraints = [
            {'type': 'eq',
             'fun': lambda w: np.sum(w) / n_opt - 1.0},
            {'type': 'ineq',
             'fun': lambda w: self.ks_bound - self._weighted_ks_distance(
                 X_opt, X_target, weights_1=w)},
        ]

        def objective(w):
            return (np.sum(w * np.log(w + 1e-10))
                    + self.ks_penalty * self._weighted_ks_distance(
                        X_opt, X_target, weights_1=w))

        result = minimize(
            objective, w0_opt, method='SLSQP', bounds=bounds,
            constraints=constraints, options={'ftol': 1e-9, 'maxiter': 10000},
        )

        if not result.success:
            print(f"Warning: entropy balancing failed ({result.message}); "
                  "falling back to uniform weights.")
            w_opt = w0_opt
        else:
            w_opt = result.x

        if sub_idx is not None:
            sort_order = np.argsort(X_control[sub_idx])
            full_weights = np.interp(
                X_control, X_control[sub_idx[sort_order]], w_opt[sort_order])
            full_weights = full_weights / full_weights.mean()
        else:
            full_weights = w_opt

        self.weights = full_weights
        self.final_ks_distance = self._weighted_ks_distance(
            X_control, X_target, weights_1=self.weights)
        self.effective_sample_size = (
            np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)
        )

    def _predict_knn(self, fp_target, scores_target, nonconformities, _chunk=512):
        n_cal  = len(self.scores_calib)
        k_eff  = n_cal if self.num_nn == -1 else min(self.num_nn, n_cal)
        use_all = k_eff == n_cal

        p_values = np.zeros(len(scores_target))
        fp_cal = self.fp_calib.astype(np.uint8)
        fp_tgt = fp_target.astype(np.uint8)

        for chunk_start in range(0, len(scores_target), _chunk):
            chunk_end  = min(chunk_start + _chunk, len(scores_target))
            dist_chunk = get_dists_between_two_sets(fp_cal, fp_tgt[chunk_start:chunk_end])
            # dist_chunk: (n_cal, chunk_size)

            if use_all:
                nn_dists  = dist_chunk                                         # (n_cal, chunk_size)
                nn_scores = np.broadcast_to(
                    self.scores_calib[:, None], dist_chunk.shape)
            else:
                nn_idx    = np.argpartition(dist_chunk, k_eff, axis=0)[:k_eff] # (k_eff, chunk_size)
                nn_dists  = np.take_along_axis(dist_chunk, nn_idx, axis=0)
                nn_scores = self.scores_calib[nn_idx]

            if self.weighted:
                raw_w   = np.exp(-self.gamma * nn_dists)                       # (k_eff, chunk_size)
                w_sum   = raw_w.sum(axis=0, keepdims=True)
                weights = np.where(w_sum > 0, raw_w * (k_eff / w_sum),
                                   np.ones_like(raw_w))
            else:
                weights = np.ones(nn_dists.shape, dtype=np.float64)

            s_chunk = scores_target[chunk_start:chunk_end]                     # (chunk_size,)
            mask    = (nn_scores <= s_chunk[None, :] if nonconformities
                       else nn_scores > s_chunk[None, :])                      # (k_eff, chunk_size)

            w_test  = 1.0
            numer   = (weights * mask).sum(axis=0) + w_test                   # (chunk_size,)
            denom   = weights.sum(axis=0) + w_test
            p_values[chunk_start:chunk_end] = numer / denom

        return p_values

    def _predict_eb(self, scores_target, nonconformities, w_test=None):
        cal = self.scores_calib
        weights = (self.weights if (self.weighted and self.weights is not None)
                   else np.ones(len(cal)))
        w_test_arr = (np.ones(len(scores_target)) if w_test is None
                      else np.asarray(w_test, dtype=np.float64))

        sort_idx   = np.argsort(cal)
        cal_sorted = cal[sort_idx]
        cum_w      = np.cumsum(weights[sort_idx])
        total_cal  = cum_w[-1]

        pos = np.searchsorted(cal_sorted, scores_target, side='right')  # (n_test,)
        cum_at_pos = np.where(pos > 0, cum_w[np.minimum(pos - 1, len(cum_w) - 1)], 0.0)

        if nonconformities:
            matched = cum_at_pos                  # sum of weights[cal <= s]
        else:
            matched = total_cal - cum_at_pos      # sum of weights[cal > s]

        return (matched + w_test_arr) / (total_cal + w_test_arr)

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
