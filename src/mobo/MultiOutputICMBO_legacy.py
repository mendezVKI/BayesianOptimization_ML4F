# -*- coding: utf-8 -*-
"""
Multi-output, multi-objective Bayesian optimization with a 2-output ICM GP
and Monte-Carlo EHVI (Expected Hypervolume Improvement) for maximization.

Outputs:
    y = [y1, y2]

Assumptions:
- two objectives, both to be maximized
- one candidate x proposed at a time (q = 1)
- EHVI is estimated by Monte Carlo from the 2D Gaussian posterior at x
- Pareto set is maintained from observed Y

Author: ChatGPT
"""

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize


class MultiOutputICMGP:
    """
    Two-output GP with intrinsic coregionalization model (ICM):

        K((x,i),(x',j)) = B_ij * k_RBF(x,x') + delta_ij sigma_i^2

    where:
        - k_RBF is an ARD RBF kernel
        - B = L L^T is a 2x2 positive semidefinite output covariance matrix
        - sigma_i^2 are output-specific noise variances

    This class also provides multi-objective BO with EHVI for 2 objectives.
    """

    def __init__(self, jitter=1e-8):
        self.jitter = jitter
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Standardization helpers
    # ------------------------------------------------------------------
    def _standardize_X(self, X):
        return (X - self.X_mean) / self.X_std

    def _standardize_Y(self, Y):
        return (Y - self.Y_mean) / self.Y_std

    def _unstandardize_Y(self, Y_scaled):
        return Y_scaled * self.Y_std + self.Y_mean

    def _unstandardize_cov_point(self, Sigma_scaled):
        """
        Unstandardize a 2x2 covariance matrix for one prediction point.
        """
        D = np.diag(self.Y_std)
        return D @ Sigma_scaled @ D

    # ------------------------------------------------------------------
    # Kernel and covariance construction
    # ------------------------------------------------------------------
    @staticmethod
    def _rbf_kernel(X1, X2, log_lengthscales, log_sigma_f):
        lengthscales = np.exp(log_lengthscales)
        sigma_f = np.exp(log_sigma_f)

        X1s = X1 / lengthscales
        X2s = X2 / lengthscales

        sqdist = (
            np.sum(X1s**2, axis=1)[:, None]
            + np.sum(X2s**2, axis=1)[None, :]
            - 2.0 * X1s @ X2s.T
        )
        sqdist = np.maximum(sqdist, 0.0)

        return sigma_f**2 * np.exp(-0.5 * sqdist)

    @staticmethod
    def _build_B(params_B):
        """
        For two outputs:
            L = [[exp(a), 0],
                 [b,      exp(c)]]
            B = L L^T
        """
        a, b, c = params_B
        L = np.array([
            [np.exp(a), 0.0],
            [b, np.exp(c)],
        ])
        return L @ L.T

    def _build_full_K(self, X1, X2, params, add_noise=False):
        """
        Builds the full covariance matrix using the packing:
            y_vec = [y1(all samples), y2(all samples)]
        """
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        D = X1.shape[1]

        log_lengthscales = params[:D]
        log_sigma_f = params[D]
        params_B = params[D + 1:D + 4]
        log_noise = params[D + 4:D + 6]

        Kx = self._rbf_kernel(X1, X2, log_lengthscales, log_sigma_f)
        B = self._build_B(params_B)

        K = np.kron(B, Kx)

        if add_noise:
            if n1 != n2:
                raise ValueError("Noise can only be added when X1 and X2 have same size.")
            noise = np.exp(log_noise) ** 2
            K += np.kron(np.diag(noise), np.eye(n1))
            K += self.jitter * np.eye(2 * n1)

        return K

    @staticmethod
    def _pack_Y(Y):
        """
        Y shape: (n_samples, 2)
        packed vector:
            [y_output_0_all_samples, y_output_1_all_samples]
        """
        return Y.T.reshape(-1)

    @staticmethod
    def _unpack_Y(y_vec, n):
        return y_vec.reshape(2, n).T

    # ------------------------------------------------------------------
    # GP fit
    # ------------------------------------------------------------------
    def fit(self, X, Y, verbose=True):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if Y.ndim != 2 or Y.shape[1] != 2:
            raise ValueError("Y must have shape (n_samples, 2).")

        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-12

        self.Y_mean = Y.mean(axis=0)
        self.Y_std = Y.std(axis=0) + 1e-12

        Xs = self._standardize_X(X)
        Ys = self._standardize_Y(Y)

        self.X_train = Xs
        self.Y_train = Ys
        self.y_train_vec = self._pack_Y(Ys)

        n, D = X.shape

        log_lengthscales0 = np.log(np.ones(D))
        log_sigma_f0 = np.log(1.0)
        params_B0 = np.array([np.log(1.0), 0.0, np.log(1.0)])
        log_noise0 = np.log(np.array([0.05, 0.05]))

        theta0 = np.concatenate([
            log_lengthscales0,
            [log_sigma_f0],
            params_B0,
            log_noise0,
        ])

        def nll(params):
            K = self._build_full_K(Xs, Xs, params, add_noise=True)

            try:
                L = cholesky(K, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                return 1e25

            alpha = solve_triangular(
                L.T,
                solve_triangular(L, self.y_train_vec, lower=True),
                lower=False,
            )

            val = 0.5 * self.y_train_vec @ alpha
            val += np.sum(np.log(np.diag(L)))
            val += 0.5 * len(self.y_train_vec) * np.log(2.0 * np.pi)

            if not np.isfinite(val):
                return 1e25
            return float(val)

        res = minimize(
            nll,
            theta0,
            method="L-BFGS-B",
            options={"maxiter": 500},
        )

        self.params = res.x

        K = self._build_full_K(Xs, Xs, self.params, add_noise=True)
        self.L = cholesky(K, lower=True, check_finite=False)
        self.alpha = solve_triangular(
            self.L.T,
            solve_triangular(self.L, self.y_train_vec, lower=True),
            lower=False,
        )

        self.is_fitted = True

        if verbose:
            print("\nGP training complete.")
            print("NLL:", res.fun)
            print("Success:", res.success)
            print("Message:", res.message)

        return self

    # ------------------------------------------------------------------
    # Posterior at one point
    # ------------------------------------------------------------------
    def point_posterior(self, x):
        """
        Posterior mean and 2x2 covariance at one input x.

        Returns
        -------
        mu : ndarray, shape (2,)
            Mean in original output units.
        Sigma : ndarray, shape (2,2)
            Covariance in original output units.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before prediction.")

        x = np.asarray(x, dtype=float).reshape(1, -1)
        xs = self._standardize_X(x)

        # cross-cov between train outputs and test outputs
        K_star = self._build_full_K(self.X_train, xs, self.params, add_noise=False)  # (2n, 2)
        mean_vec_scaled = K_star.T @ self.alpha                                       # (2,)

        K_ss_scaled = self._build_full_K(xs, xs, self.params, add_noise=False)        # (2,2)
        v = solve_triangular(self.L, K_star, lower=True)
        Sigma_scaled = K_ss_scaled - v.T @ v
        Sigma_scaled = 0.5 * (Sigma_scaled + Sigma_scaled.T)

        mu_scaled = mean_vec_scaled.reshape(2,)
        mu = self._unstandardize_Y(mu_scaled.reshape(1, 2))[0]
        Sigma = self._unstandardize_cov_point(Sigma_scaled)

        # numerical symmetrization / clipping
        Sigma = 0.5 * (Sigma + Sigma.T)
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.maximum(eigvals, 1e-16)
        Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return mu, Sigma

    def predict(self, X_star, return_cov=False):
        """
        Mean prediction for multiple points.
        If return_cov=True, returns a list of 2x2 point covariances.
        """
        X_star = np.asarray(X_star, dtype=float)
        means = []
        covs = []

        for i in range(X_star.shape[0]):
            mu, Sigma = self.point_posterior(X_star[i])
            means.append(mu)
            if return_cov:
                covs.append(Sigma)

        means = np.asarray(means)
        if return_cov:
            return means, covs
        return means

    # ------------------------------------------------------------------
    # Pareto utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _nondominated_mask(Y):
        """
        Maximization.
        Returns boolean mask for nondominated points.
        """
        Y = np.asarray(Y, dtype=float)
        n = Y.shape[0]
        mask = np.ones(n, dtype=bool)

        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(Y[j] >= Y[i]) and np.any(Y[j] > Y[i]):
                    mask[i] = False
                    break
        return mask

    @classmethod
    def pareto_front(cls, Y):
        """
        Returns nondominated observed points for maximization.
        Sorted increasingly by objective 1.
        """
        Y = np.asarray(Y, dtype=float)
        mask = cls._nondominated_mask(Y)
        P = Y[mask]

        # Sort by first objective ascending
        idx = np.argsort(P[:, 0])
        P = P[idx]

        # For a proper Pareto front in 2D maximization, after sorting by f1 ascending,
        # the f2 values should be strictly decreasing along the front.
        keep = []
        best_f2 = -np.inf
        for i in range(P.shape[0]):
            if P[i, 1] > best_f2:
                keep.append(P[i])
                best_f2 = P[i, 1]
        return np.asarray(keep)

    @staticmethod
    def hypervolume_2d_max(P, ref_point):
        """
        Hypervolume for 2D maximization relative to reference point ref_point.
        Only counts points above the reference point.

        Parameters
        ----------
        P : ndarray, shape (m,2)
            Nondominated points preferred, but function is robust to extra points.
        ref_point : array-like, shape (2,)
            Reference point, must be dominated by relevant Pareto points.

        Returns
        -------
        hv : float
        """
        P = np.asarray(P, dtype=float)
        ref = np.asarray(ref_point, dtype=float).reshape(2,)

        if P.size == 0:
            return 0.0

        # Keep only points that improve over reference
        mask = np.all(P > ref, axis=1)
        P = P[mask]
        if P.size == 0:
            return 0.0

        # Reduce to Pareto front
        mask_nd = MultiOutputICMGP._nondominated_mask(P)
        P = P[mask_nd]

        # Sort by f1 ascending
        P = P[np.argsort(P[:, 0])]

        hv = 0.0
        prev_f1 = ref[0]
        max_f2 = ref[1]

        for i in range(P.shape[0]):
            f1, f2 = P[i]
            if f2 > max_f2:
                hv += max(0.0, f1 - prev_f1) * max(0.0, f2 - ref[1])
                prev_f1 = f1
                max_f2 = f2

        return float(hv)

    @classmethod
    def hypervolume_improvement_2d(cls, y, pareto_Y, ref_point):
        """
        Hypervolume improvement of a single candidate y w.r.t. existing observations.
        Maximization, 2 objectives.
        """
        y = np.asarray(y, dtype=float).reshape(1, 2)
        pareto_Y = np.asarray(pareto_Y, dtype=float)
        ref_point = np.asarray(ref_point, dtype=float).reshape(2,)

        hv_before = cls.hypervolume_2d_max(pareto_Y, ref_point)
        hv_after = cls.hypervolume_2d_max(np.vstack([pareto_Y, y]), ref_point)
        return max(0.0, hv_after - hv_before)

    # ------------------------------------------------------------------
    # EHVI acquisition (Monte Carlo)
    # ------------------------------------------------------------------
    def acquisition_ehvi_mc(
        self,
        x,
        pareto_Y,
        ref_point,
        n_mc=512,
        rng=None,
    ):
        """
        Monte-Carlo approximation of EHVI at a single x.

        Parameters
        ----------
        x : array-like, shape (D,)
        pareto_Y : ndarray, shape (m,2)
            Current nondominated observed outputs.
        ref_point : array-like, shape (2,)
            Hypervolume reference point.
        n_mc : int
            Number of MC samples.
        rng : np.random.Generator or None
        """
        if rng is None:
            rng = np.random.default_rng()

        mu, Sigma = self.point_posterior(x)

        # Stable Cholesky for sampling
        Sigma = 0.5 * (Sigma + Sigma.T)
        jitter = 1e-12
        for _ in range(6):
            try:
                L = np.linalg.cholesky(Sigma + jitter * np.eye(2))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            # fallback if covariance is numerically problematic
            eigvals, eigvecs = np.linalg.eigh(Sigma)
            eigvals = np.maximum(eigvals, 1e-12)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        Z = rng.standard_normal((n_mc, 2))
        samples = mu[None, :] + Z @ L.T

        improvements = np.empty(n_mc, dtype=float)
        for i in range(n_mc):
            improvements[i] = self.hypervolume_improvement_2d(
                samples[i], pareto_Y, ref_point
            )

        return float(np.mean(improvements))

    # ------------------------------------------------------------------
    # Acquisition optimization
    # ------------------------------------------------------------------
    def propose_location_ehvi(
        self,
        bounds,
        pareto_Y,
        ref_point,
        n_restarts=20,
        n_mc=512,
        rng=None,
    ):
        """
        Optimize MC-EHVI over x.

        Returns
        -------
        x_best : ndarray, shape (D,)
        ehvi_best : float
        """
        bounds = np.asarray(bounds, dtype=float)
        D = bounds.shape[0]

        if rng is None:
            rng = np.random.default_rng()

        def acq_fun(x):
            val = self.acquisition_ehvi_mc(
                x=x,
                pareto_Y=pareto_Y,
                ref_point=ref_point,
                n_mc=n_mc,
                rng=rng,
            )
            return -val  # scipy minimizes

        best_res = None

        for _ in range(n_restarts):
            x0 = rng.uniform(bounds[:, 0], bounds[:, 1], size=D)

            res = minimize(
                acq_fun,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
            )

            if best_res is None or res.fun < best_res.fun:
                best_res = res

        return best_res.x, -best_res.fun

    # ------------------------------------------------------------------
    # Multi-objective BO loop with EHVI
    # ------------------------------------------------------------------
    def bayesopt_joint_mobo(
        self,
        X_init,
        Y_init,
        objective_func,
        bounds,
        n_iter=20,
        ref_point=None,
        ref_margin=0.10,
        n_restarts=20,
        n_mc_ehvi=512,
        rng=None,
        refit_each_iter=True,
        verbose=True,
    ):
        """
        Multi-objective Bayesian optimization with 2-output GP + EHVI.

        Parameters
        ----------
        X_init : ndarray, shape (n0, D)
            Initial design points.
        Y_init : ndarray, shape (n0, 2)
            Initial objective values, both maximized.
        objective_func : callable
            Function taking x and returning y of shape (2,).
        bounds : ndarray, shape (D,2)
            Search bounds.
        n_iter : int
            Number of BO iterations.
        ref_point : array-like, shape (2,) or None
            Hypervolume reference point. If None, built from initial Y.
        ref_margin : float
            If ref_point is None, set reference point slightly below current minima:
                ref = min(Y, axis=0) - ref_margin * range(Y)
        n_restarts : int
            Number of multistarts for acquisition optimization.
        n_mc_ehvi : int
            Number of MC samples for EHVI estimate.
        rng : np.random.Generator or None
        refit_each_iter : bool
            If True, refit GP after each new point.
        verbose : bool

        Returns
        -------
        results : dict
        """
        if rng is None:
            rng = np.random.default_rng()

        X = np.asarray(X_init, dtype=float).copy()
        Y = np.asarray(Y_init, dtype=float).copy()
        bounds = np.asarray(bounds, dtype=float)

        if Y.ndim != 2 or Y.shape[1] != 2:
            raise ValueError("Y_init must have shape (n_samples, 2).")

        if ref_point is None:
            y_min = np.min(Y, axis=0)
            y_max = np.max(Y, axis=0)
            span = np.maximum(y_max - y_min, 1e-12)
            ref_point = y_min - ref_margin * span
        ref_point = np.asarray(ref_point, dtype=float).reshape(2,)

        history = []

        self.fit(X, Y, verbose=False)

        pareto_Y = self.pareto_front(Y)
        hv = self.hypervolume_2d_max(pareto_Y, ref_point)

        if verbose:
            print("Initial Pareto front:")
            print(pareto_Y)
            print("Initial hypervolume:", hv)
            print("Reference point:", ref_point)

        for it in range(n_iter):
            pareto_Y = self.pareto_front(Y)

            x_next, ehvi_val = self.propose_location_ehvi(
                bounds=bounds,
                pareto_Y=pareto_Y,
                ref_point=ref_point,
                n_restarts=n_restarts,
                n_mc=n_mc_ehvi,
                rng=rng,
            )

            y_next = np.asarray(objective_func(x_next), dtype=float).reshape(2,)

            X = np.vstack([X, x_next])
            Y = np.vstack([Y, y_next])

            if refit_each_iter:
                self.fit(X, Y, verbose=False)

            pareto_Y = self.pareto_front(Y)
            hv = self.hypervolume_2d_max(pareto_Y, ref_point)

            hist_entry = {
                "iter": it + 1,
                "x_next": x_next.copy(),
                "y_next": y_next.copy(),
                "ehvi_val": float(ehvi_val),
                "pareto_Y": pareto_Y.copy(),
                "hypervolume": float(hv),
            }
            history.append(hist_entry)

            if verbose:
                print(f"\nMOBO iter {it + 1}/{n_iter}")
                print("x_next =", x_next)
                print("y_next =", y_next)
                print("EHVI =", ehvi_val)
                print("Pareto front:")
                print(pareto_Y)
                print("Hypervolume =", hv)

        return {
            "X": X,
            "Y": Y,
            "history": history,
            "pareto_Y": self.pareto_front(Y),
            "hypervolume": self.hypervolume_2d_max(self.pareto_front(Y), ref_point),
            "ref_point": ref_point,
        }