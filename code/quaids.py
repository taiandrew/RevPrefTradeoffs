"""(Quadratic) almost ideal demand system estimation via iterated SUR.

Only the AIDS branch (``quadratic=False``) is in active use, by runAIDS.py
and tableAIDS.py; the QUAIDS driver is retired to code/tags/runQUAIDS.py.

Implements the demand system of Banks, Blundell and Lewbel (1997) as estimated
in Crawford & Pendakur (2013, EJ, Section 5). Budget shares for goods
j = 1..K and observations i = 1..N are

    w_i^j = a^j + sum_k A^{jk} ln p_i^k + b^j ln x~_i
            + q^j (ln x~_i)^2 / b~_i + e_i^j,

with price indices

    ln x~_i = ln x_i - sum_k a^k ln p_i^k
              - 1/2 sum_k sum_l A^{kl} ln p_i^k ln p_i^l,
    b~_i    = prod_k (p_i^k)^{b^k},

and rationality restrictions sum_k a^k = 1, sum_k b^k = sum_k q^k = 0,
sum_k A^{kl} = 0 for all l (homogeneity) and A^{kl} = A^{lk} (symmetry).
Dropping the quadratic term (``quadratic=False``, i.e. q^j = 0) gives the
linear AIDS of Deaton & Muellbauer (1980).

Following the paper, prices and total expenditure are normalised to their
median values (so ln p = ln x = 0 at the median constraint and the a^j are
predicted budget shares there). Pass ``price_medians`` / ``exp_median`` to
normalise to a constraint shared across several estimations (e.g. a common
median across all groups, as in C&P Table 3); by default each call uses its
own data's medians. Estimation uses the iterated linear SUR of Blundell &
Robin (1999): given the price indices the system is linear, so we alternate
between fitting a symmetry-constrained SUR (adding-up is imposed by dropping
the last equation; homogeneity by using prices relative to the last good) and
updating the indices, until the parameters converge.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from linearmodels.system import SUR


@dataclass
class QuaidsResult:
    """Estimates of the (QU)AIDS parameters (indexed by good).

    ``alpha``, ``beta`` and ``lambda_`` are the intercepts a^j, expenditure
    semi-elasticities b^j and quadratic terms q^j (all zero when
    ``quadratic`` is False); ``gamma`` is the K x K symmetric price-coefficient
    matrix A. ``*_se`` are classical SUR standard errors (parameters recovered
    through the rationality restrictions get theirs through the corresponding
    linear map). ``sigma`` is the estimated error covariance of the K-1
    estimated equations, and ``sur_results`` the linearmodels results object
    from the final iteration.
    """
    alpha: pd.Series
    beta: pd.Series
    lambda_: pd.Series
    gamma: pd.DataFrame
    alpha_se: pd.Series
    beta_se: pd.Series
    lambda_se: pd.Series
    gamma_se: pd.DataFrame
    sigma: pd.DataFrame
    residuals: pd.DataFrame
    fitted: pd.DataFrame
    n_obs: int
    n_iter: int
    converged: bool
    quadratic: bool
    sur_results: object = field(repr=False)

    def summary(self) -> str:
        goods = self.alpha.index
        name = 'QUAIDS' if self.quadratic else 'AIDS'
        header = f"{'good':>12} {'alpha':>16} {'beta':>16}"
        if self.quadratic:
            header += f" {'lambda':>16}"
        lines = [
            f"{name} estimation (iterated SUR, Blundell-Robin 1999)",
            f"Observations: {self.n_obs}   goods: {len(goods)}   "
            f"iterations: {self.n_iter}   converged: {self.converged}",
            "",
            "Coefficients (standard errors), at median prices and expenditure:",
            header,
        ]
        for g in goods:
            row = (f"{g!s:>12}"
                   f" {self.alpha[g]:>8.4f} ({self.alpha_se[g]:.4f})"
                   f" {self.beta[g]:>8.4f} ({self.beta_se[g]:.4f})")
            if self.quadratic:
                row += f" {self.lambda_[g]:>8.4f} ({self.lambda_se[g]:.4f})"
            lines.append(row)
        lines += ["", "Price coefficients gamma (A):",
                  self.gamma.round(4).to_string(),
                  "", "Standard errors of gamma:",
                  self.gamma_se.round(4).to_string()]
        return "\n".join(lines)


def _price_indices(lnp: np.ndarray, lnx: np.ndarray,
                   alpha: np.ndarray, gamma: np.ndarray, beta: np.ndarray):
    """Return (ln x~, b~) given current parameters. Prices are already
    median-normalised, so alpha_0 = 0."""
    ln_a = lnp @ alpha + 0.5 * np.einsum('ik,kl,il->i', lnp, gamma, lnp)
    lnx_real = lnx - ln_a
    b_tilde = np.exp(lnp @ beta)
    return lnx_real, b_tilde


def quaids(quantities: pd.DataFrame, prices: pd.DataFrame,
           quadratic: bool = True, price_medians=None, exp_median=None,
           max_iter: int = 200, tol: float = 1e-8) -> QuaidsResult:
    """Estimate the (QU)AIDS on consumption data by iterated restricted SUR.

    ``quantities`` and ``prices`` follow the project-wide conventions (N rows,
    K goods, shared index); every observation must have strictly positive
    total expenditure. With ``quadratic=False`` the quadratic term is dropped
    and plain AIDS is estimated.

    ``price_medians`` (per good, aligned to the columns) and ``exp_median``
    fix the normalisation constraint; pass a common pair across groups to make
    the level coefficients comparable. When omitted, each call normalises to
    its own data's medians. Prints the model summary and returns a
    :class:`QuaidsResult`.
    """
    if quantities.shape != prices.shape:
        raise ValueError(
            f"quantities {quantities.shape} and prices {prices.shape} must have the same shape"
        )
    if not quantities.index.equals(prices.index):
        raise ValueError("quantities and prices must share the same index")

    q = quantities.to_numpy(dtype=float)
    p = prices.to_numpy(dtype=float)
    if np.isnan(q).any() or np.isnan(p).any():
        raise ValueError("quantities and prices must not contain NaNs")
    if (p <= 0).any():
        raise ValueError("all prices must be strictly positive")
    if (q < 0).any():
        raise ValueError("quantities must be non-negative")

    goods = list(quantities.columns)
    n, k = q.shape
    if k < 2:
        raise ValueError("(QU)AIDS needs at least two goods")

    x = (p * q).sum(axis=1)
    if (x <= 0).any():
        raise ValueError(
            "every observation must have positive total expenditure "
            "(filter out zero bundles before estimation)"
        )
    w = (p * q) / x[:, None]

    # A constant budget share (e.g. a good nobody in the sample buys) makes
    # its equation degenerate: nothing to estimate, and the SUR fit divides
    # by a zero total sum of squares.
    constant = [g for g, is_const in
                zip(goods, np.ptp(w, axis=0) == 0) if is_const]
    if constant:
        raise ValueError(
            f"budget shares are constant for goods {constant}; "
            "drop these goods or estimate on a different sample"
        )

    # Median normalisation: ln p = ln x = 0 at the (possibly shared) median
    # constraint
    if price_medians is None:
        pmed = np.median(p, axis=0)
    else:
        pm = price_medians.reindex(goods) if isinstance(price_medians, pd.Series) \
            else price_medians
        pmed = np.asarray(pm, dtype=float)
        if pmed.shape != (k,):
            raise ValueError(f"price_medians must have one value per good ({k})")
    xmed = np.median(x) if exp_median is None else float(exp_median)
    lnp = np.log(p / pmed)
    lnx = np.log(x / xmed)

    # Equation and variable labels for the K-1 estimated equations. With
    # homogeneity, sum_k A^{jk} ln p^k = sum_{l<K} A^{jl} (ln p^l - ln p^K),
    # so relative log-prices carry the free gamma coefficients.
    eq_labels = [f"w_{g}" for g in goods[:-1]]
    rp_labels = [f"lnp_{g}" for g in goods[:-1]]
    rel_lnp = lnp[:, :-1] - lnp[:, [-1]]
    exog_cols = ['const'] + rp_labels + ['lnx'] + (['lnx2'] if quadratic else [])

    # Symmetry constraints A^{jl} = A^{lj} on the stacked parameter vector
    def fit_sur(lnx_real: np.ndarray, b_tilde: np.ndarray):
        columns = [np.ones(n), rel_lnp, lnx_real]
        if quadratic:
            columns.append(lnx_real ** 2 / b_tilde)
        exog = pd.DataFrame(np.column_stack(columns), columns=exog_cols,
                            index=quantities.index)
        equations = {
            eq_labels[j]: {'dependent': pd.Series(w[:, j], index=quantities.index,
                                                  name=eq_labels[j]),
                           'exog': exog}
            for j in range(k - 1)
        }
        mod = SUR(equations)
        names = mod.param_names
        rows = []
        for j in range(k - 1):
            for l in range(j + 1, k - 1):
                row = pd.Series(0.0, index=names)
                row[f"{eq_labels[j]}_{rp_labels[l]}"] = 1.0
                row[f"{eq_labels[l]}_{rp_labels[j]}"] = -1.0
                rows.append(row)
        if rows:
            mod.add_constraints(pd.DataFrame(rows))
        res = mod.fit(method='gls', iterate=True, cov_type='unadjusted')
        return res, names

    def extract(res, names):
        """Pull (alpha, gamma, beta, lambda) for the full K goods from the
        fitted K-1 equations, using adding-up, homogeneity and symmetry."""
        est = res.params
        alpha = np.empty(k)
        beta = np.empty(k)
        lam = np.zeros(k)
        gamma = np.empty((k, k))
        for j in range(k - 1):
            alpha[j] = est[f"{eq_labels[j]}_const"]
            beta[j] = est[f"{eq_labels[j]}_lnx"]
            if quadratic:
                lam[j] = est[f"{eq_labels[j]}_lnx2"]
            for l in range(k - 1):
                gamma[j, l] = est[f"{eq_labels[j]}_{rp_labels[l]}"]
        alpha[-1] = 1 - alpha[:-1].sum()
        beta[-1] = -beta[:-1].sum()
        lam[-1] = -lam[:-1].sum()
        gamma[:-1, -1] = -gamma[:-1, :-1].sum(axis=1)   # homogeneity
        gamma[-1, :-1] = gamma[:-1, -1]                 # symmetry
        gamma[-1, -1] = -gamma[-1, :-1].sum()
        return alpha, gamma, beta, lam

    # Iterated linear SUR: start from a = 1/K, A = 0, b = 0
    alpha = np.full(k, 1 / k)
    gamma = np.zeros((k, k))
    beta = np.zeros(k)
    lam = np.zeros(k)

    converged = False
    for n_iter in range(1, max_iter + 1):
        lnx_real, b_tilde = _price_indices(lnp, lnx, alpha, gamma, beta)
        res, names = fit_sur(lnx_real, b_tilde)
        new = extract(res, names)
        delta = max(np.abs(new_v - old_v).max()
                    for new_v, old_v in zip(new, (alpha, gamma, beta, lam)))
        alpha, gamma, beta, lam = new
        if delta < tol:
            converged = True
            break

    # Standard errors: the full parameter vector is a linear map T of the
    # stacked estimated parameters (plus a constant for alpha_K), so its
    # covariance is T cov(theta) T'.
    pos = {name: i for i, name in enumerate(names)}
    n_free = len(names)

    def row(*terms):
        r = np.zeros(n_free)
        for name, coef in terms:
            r[pos[name]] += coef
        return r

    t_rows = {}
    for j in range(k - 1):
        t_rows[('alpha', j)] = row((f"{eq_labels[j]}_const", 1.0))
        t_rows[('beta', j)] = row((f"{eq_labels[j]}_lnx", 1.0))
        if quadratic:
            t_rows[('lambda', j)] = row((f"{eq_labels[j]}_lnx2", 1.0))
        for l in range(k - 1):
            t_rows[('gamma', j, l)] = row((f"{eq_labels[j]}_{rp_labels[l]}", 1.0))
    t_rows[('alpha', k - 1)] = -sum(t_rows[('alpha', j)] for j in range(k - 1))
    t_rows[('beta', k - 1)] = -sum(t_rows[('beta', j)] for j in range(k - 1))
    if quadratic:
        t_rows[('lambda', k - 1)] = -sum(t_rows[('lambda', j)] for j in range(k - 1))
    for j in range(k - 1):
        t_rows[('gamma', j, k - 1)] = -sum(t_rows[('gamma', j, l)]
                                           for l in range(k - 1))
        t_rows[('gamma', k - 1, j)] = t_rows[('gamma', j, k - 1)]
    t_rows[('gamma', k - 1, k - 1)] = sum(
        t_rows[('gamma', j, l)] for j in range(k - 1) for l in range(k - 1))

    keys = list(t_rows)
    T = np.vstack([t_rows[key] for key in keys])
    cov_full = T @ res.cov.to_numpy() @ T.T
    se = dict(zip(keys, np.sqrt(np.maximum(np.diag(cov_full), 0.0))))

    alpha_se = np.array([se[('alpha', j)] for j in range(k)])
    beta_se = np.array([se[('beta', j)] for j in range(k)])
    lam_se = np.array([se.get(('lambda', j), 0.0) for j in range(k)])
    gamma_se = np.array([[se[('gamma', j, l)] for l in range(k)]
                         for j in range(k)])

    # Fitted shares and residuals for all K goods (rows of fitted sum to 1
    # by the rationality restrictions; the lambda term vanishes when lam = 0)
    lnx_real, b_tilde = _price_indices(lnp, lnx, alpha, gamma, beta)
    fitted = (alpha[None, :] + lnp @ gamma.T + np.outer(lnx_real, beta)
              + np.outer(lnx_real ** 2 / b_tilde, lam))

    result = QuaidsResult(
        alpha=pd.Series(alpha, index=goods, name='alpha'),
        beta=pd.Series(beta, index=goods, name='beta'),
        lambda_=pd.Series(lam, index=goods, name='lambda'),
        gamma=pd.DataFrame(gamma, index=goods, columns=goods),
        alpha_se=pd.Series(alpha_se, index=goods, name='alpha_se'),
        beta_se=pd.Series(beta_se, index=goods, name='beta_se'),
        lambda_se=pd.Series(lam_se, index=goods, name='lambda_se'),
        gamma_se=pd.DataFrame(gamma_se, index=goods, columns=goods),
        sigma=pd.DataFrame(res.sigma, index=eq_labels, columns=eq_labels),
        residuals=pd.DataFrame(w - fitted, index=quantities.index, columns=goods),
        fitted=pd.DataFrame(fitted, index=quantities.index, columns=goods),
        n_obs=n,
        n_iter=n_iter,
        converged=converged,
        quadratic=quadratic,
        sur_results=res,
    )
    print(result.summary())
    return result
