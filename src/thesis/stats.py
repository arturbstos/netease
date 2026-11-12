"""Estimation and evaluation helpers: binomial log-likelihood and Brier score, cluster bootstrap over users, weighted
ROC-AUC and average precision (PR-AUC), binomial (calibration) fit and two-way fixed-effects regression with user-clustered standard errors."""
import numpy as np, pandas as pd
from thesis.config import SEED, THREADS

def ll_rows(k, n, q):
    """Binomial log-likelihood of each row (k successes out of n, probability q clipped to [1e-6, 1-1e-6])."""
    q = np.clip(q, 1e-6, 1 - 1e-6); return k * np.log(q) + (n - k) * np.log(1 - q)

def brier_rows(k, n, q):
    """Sum of squared errors of each row's n trials (k of them successes) against probability q."""
    return k * (1 - q) ** 2 + (n - k) * q ** 2

def binom_ll(k, n, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float((k * np.log(p) + (n - k) * np.log(1 - p)).sum() / n.sum())

def cluster_boot(d, B=500, seed=SEED):
    """Cluster bootstrap over users. Yields one weight per row of d: how many times its user was drawn.
    Same draws as resampling users and concatenating their rows, without copying the rows."""
    users, row_user = np.unique(d.userId.values, return_inverse=True)   # users in sorted order, as groupby gives them
    rng = np.random.default_rng(seed)
    for _ in range(B):
        pick = rng.choice(len(users), len(users), replace=True)
        yield np.bincount(pick, minlength=len(users))[row_user].astype(float)

def pmap(fn, items, workers=None):
    """[fn(x) for x in items], with the calls spread over threads (numpy releases the lock on large arrays). Results
    keep the order of items, and items are drawn one at a time from the iterator, so random draws are unchanged.
    At most 2 x workers items are held at once."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    workers = workers or THREADS
    if workers <= 1:
        return [fn(x) for x in items]
    out, pending = [], deque()
    with ThreadPoolExecutor(workers) as ex:
        for x in items:
            pending.append(ex.submit(fn, x))
            if len(pending) >= 2 * workers:
                out.append(pending.popleft().result())
        while pending:
            out.append(pending.popleft().result())
    return out

def boot(fn, d, B=500, seed=SEED):
    """fn(w) for each cluster-bootstrap weight vector w of d (see cluster_boot), in parallel threads; a DataFrame."""
    return pd.DataFrame(pmap(fn, cluster_boot(d, B, seed)))

class WeightedAUC:
    """ROC-AUC with row weights (equal to roc_auc_score on duplicated rows); scores are sorted once."""
    def __init__(self, y, p):
        o = np.argsort(-np.asarray(p), kind="mergesort")
        self.o, self.y = o, np.asarray(y, dtype=float)[o]
        ps = np.asarray(p)[o]
        self.ends = np.r_[np.nonzero(np.diff(ps))[0], len(ps) - 1]  # last row of each tied score
    def __call__(self, w):
        w = w[self.o]
        tp = np.cumsum(w * self.y)[self.ends]; fp = np.cumsum(w * (1 - self.y))[self.ends]
        tp, fp = np.r_[0, tp], np.r_[0, fp]
        return float(np.trapezoid(tp, fp) / (tp[-1] * fp[-1]))

class WeightedAP:
    """Average precision (area under the precision-recall curve, step-wise as in sklearn's average_precision_score)
    with row weights; tied scores form one threshold. Scores are sorted once."""
    def __init__(self, y, p):
        o = np.argsort(-np.asarray(p), kind="mergesort")
        self.o, self.y = o, np.asarray(y, dtype=float)[o]
        ps = np.asarray(p)[o]
        self.ends = np.r_[np.nonzero(np.diff(ps))[0], len(ps) - 1]
    def __call__(self, w):
        w = w[self.o]
        tp = np.cumsum(w * self.y)[self.ends]; al = np.cumsum(w)[self.ends]
        rec = tp / tp[-1]; prec = np.divide(tp, al, out=np.zeros_like(tp), where=al > 0)  # rows of undrawn users
        return float(np.sum(np.diff(np.r_[0, rec]) * prec))

def binom_fit(z, k, n, w=None, slope=True, iters=50):
    """Maximum likelihood for P = sigmoid(a + b*z), k successes out of n, row weights w.
    slope=False fixes b = 1 (z is an offset); returns [a] or [a, b]."""
    w = np.ones(len(z)) if w is None else w
    th = np.zeros(2) if slope else np.zeros(1)
    X = np.column_stack([np.ones(len(z)), z]) if slope else np.ones((len(z), 1))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(th[0] + (th[1] * z if slope else z))))
        step = np.linalg.solve(X.T @ (X * (w * n * p * (1 - p))[:, None]), X.T @ (w * (k - n * p))); th = th + step
        if np.abs(step).max() < 1e-10: break
    return th

def fe_ols(d, y, xs):
    """Two-way FE (user, calendar day) by within-user demeaning with day dummies; SEs clustered by user."""
    X = pd.concat([d[xs], pd.get_dummies(d.origin_day, prefix="day", drop_first=True).astype(float)], axis=1)
    g = d.userId.values
    Yd = d[y] - d.groupby("userId")[y].transform("mean"); Xd = X - X.groupby(g).transform("mean")
    Xv, yv = Xd.values, Yd.values
    XtX_inv = np.linalg.pinv(Xv.T @ Xv); b = XtX_inv @ Xv.T @ yv; u = yv - Xv @ b
    codes = pd.Categorical(g).codes; S = np.zeros((codes.max() + 1, Xv.shape[1])); np.add.at(S, codes, Xv * u[:, None])
    G = int(codes.max()) + 1; n, k = len(yv), Xv.shape[1]       # user effects are nested in the clusters: not counted in k
    V = XtX_inv @ (S.T @ S) @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V))
    return {x: {"coef": round(float(b[i]), 5), "se": round(float(se[i]), 5), "ci95": [round(float(b[i] - 1.96 * se[i]), 5), round(float(b[i] + 1.96 * se[i]), 5)]}
            for i, x in enumerate(xs)}
