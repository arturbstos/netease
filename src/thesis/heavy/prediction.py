"""Slow steps of notebook 02 (RQ2: predicting active and inactive users). Each function writes its outputs; the
notebook reads them back through thesis.cache.ensure. The code is the notebook's, moved here unchanged except that the
bootstrap replicates run in parallel threads (same draws, same results)."""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from thesis.cache import step, path, read, write_json
from thesis.config import SAMPLE, SEED, P, connect
from thesis.panel import (USER_WEIGHT, target_sets, lin, user_w, LGB_PARAMS, LGB_GRID, LOGIT_C_GRID,
                          lgb_params, logistic_C, feature_group, rule_return, SCORECARD)
from thesis.stats import binom_ll, binom_fit, boot, WeightedAUC, ll_rows
from thesis.heavy.common import panel, ci, shrunk_click_rate


def _early_stop_fit(params, d0, d1):
    return lgb.train(params, d0, 3000, valid_sets=[d1], callbacks=[lgb.early_stopping(50, verbose=False)])


# ---------------------------------------------------------------------------------------------------------------------
@step("02.correlations", ["10_feature_correlations.json"])
def correlations():
    """Spearman correlations between all pairs of inputs on training snapshots, and a few pairs the scorecard reads."""
    _, FEATS, tr, _, _ = panel()
    R_ = tr[FEATS].astype(float).rank().corr().values
    iu = np.triu_indices(len(FEATS), 1)
    r_abs = np.abs(R_[iu])
    r_abs = r_abs[~np.isnan(r_abs)]                       # inputs that never vary in training have no correlation
    PAIRS = [("w7_sessions", "h_sessions"), ("w7_visit_days", "w7_sessions"), ("w7_impr", "w7_sessions"),
             ("h_impr", "h_sessions"), ("h_clicks", "h_likes"), ("w7_visit_days", "h_visit_days")]
    idx = {f: i for i, f in enumerate(FEATS)}
    write_json("10_feature_correlations.json", {
        "inputs": len(FEATS), "pairs": int(len(r_abs)), "median_abs_corr": round(float(np.median(r_abs)), 3),
        "share_pairs_abs_corr_above_0.8": round(float((r_abs > 0.8).mean()), 4),
        "share_pairs_abs_corr_above_0.5": round(float((r_abs > 0.5).mean()), 4),
        "selected_pairs": {f"{a} ~ {b}": round(float(R_[idx[a], idx[b]]), 3) for a, b in PAIRS if a in idx and b in idx}})


# ---------------------------------------------------------------------------------------------------------------------
def valid_logloss(kb, nb, p, ub):
    """User-weighted binomial log-loss per trial: the criterion of early stopping and of every tuning choice."""
    return float(-(ub * ll_rows(kb, nb, p)).sum() / (ub * nb).sum())


def _stacked(a, feats, k, n, ua):
    """Each row as its successes (label 1) and failures (label 0), for the logistic regression."""
    X2 = pd.concat([lin(a, feats), lin(a, feats)])
    y2 = np.r_[np.ones(len(a)), np.zeros(len(a))]
    w2 = np.r_[k(a).values * ua, (n(a) - k(a)).values * ua]
    return X2, y2, w2


def _tune(target):
    _, FEATS, tr, va, te = panel()
    (a, b, _), k, n = target_sets(target, tr, va, te)
    ua, ub = user_w(a), user_w(b)
    kb, nb = k(b).values.astype(float), n(b).values.astype(float)
    out = {"lightgbm": {"grid": []}, "logistic": {"grid": []}}
    sets = {}
    for s in LGB_GRID:
        t0 = time.time()
        if s["min_data_in_leaf"] not in sets:    # one pair of datasets per leaf size (it affects binning; the tree
            dtr = lgb.Dataset(a[FEATS], label=k(a) / n(a), weight=n(a).values * ua)   # size does not)
            dva = lgb.Dataset(b[FEATS], label=k(b) / n(b), weight=n(b).values * ub, reference=dtr)
            sets[s["min_data_in_leaf"]] = (dtr, dva)
        dtr, dva = sets[s["min_data_in_leaf"]]
        m = _early_stop_fit(dict(LGB_PARAMS, **s), dtr, dva)
        p = m.predict(b[FEATS], num_iteration=m.best_iteration)
        out["lightgbm"]["grid"].append({**s, "trees": m.best_iteration,
                                        "valid_logloss": round(valid_logloss(kb, nb, p, ub), 6),
                                        "seconds": round(time.time() - t0, 1)})
    X2, y2, w2 = _stacked(a, FEATS, k, n, ua)
    Xb = lin(b, FEATS)
    for C in LOGIT_C_GRID:
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        lr.fit(X2, y2, logisticregression__sample_weight=w2)
        p = lr.predict_proba(Xb)[:, 1]
        out["logistic"]["grid"].append({"C": C, "valid_logloss": round(valid_logloss(kb, nb, p, ub), 6)})
    best = min(out["lightgbm"]["grid"], key=lambda g: g["valid_logloss"])
    out["lightgbm"]["chosen"] = {"num_leaves": best["num_leaves"], "min_data_in_leaf": best["min_data_in_leaf"]}
    out["lightgbm"]["at_grid_edge"] = [p for p in ("num_leaves", "min_data_in_leaf")
                                       if best[p] in (min(g[p] for g in LGB_GRID), max(g[p] for g in LGB_GRID))]
    out["logistic"]["chosen_C"] = min(out["logistic"]["grid"], key=lambda g: g["valid_logloss"])["C"]
    out["logistic"]["at_grid_edge"] = out["logistic"]["chosen_C"] in (min(LOGIT_C_GRID), max(LOGIT_C_GRID))
    return out


@step("02.tuning", ["10_tuning.json"])
def tuning():
    """LightGBM tree size and logistic regression penalty for each outcome, chosen on validation users."""
    write_json("10_tuning.json", {"criterion": "user-weighted binomial log-loss per trial on validation users",
                                  "R": _tune("R"), "C": _tune("C")})


# ---------------------------------------------------------------------------------------------------------------------
def logit_fit(a, feats, k, n, ua, C_):
    """Logistic regression on k successes out of n: each row enters as its successes (label 1) and failures (label 0)."""
    X2, y2, w2 = _stacked(a, feats, k, n, ua)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=C_, max_iter=2000))
    lr.fit(X2, y2, logisticregression__sample_weight=w2)
    assert np.isfinite(lr[-1].coef_).all(), "non-finite logistic regression"
    return lr


def _fit_all(target, SAVED):
    _, FEATS, tr, va, te = panel()
    (a, b, c), k, n = target_sets(target, tr, va, te)
    pooled = k(a).sum() / n(a).sum()
    preds = {"constant": np.full(len(c), pooled)}
    preds["rule"] = rule_return(a, c) if target == "R" else shrunk_click_rate(a, c)

    ua = user_w(a)
    sc_feats = SCORECARD[target]
    sc = logit_fit(a, sc_feats, k, n, ua, logistic_C(target))
    lr = logit_fit(a, FEATS, k, n, ua, logistic_C(target))
    dtr = lgb.Dataset(a[FEATS], label=k(a) / n(a), weight=n(a).values * ua)
    dva = lgb.Dataset(b[FEATS], label=k(b) / n(b), weight=n(b).values * user_w(b), reference=dtr)
    gbm = _early_stop_fit(lgb_params(target), dtr, dva)

    # validation and test predictions for calibration and the later notebooks
    def p3(d):
        return {"scorecard": sc.predict_proba(lin(d, sc_feats))[:, 1], "logistic": lr.predict_proba(lin(d, FEATS))[:, 1],
                "lightgbm": gbm.predict(d[FEATS], num_iteration=gbm.best_iteration)}
    pc = p3(c)
    preds.update(pc)
    for name, d in (("valid", b), ("test", c)):
        SAVED.append(pd.DataFrame({"target": target, "set": name, "userId": d.userId.values,
                                   "origin_day": d.origin_day.values, "k": k(d).values, "n": n(d).values,
                                   **(pc if name == "test" else p3(d))}))
    if target == "C":                                          # click propensity for EVERY test origin (section 7)
        SAVED.append(pd.DataFrame({"target": target, "set": "test_all_origins", "userId": te.userId.values,
                                   "origin_day": te.origin_day.values, "k": np.nan, "n": np.nan, **p3(te)}))
    preds = {m: preds[m] for m in ("constant", "rule", "scorecard", "logistic", "lightgbm")}   # the notebook's order

    # test accuracy: pseudo-R2 (probabilities) and AUC (ranking)
    kc, nc = k(c).values.astype(float), n(c).values.astype(float)
    ll0 = binom_ll(kc, nc, preds["constant"])
    if target == "R":
        h, ev = np.ones(len(c), bool), c.y_visit_days.values > 0
    else:
        h = c.y_impr.values >= 20
        ev = c.y_clicks.values[h] > 0
    out = {"test_rows": len(c), "test_users": int(c.userId.nunique()), "lightgbm_trees": gbm.best_iteration,
           "event": "return within 7 days" if target == "R" else "any click, origins with 20+ cards shown ahead",
           "event_base_rate": round(float(ev.mean()), 4)}
    for name, p in preds.items():
        out[name] = {"pseudo_r2": round(1 - binom_ll(kc, nc, p) / ll0, 4), "auc": round(roc_auc_score(ev, p[h]), 4)}
    if target == "R":                                          # lift: no-return rate among the 10% least likely to return
        low = preds["lightgbm"] <= np.quantile(preds["lightgbm"], 0.10)
        out["lightgbm"]["lift_no_return_bottom10"] = round(float((~ev[low]).mean() / (~ev).mean()), 3)
    names = list(sc_feats) + (["no_prev_visit_day"] if "gap_since_prev_visit" in sc_feats else [])
    out["scorecard_coefficients_per_sd"] = dict(zip(names, np.round(sc[-1].coef_[0], 4).tolist()))

    # by days of history; pseudo-R2 against the training mean of the same history group
    def cut(d):
        return pd.cut(d.history_days, [0, 1, 3, 7, 14, 30], labels=["1", "2-3", "4-7", "8-14", "15+"])

    bins, bins_a = cut(c), cut(a)
    by = {}
    for lab in bins.cat.categories:
        mk = (bins == lab).values
        cc = c[mk]
        if len(cc) < 500:
            continue
        aa = a[(bins_a == lab).values]
        l0 = binom_ll(k(cc), n(cc), np.full(len(cc), k(aa).sum() / n(aa).sum()))
        e = {"rows": int(mk.sum())}
        for name in ("lightgbm", "scorecard", "rule"):
            e[f"{name}_pseudo_r2"] = round(1 - binom_ll(k(cc), n(cc), preds[name][mk]) / l0, 4)
        if target == "R":
            for name in ("lightgbm", "rule"):
                e[f"{name}_auc_return"] = round(roc_auc_score(cc.y_visit_days > 0, preds[name][mk]), 4)
        by[lab] = e
    out["by_history_days"] = by
    nu = (c.tenure <= 1).values                                # new accounts (at most one month old)
    if nu.sum() > 500:
        l0 = binom_ll(k(c[nu]), n(c[nu]), preds["constant"][nu])
        out["new_accounts"] = {"rows": int(nu.sum()),
                               **{f"{m}_pseudo_r2": round(1 - binom_ll(k(c[nu]), n(c[nu]), preds[m][nu]) / l0, 4)
                                  for m in ("lightgbm", "scorecard", "rule")}}
        if target == "R":
            out["new_accounts"]["lightgbm_auc"] = round(roc_auc_score(c.y_visit_days[nu] > 0, preds["lightgbm"][nu]), 4)
    return out


@step("02.models", ["10_panel_models.json", "10_predictions.parquet"], needs=["02.tuning"])
def models():
    """Constant, rule, scorecard, logistic regression and LightGBM for return and clicks; test scores and predictions."""
    _, FEATS, tr, va, te = panel()
    SAVED = []
    res = {"sample_pct": SAMPLE, "equal_user_weight": USER_WEIGHT,
           "rows": {"train": len(tr), "valid": len(va), "test": len(te)}, "n_features": len(FEATS),
           "R_return": _fit_all("R", SAVED), "C_click_propensity": _fit_all("C", SAVED)}
    pd.concat(SAVED, ignore_index=True).to_parquet(path("10_predictions.parquet"), index=False)
    write_json("10_panel_models.json", res)


# ---------------------------------------------------------------------------------------------------------------------
def _nbd_ll(params, x, T):
    """Log-likelihood of the NBD model (BG/NBD without dropout): x repeat visit days in T days, gamma-Poisson rate."""
    from scipy.special import gammaln
    r, alpha = np.exp(params)
    return gammaln(r + x) - gammaln(r) + r * np.log(alpha) - (r + x) * np.log(alpha + T)


def _bgnbd_ll(params, x, tx, T):
    """Log-likelihood of the BG/NBD model (Fader, Hardie and Lee 2005): x repeat visit days, the last one at tx, in T days."""
    from scipy.special import gammaln, betaln
    r, alpha, a, b = np.exp(params)
    base = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    t1 = betaln(a, b + x) - betaln(a, b) - (r + x) * np.log(alpha + T)
    t2 = np.where(x > 0, betaln(a + 1, np.maximum(b + x - 1, 1e-12)) - betaln(a, b) - (r + x) * np.log(alpha + tx), -np.inf)
    return base + np.logaddexp(t1, t2)


def _bgnbd_expected(params, x, tx, T, tau):
    """Expected visit days in the next tau days under BG/NBD."""
    from scipy.special import hyp2f1
    r, alpha, a, b = params
    A = (a + b + x - 1) / (a - 1)
    B = 1 - ((alpha + T) / (alpha + T + tau)) ** (r + x) * hyp2f1(r + x, b + x, a + b + x - 1, tau / (alpha + T + tau))
    C = 1 + (x > 0) * a / (b + x - 1) * ((alpha + T) / (alpha + tx)) ** (r + x)
    return A * B / C


@step("02.bgnbd", ["14_bgnbd.json"], needs=["02.models"])
def bgnbd():
    """Marketing benchmark for return: the BG/NBD model on visit days alone. A 'transaction' is a visit day. The model is
    fitted to training users' days 1 to 15 (x = repeat visit days, tx = day of the last one, T = days observed, all
    counted from the user's first day) and predicts the visit days in the 7 days after each test snapshot. When the
    fitted chance of dropping out is practically zero, BG/NBD equals the NBD model and the exact NBD forecast is used.
    Compared on the same test snapshots with the main LightGBM and with LightGBM given only x and T."""
    from scipy.optimize import minimize
    df, FEATS, tr, va, te = panel()
    cal = tr.groupby("userId", observed=True).origin_day.agg(["min", "max", "size"])
    x = (cal["size"] - 1).values.astype(float)
    tx = (cal["max"] - cal["min"]).values.astype(float)
    T = (15 - cal["min"]).values.astype(float)
    opts = dict(maxiter=4000, xatol=1e-8, fatol=1e-12)
    fits = [minimize(lambda p: -_bgnbd_ll(p, x, tx, T).sum() / len(x), np.log(s), method="Nelder-Mead", options=opts)
            for s in ([1, 1, 1, 1], [0.5, 5, 0.5, 2], [2, 2, 2, 5])]
    best = min(fits, key=lambda m: m.fun)
    params = np.exp(best.x)
    nbd = minimize(lambda p: -_nbd_ll(p, x, T).sum() / len(x), np.log([1, 1]), method="Nelder-Mead", options=opts)
    r, alpha = np.exp(nbd.x)
    dropout = float(params[2] / (params[2] + params[3]))
    collapsed = bool(dropout < 1e-6)
    # forecasts at the test snapshots: the snapshot day is a visit day, so tx = T = days since first seen
    xt, Tt = (te.h_visit_days - 1).values.astype(float), (te.history_days - 1).values.astype(float)
    e_nbd = 7.0 * (r + xt) / (alpha + Tt)
    e_bg = e_nbd if collapsed else _bgnbd_expected(params, xt, Tt, Tt, 7.0)
    # LightGBM with the same two inputs
    two = lambda d: pd.DataFrame({"x": d.h_visit_days.values - 1.0, "T": d.history_days.values - 1.0})
    P2 = dict(LGB_PARAMS, feature_fraction=1.0)
    ds = lambda d, ref=None: lgb.Dataset(two(d), label=d.y_visit_days.values / 7, weight=7 * user_w(d), reference=ref)
    d0 = ds(tr)
    g = lgb.train(P2, d0, 3000, valid_sets=[ds(va, d0)], callbacks=[lgb.early_stopping(50, verbose=False)])
    e_two = g.predict(two(te), num_iteration=g.best_iteration)
    D = read("10_predictions.parquet")
    main = D[(D.target == "R") & (D.set == "test")][["userId", "origin_day", "lightgbm"]].astype({"userId": str})
    e_main = te[["userId", "origin_day"]].astype({"userId": str}).merge(main, on=["userId", "origin_day"], how="left").lightgbm.values
    assert not np.isnan(e_main).any()
    y, k = (te.y_visit_days > 0).values, te.y_visit_days.values.astype(float)
    preds = {"bgnbd": e_bg, "nbd_no_dropout": e_nbd, "lightgbm_frequency_recency_only": e_two, "lightgbm": e_main}
    A = {m: WeightedAUC(y, p) for m, p in preds.items()}
    scores = lambda w: {m: A[m](w) for m in preds}
    point, bt = scores(np.ones(len(te))), boot(scores, te)
    res = {"training_users": int(len(cal)), "bgnbd_params_r_alpha_a_b": [float(f"{v:.4g}") for v in params],
           "nbd_params_r_alpha": [float(f"{r:.4g}"), float(f"{alpha:.4g}")],
           "dropout_probability": float(f"{dropout:.4g}"), "bgnbd_equals_nbd": collapsed, "test_rows": int(len(te)),
           "visit_days_next7": {"observed_mean": round(float(k.mean()), 4), "bgnbd_mean": round(float(np.mean(e_bg)), 4),
                                "bgnbd_rmse": round(float(np.sqrt(np.mean((e_bg - k) ** 2))), 4)},
           "models": {m: {"auc_return": round(point[m], 4), "auc_ci95": ci(bt[m]),
                          "auc_minus_lightgbm_ci95": ci(bt[m] - bt["lightgbm"])} for m in preds}}
    write_json("14_bgnbd.json", res)


# ---------------------------------------------------------------------------------------------------------------------
CAL_MODELS = ["scorecard", "logistic", "lightgbm"]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@step("02.calibration", ["13_calibration.json", "13_calibration_curve.csv", "13_calibration_by_history.csv"], needs=["02.models"])
def calibration():
    """Predicted minus observed rate, calibration intercept and slope with bootstrap intervals; curves by tenth."""
    from thesis.config import TAG
    D = read("10_predictions.parquet")

    def stats(t, w=None):
        k, n = t.k.values, t.n.values
        w = np.ones(len(t)) if w is None else w
        out = {}
        for m in CAL_MODELS:
            p, z = t[m].values, _logit(t[m].values)
            out[f"{m}_pred_minus_obs_pp"] = 100 * ((w * p * n).sum() - (w * k).sum()) / (w * n).sum()
            out[f"{m}_intercept"] = binom_fit(z, k, n, w, slope=False)[0]
            out[f"{m}_slope"] = binom_fit(z, k, n, w)[1]
        return out

    res, curves = {"sample": TAG or "full"}, []
    for tgt in ("R", "C"):
        t = D[(D.target == tgt) & (D.set == "test")].reset_index(drop=True)
        point = stats(t)
        bt = boot(lambda w: stats(t, w), t)
        r = {"test_rows": len(t), "observed_rate": round(float(t.k.sum() / t.n.sum()), 4)}
        for m in CAL_MODELS:
            r[m] = {"predicted_rate": round(float((t[m] * t.n).sum() / t.n.sum()), 4),
                    **{s: round(float(point[f"{m}_{s}"]), 4) for s in ("pred_minus_obs_pp", "intercept", "slope")},
                    **{f"{s}_ci95": ci(bt[f"{m}_{s}"]) for s in ("pred_minus_obs_pp", "intercept", "slope")}}
            dec = pd.qcut(t[m].rank(method="first"), 10, labels=False) + 1
            g = t.assign(pn=t[m] * t.n, decile=dec).groupby("decile")[["pn", "k", "n"]].sum()
            curves.append(pd.DataFrame({"target": tgt, "model": m, "decile": g.index, "predicted": g.pn / g.n,
                                        "observed": g.k / g.n, "trials": g.n}))
        day = t.assign(pn=t.lightgbm * t.n).groupby("origin_day")[["pn", "k", "n"]].sum()
        r["lightgbm_by_test_day_pred_minus_obs_pp"] = {int(d): round(float(100 * (x.pn - x.k) / x.n), 3)
                                                       for d, x in day.iterrows()}
        res[tgt] = r
    write_json("13_calibration.json", res)
    pd.concat(curves).round(5).to_csv(path("13_calibration_curve.csv"), index=False)
    calibration_by_history(D).to_csv(path("13_calibration_by_history.csv"), index=False)


HISTORY_BANDS = ([0, 1, 3, 7, 15, 19, 23], ["1", "2-3", "4-7", "8-15", "16-19", "20-23"])


def calibration_by_history(D):
    """Predicted minus observed return rate (percentage points) by days since first seen, validation and test snapshots.
    Shows where the overestimate of return in the test period sits: in short histories, not in the long ones that
    lie beyond the 15 days of history seen in training."""
    from thesis.config import TAG
    con = connect()
    hist = con.execute(f"SELECT userId, origin_day, history_days FROM '{P}/panel{TAG}.parquet'").df()
    con.close()
    t = D[(D.target == "R") & D.set.isin(["valid", "test"])].merge(hist, on=["userId", "origin_day"])
    t["history_days_band"] = pd.cut(t.history_days, HISTORY_BANDS[0], labels=HISTORY_BANDS[1])
    t = t.assign(p_lgb=t.lightgbm * t.n, p_sc=t.scorecard * t.n, rows=1)
    g = t.groupby(["set", "history_days_band"], observed=True)[["rows", "k", "n", "p_lgb", "p_sc"]].sum().reset_index()
    return pd.DataFrame({"set": g.set, "history_days": g.history_days_band, "snapshots": g.rows,
                         "observed_return_pct": (100 * g.k / g.n).round(2),
                         "lightgbm_pred_minus_obs_pp": (100 * (g.p_lgb - g.k) / g.n).round(2),
                         "scorecard_pred_minus_obs_pp": (100 * (g.p_sc - g.k) / g.n).round(2)})


RECAL_FIRST_DAY, RECAL_LAST_DAY = 2, 8   # the last week whose 7-day outcomes end before the test period (day 16)
RECAL_WINDOWS = (1, 2, 5, 8)             # first days reported, each window ending on RECAL_LAST_DAY


@step("02.recalibration", ["13_recalibration.json"], needs=["02.models"])
def recalibration():
    """Return models recalibrated on the latest fully observed week: one log-odds shift per model, fitted on
    validation users' snapshots from days 2 to 8, whose outcomes end before the test period (models unchanged) and applied to the test period. Other windows are reported, not
    chosen from."""
    _, FEATS, tr, va, te = panel()
    (a, b, c), k, n = target_sets("R", tr, va, te)
    D = read("10_predictions.parquet")
    V = D[(D.target == "R") & (D.set == "valid")].reset_index(drop=True)
    T = D[(D.target == "R") & (D.set == "test")].reset_index(drop=True)
    assert (V.userId.values == b.userId.values).all() and (T.userId.values == c.userId.values).all()
    wv, day = user_w(b), V.origin_day.values
    kv, nv = V.k.values.astype(float), V.n.values.astype(float)
    kc, nc = T.k.values.astype(float), T.n.values.astype(float)
    L0 = ll_rows(kc, nc, np.full(len(T), k(a).sum() / n(a).sum()))      # against the training average, as in 02.models

    def shift(m, first):
        mk = (day >= first) & (day <= RECAL_LAST_DAY)
        return float(binom_fit(_logit(V[m].values[mk]), kv[mk], nv[mk], wv[mk], slope=False)[0])

    res = {"window": f"validation snapshots on days {RECAL_FIRST_DAY}-{RECAL_LAST_DAY}", "test_rows": len(T),
           "observed_return_rate": round(float(kc.sum() / nc.sum()), 4)}
    for m in CAL_MODELS:
        p0 = T[m].values
        s = shift(m, RECAL_FIRST_DAY)
        p1 = 1 / (1 + np.exp(-(_logit(p0) + s)))
        L_0, L_1 = ll_rows(kc, nc, p0), ll_rows(kc, nc, p1)

        def scores(w):
            return {"bias_before": 100 * ((w * p0 * nc).sum() - (w * kc).sum()) / (w * nc).sum(),
                    "bias_after": 100 * ((w * p1 * nc).sum() - (w * kc).sum()) / (w * nc).sum(),
                    "r2_before": 1 - (w @ L_0) / (w @ L0), "r2_after": 1 - (w @ L_1) / (w @ L0)}

        point, bt = scores(np.ones(len(T))), boot(scores, T)
        res[m] = {"logodds_shift": round(s, 4),
                  "pred_minus_obs_pp_before": round(float(point["bias_before"]), 3),
                  "pred_minus_obs_pp_after": round(float(point["bias_after"]), 3),
                  "pred_minus_obs_pp_after_ci95": ci(bt.bias_after, 3),
                  "pseudo_r2_before": round(float(point["r2_before"]), 4),
                  "pseudo_r2_after": round(float(point["r2_after"]), 4),
                  "pseudo_r2_gain_ci95": ci(bt.r2_after - bt.r2_before),
                  "auc_unchanged": True,                                 # one shift keeps the ranking
                  "pred_minus_obs_pp_after_by_first_validation_day": {
                      str(f): round(float(100 * ((1 / (1 + np.exp(-(_logit(p0) + shift(m, f))))) * nc).sum() / nc.sum()
                                          - 100 * kc.sum() / nc.sum()), 3) for f in RECAL_WINDOWS}}
    write_json("13_recalibration.json", res)


# ---------------------------------------------------------------------------------------------------------------------
def family(f):
    """Feature family of section 4: visits and exposure = the activity, exposure and calendar groups of
    thesis.panel.FEATURE_GROUPS; clicks; social."""
    g = feature_group(f)
    return "visits" if g in ("activity", "exposure", "calendar_account") else g


def feature_families(FEATS):
    return {name: [f for f in FEATS if family(f) == name] for name in ("visits", "clicks", "social")}


def feature_groups(FEATS):
    G = feature_families(FEATS)
    return {"visits_exposure": G["visits"],
            "visits_exposure+clicks": G["visits"] + G["clicks"],
            "clicks_social_only": G["clicks"] + G["social"] + ["history_days", "tenure", "origin_weekday"]}


def _robust(target, S10, GROUPS):
    _, FEATS, tr, va, te = panel()
    (a, b, c), k, n = target_sets(target, tr, va, te)
    par = lgb_params(target)

    def fit(feats, A=a, B=b):
        def ds(d, ref=None):
            return lgb.Dataset(d[feats], label=k(d) / n(d), weight=n(d).values * user_w(d), reference=ref)
        d0 = ds(A)
        m = _early_stop_fit(par, d0, ds(B, d0))
        return m.predict(c[feats], num_iteration=m.best_iteration)

    # models of section 2, the baseline, the feature groups and the out-of-time fits
    s10 = c[["userId", "origin_day"]].merge(S10[S10.target == target], on=["userId", "origin_day"], how="left")
    preds = {"all_features_step10": s10.lightgbm.values, "logistic_step10": s10.logistic.values,
             "scorecard_step10": s10.scorecard.values}
    preds["baseline"] = rule_return(a, c) if target == "R" else shrunk_click_rate(a, c)
    for g, fs in GROUPS.items():
        preds[g] = fit(fs)
    a8, b8 = a[a.origin_day <= 8], b[b.origin_day <= 8]
    preds["strict_out_of_time_days1_8"] = fit(FEATS, a8, b8)
    keep = pd.Series(a.userId.unique()).sample(frac=len(a8) / len(a), random_state=SEED)
    preds["same_size_control_days1_15"] = fit(FEATS, a[a.userId.isin(keep)], b)
    assert not np.isnan(preds["all_features_step10"]).any()

    # per-row log-likelihoods, so that bootstrap replicates are weighted sums
    kc, nc = k(c).values.astype(float), n(c).values.astype(float)

    def ll(q):
        return ll_rows(kc, nc, q)

    L = {m: ll(q) for m, q in preds.items()}
    if target == "R":
        h, y = np.ones(len(c), bool), (c.y_visit_days > 0).values
    else:
        h = c.y_impr.values >= 20
        y = (c.y_clicks.values > 0)[h]

    # subgroups: all rows, bands of observed history, new accounts
    def band(d):
        return pd.cut(d.history_days, [0, 1, 3, 7, 14, 30], labels=["1", "2-3", "4-7", "8-14", "15+"]).astype(str).values

    subs = {"all": (np.ones(len(c), bool), np.ones(len(a), bool))}
    for lab in ["1", "2-3", "4-7", "8-14", "15+"]:
        subs[f"history_{lab}"] = (band(c) == lab, band(a) == lab)
    subs["new_accounts"] = ((c.tenure <= 1).values, (a.tenure <= 1).values)
    L0 = {s: ll(np.full(len(c), k(a)[ma].sum() / n(a)[ma].sum())) for s, (mc, ma) in subs.items()}
    A = {(s, m): WeightedAUC(y[mc[h]], q[h][mc[h]]) for s, (mc, ma) in subs.items() for m, q in preds.items()
         if 0 < y[mc[h]].mean() < 1}

    def scores(w):
        out = {}
        for s, (mc, _) in subs.items():
            ws = w * mc
            for m in preds:
                out[(s, m, "pseudo_r2")] = 1 - (ws @ L[m]) / (ws @ L0[s])
                if (s, m) in A:
                    out[(s, m, "auc")] = A[(s, m)](ws[h][mc[h]])
        return out

    point = scores(np.ones(len(c)))
    bt = boot(scores, c)

    res = {"test_rows": len(c), "groups": {g: len(f) for g, f in GROUPS.items()}, "subsets": {}}
    for s, (mc, _) in subs.items():
        e = {"rows": int(mc.sum())}
        for m in preds:
            for met in ("pseudo_r2", "auc"):
                key = (s, m, met)
                if key not in point:
                    continue
                e[f"{m}_{met}"] = round(float(point[key]), 4)
                e[f"{m}_{met}_ci95"] = ci(bt[key])
                if m != "all_features_step10":
                    e[f"{m}_{met}_minus_all_ci95"] = ci(bt[key] - bt[(s, "all_features_step10", met)])
        res["subsets"][s] = e
    return res


def _ceiling(S10):
    """Same-week half-split ceiling for click propensity."""
    _, FEATS, tr, va, te = panel()
    con = connect()
    con.register("tu", te[["userId"]].drop_duplicates())
    daily = con.execute(f"""SELECT userId, CAST(dt AS INT) AS dt, (row_id % 2)::INT AS half, count(*) AS n,
        sum(isClick)::DOUBLE AS k
        FROM read_parquet('{P}/impressions_place/d*.parquet') JOIN tu USING (userId) WHERE dt >= 17 GROUP BY ALL""").df()
    con.close()

    # cumulative counts per [n/k, half, user, day], so any window of days is a difference
    cats = te.userId.astype("category").cat.categories
    grid = np.zeros((2, 2, len(cats), 32))
    ui = pd.Categorical(daily.userId, categories=cats).codes
    np.add.at(grid, (0, daily.half.values, ui, daily.dt.values), daily.n.values)
    np.add.at(grid, (1, daily.half.values, ui, daily.dt.values), daily.k.values)
    cs = np.concatenate([np.zeros((2, 2, len(cats), 1)), grid.cumsum(axis=3)], axis=3)
    u = pd.Categorical(te.userId, categories=cats).codes
    t = te.origin_day.values.astype(int)

    def win(a, h):
        return cs[a, h, u, t + 8] - cs[a, h, u, t + 1]           # days t+1 .. t+7

    nA, kA, nB, kB = win(0, 0), win(1, 0), win(0, 1), win(1, 1)
    ok = (nA > 0) & (nB > 0)

    # shrinkage as in the section-2 baseline
    a = tr[tr.y_impr > 0]
    pooled = a.y_clicks.sum() / a.y_impr.sum()
    m = a[a.h_impr >= 5]
    r = m.h_clicks / m.h_impr
    mu, var = r.mean(), r.var()
    kk = mu * (1 - mu) / var - 1
    s10 = te[["userId", "origin_day"]].merge(S10[S10.target == "C"], on=["userId", "origin_day"], how="left").lightgbm.values
    L0 = ll_rows(kB[ok], nB[ok], np.full(ok.sum(), pooled)).sum()

    def r2(q):
        return round(float(1 - ll_rows(kB[ok], nB[ok], q).sum() / L0), 4)

    return {"origins_with_both_halves": int(ok.sum()),
            "ceiling_same_week_half_split_pseudo_r2": r2((kA[ok] + mu * kk) / (nA[ok] + kk)),
            "step10_model_on_the_same_half_pseudo_r2": r2(s10[ok])}


@step("02.robustness", ["18_rq2_robustness.json"], needs=["02.models"])
def robustness():
    """LightGBM refitted on feature groups and on out-of-time samples, by subgroup, and the click ceiling."""
    _, FEATS, *_ = panel()
    S10 = read("10_predictions.parquet").query("set == 'test'")
    GROUPS = feature_groups(FEATS)
    write_json("18_rq2_robustness.json", {
        "R_return": _robust("R", S10, GROUPS), "C_click_propensity": _robust("C", S10, GROUPS),
        "D_predictability_ceiling_clicks": _ceiling(S10),
        "note": "history bands = days since first seen in the November log (left-censored), not account age"})


# ---------------------------------------------------------------------------------------------------------------------
LABEL_QS = [0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50]
LABEL_METRICS = ("share", "return_rate", "mean_clicks", "zero_clicks_share", "click_rate_if_returned")
LABEL_CMETRICS = ("precision", "recall", "lift")


@step("02.label", ["19_scores.parquet", "19_inactive_label.json", "19_label_composition.json",
                   "19_score_bands.csv", "19_cutoff_choice.json"], needs=["02.models"])
def label():
    """Activity score = P(return) x P(click); the bottom-10% label, its next week, composition, and score bands."""
    _, FEATS, tr, va, te = panel()
    D10 = read("10_predictions.parquet")
    key = te[["userId", "origin_day"]]
    R_ = key.merge(D10.query("target == 'R' and set == 'test'"), on=["userId", "origin_day"], how="left")
    C_ = key.merge(D10.query("target == 'C' and set == 'test_all_origins'"), on=["userId", "origin_day"], how="left")
    assert not R_.lightgbm.isna().any() and not C_.lightgbm.isna().any()
    p_ret, p_clk = R_.lightgbm.values, C_.lightgbm.values
    e = p_ret * p_clk
    e_sc = R_.scorecard.values * C_.scorecard.values

    def within_day_rank(x):
        return pd.Series(x, index=te.index).groupby(te.origin_day).rank(pct=True, method="first").values

    rank, rank_sc = within_day_rank(e), within_day_rank(e_sc)

    # next week for labelled and other user-days, at cutoffs from 5% to 50%; the label as a yes/no prediction
    ret, clicks, impr = (te.y_visit_days > 0).values, te.y_clicks.values, te.y_impr.values
    zero = clicks == 0

    def summary(lab, w):
        out = {}
        for name, m in (("inactive", lab), ("other", ~lab)):
            ww = w * m
            r = ww * ret
            out[name] = {"share": ww.sum() / w.sum(), "return_rate": r.sum() / ww.sum(),
                         "mean_clicks": (ww * clicks).sum() / ww.sum(), "zero_clicks_share": (ww * zero).sum() / ww.sum(),
                         "click_rate_if_returned": (r * clicks).sum() / max((r * impr).sum(), 1e-12)}
        return out

    TRUTHS = {"zero_clicks": zero, "no_return": ~ret}                 # what "inactive next week" can mean, observed
    A = {t: WeightedAUC(y, -e) for t, y in TRUTHS.items()}
    A_sc = {t: WeightedAUC(y, -e_sc) for t, y in TRUTHS.items()}

    def classification(lab, y, w):
        """The label as a yes/no prediction of y: precision, recall and lift (precision / base rate)."""
        tp, pos, act = (w * (lab & y)).sum(), (w * lab).sum(), (w * y).sum()
        return {"precision": tp / pos, "recall": tp / act, "lift": (tp / pos) / (act / w.sum())}

    def scores(w):
        out = {**{f"auc_{t}": A[t](w) for t in TRUTHS}, **{f"auc_scorecard_{t}": A_sc[t](w) for t in TRUTHS}}
        for q in LABEL_QS:
            s = summary(rank <= q, w)
            for g in ("inactive", "other"):
                for k, v in s[g].items():
                    out[(q, g, k)] = v
            for t, y in TRUTHS.items():
                for k, v in classification(rank <= q, y, w).items():
                    out[(q, t, k)] = v
        for g, v in summary(rank_sc <= 0.10, w).items():
            for k, x in v.items():
                out[("scorecard", g, k)] = x
        return out

    point = scores(np.ones(len(te)))
    bt = boot(scores, te)
    METRICS, CMETRICS, QS = LABEL_METRICS, LABEL_CMETRICS, LABEL_QS
    lab, lab_sc = rank <= 0.10, rank_sc <= 0.10
    res = {"test_rows": len(te), "observed_return_rate": round(float(ret.mean()), 4),
           "observed_zero_clicks_share": round(float(zero.mean()), 4),
           "ranking": {t: {"base_rate": round(float(y.mean()), 4), "auc": round(point[f"auc_{t}"], 4),
                           "auc_ci95": ci(bt[f"auc_{t}"]), "auc_scorecard": round(point[f"auc_scorecard_{t}"], 4),
                           "auc_scorecard_ci95": ci(bt[f"auc_scorecard_{t}"])} for t, y in TRUTHS.items()},
           "by_threshold": {f"bottom_{int(q * 100)}pct": {g: {k: round(float(point[(q, g, k)]), 4) for k in METRICS}
                                                            | {f"{k}_ci95": ci(bt[(q, g, k)]) for k in METRICS[1:]}
                                                            for g in ("inactive", "other")} for q in QS},
           "classification_by_threshold": {
               f"bottom_{int(q * 100)}pct": {t: {k: round(float(point[(q, t, k)]), 4) for k in CMETRICS}
                                             | {f"{k}_ci95": ci(bt[(q, t, k)]) for k in CMETRICS} for t in TRUTHS}
               for q in QS},
           "scorecard_label_bottom10": {g: {k: round(float(point[("scorecard", g, k)]), 4) for k in METRICS}
                                        for g in ("inactive", "other")}
                                       | {"share_of_lightgbm_label_also_flagged": round(float((lab & lab_sc).sum() / lab.sum()), 4)},
           "main_threshold": "bottom_10pct"}
    pd.DataFrame({"userId": te.userId.values, "origin_day": te.origin_day.values, "p_return": p_ret, "p_click": p_clk,
                  "expected_click_prob": e, "rank_within_day": rank, "inactive_bottom10": lab,
                  "scorecard_score": e_sc, "scorecard_inactive_bottom10": lab_sc}).to_parquet(
        path("19_scores.parquet"), index=False)
    write_json("19_inactive_label.json", res)

    # what fills each label: the spread of the two predictions on the log scale, and whether a labelled user-day is also
    # in the bottom 10% of its day by the return prediction alone, by the click prediction alone, by both or by neither
    comp = {"share_of_lightgbm_label_also_in_scorecard_label": round(float((lab & lab_sc).sum() / lab.sum()), 4)}
    for name, r_, c_, l_ in (("lightgbm", p_ret, p_clk, lab), ("scorecard", R_.scorecard.values, C_.scorecard.values, lab_sc)):
        low_r, low_c = within_day_rank(r_) <= 0.10, within_day_rank(c_) <= 0.10
        n_l = l_.sum()
        comp[name] = {"sd_log_p_return": round(float(np.log(r_).std()), 3), "sd_log_p_click": round(float(np.log(c_).std()), 3),
                      "labelled_low_return_only": round(float((l_ & low_r & ~low_c).sum() / n_l), 4),
                      "labelled_low_click_only": round(float((l_ & ~low_r & low_c).sum() / n_l), 4),
                      "labelled_low_both": round(float((l_ & low_r & low_c).sum() / n_l), 4),
                      "labelled_low_neither": round(float((l_ & ~low_r & ~low_c).sum() / n_l), 4),
                      "labelled_mean_p_return": round(float(r_[l_].mean()), 4),
                      "labelled_mean_p_click": round(float(c_[l_].mean()), 5),
                      "low_click_mean_p_return": round(float(r_[low_c].mean()), 4),
                      "low_return_mean_p_click": round(float(c_[low_r].mean()), 5)}
    write_json("19_label_composition.json", comp)

    # next-week outcomes by percentile band of the score, and a search for a break
    band = np.minimum((rank * 100).astype(int), 99)
    bands = pd.DataFrame({"band": band, "ret": ret, "clicks": clicks, "zero": zero, "impr": impr, "e": e})
    by_band = bands.groupby("band").agg(user_days=("ret", "size"), return_rate=("ret", "mean"),
                                        mean_clicks=("clicks", "mean"), zero_clicks_share=("zero", "mean"),
                                        mean_score=("e", "mean"))
    by_band.round(5).to_csv(path("19_score_bands.csv"))

    def break_point(yv, hi):
        """Two-segment linear fit of yv on the band over bands 0..hi-1; returns the band with the smallest squared error."""
        xs, ys = by_band.index.values[:hi].astype(float), yv[:hi]
        best = None
        for b in range(5, hi - 5):
            X = np.column_stack([np.ones_like(xs), xs, np.maximum(xs - b, 0)])
            beta = np.linalg.lstsq(X, ys, rcond=None)[0]
            sse = float(((ys - X @ beta) ** 2).sum())
            if best is None or sse < best[1]:
                best = (b, sse)
        return best[0]

    fits = {f"bands_0_{hi - 1}": break_point(by_band.return_rate.values, hi) for hi in (40, 60, 80, 100)}
    write_json("19_cutoff_choice.json", {
        "return_rate_break_point_by_fit_range": {k: int(v) for k, v in fits.items()},
        "last_band_before_zero_click_share_first_drops_below_95pct":
            int(by_band.index[(by_band.zero_clicks_share < 0.95).values.argmax()] - 1),
        "bands_0_to_24": {"return_rate_range": [round(float(by_band.return_rate[:25].min()), 4),
                                                round(float(by_band.return_rate[:25].max()), 4)],
                          "mean_clicks_range": [round(float(by_band.mean_clicks[:25].min()), 4),
                                                round(float(by_band.mean_clicks[:25].max()), 4)],
                          "zero_clicks_share_range": [round(float(by_band.zero_clicks_share[:25].min()), 4),
                                                      round(float(by_band.zero_clicks_share[:25].max()), 4)]}})


# ---------------------------------------------------------------------------------------------------------------------
HMAX, HMAX_STRICT = 15, 7
DAILY = ["impr", "clicks", "likes", "shares", "viewcomments", "viewtime", "sessions", "expected_clicks"]
HIST_ONLY = ["comments", "homepage", "video_seen", "pos_sum", "pop_sum"]
SOCIAL = {"likes", "shares", "comments", "viewcomments", "homepage"}
CLICKS = {"clicks", "ctr", "click_vs_expected", "viewtime"}
HMODELS = ("lightgbm", "scorecard", "logistic", "baseline")


def _stem(f):
    return f.split("_", 1)[1] if f.split("_")[0] in ("w1", "w3", "w7", "w14", "h") else f


@step("02.history", ["26_history_curve.json", "26_history_curve.csv", "26_early_actions.json"], needs=["02.tuning"])
def history():
    """The same new users scored after 1 to 15 days (one model per day), and early feature groups at 1, 3, 7 days."""
    df, FEATS, tr, va, te = panel()
    # cohorts of new users and their daily activity in the first fifteen days
    u = df.groupby("userId").agg(first_day=("origin_day", "min"), bucket=("bucket", "first"), tenure=("tenure", "first"))
    u["set"] = np.select([u.bucket < 70, u.bucket < 85], ["train", "valid"], "test")
    u.loc[(u.set == "test") & u.first_day.between(16, 17), "set"] = "test_strict"
    cohort = u[u.first_day.between(2, 9) | (u.set == "test_strict")]
    D = df.loc[df.userId.isin(cohort.index), ["userId", "origin_day"] + [f"w1_{c}" for c in DAILY] + [f"h_{c}" for c in HIST_ONLY]]
    D = D.merge(cohort.first_day, left_on="userId", right_index=True)
    D = D[D.origin_day <= D.first_day + HMAX - 1].sort_values(["userId", "origin_day"])
    for c in DAILY:                                      # one day's totals = the 1-day window at a visit day
        D[c] = D.pop(f"w1_{c}")
    for c in HIST_ONLY:                                  # one day's totals = the change in the running history total
        D[c] = D.groupby("userId")[f"h_{c}"].diff().fillna(D[f"h_{c}"])
        D.pop(f"h_{c}")
    assert (D.origin_day <= 23).all()

    # outcomes: cards shown and clicks per user and day, from the impressions (days up to 30)
    con = connect()
    con.register("cu", cohort.reset_index()[["userId"]])
    Y = con.execute(f"""SELECT userId, CAST(dt AS INT) AS dt, count(*) AS impr, sum(isClick)::DOUBLE AS clicks
        FROM read_parquet('{P}/impressions_place/d*.parquet') JOIN cu USING (userId) GROUP BY ALL""").df()
    con.close()
    Y = Y.merge(cohort.first_day, left_on="userId", right_index=True)

    # features at horizon d (end of the user's d-th day since first seen) and outcomes over the next 7 days
    def horizon(d, users):
        t = users.first_day + d - 1
        x = D[D.userId.isin(users.index) & (D.origin_day <= D.first_day + d - 1)]
        out = pd.DataFrame(index=users.index)
        for w in (1, 3, 7, 14):
            xw = x[x.origin_day > x.first_day + d - 1 - w]
            g = xw.groupby("userId")
            out = out.join(g[DAILY].sum().add_prefix(f"w{w}_")).join(g.size().rename(f"w{w}_visit_days"))
        g = x.groupby("userId")
        out = out.join(g[DAILY + HIST_ONLY].sum().add_prefix("h_")).join(g.size().rename("h_visit_days"))
        out = out.join((t - g.origin_day.max()).rename("days_since_last_visit"))
        fill = [c for c in out.columns if c != "days_since_last_visit"]
        out[fill] = out[fill].fillna(0)
        out["history_days"] = d
        out["tenure"] = users.tenure
        out["origin_weekday"] = ((pd.Timestamp("2019-10-31") + pd.to_timedelta(t, "D")).dt.dayofweek + 1) % 7
        for w in ("w7", "h"):                                       # the ratios of thesis.panel.load
            out[f"{w}_ctr"] = out[f"{w}_clicks"] / out[f"{w}_impr"]
            out[f"{w}_click_vs_expected"] = (out[f"{w}_clicks"] + 1) / (out[f"{w}_expected_clicks"] + 1)
        out["h_mean_pos"] = out.pop("h_pos_sum") / out.h_impr
        out["h_video_share"] = out.pop("h_video_seen") / out.h_impr
        out["h_mean_card_pop"] = out.pop("h_pop_sum") / out.h_impr
        out["h_shown_card_ctr"] = out.h_expected_clicks / out.h_impr
        yy = Y[Y.userId.isin(users.index) & (Y.dt > Y.first_day + d - 1) & (Y.dt <= Y.first_day + d + 6)].groupby("userId")
        out = out.join(yy.size().rename("y_visit_days")).join(yy[["impr", "clicks"]].sum().add_prefix("y_"))
        out[["y_visit_days", "y_impr", "y_clicks"]] = out[["y_visit_days", "y_impr", "y_clicks"]].fillna(0)
        assert (t + 7 <= 30).all()                                  # every outcome week is fully observed
        return out.join(users.set)

    # one LightGBM model, scorecard, logistic regression and baseline per horizon and outcome
    TEST = {"user_split": cohort.index[cohort.set == "test"], "strict_time_split": cohort.index[cohort.set == "test_strict"]}
    PRED, TREES, HT = {}, {}, {}
    EARLY_AB = {}                                   # training and validation users at 1, 3 and 7 days (section 7.2)
    for d in range(1, HMAX + 1):
        users = cohort[(cohort.set != "test_strict") | (d <= HMAX_STRICT)]
        h = horizon(d, users)
        a, b = h[h.set == "train"], h[h.set == "valid"]
        if d <= HMAX_STRICT:
            assert (a.index.map(cohort.first_day) + d - 1).max() <= 15        # training predictions on days 1 to 15
        if d in (1, 3, 7):
            EARLY_AB[d] = (a, b)
        for design, idx in TEST.items():
            if design == "user_split" or d <= HMAX_STRICT:
                HT[(design, d)] = h.loc[idx]
        if d == 1:
            HFEATS = [c for c in h.columns if c not in ("set", "y_visit_days", "y_impr", "y_clicks")]
            assert h.h_impr.min() > 0                                         # everyone visits on the first day
            # the step-by-step recency count stands in for the panel's gap since the previous visit day
            sc_f = {t: [("days_since_last_visit" if f == "gap_since_prev_visit" else f) for f in SCORECARD[t]] for t in ("R", "C")}
            assert all(f in HFEATS for t in sc_f for f in sc_f[t])
            # columns that the panel does not have are log-transformed by the same rule as thesis.panel.load
            import thesis.panel as _panel
            _panel.LOG_COLS.update(c for c in set(HFEATS) - set(FEATS) if a[c].astype(float).max() > 1.5)
        for target in ("R", "C"):
            if target == "R":
                k_, n_ = (lambda z: (z.y_visit_days > 0).astype(float)), (lambda z: pd.Series(1.0, index=z.index))
                A_, B_ = a, b
            else:
                k_, n_ = (lambda z: z.y_clicks), (lambda z: z.y_impr)
                A_, B_ = a[a.y_impr > 0], b[b.y_impr > 0]
            d0 = lgb.Dataset(A_[HFEATS], label=k_(A_) / n_(A_), weight=n_(A_).values)
            d1 = lgb.Dataset(B_[HFEATS], label=k_(B_) / n_(B_), weight=n_(B_).values, reference=d0)
            m = _early_stop_fit(lgb_params(target), d0, d1)
            TREES[(target, d)] = m.best_iteration
            ones = np.ones(len(A_))
            sc = logit_fit(A_, sc_f[target], k_, n_, ones, logistic_C(target))
            lr = logit_fit(A_, HFEATS, k_, n_, ones, logistic_C(target))
            pooled = k_(A_).sum() / n_(A_).sum()
            for design in TEST:
                if (design, d) not in HT:
                    continue
                c = HT[(design, d)]
                base = rule_return(A_, c, "h_visit_days") if target == "R" else shrunk_click_rate(A_, c)
                PRED[(design, target, d)] = {"lightgbm": m.predict(c[HFEATS], num_iteration=m.best_iteration),
                                             "scorecard": sc.predict_proba(lin(c, sc_f[target]))[:, 1],
                                             "logistic": lr.predict_proba(lin(c, HFEATS))[:, 1],
                                             "baseline": base, "constant": np.full(len(c), pooled)}
        del h, a, b

    # accuracy at each horizon with bootstrap intervals, and the gain from each extra day
    def curve(design):
        ds = sorted(d for (g, d) in HT if g == design)
        LL, AUCS = {}, {}
        for d in ds:
            c = HT[(design, d)]
            kR, nR = (c.y_visit_days.values > 0).astype(float), np.ones(len(c))
            hC = c.y_impr.values > 0
            kC, nC = c.y_clicks.values.astype(float), c.y_impr.values.astype(float)
            a20 = c.y_impr.values >= 20
            for m in HMODELS + ("constant",):
                pR, pC = PRED[(design, "R", d)][m], PRED[(design, "C", d)][m]
                LL[("R", d, m)] = ll_rows(kR, nR, pR)
                LL[("C", d, m)] = np.where(hC, ll_rows(kC, np.maximum(nC, 1), pC), 0.0)
                if m != "constant":
                    AUCS[("R", d, m)] = (WeightedAUC(kR > 0, pR), np.ones(len(c), bool))
                    AUCS[("C", d, m)] = (WeightedAUC(kC[a20] > 0, pC[a20]), a20)

        def scores(w):
            out = {}
            for (tgt, d, m), l in LL.items():
                if m != "constant":
                    out[(tgt, d, m, "pseudo_r2")] = 1 - (w @ l) / (w @ LL[(tgt, d, "constant")])
            for (tgt, d, m), (A_, mk) in AUCS.items():
                out[(tgt, d, m, "auc")] = A_(w[mk])
            return out

        users = HT[(design, ds[0])].index.to_frame(index=False)              # same users, same order, at every horizon
        assert all((HT[(design, d)].index == users.userId).all() for d in ds)
        point = scores(np.ones(len(users)))
        bt = boot(scores, users)
        res = {"test_users": len(users)}
        for tgt in ("R", "C"):
            for d in ds:
                c = HT[(design, d)]
                e = {"trees": TREES[(tgt, d)], "return_next7": round(float((c.y_visit_days > 0).mean()), 4),
                     "users_20plus_cards_next7": int((c.y_impr >= 20).sum())}
                for m in HMODELS:
                    for met in ("pseudo_r2", "auc"):
                        key = (tgt, d, m, met)
                        e[f"{m}_{met}"] = round(float(point[key]), 4)
                        e[f"{m}_{met}_ci95"] = ci(bt[key])
                        if d > 1:
                            e[f"{m}_{met}_gain_from_previous_day_ci95"] = ci(bt[key] - bt[(tgt, d - 1, m, met)])
                res.setdefault({"R": "R_return", "C": "C_click_propensity"}[tgt], {})[d] = e
        return res

    CURVES = {design: curve(design) for design in TEST}
    res = {"cohort_first_seen_days": {"user_split": [2, 9], "strict_time_split": [16, 17]},
           "training_users": int((cohort.set == "train").sum()), "validation_users": int((cohort.set == "valid").sum()),
           "auc_meaning": {"R": "return within 7 days", "C": "any click, users with 20+ cards shown in the 7 days"},
           **CURVES}
    write_json("26_history_curve.json", res)
    pd.DataFrame([{"design": g, "target": t, "horizon_days": d, **{k: v for k, v in e.items() if not isinstance(v, list)}}
                  for g in CURVES for t in ("R_return", "C_click_propensity")
                  for d, e in CURVES[g][t].items()]).to_csv(path("26_history_curve.csv"), index=False)

    # early actions among users in their first days: return model refitted on feature groups at 1, 3 and 7 days
    HG = {"visits_exposure": [f for f in HFEATS if _stem(f) not in SOCIAL | CLICKS]}
    HG["visits_exposure_clicks"] = HG["visits_exposure"] + [f for f in HFEATS if _stem(f) in CLICKS]
    early = {}
    for d in (1, 3, 7):
        (a, b), c = EARLY_AB[d], HT[("user_split", d)]
        y = (c.y_visit_days > 0).values
        P_ = {"all": PRED[("user_split", "R", d)]["lightgbm"]}
        for g, fs in HG.items():
            d0 = lgb.Dataset(a[fs], label=(a.y_visit_days > 0).astype(float))
            d1 = lgb.Dataset(b[fs], label=(b.y_visit_days > 0).astype(float), reference=d0)
            m = _early_stop_fit(lgb_params("R"), d0, d1)
            P_[g] = m.predict(c[fs], num_iteration=m.best_iteration)
        AU = {g: WeightedAUC(y, p) for g, p in P_.items()}
        L = {g: ll_rows(y.astype(float), np.ones(len(y)), p) for g, p in P_.items()}
        L0 = ll_rows(y.astype(float), np.ones(len(y)), np.full(len(y), y.mean()))
        users = c.index.to_frame(index=False)
        bt = boot(lambda w: {**{(g, "auc"): AU[g](w) for g in P_}, **{(g, "r2"): 1 - (w @ L[g]) / (w @ L0) for g in P_}},
                  users)
        early[d] = {g: {"auc": round(float(AU[g](np.ones(len(y)))), 4),
                        "pseudo_r2": round(float(1 - L[g].sum() / L0.sum()), 4),
                        **({} if g == "all" else {
                            "auc_minus_all_ci95": ci(bt[(g, "auc")] - bt[("all", "auc")]),
                            "pseudo_r2_minus_all_ci95": ci(bt[(g, "r2")] - bt[("all", "r2")])})}
                    for g in P_}
    write_json("26_early_actions.json", early)
