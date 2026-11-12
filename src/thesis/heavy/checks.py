"""Slow steps of appendix A1 (model checks). Each function writes its outputs; the notebook reads them back through
thesis.cache.ensure. The code is the notebook's, moved here unchanged except that bootstrap replicates run in parallel
threads (same draws, same results) and the training curves score their checkpoints in one pass."""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from thesis.cache import step, path, read, write_json
from thesis.config import SEED, P, connect
from thesis.panel import USER_WEIGHT, past_user_w, target_sets, lin, user_w, lgb_params, logistic_C
from thesis.stats import boot, WeightedAUC, ll_rows
from thesis.heavy.common import panel, drop_full_panel, ci

RF_TREES, RF_SAMPLE = 200, 0.2
RF_GRID = [(1 / 3, 30), (2 / 3, 100)]   # (share of features, min leaf size): the choices of the earlier 8-setting search
# XGBoost search: start on depth (4, 8, 12) x min child weight (10, 50, 250); whenever the best setting lies on the
# edge of what has been searched, the next value along that ladder is added, until the best setting is interior or
# the ladder ends
XGB_DEPTHS = (2, 4, 8, 12, 16, 20)
XGB_MCWS = (2, 10, 50, 250, 1000, 5000, 25000)
XGB_START = ((1, 3), (1, 3))           # index ranges [lo, hi] of the starting grid in each ladder


def _xgb(depth, mcw, callbacks=None):
    return xgb.XGBRegressor(objective="reg:logistic", n_estimators=3000, learning_rate=0.05, max_depth=depth,
                            min_child_weight=mcw, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                            early_stopping_rounds=50, eval_metric="logloss", random_state=SEED, n_jobs=-1,
                            callbacks=callbacks)


def _compare(TARGET):
    _, FEATS, tr, va, te = panel()
    (a, b, c), k, n = target_sets(TARGET, tr, va, te)
    ya, yb = (k(a) / n(a)).values, (k(b) / n(b)).values
    ua, ub = user_w(a), user_w(b)
    wa, wb = n(a).values.astype(float) * ua, n(b).values.astype(float) * ub
    preds_v, preds_t, secs = {}, {}, {}
    Xa, Xb, Xc = a[FEATS], b[FEATS], c[FEATS]

    def stacked2(d):
        """Each origin as two weighted rows (successes, failures), for learners that need 0/1 labels."""
        u = user_w(d)
        return np.r_[np.ones(len(d)), np.zeros(len(d))], np.r_[k(d).values * u, (n(d) - k(d)).values * u]

    # logistic regression
    t0 = time.time()
    y2, w2 = stacked2(a)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=logistic_C(TARGET), max_iter=2000))
    lr.fit(pd.concat([lin(a, FEATS)] * 2), y2, logisticregression__sample_weight=w2)
    preds_v["logistic"] = lr.predict_proba(lin(b, FEATS))[:, 1]
    preds_t["logistic"] = lr.predict_proba(lin(c, FEATS))[:, 1]
    secs["logistic"] = time.time() - t0

    # LightGBM (settings chosen in notebook 02)
    t0 = time.time()
    dtr = lgb.Dataset(Xa, label=ya, weight=wa)
    dva = lgb.Dataset(Xb, label=yb, weight=wb, reference=dtr)
    g = lgb.train(lgb_params(TARGET), dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    preds_v["lightgbm"] = g.predict(Xb, num_iteration=g.best_iteration)
    preds_t["lightgbm"] = g.predict(Xc, num_iteration=g.best_iteration)
    secs["lightgbm"] = time.time() - t0

    def vloss(pv):
        return float(-(ub * ll_rows(k(b).values, n(b).values, pv)).sum() / (ub * n(b).values).sum())

    # random forest: share of features and minimum leaf size chosen on validation users (user-weighted log-loss)
    t0 = time.time()
    rf_val, best = {}, None
    for mf, leaf in RF_GRID:
        rf = RandomForestRegressor(n_estimators=RF_TREES, max_features=mf, min_samples_leaf=leaf,
                                   max_samples=RF_SAMPLE, n_jobs=-1, random_state=SEED)
        rf.fit(Xa, ya, sample_weight=wa)
        pv = rf.predict(Xb)
        rf_val[(mf, leaf)] = vloss(pv)
        if best is None or rf_val[(mf, leaf)] < rf_val[best[0]]:
            best = ((mf, leaf), rf, pv)
        else:
            del rf
    chosen, rf, preds_v["random_forest"] = best
    preds_t["random_forest"] = rf.predict(Xc)
    secs["random_forest"] = time.time() - t0
    del rf, best

    # XGBoost: maximum depth and minimum child weight chosen on validation users (same log-loss), early stopping
    t0 = time.time()
    xgb_val, xbest = {}, None
    (dlo, dhi), (mlo, mhi) = XGB_START
    while True:
        for depth in XGB_DEPTHS[dlo:dhi + 1]:
            for mcw in XGB_MCWS[mlo:mhi + 1]:
                if (depth, mcw) in xgb_val:
                    continue
                xm = _xgb(depth, mcw)
                xm.fit(Xa, ya, sample_weight=wa, eval_set=[(Xb, yb)], sample_weight_eval_set=[wb], verbose=False)
                pv = xm.predict(Xb)
                xgb_val[(depth, mcw)] = (vloss(pv), int(xm.best_iteration) + 1)
                if xbest is None or xgb_val[(depth, mcw)][0] < xgb_val[xbest[0]][0]:
                    xbest = ((depth, mcw), xm, pv)
                else:
                    del xm
        di, mi = XGB_DEPTHS.index(xbest[0][0]), XGB_MCWS.index(xbest[0][1])
        new = (dlo - (di == dlo and dlo > 0), dhi + (di == dhi and dhi < len(XGB_DEPTHS) - 1),
               mlo - (mi == mlo and mlo > 0), mhi + (mi == mhi and mhi < len(XGB_MCWS) - 1))
        if new == (dlo, dhi, mlo, mhi):
            break
        dlo, dhi, mlo, mhi = new
    xchosen, xm, preds_v["xgboost"] = xbest
    xgb_edge = [p for p, v, ladder, lo, hi in (("max_depth", xchosen[0], XGB_DEPTHS, dlo, dhi),
                                                ("min_child_weight", xchosen[1], XGB_MCWS, mlo, mhi))
                if v in (ladder[lo], ladder[hi])]                   # only possible at the end of a ladder
    preds_t["xgboost"] = xm.predict(Xc)
    secs["xgboost"] = time.time() - t0
    del xm, xbest

    # pseudo-R2 and AUC; for clicks, AUC of any click among origins with 20+ cards shown next week
    pooled = k(a).sum() / n(a).sum()

    def point_scores(d, preds):
        """Pseudo-R² against the training mean, and AUC, with every origin counted once (as on the test set)."""
        kd, nd = k(d).values.astype(float), n(d).values.astype(float)
        l0 = ll_rows(kd, nd, np.full(len(d), pooled)).sum()
        if TARGET == "R":
            hd, yd = np.ones(len(d), bool), d.y_visit_days.values > 0
        else:
            hd = d.y_impr.values >= 20
            yd = d.y_clicks.values[hd] > 0
        return {m: {"pseudo_r2": round(float(1 - ll_rows(kd, nd, q).sum() / l0), 4),
                    "auc": round(float(roc_auc_score(yd, q[hd])), 4)} for m, q in preds.items()}

    validation = point_scores(b, preds_v)          # the model choice rests on validation users
    kc, nc = k(c).values.astype(float), n(c).values.astype(float)
    L0 = ll_rows(kc, nc, np.full(len(c), pooled))
    L = {m: ll_rows(kc, nc, q) for m, q in preds_t.items()}
    if TARGET == "R":
        h, yh = np.ones(len(c), bool), c.y_visit_days.values > 0
    else:
        h = c.y_impr.values >= 20
        yh = c.y_clicks.values[h] > 0
    A = {m: WeightedAUC(yh, q[h]) for m, q in preds_t.items()}

    def scores(w):
        """w = number of times each test row's user is drawn."""
        out = {}
        for m in preds_t:
            out[f"{m}_r2"] = 1 - (w @ L[m]) / (w @ L0)
            out[f"{m}_auc"] = A[m](w[h])
        return out

    point = scores(np.ones(len(c)))
    bt = boot(scores, c)
    users = c.userId.unique()
    res = {"target": TARGET, "equal_user_weight": USER_WEIGHT, "test_rows": len(c), "test_users": len(users),
           "random_forest": {"trees": RF_TREES, "subsample": RF_SAMPLE,
                             "chosen": {"max_features": round(chosen[0], 4), "min_samples_leaf": chosen[1]},
                             "validation_logloss": {f"max_features={mf:.3f}, min_samples_leaf={l}": round(v, 6)
                                                    for (mf, l), v in rf_val.items()}},
           "xgboost": {"chosen": {"max_depth": xchosen[0], "min_child_weight": xchosen[1],
                                  "trees": xgb_val[xchosen][1]},
                       "settings_searched": len(xgb_val), "at_end_of_ladder": xgb_edge,
                       "validation_logloss": {f"max_depth={d}, min_child_weight={m}": round(v[0], 6)
                                              for (d, m), v in xgb_val.items()}},
           "auc_meaning": "return in next 7 days" if TARGET == "R" else "any click, origins with 20+ impressions ahead",
           "event_base_rate": round(float(yh.mean()), 4),
           "validation": validation,
           "seconds": {m: round(s, 1) for m, s in secs.items()},
           "models": {m: {"pseudo_r2": round(point[f"{m}_r2"], 4), "auc": round(point[f"{m}_auc"], 4),
                          **{f"{s}_minus_lightgbm_ci95": ci(bt[f"{m}_{s}"] - bt[f"lightgbm_{s}"]) for s in ("r2", "auc")}}
                      for m in preds_t}}
    write_json(f"11_compare_{TARGET}.json", res)


@step("A1.compare", ["11_compare_R.json", "11_compare_C.json"], needs=["02.tuning"])
def compare():
    """Logistic regression, LightGBM, random forest and XGBoost on the same data; differences from LightGBM."""
    for t in ("R", "C"):
        _compare(t)


# ---------------------------------------------------------------------------------------------------------------------
class _Clock(xgb.callback.TrainingCallback):
    def __init__(self, store):
        self.store = store

    def after_iteration(self, model, epoch, evals_log):
        self.store.append(time.perf_counter())
        return False


def _training_curve(TARGET):
    _, FEATS, tr, va, te = panel()
    RES = read(f"11_compare_{TARGET}.json")
    (a, b, c), k, n = target_sets(TARGET, tr, va, te)
    ya, yb = (k(a) / n(a)).values, (k(b) / n(b)).values
    wa, wb = n(a).values.astype(float) * user_w(a), n(b).values.astype(float) * user_w(b)
    if TARGET == "R":
        h, yh = np.ones(len(c), bool), c.y_visit_days.values > 0
    else:
        h = c.y_impr.values >= 20
        yh = c.y_clicks.values[h] > 0
    Xa, Xb, Xc = a[FEATS], b[FEATS], c[FEATS]

    def auc(p):
        return round(float(roc_auc_score(yh, p[h])), 5)

    rows = []

    def checkpoints(total):
        pts = sorted(set(np.unique(np.geomspace(1, total, 40).astype(int))) | {total})
        return [p for p in pts if p >= 1]

    # logistic regression: one point
    t0 = time.perf_counter()
    y2 = np.r_[np.ones(len(a)), np.zeros(len(a))]
    w2 = np.r_[k(a).values * user_w(a), (n(a) - k(a)).values * user_w(a)]
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=logistic_C(TARGET), max_iter=2000))
    lr.fit(pd.concat([lin(a, FEATS)] * 2), y2, logisticregression__sample_weight=w2)
    rows.append(("logistic", 1, time.perf_counter() - t0, auc(lr.predict_proba(lin(c, FEATS))[:, 1])))

    # LightGBM: chosen tree size, trees up to the early-stopping point
    stamps = []
    t0 = time.perf_counter()
    dtr = lgb.Dataset(Xa, label=ya, weight=wa)
    dva = lgb.Dataset(Xb, label=yb, weight=wb, reference=dtr)
    g = lgb.train(lgb_params(TARGET), dtr, 3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False), lambda env: stamps.append(time.perf_counter())])
    # scores at every checkpoint from one pass: the raw score of the first `it` trees is a running sum over trees
    pts = checkpoints(g.best_iteration)
    raw = np.zeros(len(c))
    prev = 0
    for it in pts:
        raw += g.predict(Xc, start_iteration=prev, num_iteration=it - prev, raw_score=True)
        prev = it
        rows.append(("lightgbm", it, stamps[it - 1] - t0, auc(raw)))      # AUC depends only on the ranking

    # XGBoost: chosen depth and minimum child weight, trees up to the early-stopping point
    xc = RES["xgboost"]["chosen"]
    stamps = []
    t0 = time.perf_counter()
    xm = _xgb(xc["max_depth"], xc["min_child_weight"], callbacks=[_Clock(stamps)])
    xm.fit(Xa, ya, sample_weight=wa, eval_set=[(Xb, yb)], sample_weight_eval_set=[wb], verbose=False)
    for it in checkpoints(int(xm.best_iteration) + 1):
        rows.append(("xgboost", it, stamps[it - 1] - t0, auc(xm.predict(Xc, iteration_range=(0, it)))))

    # random forest: chosen setting, grown 10 trees at a time
    rc = RES["random_forest"]["chosen"]
    rf = RandomForestRegressor(n_estimators=0, max_features=rc["max_features"], min_samples_leaf=rc["min_samples_leaf"],
                               max_samples=RF_SAMPLE, n_jobs=-1, random_state=SEED, warm_start=True)
    spent = 0.0
    for nt in range(10, RF_TREES + 1, 10):
        rf.set_params(n_estimators=nt)
        t0 = time.perf_counter()
        rf.fit(Xa, ya, sample_weight=wa)
        spent += time.perf_counter() - t0
        rows.append(("random_forest", nt, spent, auc(rf.predict(Xc))))
    return pd.DataFrame(rows, columns=["model", "trees", "seconds", "test_auc"]).assign(target=TARGET)


@step("A1.training_curve", ["11_training_curve.csv"], needs=["A1.compare"])
def training_curve():
    """Test AUC against training time for each learner, refitted once with the settings chosen in A1.compare."""
    CURVE = pd.concat([_training_curve(t) for t in ("R", "C")], ignore_index=True)
    CURVE.round({"seconds": 2}).to_csv(path("11_training_curve.csv"), index=False)


# ---------------------------------------------------------------------------------------------------------------------
@step("A1.weighting", ["12_auc_experiments.json"], needs=["02.tuning"])
def weighting():
    """LightGBM for return with and without equal user weights."""
    _, FEATS, tr, va, te = panel()

    def fit(user_weight):
        def ds(x, ref=None):
            w = np.ones(len(x))
            if user_weight:
                w = past_user_w(x)
            return lgb.Dataset(x[FEATS], label=(x.y_visit_days > 0).astype(float), weight=w, reference=ref)

        dtr = ds(tr)
        m = lgb.train(lgb_params("R"), dtr, 5000, valid_sets=[ds(va, dtr)], callbacks=[lgb.early_stopping(100, verbose=False)])
        return m.predict(te[FEATS], num_iteration=m.best_iteration)

    y = (te.y_visit_days > 0).values
    preds = {"no_user_weight": fit(False), "equal_user_weight": fit(True)}
    A = {m: WeightedAUC(y, p) for m, p in preds.items()}
    bt = boot(lambda w: {m: A[m](w) for m in preds}, te)
    users = te.userId.unique()
    write_json("12_auc_experiments.json", {
        "test_rows": len(te), "test_users": len(users),
        "auc": {m: round(roc_auc_score(y, p), 4) for m, p in preds.items()},
        "minus_no_user_weight_ci95": {m: ci(bt[m] - bt.no_user_weight) for m in preds}})


# ---------------------------------------------------------------------------------------------------------------------
def _extra_check(target, XF, XR, label):
    """Full LightGBM with and without the extra features XR (columns of XF, indexed by user and origin day), overall and
    by days since first seen; 500 bootstrap draws over users."""
    _, FEATS, tr, va, te = panel()

    def add(d):
        key = pd.MultiIndex.from_arrays([d.userId.values, d.origin_day.values.astype("int64")])
        return pd.concat([d[FEATS].reset_index(drop=True), XF.reindex(key)[XR].reset_index(drop=True)], axis=1)
    (a, b, c), k, n = target_sets(target, tr, va, te)
    Xa, Xb, Xc = add(a), add(b), add(c)
    kc, nc = k(c).values.astype(float), n(c).values.astype(float)
    preds, gain = {}, None
    for name, feats in (("all_59", FEATS), (label, FEATS + XR)):
        d0 = lgb.Dataset(Xa[feats], label=k(a) / n(a), weight=n(a).values * user_w(a))
        d1 = lgb.Dataset(Xb[feats], label=k(b) / n(b), weight=n(b).values * user_w(b), reference=d0)
        m = lgb.train(lgb_params(target), d0, 5000, valid_sets=[d1], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds[name] = m.predict(Xc[feats], num_iteration=m.best_iteration)
        if name == label:
            gain = pd.Series(m.feature_importance("gain"), index=feats)
    h = np.ones(len(c), bool) if target == "R" else nc >= 20
    ev = kc > 0
    L0 = ll_rows(kc, nc, np.full(len(c), k(a).sum() / n(a).sum()))
    L = {m: ll_rows(kc, nc, p) for m, p in preds.items()}
    A = {m: WeightedAUC(ev[h], p[h]) for m, p in preds.items()}
    band = pd.cut(c.history_days, [0, 1, 7, 14, 30], labels=["1", "2-7", "8-14", "15+"]).astype(str).values
    subs = {"all": np.ones(len(c), bool), **{f"history_{x}": band == x for x in ("1", "2-7", "8-14", "15+")}}

    def scores(w):
        return {(s, m, met): (1 - (w * mk) @ L[m] / ((w * mk) @ L0)) if met == "pseudo_r2" else A[m]((w * mk)[h])
                for s, mk in subs.items() for m in preds for met in ("pseudo_r2", "auc")}
    pt = scores(np.ones(len(c)))
    bt = boot(scores, c)
    out = {s: {"rows": int(mk.sum()),
               **{f"{met}_{m}": round(float(pt[(s, m, met)]), 4) for m in preds for met in ("pseudo_r2", "auc")},
               **{f"{met}_gain_ci95": ci(bt[(s, label, met)] - bt[(s, "all_59", met)]) for met in ("pseudo_r2", "auc")}}
           for s, mk in subs.items()}
    out["creator_share_of_gain" if label == "plus_creator" else "extra_share_of_gain"] = \
        {f: round(float(gain[f] / gain.sum()), 4) for f in XR}
    return out


@step("A1.creator", ["12_creator_features.json"], needs=["02.tuning"])
def creator():
    """Creator features per user-day (up to the origin day) and whether they add anything to the full LightGBM."""
    drop_full_panel()
    K_ = 100.0
    con = connect(keep_order=False)
    g_ = con.execute(f"SELECT sum(CAST(userClickCount AS BIGINT))::DOUBLE / sum(CAST(userImprssionCount AS BIGINT)) FROM '{P}/mlog_stats.parquet'").fetchone()[0]
    con.execute(f"CREATE TABLE card AS SELECT mlogId, creatorId FROM '{P}/mlog_demographics.parquet'")
    con.execute(f"""CREATE TABLE crp AS SELECT creatorId, dt_apply,
        (sum(clicks) OVER w + {K_} * {g_}) / (sum(impr) OVER w + {K_}) AS cr_ctr
      FROM (SELECT c.creatorId, CAST(s.dt AS INT) + 1 AS dt_apply, sum(CAST(s.userImprssionCount AS BIGINT)) AS impr,
                   sum(CAST(s.userClickCount AS BIGINT)) AS clicks
            FROM '{P}/mlog_stats.parquet' s JOIN card c USING (mlogId) GROUP BY 1, 2)
      WINDOW w AS (PARTITION BY creatorId ORDER BY dt_apply)""")
    con.execute(f"""CREATE TABLE crposts AS SELECT creatorId, CAST(dt AS INT) + 1 AS dt_apply,
        ln(1 + sum(CAST(PushlishMlogCnt AS BIGINT)) OVER (PARTITION BY creatorId ORDER BY CAST(dt AS INT))) AS log_posts
      FROM '{P}/creator_stats.parquet'""")
    con.execute(f"""CREATE TABLE imp AS SELECT i.userId, CAST(i.dt AS INT) AS dt, c.creatorId, i.isClick::INT AS y
      FROM read_parquet('{P}/impressions_place/d*.parquet') i LEFT JOIN card c USING (mlogId) WHERE CAST(i.dt AS INT) <= 23""")
    con.execute("CREATE TABLE first_click AS SELECT userId, creatorId, min(dt) AS f FROM imp WHERE y = 1 AND creatorId IS NOT NULL GROUP BY 1, 2")
    con.execute(f"""CREATE TABLE daily AS
      SELECT i.userId, i.dt, count(*) AS n, sum((fc.f IS NOT NULL AND fc.f < i.dt)::INT) AS n_from_clicked,
             sum(coalesce(p.cr_ctr, {g_})) AS s_ctr, sum(coalesce(q.log_posts, 0)) AS s_posts
      FROM imp i LEFT JOIN first_click fc ON fc.userId = i.userId AND fc.creatorId = i.creatorId
      ASOF LEFT JOIN crp p ON p.creatorId = i.creatorId AND i.dt >= p.dt_apply
      ASOF LEFT JOIN crposts q ON q.creatorId = i.creatorId AND i.dt >= q.dt_apply
      GROUP BY 1, 2""")
    con.execute("DROP TABLE imp")
    CF = con.execute("""SELECT d.userId, d.dt AS origin_day,
        sum(n_from_clicked) OVER wh / sum(n) OVER wh AS h_cr_from_clicked_share,
        sum(n_from_clicked) OVER w7 / sum(n) OVER w7 AS w7_cr_from_clicked_share,
        sum(s_ctr) OVER wh / sum(n) OVER wh AS h_cr_ctr_shown,
        sum(s_posts) OVER wh / sum(n) OVER wh AS h_cr_posts_shown,
        sum(coalesce(nc.k, 0)) OVER wh AS h_cr_distinct_clicked
      FROM daily d LEFT JOIN (SELECT userId, f AS dt, count(*) AS k FROM first_click GROUP BY 1, 2) nc USING (userId, dt)
      WINDOW wh AS (PARTITION BY d.userId ORDER BY d.dt ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
             w7 AS (PARTITION BY d.userId ORDER BY d.dt RANGE BETWEEN 6 PRECEDING AND CURRENT ROW)""").df()
    con.close()
    CR = [c for c in CF.columns if c not in ("userId", "origin_day")]
    CF = CF.astype({c: "float32" for c in CR}).set_index(["userId", CF.origin_day.astype("int64")]).drop(columns="origin_day")
    write_json("12_creator_features.json", {
        "features": CR, "mean_all_user_days": {f: round(float(CF[f].mean()), 4) for f in CR},
        "describe": CF.describe().T.round(4).to_dict("index"),
        "note": "pseudo-R2 against the training average of all rows; AUC for clicks among origins with 20+ cards shown ahead",
        **{t: _extra_check(t, CF, CR, "plus_creator") for t in ("R", "C")}})


@step("A1.variety", ["12_variety_features.json"], needs=["02.tuning"])
def variety():
    """Variety of what the user clicked (distinct topics) and whether it adds anything to the full LightGBM."""
    drop_full_panel()
    con = connect(keep_order=False)
    con.execute(f"CREATE TABLE card_t AS SELECT mlogId, talkId FROM '{P}/mlog_demographics.parquet'")
    con.execute(f"""CREATE TABLE od AS SELECT DISTINCT userId, CAST(dt AS INT) AS dt
                    FROM read_parquet('{P}/impressions_place/d*.parquet') WHERE CAST(dt AS INT) <= 23""")
    con.execute(f"""CREATE TABLE clk AS SELECT i.userId, CAST(i.dt AS INT) AS dt, c.talkId
                    FROM read_parquet('{P}/impressions_place/d*.parquet') i JOIN card_t c USING (mlogId)
                    WHERE i.isClick = 1 AND CAST(i.dt AS INT) <= 23 AND c.talkId IS NOT NULL""")
    VF = con.execute("""
      WITH ft AS (SELECT userId, talkId, min(dt) AS f FROM clk GROUP BY 1, 2),
           nt AS (SELECT userId, f AS dt, count(*) AS k FROM ft GROUP BY 1, 2),
           nc AS (SELECT userId, dt, count(*) AS c FROM clk GROUP BY 1, 2),
           t7 AS (SELECT o.userId, o.dt, count(DISTINCT k.talkId) AS t7
                  FROM od o JOIN clk k ON k.userId = o.userId AND k.dt BETWEEN o.dt - 6 AND o.dt GROUP BY 1, 2)
      SELECT o.userId, o.dt AS origin_day,
             sum(coalesce(nt.k, 0)) OVER wh AS h_topics_clicked,
             coalesce(t7.t7, 0) AS w7_topics_clicked,
             sum(coalesce(nt.k, 0)) OVER wh / greatest(sum(coalesce(nc.c, 0)) OVER wh, 1) AS h_topics_per_click
      FROM od o LEFT JOIN nt USING (userId, dt) LEFT JOIN nc USING (userId, dt) LEFT JOIN t7 USING (userId, dt)
      WINDOW wh AS (PARTITION BY o.userId ORDER BY o.dt ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)""").df()
    con.close()
    VR = [c for c in VF.columns if c not in ("userId", "origin_day")]
    VF = VF.astype({c: "float32" for c in VR}).set_index(["userId", VF.origin_day.astype("int64")]).drop(columns="origin_day")
    write_json("12_variety_features.json", {
        "features": VR, "mean_all_user_days": {f: round(float(VF[f].mean()), 4) for f in VR},
        "describe": VF.describe().T.round(4).to_dict("index"),
        "note": "pseudo-R2 against the training average of all rows; AUC for clicks among origins with 20+ cards shown ahead",
        **{t: _extra_check(t, VF, VR, "plus_extra") for t in ("R", "C")}})
