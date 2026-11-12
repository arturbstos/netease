"""Slow steps of notebook 03 (RQ3: recommender design). Each function writes its outputs; the notebook reads them back
through thesis.cache.ensure. The code is the notebook's, moved here unchanged except that bootstrap replicates run in
parallel threads (same draws, same results) and that the impression table built for the exposure model is kept open
for the click model when both run in the same process (it is rebuilt otherwise: same users, same codes)."""
import gc, hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import thesis.click_model as cm
from thesis.cache import step, path, read, write_json
from thesis.config import P, O, TAG, SEED, connect
from thesis.panel import load, target_sets, user_w, lgb_params
from thesis.stats import boot, pmap, WeightedAUC, binom_fit, ll_rows, fe_ols
from thesis.heavy.common import ci

_CTX = {}   # DuckDB connection with the impression table of all panel users, its sizes and user ids, the exposure model


def _prepared(df=None):
    """Connection with cm.prepare() run for every panel user (as in notebook section 2), reused within a process."""
    if "con" not in _CTX:
        if df is None:
            df = load()[0]
        con = connect()
        sizes, user_ids = cm.prepare(con, P, df[["userId", "bucket"]].drop_duplicates("userId"))
        _CTX.update(con=con, sizes=sizes, user_ids=user_ids)
    return _CTX["con"], _CTX["sizes"], _CTX["user_ids"]


def _close():
    if "con" in _CTX:
        _CTX["con"].close()
    _CTX.clear()
    gc.collect()


# ---------------------------------------------------------------------------------------------------------------------
@step("03.exposure", ["15_exposure_model.parquet", "15_exposure_model.json", "15_predictions.parquet",
                      "15_exposure.json"], needs=["02.models"])
def exposure():
    """Exposure model (card x slot x page depth, pooled slots); expected clicks per user-day; four click models."""
    df, FEATS, tr, va, te = load()
    con, sizes, user_ids = _prepared(df)
    trn = cm.fetch(con, "bucket < 70 AND dt <= 15")
    trn["grp"] = 0
    m = cm.fit(sizes, trn, by_group=False)
    del trn
    cm.save_model(con, m, str(O / f"15_exposure_model{TAG}"))                # reused in section 5
    _CTX["m_exposure"] = m

    parts = []
    for lo in range(0, 100, 10):
        ch = cm.fetch(con, f"bucket >= {lo} AND bucket < {lo + 10}")
        ch["e"] = np.minimum(m["base"] * cm.card_effect(m, ch) * cm.slot_factor(m, ch) * m["gf"][ch.depth.values], 0.99)
        parts.append(ch.groupby(["u", "dt"], sort=False).e.sum().reset_index())
        del ch
    daily = pd.concat(parts, ignore_index=True)
    del parts
    daily["userId"] = user_ids.values[daily.u.values]
    daily = daily.rename(columns={"e": "e_adj"})

    # expected clicks over the past 7 days, all history and the next 7 days; fit and score the four models
    codes = df.userId.astype("category")
    cats = codes.cat.categories
    grid = np.zeros((len(cats), 31 + 8))
    grid[pd.Categorical(daily.userId, categories=cats).codes, daily.dt.values] = daily.e_adj.values
    cs = np.concatenate([np.zeros((len(cats), 1)), grid.cumsum(axis=1)], axis=1)   # cs[:, d+1] = sum of days 0..d
    ui, t = codes.cat.codes.values, df.origin_day.values.astype(int)
    df["w7_expected_adj"] = cs[ui, t + 1] - cs[ui, np.maximum(t - 6, 0)]
    df["h_expected_adj"] = cs[ui, t + 1]
    df["y_expected_adj"] = cs[ui, t + 8] - cs[ui, t + 1]
    df["h_click_vs_expected_adj"] = (df.h_clicks + 1) / (df.h_expected_adj + 1)
    df["w7_click_vs_expected_adj"] = (df.w7_clicks + 1) / (df.w7_expected_adj + 1)
    del daily, grid, cs, ui, t, codes
    tr, va, te = (df.loc[x.index] for x in (tr, va, te))
    (a, b, c), k, n = target_sets("C", tr, va, te)

    def off(d):
        """Log-odds of next week's expected click rate (the recommender's part)."""
        r = np.clip(d.y_expected_adj / d.y_impr, 1e-6, 0.99)
        return np.log(r / (1 - r)).values

    def sig(z):
        return 1 / (1 + np.exp(-z))

    par = lgb_params("C")                                  # tree size chosen for click propensity in notebook 02

    def gbm(feats, offset):
        def ds(d, ref=None):
            return lgb.Dataset(d[feats], label=k(d) / n(d), weight=n(d).values * user_w(d), reference=ref,
                               init_score=off(d) if offset else None)
        d0 = ds(a)
        mm = lgb.train(par, d0, 3000, valid_sets=[ds(b, d0)], callbacks=[lgb.early_stopping(50, verbose=False)])
        raw = mm.predict(c[feats], num_iteration=mm.best_iteration, raw_score=True)
        return 1 / (1 + np.exp(-(raw + (off(c) if offset else 0)))), raw, mm.best_iteration

    c0 = binom_fit(off(a), k(a).values, n(a).values, user_w(a), slope=False)[0]
    s10 = read("10_predictions.parquet").query("target == 'C' and set == 'test'")
    NEW = ["w7_expected_adj", "h_expected_adj", "h_click_vs_expected_adj", "w7_click_vs_expected_adj"]
    p_hist, _, it_hist = gbm(FEATS + NEW, False)
    p_ux, raw_ux, it_ux = gbm(FEATS + NEW, True)
    preds = {"exposure_only": sig(c0 + off(c)),
             "lightgbm_step10": c[["userId", "origin_day"]].merge(s10, on=["userId", "origin_day"], how="left").lightgbm.values,
             "lightgbm_plus_adjusted_history": p_hist,
             "user_x_exposure": p_ux}
    assert not np.isnan(preds["lightgbm_step10"]).any()

    # pseudo-R2 and AUC with bootstrap intervals for the difference from the notebook-02 model
    kc, nc = k(c).values.astype(float), n(c).values.astype(float)
    pooled = k(a).sum() / n(a).sum()
    L0 = ll_rows(kc, nc, np.full(len(c), pooled))
    L = {mm: ll_rows(kc, nc, q) for mm, q in preds.items()}
    h = c.y_impr.values >= 20
    A = {mm: WeightedAUC(c.y_clicks.values[h] > 0, q[h]) for mm, q in preds.items()}

    def scores(w):
        return {**{f"{mm}_r2": 1 - (w @ L[mm]) / (w @ L0) for mm in preds},
                **{f"{mm}_auc": A[mm](w[h]) for mm in preds}}

    point = scores(np.ones(len(c)))
    bt = boot(scores, c)
    res = {"test_rows": len(c), "exposure_model_slot_factors": np.round(m["sf"], 3).tolist(),
           "exposure_model_page_depth_factors": np.round(m["gf"], 3).tolist(), "exposure_only_intercept": round(float(c0), 4),
           "trees": {"lightgbm_plus_adjusted_history": it_hist, "user_x_exposure": it_ux},
           "corr_expected_vs_observed_clicks_next7_test": round(float(np.corrcoef(c.y_expected_adj, c.y_clicks)[0, 1]), 4),
           "models": {mm: {"pseudo_r2": round(point[f"{mm}_r2"], 4), "auc_any_click_20plus": round(point[f"{mm}_auc"], 4),
                           "r2_minus_step10_ci95": ci(bt[f"{mm}_r2"] - bt["lightgbm_step10_r2"]),
                           "auc_minus_step10_ci95": ci(bt[f"{mm}_auc"] - bt["lightgbm_step10_auc"])} for mm in preds}}
    pd.DataFrame({"userId": c.userId.values, "origin_day": c.origin_day.values, "user_logodds_net_of_exposure": raw_ux,
                  "exposure_logodds": off(c)}).to_parquet(path("15_predictions.parquet"), index=False)
    write_json("15_exposure.json", res)


# ---------------------------------------------------------------------------------------------------------------------
@step("03.click_model", ["21_click_model_bounds.json", "21_rerank.json", "21_user_gains.parquet"],
      needs=["03.exposure", "02.label"])
def click_model():
    """Nested click models chosen on validation users, slot-factor intervals, bounds, and re-ordering within pages."""
    con, sizes, user_ids = _prepared()
    m_exposure = _CTX.get("m_exposure")
    if m_exposure is None:                             # the saved model, on the codes of this same prepare()
        m_exposure = cm.load_model(con, str(O / f"15_exposure_model{TAG}"), sizes)
    grp = cm.scroll_groups_sql(con)                    # same users and impression table as section 2
    imp = cm.fetch(con, "(bucket < 70 AND dt <= 15) OR bucket >= 70")
    _close()
    imp["grp"] = grp[imp.u.values]
    NU = sizes["userId"]
    trn = imp[(imp.bucket < 70) & (imp.dt <= 15)]
    models = {"user_x_card": cm.fit(sizes, trn, slot=False, depth=False, by_group=False, hierarchy=False),
              "plus_page_depth": cm.fit(sizes, trn, slot=False, by_group=False, hierarchy=False),
              "plus_slot": cm.fit(sizes, trn, by_group=False, hierarchy=False)}
    # the scroll-depth models need pooled slot factors for users without days 1-15 history: those are exactly the
    # by_group=False fits already done (plus_slot without, and the section-2 exposure model with the creator/topic fallback)
    models["plus_slot_by_scroll_depth"] = cm.fit(sizes, trn, hierarchy=False, pooled_sf=models["plus_slot"]["sf"])
    models["plus_creator_topic_fallback"] = cm.fit(sizes, trn, pooled_sf=m_exposure["sf"])
    seen_train = np.bincount(trn.c.values, minlength=sizes["mlogId"]) > 0

    # out-of-sample check on validation and test users, days 16-30; choose the model on validation users
    def evaluate(lo, hi):
        early = imp[(imp.bucket >= lo) & (imp.bucket < hi) & (imp.dt <= 15)]
        late_ = imp[(imp.bucket >= lo) & (imp.bucket < hi) & (imp.dt >= 16)]
        yy, new, out = late_.y.values, ~seen_train[late_.c.values], {}
        for name, mod in models.items():
            q = (mod["base"] * cm.user_factor(mod, early, NU)[late_.u.values] * cm.card_effect(mod, late_)
                 * cm.slot_factor(mod, late_) * mod["gf"][late_.depth.values])
            q = np.clip(q * yy.sum() / q.sum(), 1e-6, 1 - 1e-6)
            ll = -(yy * np.log(q) + (1 - yy) * np.log(1 - q))
            out[name] = {"auc": round(float(roc_auc_score(yy, q)), 4), "logloss": round(float(ll.mean()), 5),
                         "new_cards_auc": round(float(roc_auc_score(yy[new], q[new])), 4),
                         "new_cards_logloss": round(float(ll[new].mean()), 5)}
        out["share_cards_new"] = round(float(new.mean()), 4)
        out["click_rate_new_vs_seen_cards"] = [round(float(yy[new].mean()), 4), round(float(yy[~new].mean()), 4)]
        return out

    check_valid, check = evaluate(70, 85), evaluate(85, 100)
    cands = ("plus_slot", "plus_slot_by_scroll_depth", "plus_creator_topic_fallback")
    best = min(cands, key=lambda key: check_valid[key]["logloss"])           # chosen on validation users
    m = models[best]
    print("Model chosen on validation users:", best)

    # 95% intervals of the chosen model's slot factors: training users are resampled (Poisson weights per user, 200
    # replicates) and the slot factors re-estimated with the user, card and page-depth effects held at their fitted
    # values. The intervals therefore show the sampling noise of the slot factors themselves, not of the whole model.
    uf_tr = cm.user_factor(m, trn, NU)
    x_tr = m["base"] * uf_tr[trn.u.values] * cm.card_effect(m, trn) * m["gf"][trn.depth.values]
    sg_tr = (np.minimum(trn.grp.values, 2) * 6 + trn.slot.values) if m["by_group"] else trn.slot.values
    ns = 18 if m["by_group"] else 6
    rng_sf = np.random.default_rng(SEED)
    y_tr, u_tr = trn.y.values, trn.u.values

    def sf_rep(w_user):
        w = w_user[u_tr]
        f = np.bincount(sg_tr, w * y_tr, ns) / np.bincount(sg_tr, w * x_tr, ns)
        return (f.reshape(-1, 6) / f.reshape(-1, 6)[:, :1]).ravel()

    boot_sf = pmap(sf_rep, (rng_sf.poisson(1.0, NU) for _ in range(200)))
    sf_ci = np.quantile(np.array(boot_sf), [0.025, 0.975], axis=0)
    del uf_tr, x_tr, sg_tr, boot_sf, y_tr, u_tr

    # bounds: the click model is a rate model, so a predicted rate can exceed 1. How often does that happen for test
    # users' impressions on days 16-30 (rates rescaled to the observed total, as in the evaluation, before any capping)?
    early_ = imp[(imp.bucket >= 85) & (imp.dt <= 15)]
    late_ = imp[(imp.bucket >= 85) & (imp.dt >= 16)]
    q_ = (m["base"] * cm.user_factor(m, early_, NU)[late_.u.values] * cm.card_effect(m, late_)
          * cm.slot_factor(m, late_) * m["gf"][late_.depth.values])
    q_ = q_ * late_.y.sum() / q_.sum()
    write_json("21_click_model_bounds.json", {
        "model": best, "test_impressions": int(len(q_)), "share_rate_above_1": round(float((q_ > 1).mean()), 6),
        "share_rate_above_0.5": round(float((q_ > 0.5).mean()), 6), "max_rate": round(float(q_.max()), 3),
        "share_of_expected_clicks_on_rows_above_1": round(float(q_[q_ > 1].sum() / q_.sum()), 6),
        "rate_percentiles": {f"p{p}": round(float(np.quantile(q_, p / 100)), 4) for p in (50, 90, 99, 99.9)}})
    del early_, late_, q_

    # re-rank within pages and compute expected clicks under each order
    late = imp[(imp.bucket >= 85) & (imp.dt >= 16)].copy()                  # rows sorted by user, page, slot
    del imp, trn
    gc.collect()
    late["page"] = late.groupby(["u", "impressTime"], sort=False).ngroup().values
    late["cf"] = cm.card_effect(m, late)
    late["f"] = cm.slot_factor(m, late)
    f_sorted = np.lexsort((-late.f.values, late.page.values))               # the same for every score

    def reassign(score):
        """Within each page, the card with the k-th highest score takes the k-th largest slot factor."""
        order = np.lexsort((-score, late.page.values))
        out = np.empty(len(late))
        out[order] = late.f.values[f_sorted]
        return out

    rng = np.random.default_rng(SEED)
    POL = {"platform_rate": late.card_ctr.values, "platform_rate_creator_aware": late.card_ctr_cr.values,
           "click_model_appeal": late.cf.values, "random": rng.random(len(late))}
    cols = ["x_logged"]
    late["x_logged"] = late.cf * late.f
    for name, score in POL.items():
        late["x_" + name] = late.cf * reassign(score)
        cols.append("x_" + name)
    per_user = late.groupby("u", sort=True)[cols].sum()
    per_user.index = user_ids.values[per_user.index.values]
    pu = per_user.values
    Bw = [np.bincount(rng.integers(0, len(pu), len(pu)), minlength=len(pu)) for _ in range(500)]   # bootstrap over users

    pages = late.groupby("page").agg(n=("f", "size"))
    multi = late[late.groupby("page").f.transform("size").values > 1]
    sim = {"model_used": best, "test_cards": len(late), "pages": len(pages),
           "share_pages_2plus_cards": round(float((pages.n > 1).mean()), 4),
           "rank_corr_logged_slot_vs_card_ctr_within_pages": round(float(
               multi.groupby("page")[["slot", "card_ctr"]].corr(method="spearman").xs("slot", level=1).card_ctr.mean()), 4)}
    for j, name in enumerate(POL, start=1):
        sim[f"expected_clicks_change_pct_{name}"] = round(float(100 * (pu[:, j].sum() / pu[:, 0].sum() - 1)), 2)
        sim[f"expected_clicks_change_pct_{name}_ci95"] = [
            round(float(np.quantile([100 * ((w @ pu[:, j]) / (w @ pu[:, 0]) - 1) for w in Bw], q)), 2) for q in (0.025, 0.975)]

    # effect on the inactive label: click probability scaled by the user's expected gain
    S = read("19_scores.parquet")
    cut = S.groupby("origin_day").expected_click_prob.transform(lambda s: s.quantile(0.10))
    act = {"test_origins": len(S), "inactive_share_before": round(float((S.expected_click_prob <= cut).mean()), 4)}
    for name in ("platform_rate", "platform_rate_creator_aware"):
        gain = per_user["x_" + name] / per_user.x_logged
        e_new = S.p_return * np.clip(S.p_click * S.userId.map(gain).fillna(1.0), 0, 0.99)
        act[f"inactive_share_after_{name}"] = round(float((e_new <= cut).mean()), 4)

    (per_user[[c for c in per_user.columns if c != "x_logged"]].div(per_user.x_logged, axis=0)
     .rename(columns=lambda c: "click_gain_" + c[2:]).rename_axis("userId").reset_index()
     .to_parquet(path("21_user_gains.parquet"), index=False))                # per test user, used in section 6
    write_json("21_rerank.json", {
        "model_choice_validation_users_days16_30": check_valid, "out_of_sample_check_test_days16_30": check,
        "slot_factors": np.round(m["sf"], 3).tolist(),
        "slot_factors_ci95": {"low": np.round(sf_ci[0], 3).tolist(), "high": np.round(sf_ci[1], 3).tolist(),
                              "note": "training users resampled, other effects held at their fitted values"},
        "page_depth_factors": np.round(m["gf"], 3).tolist(),
        "rerank_within_pages": sim, "active_users": act,
        "assumptions": "examination model; same cards and slots; users do not scroll or return differently because of the order"})


# ---------------------------------------------------------------------------------------------------------------------
def _user_ids():
    """The user ids of cm.prepare() for every panel user, in code order (without building the impression table)."""
    if "user_ids" in _CTX:
        return _CTX["user_ids"]
    df = load()[0]
    con = connect()
    con.register("users", df[["userId", "bucket"]].drop_duplicates("userId"))
    ids = con.execute(f"""SELECT v AS userId FROM (SELECT DISTINCT coalesce(i.userId, '~missing') AS v
                          FROM read_parquet('{P}/impressions_place/d*.parquet') i JOIN users u USING (userId)) ORDER BY v""").df().userId
    con.close()
    return ids


ESTABLISHED = (100, 1000, 10000)   # minimum earlier impressions on the platform for a card to count as established


@step("03.observed", ["21_observed_rerank.json", "21_observed_balance.csv", "21_slot_cards.csv"])
def observed():
    """Gain in the observed clicks from sorting each page by a score (pages of 2-6 cards), with a random-order check."""
    user_ids = _user_ids()
    K_ = 100.0
    users = pd.DataFrame({"userId": user_ids.values})
    users["bucket"] = [int(hashlib.md5(x.encode()).hexdigest()[:8], 16) % 100 for x in users.userId]
    users["uid"] = np.arange(len(users), dtype=np.int32)
    con = connect(keep_order=False)
    con.register("users", users)
    g_ = con.execute(f"SELECT sum(CAST(userClickCount AS BIGINT))::DOUBLE / sum(CAST(userImprssionCount AS BIGINT)) FROM '{P}/mlog_stats.parquet'").fetchone()[0]
    con.execute(f"CREATE TABLE cp AS {cm.card_prior_sql(P)}")
    con.execute(f"""CREATE TABLE crp AS SELECT creatorId, dt_apply,
        sum(impr) OVER (PARTITION BY creatorId ORDER BY dt_apply) AS cr_impr, sum(clicks) OVER (PARTITION BY creatorId ORDER BY dt_apply) AS cr_clicks
      FROM (SELECT m.creatorId, CAST(s.dt AS INT) + 1 AS dt_apply, sum(CAST(s.userImprssionCount AS BIGINT)) AS impr, sum(CAST(s.userClickCount AS BIGINT)) AS clicks
            FROM '{P}/mlog_stats.parquet' s JOIN '{P}/mlog_demographics.parquet' m USING (mlogId) GROUP BY 1, 2)""")
    con.execute(f"""CREATE TABLE ob AS SELECT u.uid, i.impressTime, i.impressPosition, i.mlogId, CAST(i.isClick AS INT) AS y,
        coalesce(cp.cum_impr, 0) AS card_impr,
        coalesce((cp.cum_clicks + {K_} * {g_}) / (cp.cum_impr + {K_}), {g_}) AS card_ctr,
        (coalesce(cp.cum_clicks, 0) + {K_} * (({g_} * {K_} + coalesce(cr.cr_clicks, 0) - coalesce(cp.cum_clicks, 0))
           / ({K_} + coalesce(cr.cr_impr, 0) - coalesce(cp.cum_impr, 0)))) / (coalesce(cp.cum_impr, 0) + {K_}) AS card_ctr_cr
      FROM read_parquet('{P}/impressions_place/d*.parquet') i JOIN users u USING (userId)
      LEFT JOIN '{P}/mlog_demographics.parquet' m USING (mlogId)
      ASOF LEFT JOIN cp ON cp.mlogId = i.mlogId AND CAST(i.dt AS INT) >= cp.dt_apply
      ASOF LEFT JOIN crp cr ON cr.creatorId = m.creatorId AND CAST(i.dt AS INT) >= cr.dt_apply
      WHERE CAST(i.dt AS INT) >= 16""")
    con.execute("""CREATE TABLE pg AS SELECT uid, impressTime, y::TINYINT AS y, card_ctr, card_impr,
        (row_number() OVER w0 - 1)::TINYINT AS s, least(count(*) OVER (PARTITION BY uid, impressTime), 100)::TINYINT AS k,
        (row_number() OVER (PARTITION BY uid, impressTime ORDER BY card_ctr DESC, hash(mlogId, impressTime)) - 1)::TINYINT AS r1,
        (row_number() OVER (PARTITION BY uid, impressTime ORDER BY card_ctr_cr DESC, hash(mlogId, impressTime)) - 1)::TINYINT AS r2,
        (row_number() OVER (PARTITION BY uid, impressTime ORDER BY hash(mlogId, uid, impressTime)) - 1)::TINYINT AS r0
      FROM ob WINDOW w0 AS (PARTITION BY uid, impressTime ORDER BY impressPosition, mlogId)""")
    con.execute("DROP TABLE ob")
    tot = con.execute("SELECT uid, sum(y) AS clicks, count(*) AS cards FROM pg GROUP BY uid ORDER BY uid").df()
    tot["bucket"] = users.bucket.values[tot.uid.values]
    cells_ = {r: con.execute(f"""SELECT uid, k, {r} AS r, s, sum(y)::INT AS yy, count(*)::INT AS n
                                 FROM pg WHERE k BETWEEN 2 AND 6 GROUP BY ALL""").df() for r in ("r1", "r2", "r0")}
    balance = con.execute("""SELECT k, r1 AS r, s, count(*) AS cards, avg(card_ctr) AS mean_card_ctr, avg(y) AS click_rate
                             FROM pg WHERE k BETWEEN 2 AND 6 GROUP BY ALL ORDER BY k, r, s""").df()
    slot_cards = con.execute(f"""SELECT r1 AS r, s, count(*) AS cards, avg(card_impr) AS mean_prior_impressions,
                                   median(card_impr) AS median_prior_impressions,
                                   avg((card_impr < {K_})::INT) AS share_under_100_impressions, avg(y) AS click_rate
                                FROM pg WHERE k = 4 GROUP BY ALL ORDER BY r, s""").df()
    slot_cards.to_csv(path("21_slot_cards.csv"), index=False)
    page_sizes = con.execute("SELECT k, count(*) // k AS pages, sum(y) AS clicks FROM pg GROUP BY k ORDER BY k").df()
    # Established cards only: cards with fewer than T earlier impressions on the platform keep their slot (the platform
    # tries new cards out, mostly in the second slot), and the other cards of the page are sorted by the creator-aware
    # rate into the slots those cards occupied. t = the slot an established card would move to.
    est_cells, est_targets, est_shares = {}, {}, {}
    for T in ESTABLISHED:
        con.execute(f"""CREATE OR REPLACE TABLE es AS SELECT uid, k, r2 AS r, s, y,
            (array_agg(s) OVER (PARTITION BY uid, impressTime ORDER BY s
                                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING))[
                row_number() OVER (PARTITION BY uid, impressTime ORDER BY r2)] AS t
          FROM pg WHERE k BETWEEN 2 AND 6 AND card_impr >= {T}""")
        est_cells[T] = con.execute("SELECT uid, k, r, s, sum(y)::INT AS yy, count(*)::INT AS n FROM es GROUP BY ALL").df()
        est_targets[T] = con.execute("SELECT uid, k, r, t AS s, 0 AS yy, count(*)::INT AS n FROM es GROUP BY ALL").df()
        n_all = con.execute("SELECT count(*) FROM pg WHERE k BETWEEN 2 AND 6").fetchone()[0]
        n_est, n_moved = con.execute("SELECT count(*), sum((t <> s)::INT) FROM es").fetchone()
        est_shares[T] = {"share_of_cards_established": round(n_est / n_all, 4), "share_of_cards_moved": round(n_moved / n_all, 4)}
    con.execute("DROP TABLE IF EXISTS es")
    con.close()
    uidx = pd.Index(tot.uid)
    NC = 7 * 7 * 7

    def prep(d):
        ci_ = (d.k.values.astype(np.int64) * 7 + d.r.values.astype(np.int64)) * 7 + d.s.values.astype(np.int64)
        return uidx.get_indexer(d.uid), ci_, d.yy.values.astype(float), d.n.values.astype(float)

    def estimate(ui, ci_, yy, n, w, detail=True):
        """Gain in clicks (share of all clicks) if every page of 2-6 cards were sorted, and details by page size.
        Two assignments of the sorted cards to slots: in feed order (best card in the first slot, second best in the
        second, ...) and by the slots' own click rates (best card in the most-clicked slot, ...). They differ because the
        feed is a grid of two columns: on the first page the top-right card is clicked least (thesis.positions)."""
        Y = np.bincount(ci_, yy * w[ui], NC).reshape(7, 7, 7)
        N = np.bincount(ci_, n * w[ui], NC).reshape(7, 7, 7)
        gain, gain_layout, out = 0.0, 0.0, {}
        for k in range(2, 7):
            Mk = np.divide(Y[k, :k, :k], N[k, :k, :k], out=np.zeros((k, k)), where=N[k, :k, :k] > 0)
            pages = N[k, 0, :k].sum()
            slot_rate = np.divide(Y[k, :k, :k].sum(0), N[k, :k, :k].sum(0), out=np.zeros(k), where=N[k, :k, :k].sum(0) > 0)
            order = np.argsort(-slot_rate, kind="stable")              # slots from most to least clicked
            gain += pages * np.trace(Mk) - Y[k, :k, :k].sum()
            gain_layout += pages * Mk[np.arange(k), order].sum() - Y[k, :k, :k].sum()
            if detail:
                out[k] = {"pages": round(float(pages)),
                          "gain_pct_of_these_pages": round(float(100 * (pages * np.trace(Mk) / Y[k, :k, :k].sum() - 1)), 2),
                          "gain_pct_of_these_pages_slots_by_click_rate": round(float(100 * (pages * Mk[np.arange(k), order].sum() / Y[k, :k, :k].sum() - 1)), 2),
                          "slots_from_most_to_least_clicked": order.tolist(),
                          "share_best_in_top_slot": round(float(N[k, 0, 0] / pages), 4),
                          "click_rate_by_slot_relative_to_first": (Y[k, :, :k].sum(0) / N[k, :, :k].sum(0)
                                                                   / (Y[k, :, 0].sum() / N[k, :, 0].sum())).round(3).tolist()}
        tot_clicks = (tot.clicks.values * w).sum()
        return gain / tot_clicks, gain_layout / tot_clicks, out

    def estimate_established(a, b, w):
        """Gain in clicks (share of all clicks) when only established cards are sorted. The click rate of a card of
        rank r in slot s is measured on established cards alone, so new cards never enter the comparison."""
        Y = np.bincount(a[1], a[2] * w[a[0]], NC)
        N = np.bincount(a[1], a[3] * w[a[0]], NC)
        Tn = np.bincount(b[1], b[3] * w[b[0]], NC)
        M = np.divide(Y, N, out=np.zeros(NC), where=N > 0)
        return (Tn * M).sum() - Y.sum(), Tn[N == 0].sum() / max(Tn.sum(), 1.0)

    rng = np.random.default_rng(SEED)
    obs = {}
    prepped = {r: prep(cells_[r]) for r in ("r1", "r2", "r0")}
    del cells_
    for pop, mask in (("test_users", tot.bucket.values >= 85), ("all_users", np.ones(len(tot), bool))):
        base_w = mask.astype(float)
        W = [base_w * rng.poisson(1.0, len(tot)) for _ in range(500)]
        obs[pop] = {"users": int(mask.sum()), "clicks": int(tot.clicks.values[mask].sum())}
        for r, name in (("r1", "platform_rate"), ("r2", "platform_rate_creator_aware"), ("r0", "random_order")):
            ui, ci_, yy, n = prepped[r]
            g0, g0_layout, detail = estimate(ui, ci_, yy, n, base_w)
            boots = np.array(pmap(lambda w: estimate(ui, ci_, yy, n, w, detail=False)[:2], W))
            obs[pop][name] = {"gain_pct": round(100 * float(g0), 2),
                              "ci95": [round(100 * float(np.quantile(boots[:, 0], q)), 2) for q in (0.025, 0.975)],
                              "gain_pct_slots_by_click_rate": round(100 * float(g0_layout), 2),
                              "ci95_slots_by_click_rate": [round(100 * float(np.quantile(boots[:, 1], q)), 2) for q in (0.025, 0.975)],
                              "by_page_size": detail}
        obs[pop]["established_cards_only"] = {}
        for T in ESTABLISHED:
            a, b = prep(est_cells[T]), prep(est_targets[T])
            tot_clicks = lambda w: (tot.clicks.values * w).sum()
            g0, uncovered = estimate_established(a, b, base_w)
            boots = np.array(pmap(lambda w: estimate_established(a, b, w)[0] / tot_clicks(w), W))
            obs[pop]["established_cards_only"][str(T)] = {
                "min_prior_impressions": T, "gain_pct": round(100 * float(g0 / tot_clicks(base_w)), 2),
                "ci95": [round(100 * float(np.quantile(boots, q)), 2) for q in (0.025, 0.975)],
                "share_of_moved_cards_without_estimate": round(float(uncovered), 4), **est_shares[T]}
        del W
    tu = tot[tot.bucket >= 85]
    obs["test_user_clicks_days16_30"] = {"mean": round(float(tu.clicks.mean()), 4), "sd": round(float(tu.clicks.std()), 4),
                                         "users": len(tu)}
    obs["page_sizes_all_users"] = page_sizes.astype(int).to_dict("records")
    balance.round(5).to_csv(path("21_observed_balance.csv"), index=False)
    write_json("21_observed_rerank.json", obs)


# ---------------------------------------------------------------------------------------------------------------------
FIRST = ["quality", "video", "popularity", "age"]
MIX = FIRST + ["creator_variety", "topic_variety"]
FEAS_COLS = ["q_bestq", "v_bestq", "q_bestv", "v_bestv"]


@step("03.first_page", ["23_first_page_click_gains.parquet", "23_content_mix.json", "23_first_page_options.parquet"],
      needs=["03.exposure"])
def first_page():
    """First-page features per user-day, the best first page buildable from that day's cards, and the within-user
    (two-way fixed effects) comparison of first pages and return."""
    df, FEATS, tr, va, te = load()
    con = connect()
    # only validation and test users are needed; card effects come from the saved model, matched by id
    sizes, user_ids = cm.prepare(con, P, df[df.bucket >= 70][["userId", "bucket"]].drop_duplicates("userId"))
    m = cm.load_model(con, str(O / f"15_exposure_model{TAG}"), sizes)
    parts = []
    NUc = sizes["userId"]
    fp_logged, fp_new = np.zeros(NUc), {"best_quality": np.zeros(NUc), "most_video": np.zeros(NUc)}
    for lo in range(70, 100, 10):
        ch = cm.fetch(con, f"bucket >= {lo} AND bucket < {lo + 10}")
        ch["quality"] = np.log(cm.card_effect(m, ch))
        ch["popularity"] = np.log1p(ch.card_impr)
        ch["age"] = np.log1p(ch.card_age.astype(float))
        first = ch[ch.depth == 0].groupby(["u", "dt"], sort=False)[["quality", "video", "popularity", "age"]].mean()
        day = ch.groupby(["u", "dt"], sort=False).agg(quality_all=("quality", "mean"), n=("c", "size"),
                                                      nk=("k", "nunique"), nt=("t", "nunique"), pages=("impressTime", "nunique"))
        day["creator_variety"] = day.nk / day.n
        day["topic_variety"] = day.nt / day.n
        # feasibility (section 6.2): the best first page that could be built from the cards the user was shown that
        # day, keeping the first page's number of cards. Best for quality: the highest card effects. Most video: video
        # cards first, then by card effect.
        keys = [ch.u, ch.dt]
        n0 = (ch.depth == 0).groupby(keys, sort=False).transform("sum")
        rq = ch.quality.groupby(keys, sort=False).rank(ascending=False, method="first")
        rv = (ch.video * 100 + ch.quality).groupby(keys, sort=False).rank(ascending=False, method="first")
        bq = ch[rq <= n0].groupby(["u", "dt"], sort=False).agg(q_bestq=("quality", "mean"), v_bestq=("video", "mean"))
        bv = ch[rv <= n0].groupby(["u", "dt"], sort=False).agg(q_bestv=("quality", "mean"), v_bestv=("video", "mean"))
        # click side of the rebuilt first page (test users, days 16-30): chosen first-page cards keep their places, each
        # incoming card takes the place of a displaced first-page card and that card takes the incoming card's place
        # (paired in logged order within the user-day); expected clicks = card effect x slot x depth
        tst = ((ch.bucket >= 85) & (ch.dt >= 16)).values
        if tst.any():
            gid = ch.groupby(keys, sort=False).ngroup().values
            order = np.arange(len(ch))
            cf_ = np.exp(ch.quality.values)
            pf_ = m["sf"][ch.slot.values] * m["gf"][ch.depth.values]
            fp = ch.depth.values == 0
            fp_logged += np.bincount(ch.u.values[tst], (cf_ * pf_)[tst], NUc)
            for name, sel in (("best_quality", (rq <= n0).values), ("most_video", (rv <= n0).values)):
                inc = np.flatnonzero(sel & ~fp); inc = inc[np.lexsort((order[inc], gid[inc]))]   # cards brought in
                out = np.flatnonzero(fp & ~sel); out = out[np.lexsort((order[out], gid[out]))]   # cards they replace
                assert len(inc) == len(out) and (gid[inc] == gid[out]).all()
                pf_new = pf_.copy(); pf_new[inc], pf_new[out] = pf_[out], pf_[inc]
                fp_new[name] += np.bincount(ch.u.values[tst], (cf_ * pf_new)[tst], NUc)
        parts.append(day[["quality_all", "creator_variety", "topic_variety", "pages"]].join(first, how="left")
                     .join(bq, how="left").join(bv, how="left").reset_index())
        del ch
    con.close()
    has_fp = fp_logged > 0
    pd.DataFrame({"userId": user_ids.values[has_fp], **{f"gain_{k}": v[has_fp] / fp_logged[has_fp] for k, v in fp_new.items()}}
                 ).to_parquet(path("23_first_page_click_gains.parquet"), index=False)

    q = pd.concat(parts, ignore_index=True)
    q["userId"] = user_ids.values[q.u.values]
    base = df[df.bucket >= 70].merge(q.rename(columns={"dt": "origin_day"})[["userId", "origin_day", "quality_all", "pages"] + MIX + FEAS_COLS],
                                     on=["userId", "origin_day"], how="inner")
    del df, tr, va, te, q, parts
    base["returned"] = (base.y_visit_days > 0).astype(float)
    base["past6_visit_days"] = base.w7_visit_days - 1
    base["log_cards_today"] = np.log(base.w1_impr)
    base["log_prev_week_visits"] = np.log1p(base.w7_visit_days - 1)
    base["log_clicks_today"] = np.log1p(base.w1_clicks)
    ctrl = ["log_cards_today", "log_prev_week_visits"]

    def zscore(d, cols, res):
        """Standardise each feature within user: deviation from the user's mean, divided by the SD of those deviations."""
        for f in cols:
            dev = d[f] - d.groupby("userId")[f].transform("mean")
            res.setdefault("within_user_sd", {})[f] = round(float(dev.std()), 4)
            d[f + "_z"] = dev / dev.std()

    # A: quality only, and B: joint model (two-way fixed effects, user-clustered standard errors)
    res = {}
    d = base[base.quality.notna()]                                           # user-days with a first page
    d = d[d.groupby("userId").userId.transform("size") > 1].copy()           # users with 2+ such days
    A = {"rows": len(d), "users": int(d.userId.nunique()),
         "outcome_means": {"returned": round(float(d.returned.mean()), 4),
                           "visit_days_next7": round(float(d.y_visit_days.mean()), 4)}}
    zscore(d, ["quality", "quality_all"], A)
    for f in ("quality", "quality_all"):
        z = [f + "_z"] + ctrl
        A[f] = {"return_next7": fe_ols(d, "returned", z)[f + "_z"],
                "visit_days_next7": fe_ols(d, "y_visit_days", z)[f + "_z"],
                "reverse_check_past6_visit_days": fe_ols(d, "past6_visit_days", [f + "_z", "log_cards_today"])[f + "_z"],
                "return_next7_controlling_clicks_today": fe_ols(d, "returned", z + ["log_clicks_today"])[f + "_z"],
                "return_next7_without_previous_week_control": fe_ols(d, "returned", [f + "_z", "log_cards_today"])[f + "_z"]}
    res["A_quality_only"] = A

    d = base.dropna(subset=MIX)                                              # all features available
    d = d[d.groupby("userId").userId.transform("size") > 1].copy()
    B = {"rows": len(d), "users": int(d.userId.nunique())}
    zscore(d, MIX, B)
    Z = [f + "_z" for f in MIX]
    B["return_next7"] = fe_ols(d, "returned", Z + ctrl)
    B["visit_days_next7"] = fe_ols(d, "y_visit_days", Z + ctrl)
    B["reverse_check_past6_visit_days"] = fe_ols(d, "past6_visit_days", Z + ["log_cards_today"])
    B["return_next7_without_previous_week_control"] = fe_ols(d, "returned", Z + ["log_cards_today"])
    res["B_joint"] = B
    res["note"] = "coefficients per +1 within-user SD; linear probability for return; the policy simulation uses B_joint"
    base[["userId", "origin_day", "quality", "video", "pages"] + FEAS_COLS].to_parquet(
        path("23_first_page_options.parquet"), index=False)                      # used by the policies (section 7)
    write_json("23_content_mix.json", res)
