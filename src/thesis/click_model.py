"""Shared click model for RQ3 (notebooks 20, 21, 30) and card click rates known the day before (also used by notebook 03).
Impressions are read with: slot (rank of the card within its page = cards sharing one impressTime), page depth (band of
the page's first feed position), the user's scroll-depth group (share of cards shown beyond the first page on days
1-15; 3 = no days 1-15 history), and two platform click rates known the day before:
  card_ctr      : the card's own platform rate, shrunk toward the platform average (K = 100 impressions, as in the panel)
  card_ctr_cr   : the same, shrunk toward the CREATOR's platform rate (the creator's other cards, up to the day before)
Extra card columns: video (type 2), card_impr (platform impressions up to the day before), card_age (days since
publication on the impression day; publishTime = days from publication to Dec 1st 2019).
Model (examination form, Poisson-style multiplicative fit by alternating updates, shrinkage of A pseudo-clicks):
  clicks ~ base x user x card x slot[scroll-depth group] x page depth
with card effects shrunk toward their creator's effect and creators toward their card type (video or not). A card not
seen in training takes its creator's effect, else its topic's (talkId), else its type's. Fitted on training users
(md5 bucket < 70), days 1-15."""
import numpy as np, pandas as pd

K, A = 100.0, 2.0
POS_DEPTH = """CASE WHEN min(impressPosition) OVER (PARTITION BY userId, impressTime) <= 4 THEN 0
         WHEN min(impressPosition) OVER (PARTITION BY userId, impressTime) <= 9 THEN 1
         WHEN min(impressPosition) OVER (PARTITION BY userId, impressTime) <= 20 THEN 2
         WHEN min(impressPosition) OVER (PARTITION BY userId, impressTime) <= 50 THEN 3 ELSE 4 END"""

def platform_rate(con, P):
    """Platform-wide click rate over the whole month (the shrinkage target for card rates; panel and click model)."""
    return con.execute(f"SELECT sum(CAST(userClickCount AS BIGINT))::DOUBLE / sum(CAST(userImprssionCount AS BIGINT)) FROM '{P}/mlog_stats.parquet'").fetchone()[0]

def card_prior_sql(P):
    """Each card's cumulative platform impressions and clicks, applicable from the day AFTER (no same-day leakage)."""
    return f"""SELECT mlogId, CAST(dt AS INT) + 1 AS dt_apply,
        sum(CAST(userImprssionCount AS BIGINT)) OVER (PARTITION BY mlogId ORDER BY CAST(dt AS INT)) AS cum_impr,
        sum(CAST(userClickCount AS BIGINT)) OVER (PARTITION BY mlogId ORDER BY CAST(dt AS INT)) AS cum_clicks
      FROM '{P}/mlog_stats.parquet'"""

def prepare(con, P, users):
    """Build the impression table for these users (userId, bucket) inside DuckDB, with page slot, page depth and the two
    platform click rates, plus integer lookup tables for every id (so text ids never enter Python)."""
    con.register("users", users)
    g = platform_rate(con, P)
    con.execute(f"CREATE OR REPLACE TABLE card_prior AS {card_prior_sql(P)}")
    con.execute(f"""CREATE OR REPLACE TABLE creator_prior AS SELECT creatorId, dt_apply,
        sum(impr) OVER (PARTITION BY creatorId ORDER BY dt_apply) AS cr_impr, sum(clicks) OVER (PARTITION BY creatorId ORDER BY dt_apply) AS cr_clicks
      FROM (SELECT m.creatorId, CAST(s.dt AS INT) + 1 AS dt_apply, sum(CAST(s.userImprssionCount AS BIGINT)) AS impr, sum(CAST(s.userClickCount AS BIGINT)) AS clicks
            FROM '{P}/mlog_stats.parquet' s JOIN '{P}/mlog_demographics.parquet' m USING (mlogId) GROUP BY 1, 2)""")
    con.execute(f"""CREATE OR REPLACE TABLE imp AS SELECT row_id, userId, bucket, dt, mlogId, creatorId, talkId, card_type, impressTime, impressPosition, y, card_ctr,
        (card_type = '2')::INT AS video, card_impr, greatest(dt + publish_days - 31, 0) AS card_age, has_detail,
        (card_clicks + {K} * creator_rate) / (card_impr + {K}) AS card_ctr_cr,
        least(row_number() OVER (PARTITION BY userId, impressTime ORDER BY impressPosition, mlogId, row_id), 6) - 1 AS slot,
        {POS_DEPTH} AS depth
      FROM (SELECT i.row_id, i.userId, u.bucket, CAST(i.dt AS INT) AS dt, i.mlogId, m.creatorId, m.talkId, coalesce(m.type, '?') AS card_type,
                   TRY_CAST(m.publishTime AS INT) AS publish_days,                -- days from publication to Dec 1st 2019
                   i.has_detail,                                                  -- the card has a swipe-down list
                   i.impressTime, i.impressPosition, i.isClick::DOUBLE AS y,
                   coalesce(cp.cum_impr, 0) AS card_impr, coalesce(cp.cum_clicks, 0) AS card_clicks,
                   coalesce((cp.cum_clicks + {K} * {g}) / (cp.cum_impr + {K}), {g}) AS card_ctr,
                   -- creator's rate from its OTHER cards up to the day before, shrunk toward the platform average
                   ({g} * {K} + coalesce(cr.cr_clicks, 0) - coalesce(cp.cum_clicks, 0)) / ({K} + coalesce(cr.cr_impr, 0) - coalesce(cp.cum_impr, 0)) AS creator_rate
            FROM read_parquet('{P}/impressions_place/d*.parquet') i JOIN users u USING (userId)
            LEFT JOIN '{P}/mlog_demographics.parquet' m USING (mlogId)
            ASOF LEFT JOIN card_prior cp ON cp.mlogId = i.mlogId AND CAST(i.dt AS INT) >= cp.dt_apply
            ASOF LEFT JOIN creator_prior cr ON cr.creatorId = m.creatorId AND CAST(i.dt AS INT) >= cr.dt_apply)""")
    for col in ("userId", "mlogId", "creatorId", "talkId", "card_type"):
        con.execute(f"""CREATE OR REPLACE TABLE map_{col} AS SELECT v, (row_number() OVER (ORDER BY v) - 1)::INT AS code
                        FROM (SELECT DISTINCT coalesce({col}, '~missing') AS v FROM imp)""")
    sizes = {col: con.execute(f"SELECT count(*) FROM map_{col}").fetchone()[0] for col in ("userId", "mlogId", "creatorId", "talkId", "card_type")}
    user_ids = con.execute("SELECT v AS userId FROM map_userId ORDER BY code").df().userId
    return sizes, user_ids

def fetch(con, where):
    """Impressions matching a SQL filter on imp (e.g. 'bucket < 70 AND dt <= 15'), integer-coded, sorted by user, page, slot."""
    j = lambda col: f"JOIN map_{col} ON map_{col}.v = coalesce(imp.{col}, '~missing')"
    return con.execute(f"""SELECT map_userId.code AS u, bucket::SMALLINT AS bucket, dt::SMALLINT AS dt, map_mlogId.code AS c,
        map_creatorId.code AS k, map_talkId.code AS t, map_card_type.code AS ty, impressTime, impressPosition, y, card_ctr, card_ctr_cr,
        video::SMALLINT AS video, card_impr, card_age,
        slot::SMALLINT AS slot, depth::SMALLINT AS depth
        FROM imp {j('userId')} {j('mlogId')} {j('creatorId')} {j('talkId')} {j('card_type')}
        WHERE {where} ORDER BY u, impressTime, slot, row_id""").df()   # row_id is unique: the row order is fully determined

def scroll_groups_sql(con):
    """Scroll-depth group per user code, from the share of cards shown beyond the first page (place 4 or later) on days
    1-15: 0 = first page only, 1 = beyond it for under half of the cards, 2 = for half or more; 3 = no days 1-15 history.
    (Places come from thesis.positions; most users never go beyond the first page, so equal-sized terciles of mean
    position would split them arbitrarily.)"""
    e = con.execute("""SELECT m.code AS u, avg((impressPosition >= 4)::INT) AS beyond
                       FROM imp JOIN map_userId m ON m.v = imp.userId WHERE dt <= 15 GROUP BY 1""").df()
    n = con.execute("SELECT count(*) FROM map_userId").fetchone()[0]
    grp = np.full(n, 3); grp[e.u.values] = np.where(e.beyond.values == 0, 0, np.where(e.beyond.values < 0.5, 1, 2))
    return grp

def fit(sizes, d, slot=True, depth=True, by_group=True, hierarchy=True, iters=80, pooled_sf=None):
    """Fit on rows d (needs columns u, c, k, t, ty, slot, depth, grp, y). sizes = number of codes per id (from prepare).
    pooled_sf: slot factors of the same model fitted with by_group=False, if already available (otherwise fitted here)."""
    NU, NC, NK, NT, NY = (sizes[x] for x in ("userId", "mlogId", "creatorId", "talkId", "card_type"))
    u, c, k, s, gd, y = d.u.values, d.c.values, d.k.values, d.slot.values, d.depth.values, d.y.values
    t, ty = d.t.values, d.ty.values
    sg = (np.minimum(d.grp.values, 2) * 6 + s) if by_group else s; ns = 18 if by_group else 6
    card_k = np.zeros(NC, int); card_k[c] = k
    k_ty = np.zeros(NK, int); k_ty[k] = ty          # a creator's / topic's type: that of (one of) its cards
    t_ty = np.zeros(NT, int); t_ty[t] = ty
    # clicks per level do not change between iterations: count them once
    Y_ty, Y_k, Y_t, Y_c = np.bincount(ty, y, NY), np.bincount(k, y, NK), np.bincount(t, y, NT), np.bincount(c, y, NC)
    Y_u, Y_sg, Y_gd = np.bincount(u, y, NU), np.bincount(sg, y, ns), np.bincount(gd, y, 5)
    base = y.mean(); uf, cf, sf, gf = np.ones(NU), np.ones(NC), np.ones(ns), np.ones(5)
    kf, tyf, tf = np.ones(NK), np.ones(NY), np.ones(NT)
    UF, SF, GF = uf[u], sf[sg], gf[gd]                                   # current factors of every row
    for _ in range(iters):
        x = base * UF * SF * GF                                         # expected clicks before the card effect
        if hierarchy:
            tyf = (Y_ty + A) / (np.bincount(ty, x, NY) + A)
            kf = (Y_k + A * tyf[k_ty]) / (np.bincount(k, x, NK) + A)
            tf = (Y_t + A * tyf[t_ty]) / (np.bincount(t, x, NT) + A)
            cf = (Y_c + A * kf[card_k]) / (np.bincount(c, x, NC) + A)
        else:
            cf = (Y_c + A) / (np.bincount(c, x, NC) + A)
        CF = cf[c]
        uf = (Y_u + A) / (np.bincount(u, base * CF * SF * GF, NU) + A); UF = uf[u]
        if slot:
            sf = Y_sg / np.maximum(np.bincount(sg, base * UF * CF * GF, ns), 1e-12)
            sf = (sf.reshape(-1, 6) / sf.reshape(-1, 6)[:, :1]).ravel(); SF = sf[sg]
        if depth:
            gf = Y_gd / np.bincount(gd, base * UF * CF * SF, 5); gf = gf / gf[0]; GF = gf[gd]
    seen_c = np.bincount(c, minlength=NC) > 0; seen_k = np.bincount(k, minlength=NK) > 0; seen_t = np.bincount(t, minlength=NT) > 0
    pooled = None
    if by_group and slot:  # pooled slot factors for users with no days 1-15 history
        pooled = pooled_sf if pooled_sf is not None else \
            fit(sizes, d, slot=True, depth=depth, by_group=False, hierarchy=hierarchy, iters=iters)["sf"]
    return dict(base=base, cf=cf, kf=kf, tf=tf, tyf=tyf, sf=sf, gf=gf, by_group=by_group, hierarchy=hierarchy,
                seen_c=seen_c, seen_k=seen_k, seen_t=seen_t, pooled_sf=pooled)

def card_effect(m, d):
    """Card effect for rows d; unseen cards fall back to creator, topic, type (hierarchy) or 1 (flat)."""
    c = d.c.values; out = m["cf"][c].copy(); new = ~m["seen_c"][c]
    if m["hierarchy"]:
        kk, tt = d.k.values, d.t.values
        fk = m["seen_k"][kk]; ft = m["seen_t"][tt]
        out[new] = np.where(fk[new], m["kf"][kk][new], np.where(ft[new], m["tf"][tt][new], m["tyf"][d.ty.values][new]))
    else:
        out[new] = 1.0
    return out

def slot_factor(m, d):
    if not m["by_group"]: return m["sf"][d.slot.values]
    gi = np.minimum(d.grp.values, 2) * 6 + d.slot.values
    return np.where(d.grp.values == 3, m["pooled_sf"][d.slot.values], m["sf"][gi])

def user_factor(m, early, NU):
    """User effect from the user's own rows (e.g. days 1-15), with cards, slots and depth fixed; shrunk toward 1."""
    x = m["base"] * card_effect(m, early) * slot_factor(m, early) * m["gf"][early.depth.values]
    return (np.bincount(early.u.values, early.y.values, NU) + A) / (np.bincount(early.u.values, x, NU) + A)

def save_model(con, m, path):
    """Save a fitted model keyed by ids (not by codes, which depend on the users a script loads): one parquet of card,
    creator, topic and type effects, and a json of the slot, page-depth and base factors."""
    import json
    rows = []
    for col, f, seen in (("mlogId", "cf", "seen_c"), ("creatorId", "kf", "seen_k"), ("talkId", "tf", "seen_t"), ("card_type", "tyf", None)):
        ids = con.execute(f"SELECT v FROM map_{col} ORDER BY code").df().v
        rows.append(pd.DataFrame({"level": col, "id": ids.values, "effect": m[f][:len(ids)],
                                  "seen": m[seen][:len(ids)] if seen else True}))
    pd.concat(rows, ignore_index=True).to_parquet(f"{path}.parquet", index=False)
    meta = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()
            if k in ("base", "sf", "gf", "by_group", "hierarchy", "pooled_sf")}
    with open(f"{path}.json", "w") as fh: json.dump(meta, fh)

def load_model(con, path, sizes):
    """Load a model saved by save_model onto the codes of the current prepare() (ids not seen when saving: unseen)."""
    import json
    with open(f"{path}.json") as fh: meta = json.load(fh)
    m = {k: (np.array(v) if isinstance(v, list) else v) for k, v in meta.items()}
    eff = pd.read_parquet(f"{path}.parquet")
    for col, f, seen in (("mlogId", "cf", "seen_c"), ("creatorId", "kf", "seen_k"), ("talkId", "tf", "seen_t"), ("card_type", "tyf", None)):
        n = sizes[col]; arr = np.ones(n); flag = np.zeros(n, bool)
        mp = con.execute(f"SELECT v, code FROM map_{col}").df().merge(eff[eff.level == col], left_on="v", right_on="id", how="inner")
        arr[mp.code.values] = mp.effect.values; flag[mp.code.values] = mp.seen.values.astype(bool)
        m[f] = arr
        if seen: m[seen] = flag
    return m
