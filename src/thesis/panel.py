"""Panel data (built in notebook 03): loading, engineered ratios, feature list, user and time splits, inputs for the
linear model, user weights and the LightGBM settings shared by every panel model.
Split: md5(userId) bucket < 70 train, 70-84 validation, >= 85 test; train and validation use origin days 1-15,
test uses origin days 16-23 (unseen users, later period)."""
import hashlib, json, os
import numpy as np, pandas as pd
from thesis.config import P, O, TAG, SEED

DROP = {"userId", "origin_day", "bucket", "y_visit_days", "y_impr", "y_clicks", "h_pos_sum", "h_video_seen", "h_pop_sum"}

def load():
    df = pd.read_parquet(P / f"panel{TAG}.parquet")
    u = df.userId.astype("category")  # hash each user once, not each row
    b = [int(hashlib.md5(x.encode()).hexdigest()[:8], 16) % 100 for x in u.cat.categories]
    df["bucket"] = np.asarray(b, dtype=np.int64)[u.cat.codes.values]
    for w in ("w7", "h"):
        df[f"{w}_ctr"] = df[f"{w}_clicks"] / df[f"{w}_impr"]
        df[f"{w}_click_vs_expected"] = (df[f"{w}_clicks"] + 1) / (df[f"{w}_expected_clicks"] + 1)
    df["h_mean_pos"] = df.h_pos_sum / df.h_impr
    df["h_video_share"] = df.h_video_seen / df.h_impr
    df["h_mean_card_pop"] = df.h_pop_sum / df.h_impr
    df["h_shown_card_ctr"] = df.h_expected_clicks / df.h_impr
    feats = [c for c in df.columns if c not in DROP]
    tr = df[(df.bucket < 70) & (df.origin_day <= 15)]
    va = df[(df.bucket >= 70) & (df.bucket < 85) & (df.origin_day <= 15)]
    te = df[(df.bucket >= 85) & (df.origin_day >= 16)]
    LOG_COLS.clear(); LOG_COLS.update(c for c in feats if tr[c].astype(float).max() > 1.5)
    return df, feats, tr, va, te

RATIOS = ["w7_ctr", "w7_click_vs_expected", "h_ctr", "h_click_vs_expected", "h_mean_pos", "h_video_share",
          "h_mean_card_pop", "h_shown_card_ctr"]   # the columns load() adds, in its order


def feature_names():
    """The feature list of load() (FEATS), read from the panel's schema without loading the panel."""
    import pyarrow.parquet as pq
    cols = pq.read_schema(P / f"panel{TAG}.parquet").names
    cols = cols + [c for c in ["bucket"] + RATIOS if c not in cols]
    return [c for c in cols if c not in DROP]

def target_sets(target, tr, va, te):
    """R: return (at least one visit day in the next 7), all origins. C: clicks out of cards shown, origins with at least
    one card shown ahead. Both as k successes out of n trials (n = 1 for return)."""
    if target == "R":
        return (tr, va, te), (lambda d: (d.y_visit_days > 0).astype(float)), (lambda d: pd.Series(1.0, index=d.index))
    return tuple(d[d.y_impr > 0] for d in (tr, va, te)), (lambda d: d.y_clicks), (lambda d: d.y_impr)


def rule_return(a, c, col="w7_visit_days"):
    """One-feature rule for return: the training return rate of users with the same number of visit days (col)."""
    rate = (a.y_visit_days > 0).groupby(a[col]).mean()
    return c[col].map(rate).fillna((a.y_visit_days > 0).mean()).clip(1e-4, 1 - 1e-4).values


# Scorecard: a logistic regression on a few counts the platform already keeps (return: 10, clicks: 12)
SCORECARD = {"R": ["w7_visit_days", "h_visit_days", "w7_sessions", "h_sessions", "gap_since_prev_visit", "history_days",
                   "w7_impr", "h_impr", "h_ctr", "h_mean_pos"]}
SCORECARD["C"] = SCORECARD["R"] + ["h_click_vs_expected", "h_shown_card_ctr"]

# LightGBM settings shared by every panel model; notebooks override only what differs. The tree size (num_leaves,
# min_data_in_leaf) is chosen per outcome on validation users in notebook 02 and read back with lgb_params(target).
LGB_PARAMS = dict(objective="cross_entropy", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, seed=SEED, verbose=-1,
                  deterministic=True, force_col_wise=True)   # same result on every run with the same number of threads

# Grids searched on validation users in notebook 02 (LightGBM tree size, logistic regression penalty)
LGB_GRID = [dict(num_leaves=nl, min_data_in_leaf=md) for nl in (7, 15, 63, 255) for md in (50, 200, 1000, 5000)]
LOGIT_C_GRID = (0.01, 0.1, 1.0, 10.0)


def _tuning():
    f = O / f"10_tuning{TAG}.json"
    if not f.exists():
        raise FileNotFoundError(f"{f.name} not found: run notebook 02 (section 2) first")
    return json.loads(f.read_text())


def lgb_params(target):
    """LightGBM settings for outcome 'R' or 'C': the shared settings with the tree size chosen in notebook 02."""
    return dict(LGB_PARAMS, **_tuning()[target]["lightgbm"]["chosen"])


def logistic_C(target):
    """Inverse L2 penalty of the logistic regression for outcome 'R' or 'C', chosen in notebook 02."""
    return _tuning()[target]["logistic"]["chosen_C"]

# Feature groups (Table 4.2 of the thesis), by the feature's name without its window prefix (w1_, w3_, w7_, w14_, h_)
FEATURE_GROUPS = {
    "activity": {"impr", "sessions", "visit_days"},
    "clicks": {"clicks", "ctr", "click_vs_expected", "viewtime"},
    "social": {"likes", "shares", "comments", "viewcomments", "homepage"},
    "exposure": {"expected_clicks", "video_share", "mean_pos", "mean_card_pop", "shown_card_ctr"},
    "calendar_account": {"history_days", "gap_since_prev_visit", "tenure", "origin_weekday"}}


def feature_group(f):
    stem = f.split("_", 1)[1] if f.split("_")[0] in ("w1", "w3", "w7", "w14", "h") else f
    for name, stems in FEATURE_GROUPS.items():
        if stem in stems:
            return name
    raise ValueError(f"feature {f} has no group")

LOG_COLS = set()  # count-like columns to log-transform for the linear model; decided on the training split in load()

def lin(d, feats):
    """Inputs for the linear model: log1p of count-like columns (chosen on training data), missing -> 0, plus an
    indicator for 'no earlier visit day in the log' so that a missing gap is not read as a gap of 0 days."""
    X = d[feats].copy().astype(float)
    for c in feats:
        if c in LOG_COLS: X[c] = np.log1p(X[c].clip(lower=0))
    if "gap_since_prev_visit" in feats:
        X["no_prev_visit_day"] = X.gap_since_prev_visit.isna().astype(float)
    return X.fillna(0)

# User weighting in training (appendix A1 tests this choice). Off by default; set THESIS_USER_WEIGHT=1 to switch on.
USER_WEIGHT = os.environ.get("THESIS_USER_WEIGHT", "0") == "1"

def past_user_w(d):
    """1 / (the user's visit days in the log up to and including the prediction date). The weight uses only the past, so
    it is known on the prediction date and does not depend on whether the user comes back later. Frequent users still
    weigh more than rare ones, but far less than with one weight per row (a user's total weight grows with the log of
    the visit days rather than with their number)."""
    return 1.0 / d.h_visit_days.clip(lower=1).values

def user_w(d):
    """Multiplier per row: past_user_w(d), or 1 when user weighting is off."""
    if not USER_WEIGHT:
        return np.ones(len(d))
    return past_user_w(d)


def first_origin_tiers(S, n_tiers=10):
    """One row per user at the user's first origin in S (scores of notebook 02), with the activity tier: the tenth of
    users by expected click probability, ranked against the other users whose first origin is on the same day
    (1 = least active, 10 = most active). Used by notebook 04 and appendix A2."""
    first = S.sort_values(["userId", "origin_day"]).drop_duplicates("userId").copy()
    r = first.groupby("origin_day").expected_click_prob.rank(pct=True, method="first")
    first["tier"] = np.minimum(np.ceil(r * n_tiers).astype(int), n_tiers)
    return first
