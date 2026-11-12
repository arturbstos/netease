"""Shared state of the slow steps: the panel, loaded once per process."""
import numpy as np
from thesis.panel import load

_PANEL = {}


def panel():
    """(df, FEATS, tr, va, te) from thesis.panel.load(), loaded once per process. Do not modify the frames."""
    if _PANEL.get("p") is None or _PANEL["p"][0] is None:
        _PANEL["p"] = load()
    return _PANEL["p"]


def drop_full_panel():
    """Free the full panel frame (the splits stay loaded); a later panel() call reloads it."""
    if _PANEL.get("p") is not None:
        _PANEL["p"] = (None,) + tuple(_PANEL["p"][1:])


def ci(col, nd=4):
    return [round(float(np.quantile(col, q)), nd) for q in (0.025, 0.975)]


def shrunk_click_rate(a, c):
    """The click baseline: each user's past click rate shrunk toward the training mean (beta prior from the spread of
    click rates among training origins with 5+ cards shown)."""
    m = a[a.h_impr >= 5]
    r = m.h_clicks / m.h_impr
    mu, var = r.mean(), r.var()
    kk = mu * (1 - mu) / var - 1
    return ((c.h_clicks + mu * kk) / (c.h_impr + kk)).values
