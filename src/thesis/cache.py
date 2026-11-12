"""Saved results of the slow steps (model fits, bootstraps, impression scans).

Each slow step is a plain function in thesis.heavy that writes its results to outputs/. A notebook asks for a step
with ensure("02.models"): if every output file of the step exists, the saved results are read back; if any is
missing, the step runs (after the steps it needs) and saves them. Nothing is recomputed while the files exist, so
after a change to the code or the data, recompute with  python run_heavy.py --force <step>  (or delete the files).

    from thesis.cache import ensure, read
    R = ensure("02.models")              # {"10_panel_models": {...}, "10_predictions": Path(...)}
    D = read("10_predictions.parquet")

Command line (also used by run_heavy.py):  python -m thesis.cache [--force] [--list] step-or-prefix ...
THESIS_FORCE=1 recomputes every step a notebook asks for directly (not the steps it needs), once per process.
"""
import json, os, sys, time
from pathlib import Path
import pandas as pd
from thesis.config import O, TAG

STEPS = {}       # name -> dict(fn, outputs, needs, doc)
_DONE = set()    # steps computed in this process (so a forced step runs once)
FORCE = os.environ.get("THESIS_FORCE", "0") == "1"


def step(name, outputs, needs=()):
    """Register fn as the slow step `name`, writing `outputs` (file names without the sample suffix)."""
    def deco(fn):
        STEPS[name] = dict(fn=fn, outputs=list(outputs), needs=list(needs), doc=(fn.__doc__ or "").strip())
        return fn
    return deco


def _register():
    import thesis.heavy  # noqa: F401  (importing registers every step)


def path(fname):
    """Output path of a file name such as '10_tuning.json' (the sample suffix is added: 10_tuning_s10.json)."""
    p = Path(fname)
    return O / f"{p.stem}{TAG}{p.suffix}"


def read(fname):
    """Read one output: json -> dict, csv -> DataFrame, parquet -> DataFrame."""
    p = path(fname)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    if p.suffix == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def write_json(fname, obj):
    path(fname).write_text(json.dumps(obj, indent=2, default=str))


def missing(name):
    _register()
    return [f for f in STEPS[name]["outputs"] if not path(f).exists()]


def run(name):
    """Compute a step now (its missing prerequisites first) and save its outputs."""
    _register()
    for dep in STEPS[name]["needs"]:
        ensure(dep, force=False)
    t0 = time.time()
    print(f"[{name}] computing", flush=True)
    STEPS[name]["fn"]()
    _DONE.add(name)
    print(f"[{name}] saved in {time.time() - t0:.0f} s", flush=True)


def ensure(name, force=None):
    """Saved results of a step, computed first if any output is missing (or if force). Returns {file stem: result}
    with json read as dicts and csv as DataFrames; parquet files are returned as paths (read them with read()). When two
    outputs share a stem (a .json and a .csv), both are keyed by their full file name."""
    _register()
    force = FORCE if force is None else force
    if name not in _DONE and (force or missing(name)):
        run(name)
    else:
        print(f"[{name}] saved results loaded", flush=True)
    outs = STEPS[name]["outputs"]
    stems = [Path(f).stem for f in outs]
    out = {}
    for f, st in zip(outs, stems):
        key = st if stems.count(st) == 1 else f          # e.g. 26_history_curve.json and 26_history_curve.csv
        out[key] = path(f) if f.endswith(".parquet") else read(f)
    return out


def select(args):
    """Step names matching each argument: an exact name, or a prefix such as '02' or '03.click'."""
    _register()
    out = []
    for a in args:
        hits = [s for s in STEPS if s == a or s.startswith(a)]
        if not hits:
            raise SystemExit(f"no step matches {a!r}; steps: {', '.join(STEPS)}")
        out += [h for h in hits if h not in out]
    return out


def cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m thesis.cache")
    ap.add_argument("steps", nargs="*")
    ap.add_argument("--force", action="store_true", help="recompute even if the outputs exist")
    ap.add_argument("--list", action="store_true", help="list the steps and whether their outputs exist")
    a = ap.parse_args(argv)
    _register()
    names = select(a.steps) if a.steps else list(STEPS)
    if a.list:
        for s in names:
            miss = missing(s)
            print(f"{s:24s} {'missing: ' + ', '.join(miss) if miss else 'saved'}")
        return
    for s in names:
        if a.force or missing(s):
            run(s)
        else:
            print(f"[{s}] already saved")


if __name__ == "__main__":
    from thesis import cache as _c   # use the imported module, so the steps register in one registry
    _c.cli(sys.argv[1:])
