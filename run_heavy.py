#!/usr/bin/env python3
"""Compute the slow steps of the analysis and save their results in outputs/, without running the notebooks.

    python run_heavy.py                   every step whose results are missing, in notebook order
    python run_heavy.py 02 A1             the steps of notebooks 02 and A1 (missing results only)
    python run_heavy.py --force 03.click  recompute a step (or every step starting with the name) even if saved
    python run_heavy.py --list            the steps and whether their results are saved

The notebooks read the same saved results (thesis.cache.ensure), so after this they run in seconds. The steps of each
notebook run in their own process, so memory is freed between notebooks. Settings as for run_all.sh (THESIS_DATA,
THESIS_SAMPLE, THESIS_MEM, THESIS_THREADS, ...). Results are not recomputed when the code or data change: use --force.
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from thesis import cache  # noqa: E402

ORDER = ("02", "03", "A1")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", help="step names or prefixes (default: all)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    cache._register()
    names = cache.select(a.steps) if a.steps else list(cache.STEPS)
    if a.list:
        cache.cli(names + ["--list"])
        return
    # everything printed (also by the step processes) goes to the terminal and to logs/heavy_<time>.log
    logs = Path(__file__).resolve().parent / "logs"
    logs.mkdir(exist_ok=True)
    log = open(logs / f"heavy_{time.strftime('%Y%m%d_%H%M%S')}.log", "w")
    print(f"== run_heavy.py {' '.join(sys.argv[1:])} | log {log.name}", flush=True)
    t0 = time.time()
    for nb in ORDER:
        group = [s for s in names if s.split(".")[0] == nb]
        if not a.force:
            group = [s for s in group if cache.missing(s)]
        if not group:
            continue
        print(f"\n== {nb}: {', '.join(group)}", flush=True)
        log.write(f"\n== {nb}: {', '.join(group)}\n"); log.flush()
        cmd = [sys.executable, "-m", "thesis.cache", *group] + (["--force"] if a.force else [])
        src = str(Path(__file__).resolve().parent / "src")
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH")) if p)}
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line); log.write(line); log.flush()
        if p.wait():
            log.write(f"\n== {nb} failed (exit {p.returncode})\n"); log.close()
            sys.exit(f"== {nb} failed (exit {p.returncode}); log in logs/{log.name}")
    msg = f"\n== done in {time.time() - t0:.0f} s"
    print(msg, flush=True); log.write(msg + "\n"); log.close()


if __name__ == "__main__":
    main()
