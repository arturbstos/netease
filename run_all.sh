#!/usr/bin/env bash
# Execute every notebook top to bottom, in order, saving the results into the notebooks and outputs/.
#
#   ./run_all.sh                 all notebooks (creates .venv on the first run)
#   ./run_all.sh --from 03       start at notebook 03 (earlier outputs must exist)
#   ./run_all.sh --only A1       a single notebook
#   ./run_all.sh --sample 10     panel notebooks on the 10% user sample (quick test; results get the suffix _s10)
#   ./run_all.sh --force         recompute the slow steps even if their results are saved in outputs/
#                                (without it, a notebook reads saved results and computes only the missing ones;
#                                see run_heavy.py to run the slow steps without the notebooks)
#
# Settings (optional): THESIS_DATA (folder with csv_data/ and parquet/), THESIS_MEM (default 8GB),
# THESIS_THREADS (default 8). On a Mac, keep it awake: caffeinate -i ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

FROM=""; ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --sample) export THESIS_SAMPLE="$2"; shift 2 ;;
    --force) export THESIS_FORCE=1; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done
ORDER=(01 02 03 04 A1 A2 05)   # the figures (05) come last: they read the results of every other notebook

# ---- Python environment -------------------------------------------------------
PY_BIN="${PYTHON:-python3}"
"$PY_BIN" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' || {
  echo "Python 3.10 or newer is required. On a Mac: brew install python@3.12"; exit 1; }
if [[ "$(uname)" == "Darwin" ]] && ! [[ -e /opt/homebrew/opt/libomp/lib/libomp.dylib || -e /usr/local/opt/libomp/lib/libomp.dylib ]]; then
  echo "LightGBM needs OpenMP on macOS. Install it once with: brew install libomp"; exit 1
fi
if ! [[ -x .venv/bin/python ]]; then
  echo "== creating .venv"; "$PY_BIN" -m venv .venv; .venv/bin/pip install --quiet --upgrade pip
fi
.venv/bin/pip install --quiet -r requirements.txt -e .
PY=.venv/bin/python

mkdir -p logs
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== run_all.sh $(date) | sample ${THESIS_SAMPLE:-100}% | memory ${THESIS_MEM:-8GB} | threads ${THESIS_THREADS:-8} | recompute saved steps: ${THESIS_FORCE:-0}"

run_nb () {  # execute one notebook in place; on an error the notebook is saved with the failing cell's output
  "$PY" - "$1" <<'PYEOF'
import sys, nbformat
from nbclient import NotebookClient
path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
try:
    NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": "notebooks"}}).execute()
finally:
    nbformat.write(nb, path)
PYEOF
}

started=0; [[ -z "$FROM" ]] && started=1
for id in "${ORDER[@]}"; do
  [[ "$id" == "$FROM" ]] && started=1
  if [[ -n "$ONLY" && "$id" != "$ONLY" ]] || [[ -z "$ONLY" && $started -eq 0 ]]; then continue; fi
  nb=$(ls notebooks/${id}_*.ipynb)
  echo; echo "== $nb ($(date +%H:%M:%S))"; t0=$SECONDS
  run_nb "$nb"
  echo "== finished in $((SECONDS - t0)) s"
done
echo; echo "== all done $(date). Results in outputs/, log in $LOG"
