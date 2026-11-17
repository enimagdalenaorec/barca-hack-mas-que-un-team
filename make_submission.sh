#!/usr/bin/env bash
# Regenerate the submission from scratch and check that a clean copy of the code reproduces it.
#
#     ./make_submission.sh "<team name>"            # build, validate, hash
#     ./make_submission.sh "<team name>" --clean    # also rebuild in a clean copy and compare
#
# Data: ./data, or MTAH_DATA_DIR if set (tactical/ with the match files, external/ with the
# training matches). Every command's output goes to logs/ and is NOT printed: the match data and
# anything derived from it stay on this machine. The terminal shows step names, ok/FAIL and
# SHA-256 hashes only. The configuration the pipeline chose is in logs/pipeline.log.
set -u
TEAM="${1:?usage: ./make_submission.sh \"<team name>\" [--clean]}"
CLEAN="${2:-}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA="${MTAH_DATA_DIR:-$ROOT/data}"
PY="$ROOT/.venv/bin/python"
LOGS="$ROOT/logs"
OUT="$ROOT/evaluation/results/$TEAM.csv"
mkdir -p "$LOGS" "$ROOT/evaluation/results"
cd "$ROOT"

step() {  # step <name> <command...>: run quietly, log everything, stop on failure
    local name="$1"; shift
    if "$@" >"$LOGS/$name.log" 2>&1; then
        echo "ok    $name"
    else
        echo "FAIL  $name   (see logs/$name.log)"; exit 1
    fi
}

# Fresh models every time: cached ones may come from older code or the other configuration.
rm -f "$DATA/cache/model.pkl" "$DATA/cache/dead_model.pkl"
step selftest   "$PY" -m solution.selftest
step external   "$PY" -m solution.download_external
step pipeline   "$PY" -m solution.pipeline --name "$TEAM"
step validate   "$PY" evaluation/validate_submission.py "$OUT"
HASH="$(shasum -a 256 "$OUT" | cut -d' ' -f1)"
echo "sha256 $HASH  evaluation/results/$TEAM.csv"

if [ "$CLEAN" = "--clean" ]; then
    # Tracked and new source files only (no data, caches or venv), a fresh virtual environment
    # from the pinned requirements, the raw data linked in, every cache rebuilt from scratch.
    TMP="$(mktemp -d)"
    git ls-files --cached --others --exclude-standard -z | rsync -a --from0 --files-from=- ./ "$TMP/"
    mkdir -p "$TMP/data/external"
    for src in metrica idsse; do
        [ -d "$DATA/external/$src" ] && ln -s "$DATA/external/$src" "$TMP/data/external/$src"
    done
    for d in "$DATA"/*/; do
        name="$(basename "$d")"
        case "$name" in external|cache) ;; *) ln -s "$d" "$TMP/data/$name" ;; esac
    done
    cd "$TMP"
    unset MTAH_DATA_DIR
    step clean_venv     python3 -m venv .venv
    step clean_install  .venv/bin/pip install -r requirements-solution.txt
    step clean_external .venv/bin/python -m solution.download_external
    step clean_pipeline .venv/bin/python -m solution.pipeline --name "$TEAM"
    HASH2="$(shasum -a 256 "$TMP/evaluation/results/$TEAM.csv" | cut -d' ' -f1)"
    echo "sha256 $HASH2  (clean copy)"
    cd "$ROOT"; rm -rf "$TMP"
    if [ "$HASH" = "$HASH2" ]; then echo "ok    reproduced byte for byte"; else echo "FAIL  clean copy differs"; exit 1; fi
fi
echo "upload: evaluation/results/$TEAM.csv"
