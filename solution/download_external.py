"""Fetch the external training matches, convert each one, delete the raw download.

    python -m solution.download_external            # everything in MATCHES not yet converted
    python -m solution.download_external --list     # what is converted, and its size

One match at a time: download -> convert to data/cache/ext_*.npz -> delete the raw files. Peak
extra disk is one raw match (~400 MB for IDSSE); what stays is ~30 MB per match.

MATCHES is deliberately a subset (decided 2026-11-07 to keep the download near 1.9 GB): 4 of the
7 IDSSE matches, two per division, and 5 of the 10 SkillCorner matches. The rest are listed
commented out -- add them back if the learning curve says more data still helps.
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import external_games as ext  # noqa: E402

# Params
IDSSE_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
SC_RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/"
SC_LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/"
MATCHES = {
    "idsse": [
        ("DFL-COM-000001", "J03WMX"), ("DFL-COM-000001", "J03WN1"),   # Bundesliga
        ("DFL-COM-000002", "J03WOH"), ("DFL-COM-000002", "J03WOY"),   # 2. Bundesliga
        # ("DFL-COM-000002", "J03WPY"), ("DFL-COM-000002", "J03WQQ"),
        # ("DFL-COM-000002", "J03WR9"),
    ],
    "sc": [
        "1874553", "1886347", "1899585", "1925299", "1927964",
        # "1953632", and the other four listed in SkillCorner's data/matches.json
    ],
}


def fetch(url, path):
    """curl, because it follows the LFS / CDN redirects and shows a clear failure."""
    subprocess.run(["curl", "-sfL", "--retry", "3", url, "-o", path], check=True)
    return path


def get_idsse(competition, match):
    out = ext.cache_path("idsse", match)
    if os.path.isfile(out):
        return out
    stem = "{}_DFL-MAT-{}.xml".format(competition, match)
    info = fetch(IDSSE_URL + "DFL_02_01_matchinformation_" + stem,
                 os.path.join(ext.RAW_DIR, "idsse_{}_info.xml".format(match)))
    positions = fetch(IDSSE_URL + "DFL_04_03_positions_raw_observed_" + stem,
                      os.path.join(ext.RAW_DIR, "idsse_{}_positions.xml".format(match)))
    try:
        arrays = ext.convert_idsse(info, positions)
        ext.np.savez_compressed(out, **arrays)
    finally:
        for path in (info, positions):
            os.remove(path)
    return out


def get_skillcorner(match):
    out = ext.cache_path("sc", match)
    if os.path.isfile(out):
        return out
    meta = fetch(SC_RAW + "{0}/{0}_match.json".format(match),
                 os.path.join(ext.RAW_DIR, "sc_{}_match.json".format(match)))
    tracking = fetch(SC_LFS + "{0}/{0}_tracking_extrapolated.jsonl".format(match),
                     os.path.join(ext.RAW_DIR, "sc_{}_tracking.jsonl".format(match)))
    try:
        arrays = ext.convert_skillcorner(meta, tracking)
        ext.np.savez_compressed(out, **arrays)
    finally:
        for path in (meta, tracking):
            os.remove(path)
    return out


def converted():
    out = []
    for competition, match in MATCHES["idsse"]:
        out.append(("idsse", match))
    out += [("sc", match) for match in MATCHES["sc"]]
    return [(s, m) for s, m in out if os.path.isfile(ext.cache_path(s, m))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if not args.list:
        os.makedirs(ext.RAW_DIR, exist_ok=True)
        os.makedirs(ext.CACHE_DIR, exist_ok=True)
        for competition, match in MATCHES["idsse"]:
            print("idsse {} ...".format(match), flush=True)
            get_idsse(competition, match)
        for match in MATCHES["sc"]:
            print("sc {} ...".format(match), flush=True)
            get_skillcorner(match)

    total = 0
    for source, match in converted():
        size = os.path.getsize(ext.cache_path(source, match))
        total += size
        print("  {:<6} {:<8} {:>6.1f} MB".format(source, match, size / 1e6))
    print("  total {:.0f} MB".format(total / 1e6))
