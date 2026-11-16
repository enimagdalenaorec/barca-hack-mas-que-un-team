"""Fetch every training match, convert each one, delete the raw download.

    python -m solution.download_external            # Metrica + IDSSE, whatever is missing
    python -m solution.download_external --list     # what is converted, and its size

Raw files already on disk are used in place and never deleted: data/external/metrica/
(Sample_Game_1..3, original names) and data/external/idsse/ (DFL_02_01_matchinformation_* and
DFL_04_03_positions_raw_observed_*, original names). Anything missing is downloaded from the
pinned revisions below, one match at a time: download -> convert -> delete the raw files. Peak
extra disk is one raw match (~400 MB for IDSSE); what stays is ~30 MB per match.

All 7 IDSSE matches are listed; which of them the model trains on is model.IDSSE_TRAIN.
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import external_games as ext  # noqa: E402

# Params
# Pinned so the training set cannot change under the submission.
METRICA_URL = ("https://raw.githubusercontent.com/metrica-sports/sample-data/"
               "e706dd506b360d69d9d123d5b8026e7294b13996/data/")
METRICA_FILES = {
    1: ["Sample_Game_1_RawEventsData.csv", "Sample_Game_1_RawTrackingData_Home_Team.csv",
        "Sample_Game_1_RawTrackingData_Away_Team.csv"],
    2: ["Sample_Game_2_RawEventsData.csv", "Sample_Game_2_RawTrackingData_Home_Team.csv",
        "Sample_Game_2_RawTrackingData_Away_Team.csv"],
    3: ["Sample_Game_3_events.json", "Sample_Game_3_metadata.xml", "Sample_Game_3_tracking.txt"],
}
# Pinned so the training set cannot change under the submission.
IDSSE_URL = ("https://huggingface.co/datasets/pysport/idsse-data/resolve/"
             "a715a38dfbaf5f58e431727c2b78d174101a703c/")
SC_RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/"
SC_LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/"
MATCHES = {
    "idsse": [
        ("DFL-COM-000001", "J03WMX"), ("DFL-COM-000001", "J03WN1"),   # Bundesliga
        ("DFL-COM-000002", "J03WOH"), ("DFL-COM-000002", "J03WOY"),   # 2. Bundesliga
        ("DFL-COM-000002", "J03WPY"), ("DFL-COM-000002", "J03WQQ"),
        ("DFL-COM-000002", "J03WR9"),
    ],
    # SkillCorner was tried and dropped: training without it is better on every held-out match
    # (solution/model.py, SOURCES). Kept fetchable with --skillcorner to rerun that experiment.
    "sc": [
        "1874553", "1886347", "1899585", "1925299", "1927964",
    ],
}


def fetch(url, path):
    """curl, because it follows the LFS / CDN redirects and shows a clear failure."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    subprocess.run(["curl", "-sfL", "--retry", "3", url, "-o", path], check=True)
    return path


def get_metrica(n):
    """Metrica sample game n: download into data/sample/, convert (solution/sample_games.py),
    then delete the raw files like every other source."""
    from solution import sample_games
    if os.path.isfile(sample_games.cache_path(n)):
        return
    local = os.path.join(sample_games.LOCAL_DIR, "Sample_Game_{}".format(n))
    if all(os.path.isfile(os.path.join(local, f)) for f in METRICA_FILES[n]):
        sample_games.SAMPLE_DIR = sample_games.LOCAL_DIR
        sample_games.load(n)  # converts and caches; the local raw files are kept
        return
    sample_games.SAMPLE_DIR = os.path.join(sample_games.DATA_DIR, "sample")
    folder = os.path.join(sample_games.SAMPLE_DIR, "Sample_Game_{}".format(n))
    os.makedirs(folder, exist_ok=True)
    paths = [fetch(METRICA_URL + "Sample_Game_{}/{}".format(n, f), os.path.join(folder, f))
             for f in METRICA_FILES[n]]
    try:
        sample_games.load(n)
    finally:
        for path in paths:
            os.remove(path)


def get_idsse(competition, match):
    out = ext.cache_path("idsse", match)
    if os.path.isfile(out):
        return out
    local = ext.local_idsse(competition, match)
    if local:
        arrays = ext.convert_idsse(*local)  # the local raw files are kept
        ext.np.savez_compressed(out, **arrays)
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
    parser.add_argument("--skillcorner", action="store_true",
                        help="also fetch SkillCorner (only for rerunning experiments)")
    args = parser.parse_args()

    if not args.list:
        os.makedirs(ext.CACHE_DIR, exist_ok=True)
        for n in METRICA_FILES:
            print("metrica {} ...".format(n), flush=True)
            get_metrica(n)
        for competition, match in MATCHES["idsse"]:
            print("idsse {} ...".format(match), flush=True)
            get_idsse(competition, match)
        for match in (MATCHES["sc"] if args.skillcorner else []):
            print("sc {} ...".format(match), flush=True)
            get_skillcorner(match)

    total = 0
    for source, match in converted():
        size = os.path.getsize(ext.cache_path(source, match))
        total += size
        print("  {:<6} {:<8} {:>6.1f} MB".format(source, match, size / 1e6))
    print("  total {:.0f} MB".format(total / 1e6))
