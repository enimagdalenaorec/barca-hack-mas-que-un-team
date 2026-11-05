"""Video frame <-> time in the tactical video, for finding a moment in the Drive player.

    python -m solution.timecode 18684           # -> 0:12:27.36  (1st half, 05:00.0 of play)
    python -m solution.timecode 1:23:45         # -> frame 125625
    python -m solution.timecode 0:12:27.36

Frame f is shown at f / 25 seconds, because the submission row, the homography row and the video
frame are the same index (DATA.md). "Match clock" counts from each half's kickoff, which is how a
moment is usually described ("Barcelona scored in the 23rd minute").
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, anchor_video_frame, half_of_video_frame  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
# Kickoff video frames. The first half is graded from its kickoff; the second half is graded from
# 110808 but kicks off at 113742 (FAQ.md), and the match clock starts at the kickoff.
KICKOFFS = {1: anchor_video_frame(GAMES[GAME]["kickoff"]),
            2: anchor_video_frame(GAMES[GAME]["kickoff_2nd"])}
HALF_START_MINUTE = {1: 0, 2: 45}


def frame_to_time(frame):
    """'H:MM:SS.ss' position of a frame in the video."""
    seconds = frame / float(FPS)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return "{:d}:{:02d}:{:05.2f}".format(int(hours), int(minutes), seconds)


def time_to_frame(text):
    """Frame shown at 'H:MM:SS(.ss)', 'MM:SS(.ss)' or plain seconds."""
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return int(round(seconds * FPS))


def match_clock(frame, game=GAME):
    """('1st half', "23:05") style description, or None outside the two halves."""
    half = half_of_video_frame(game, frame)
    if half is None:
        return None
    elapsed = (frame - KICKOFFS[half]) / float(FPS)
    if elapsed <= -1:  # the 1st-half anchor sits one frame after the first graded frame
        return "{} half, {:.1f} s before kickoff".format("1st" if half == 1 else "2nd", -elapsed)
    elapsed = max(elapsed, 0.0)
    minutes, seconds = divmod(elapsed, 60)
    return "{} half, match clock {:02d}:{:04.1f}".format(
        "1st" if half == 1 else "2nd", int(minutes) + HALF_START_MINUTE[half], seconds)


def describe(frame):
    clock = match_clock(frame)
    return "frame {}  ->  video {}   ({})".format(
        frame, frame_to_time(frame), clock or "not graded: before, between or after the halves")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("values", nargs="+", help="frame numbers or video times with ':'")
    for value in parser.parse_args().values:
        print(describe(time_to_frame(value) if ":" in value else int(value)))
