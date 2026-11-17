"""Our possession pipeline. Everything here reads the provided data and nothing else.

    python -m solution.tracking         # parse the ATD once and cache it as numpy arrays
    python -m solution.possession_demo  # run the dead-ball/event state machine and show changes
    python -m solution.make_submission  # write the telemetry-driven frame,label CSV
    python -m solution.diagnostics F    # summarise a submission without any ground truth
    python -m solution.timecode N       # video frame <-> time in the video, for checking in Drive
"""
