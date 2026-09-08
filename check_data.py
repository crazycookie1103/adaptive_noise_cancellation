"""
Sanity-checks every .wav file under data/<class>/ before you run train.py --
catches sample-rate/channel mismatches, too-short clips, and silent files
up front instead of failing partway through a training run.

Usage:
    python3 check_data.py data/            # just report problems
    python3 check_data.py data/ --fix      # also auto-convert fixable files
                                            # (wrong sample rate / channel
                                            # count) in place, via ffmpeg
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

REQUIRED_SR = 16000
MIN_DURATION_S = 0.3  # matches WINDOW_MS in features.py
CLASSES = ["stationary", "non_stationary", "speech"]


def convert_file(path):
    """
    Re-encodes `path` in place to 16kHz mono via ffmpeg. Writes to a temp
    file first and only replaces the original once ffmpeg succeeds, so a
    failed conversion never leaves you with a corrupted/truncated file.
    Returns True on success.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ar", str(REQUIRED_SR), "-ac", "1", tmp_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            print(f"    ffmpeg failed on {path}: {result.stderr.decode(errors='replace')[-300:]}")
            os.remove(tmp_path)
            return False

        # Verify the conversion actually landed at the right rate before
        # committing -- don't trust ffmpeg's exit code alone.
        info = sf.info(tmp_path)
        if info.samplerate != REQUIRED_SR or info.channels != 1:
            print(f"    conversion of {path} produced unexpected format, skipping")
            os.remove(tmp_path)
            return False

        shutil.move(tmp_path, path)
        return True
    except Exception as e:
        print(f"    error converting {path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def check_dataset(data_dir, fix=False):
    if fix and shutil.which("ffmpeg") is None:
        print("[error] --fix requires ffmpeg, but it's not on PATH.")
        print("        Install it with: sudo apt install ffmpeg")
        sys.exit(1)

    problems = []
    fixable_paths = []
    counts = {}
    fixed_count = 0

    for class_name in CLASSES:
        files = sorted(glob.glob(os.path.join(data_dir, class_name, "*.wav")))
        counts[class_name] = len(files)

        if not files:
            problems.append(f"[{class_name}] no .wav files found")
            continue

        total_duration = 0.0
        for path in files:
            try:
                info = sf.info(path)
            except Exception as e:
                problems.append(f"[{class_name}] {path}: couldn't read ({e})")
                continue

            needs_fix = False
            if info.samplerate != REQUIRED_SR:
                problems.append(
                    f"[{class_name}] {path}: sample rate {info.samplerate} Hz, "
                    f"need {REQUIRED_SR} Hz"
                )
                needs_fix = True
            if info.channels != 1:
                problems.append(
                    f"[{class_name}] {path}: {info.channels} channels, need mono"
                )
                needs_fix = True
            if needs_fix:
                fixable_paths.append(path)

            duration = info.frames / info.samplerate
            if duration < MIN_DURATION_S:
                problems.append(
                    f"[{class_name}] {path}: only {duration:.2f}s, "
                    f"need >= {MIN_DURATION_S}s (NOT auto-fixable -- too short "
                    f"means the recording itself needs to be longer)"
                )
            total_duration += duration

            # quick silence check
            audio, _ = sf.read(path, dtype="float32")
            if np.abs(audio).max() < 1e-4:
                problems.append(
                    f"[{class_name}] {path}: appears to be silent "
                    f"(NOT auto-fixable -- re-record or discard)"
                )

        print(f"{class_name}: {len(files)} files, {total_duration:.1f}s total")

    if fix and fixable_paths:
        print(f"\nConverting {len(fixable_paths)} file(s) to {REQUIRED_SR} Hz mono...")
        for path in fixable_paths:
            print(f"  fixing {path} ...", end=" ")
            if convert_file(path):
                print("done")
                fixed_count += 1
            else:
                print("FAILED (left unchanged)")

    print()
    if problems:
        print(f"Found {len(problems)} problem(s) on this pass:\n")
        for p in problems:
            print(" -", p)
        if fix:
            print(f"\nAuto-fixed {fixed_count}/{len(fixable_paths)} "
                  f"sample-rate/channel issues.")
            if fixed_count < len(fixable_paths) or any(
                "NOT auto-fixable" in p for p in problems
            ):
                print("Re-run without --fix to confirm remaining issues, "
                      "and handle the NOT auto-fixable ones manually "
                      "(re-record, discard, or trim).")
        else:
            print("\nRun with --fix to auto-convert sample-rate/channel "
                  "issues (silence and too-short files still need manual "
                  "attention).")
    else:
        print("All files look good.")

    # Rough class balance check
    total = sum(counts.values())
    if total > 0:
        print("\nClass balance (file counts, not duration):")
        for c, n in counts.items():
            print(f"  {c}: {n} ({100*n/total:.0f}%)")
        if max(counts.values()) > 3 * max(1, min(counts.values())):
            print("\n[warn] one class has 3x+ more files than another -- "
                  "train.py applies class weighting to compensate, but more "
                  "balanced data is still better if you can get it.")


if __name__ == "__main__":
    args = sys.argv[1:]
    fix = "--fix" in args
    positional = [a for a in args if a != "--fix"]
    data_dir = positional[0] if positional else "data/"
    check_dataset(data_dir, fix=fix)
