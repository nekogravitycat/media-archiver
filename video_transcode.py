#!/usr/bin/env python3
"""
video_transcode.py -- Transcode videos to HEVC using HandBrakeCLI (NVIDIA NVENC).

Encodes files whose bitrate exceeds --max-bitrate and/or whose resolution
exceeds --max-resolution. Other files are skipped. Audio tracks are copied
without re-encoding.

Without --output-dir, originals are replaced in place: each file is encoded to
a temporary file alongside it, verified (exit code, non-empty output, and
duration matching the source), and only then does the original get deleted.
If the result is not smaller than the original, the original is kept.

Usage:
    python video_transcode.py <folder> [--max-bitrate <kbps>] [--max-resolution <res>] [options]

Options:
    --max-bitrate <kbps>    Target maximum bitrate in kbps
    --max-resolution <res>  Maximum resolution, e.g. 1080p, 4k, or 1920x1080.
                            The aspect ratio is kept and the box follows the
                            video's orientation (1920x1080 box for landscape,
                            1080x1920 for portrait). Never upscales.
                            (at least one of --max-bitrate / --max-resolution is required)
    --min-saving <percent>  Skip files whose estimated saving is below this percentage
    --output-dir <path>     Output directory (default: replace originals in place)
    --recursive             Process subdirectories recursively
    --execute               Apply transcoding (default: preview only)
    --dry-run               Print the exact HandBrakeCLI commands without running them
    --skip-errors           Continue past files that fail
    --handbrake <path>      Path to HandBrakeCLI binary (default: auto-detect)

Examples:
    python video_transcode.py ~/Videos --max-bitrate 8000
    python video_transcode.py ~/Videos --max-bitrate 8000 --max-resolution 1080p --execute
    python video_transcode.py ~/Videos --max-resolution 1280x720 --min-saving 20
    python video_transcode.py ~/Videos --max-bitrate 8000 --output-dir ~/Videos/out --execute
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# -- Supported extensions -----------------------------------------------------
SUPPORTED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".m4v", ".wmv",
    ".flv", ".mts", ".m2ts", ".3gp", ".webm",
}

# Suffix used for the in-progress file written next to (or in place of) the output.
TEMP_SUFFIX = ".transcoding.tmp.mp4"

# Output duration may differ slightly from the source; anything beyond this is
# treated as a failed transcode and the original is never deleted.
DURATION_TOLERANCE_SEC = 1.0

# Assumed bitrate for audio streams that report none (common in mkv/avi).
DEFAULT_AUDIO_KBPS = 128.0

# -- Tool detection -----------------------------------------------------------

def find_tool(name: str, override: str | None) -> str:
    if override:
        if not Path(override).is_file():
            print(f"[ERROR] Specified binary not found: {override}")
            sys.exit(1)
        return override
    found = shutil.which(name)
    if not found:
        print(f"[ERROR] '{name}' not found in PATH.")
        print(f"        Install HandBrake and ensure HandBrakeCLI is accessible.")
        sys.exit(1)
    return found

def find_ffprobe() -> str:
    found = shutil.which("ffprobe")
    if not found:
        print("[ERROR] 'ffprobe' not found in PATH.")
        print("        Install ffmpeg (which includes ffprobe) to probe video bitrates.")
        sys.exit(1)
    return found

# -- Probing ------------------------------------------------------------------

def probe_format(ffprobe: str, filepath: Path) -> tuple[float | None, float | None]:
    """Return (bitrate in kbps, duration in seconds) via ffprobe; None for either on failure."""
    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(filepath),
    ]
    bitrate, duration = None, None
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        fmt = json.loads(result.stdout).get("format", {})
        if fmt.get("bit_rate"):
            bitrate = float(fmt["bit_rate"]) / 1000  # bps -> kbps
        if fmt.get("duration"):
            duration = float(fmt["duration"])
    except Exception:
        pass
    return bitrate, duration

def _stream_rotation(stream: dict) -> int:
    """Rotation in degrees (0/90/180/270) from stream tags or display-matrix side data."""
    rot = (stream.get("tags") or {}).get("rotate")
    if rot is None:
        for sd in stream.get("side_data_list") or []:
            if "rotation" in sd:
                rot = sd["rotation"]
                break
    try:
        return int(round(float(rot))) % 360
    except (TypeError, ValueError):
        return 0

def probe_streams(ffprobe: str, filepath: Path) -> dict:
    """
    Return {"width", "height", "audio_kbps"}. width/height are the *display*
    dimensions (rotation applied) of the first video stream, or None on failure.
    audio_kbps is the summed bitrate of all audio streams.
    """
    info = {"width": None, "height": None, "audio_kbps": 0.0}
    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(filepath),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        streams = json.loads(result.stdout).get("streams", [])
    except Exception:
        return info

    for s in streams:
        kind = s.get("codec_type")
        if kind == "video" and info["width"] is None:
            if (s.get("disposition") or {}).get("attached_pic"):
                continue  # cover art, not a real video stream
            w, h = s.get("width"), s.get("height")
            if w and h:
                if _stream_rotation(s) in (90, 270):
                    w, h = h, w
                info["width"], info["height"] = int(w), int(h)
        elif kind == "audio":
            br = s.get("bit_rate")
            info["audio_kbps"] += float(br) / 1000 if br else DEFAULT_AUDIO_KBPS
    return info

# -- Resolution limiting ------------------------------------------------------

def parse_max_resolution(text: str) -> tuple[int, int]:
    """
    Parse '1080p' / '4k' / '1920x1080' into (long_side, short_side).
    'Np' means a 16:9 box with N as the short side (1080p -> 1920x1080).
    """
    t = text.strip().lower()
    m = re.fullmatch(r"(\d+)\s*[x*×]\s*(\d+)", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
    else:
        m = re.fullmatch(r"(\d+)p", t)
        if m:
            short = int(m.group(1))
        elif t in ("4k", "uhd"):
            short = 2160
        elif t in ("8k",):
            short = 4320
        else:
            raise ValueError(f"invalid resolution '{text}' (use e.g. 1080p, 4k, 1920x1080)")
        a, b = round(short * 16 / 9) // 2 * 2, short
    if a <= 0 or b <= 0:
        raise ValueError(f"invalid resolution '{text}'")
    return max(a, b), min(a, b)

def box_for(width: int, height: int, limit: tuple[int, int]) -> tuple[int, int]:
    """Orient the (long, short) limit to match the video: landscape or portrait."""
    long_side, short_side = limit
    return (long_side, short_side) if width >= height else (short_side, long_side)

def fit_within(width: int, height: int, box: tuple[int, int]) -> tuple[int, int] | None:
    """
    Scale (width, height) down to fit inside box, keeping the aspect ratio and
    rounding down to even numbers. Return None when it already fits (never upscales).
    """
    scale = min(box[0] / width, box[1] / height)
    if scale >= 1:
        return None
    new_w = max(2, int(width * scale) // 2 * 2)
    new_h = max(2, int(height * scale) // 2 * 2)
    return new_w, new_h

# -- Formatting ---------------------------------------------------------------

def fmt_size(num_bytes: float) -> str:
    gb = num_bytes / 1024 ** 3
    if abs(gb) >= 1:
        return f"{gb:.2f} GB"
    return f"{num_bytes / 1024 ** 2:.1f} MB"

# -- File collection ----------------------------------------------------------

def collect_files(folder: Path, recursive: bool, output_dir: Path | None) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = []
    for p in folder.glob(pattern):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        # Skip leftovers from an interrupted run
        if p.name.endswith(TEMP_SUFFIX):
            continue
        # Skip files that are already inside the output directory
        if output_dir is not None:
            try:
                p.relative_to(output_dir)
                continue
            except ValueError:
                pass
        files.append(p)
    return sorted(files)

# -- Plan construction --------------------------------------------------------

def build_plan(
    files: list[Path],
    ffprobe: str,
    max_bitrate: float | None,
    max_resolution: tuple[int, int] | None,
    min_saving: float | None,
    output_dir: Path | None,
    folder: Path,
    recursive: bool,
    postfix: str = "",
) -> list[dict]:
    """
    Return a list of entries:
      {
        "path":           Path,
        "bitrate":        float | None,        # kbps
        "duration":       float | None,        # seconds
        "size":           int | None,          # bytes
        "width"/"height": int | None,          # display dimensions
        "action":         "transcode" | "skip" | "low_saving" | "probe_failed",
        "output":         Path | None,
        "target_bitrate": float | None,        # kbps passed to the encoder
        "box":            (w, h) | None,       # resolution limit box, set only when scaling
        "scale_to":       (w, h) | None,       # expected output resolution
        "est_size":       float | None,        # estimated output bytes
      }

    With output_dir None, each file is replaced in place: the output lands next
    to the source as .mp4 (the container HandBrake writes), so a non-mp4 source
    such as foo.mkv becomes foo.mp4 and the original is removed afterwards.
    """
    in_place = output_dir is None
    plan = []
    claimed: set[Path] = set()   # output paths already handed out by this plan
    total = len(files)
    for i, filepath in enumerate(files, 1):
        print(f"  Probing {i}/{total}: {filepath.name} ...", end="\r", flush=True)
        bitrate, duration = probe_format(ffprobe, filepath)
        try:
            size = filepath.stat().st_size
        except OSError:
            size = None

        entry = {
            "path": filepath,
            "bitrate": bitrate,
            "duration": duration,
            "size": size,
            "width": None,
            "height": None,
            "action": "skip",
            "output": None,
            "target_bitrate": None,
            "box": None,
            "scale_to": None,
            "est_size": None,
        }

        if bitrate is None:
            entry["action"] = "probe_failed"
            plan.append(entry)
            continue

        streams = probe_streams(ffprobe, filepath)
        w, h = streams["width"], streams["height"]
        entry["width"], entry["height"] = w, h

        box = scale_to = None
        if max_resolution and w and h:
            box = box_for(w, h, max_resolution)
            scale_to = fit_within(w, h, box)

        over_bitrate = max_bitrate is not None and bitrate > max_bitrate
        if not over_bitrate and scale_to is None:
            plan.append(entry)   # action stays "skip"
            continue

        # Target bitrate: never raise it above what the source already has.
        # Resolution-only runs scale the source bitrate by the pixel ratio.
        if max_bitrate is not None:
            target = min(max_bitrate, bitrate)
        else:
            target = bitrate * (scale_to[0] * scale_to[1]) / (w * h)
        target = max(target, 1.0)

        entry["target_bitrate"] = target
        entry["scale_to"] = scale_to
        entry["box"] = box if scale_to else None

        # Estimate: duration x (video target + copied audio). NVENC ABR often
        # lands at or below the target, so this is an upper-bound estimate.
        if duration:
            entry["est_size"] = duration * (target + streams["audio_kbps"]) * 1000 / 8

        if (min_saving is not None and size and entry["est_size"] is not None
                and (size - entry["est_size"]) / size * 100 < min_saving):
            entry["action"] = "low_saving"
            plan.append(entry)
            continue

        # Determine output path, mirroring subfolder structure if recursive
        stem = filepath.stem + postfix
        if in_place:
            out_path = filepath.parent / (stem + ".mp4")
            # The source is only a "conflict" with itself when it is already .mp4
            # and no postfix was given -- that is exactly the overwrite we want.
            if out_path != filepath:
                out_path = resolve_conflict(out_path, claimed)
        elif recursive:
            rel = filepath.relative_to(folder)
            out_path = resolve_conflict(output_dir / rel.parent / (stem + ".mp4"), claimed)
        else:
            out_path = resolve_conflict(output_dir / (stem + ".mp4"), claimed)
        claimed.add(out_path)

        entry["action"] = "transcode"
        entry["output"] = out_path
        plan.append(entry)

    print(" " * 60, end="\r")  # clear progress line
    return plan

def resolve_conflict(target: Path, claimed: set[Path]) -> Path:
    """Return target, or the first _2/_3/... variant that neither exists nor is claimed."""
    if not target.exists() and target not in claimed:
        return target
    counter = 2
    while True:
        candidate = target.parent / f"{target.stem}_{counter}{target.suffix}"
        if not candidate.exists() and candidate not in claimed:
            return candidate
        counter += 1

def temp_path_for(dst: Path) -> Path:
    """In-progress path written by HandBrake, promoted to dst only after verification."""
    return dst.parent / (dst.stem + TEMP_SUFFIX)

# -- Preview ------------------------------------------------------------------

def _describe(e: dict) -> str:
    parts = [f"{e['bitrate']:.0f} kbps"]
    if e["width"] and e["height"]:
        parts.append(f"{e['width']}x{e['height']}")
    if e["size"] is not None:
        parts.append(fmt_size(e["size"]))
    return ", ".join(parts)

def print_preview(
    plan: list[dict],
    max_bitrate: float | None,
    max_resolution: tuple[int, int] | None,
    min_saving: float | None,
    output_dir: Path | None,
):
    transcode  = [e for e in plan if e["action"] == "transcode"]
    skip       = [e for e in plan if e["action"] == "skip"]
    low_saving = [e for e in plan if e["action"] == "low_saving"]
    failed     = [e for e in plan if e["action"] == "probe_failed"]

    print(f"\n{'─'*72}")
    print(f"  max-bitrate    : " + (f"{max_bitrate:.0f} kbps" if max_bitrate is not None else "(none)"))
    if max_resolution:
        print(f"  max-resolution : {max_resolution[0]}x{max_resolution[1]} "
              f"(orientation follows the video)")
    else:
        print("  max-resolution : (none)")
    if min_saving is not None:
        print(f"  min-saving     : {min_saving:g}%")
    if output_dir is None:
        print("  output-dir     : (none -- originals are replaced in place)")
    else:
        print(f"  output-dir     : {output_dir}")
    print(f"  Total files    : {len(plan)}")
    print(f"{'─'*72}")

    if transcode:
        print(f"\n  To be transcoded ({len(transcode)}):\n")
        if output_dir is None:
            print("    [WARN] Originals are deleted once the new file is verified.\n")
        for e in transcode:
            print(f"    {e['path'].name}  [{_describe(e)}]")
            if e["output"] == e["path"]:
                print(f"      -> {e['output'].name}  (replaced in place)")
            else:
                print(f"      -> {e['output'].name}")
            if e["scale_to"]:
                print(f"      resolution: {e['width']}x{e['height']} -> "
                      f"{e['scale_to'][0]}x{e['scale_to'][1]}")
            print(f"      bitrate   : {e['bitrate']:.0f} -> {e['target_bitrate']:.0f} kbps (video target)")
            if e["est_size"] is not None and e["size"]:
                saved = e["size"] - e["est_size"]
                if saved > 0:
                    print(f"      est. size : ~{fmt_size(e['est_size'])} "
                          f"(saves ~{fmt_size(saved)}, -{saved / e['size'] * 100:.0f}%)")
                else:
                    note = "; original would be kept" if output_dir is None else ""
                    print(f"      est. size : ~{fmt_size(e['est_size'])} "
                          f"(not expected to shrink{note})")
            else:
                print("      est. size : unknown")

        estimated = [e for e in transcode if e["est_size"] is not None and e["size"]]
        if estimated:
            before = sum(e["size"] for e in estimated)
            saved  = sum(max(0.0, e["size"] - e["est_size"]) for e in estimated)
            after  = before - saved
            print(f"\n    Estimated total: {fmt_size(before)} -> ~{fmt_size(after)}  "
                  f"(saves ~{fmt_size(saved)}, -{saved / before * 100:.0f}%)")
            if len(estimated) < len(transcode):
                print(f"    ({len(transcode) - len(estimated)} file(s) without a duration are not included)")
            print("    Estimates assume duration x (target video bitrate + audio); actual size may differ.")

    if low_saving:
        print(f"\n  Estimated saving below {min_saving:g}%, skipped ({len(low_saving)}):\n")
        for e in low_saving:
            saved = e["size"] - e["est_size"]
            print(f"    {e['path'].name}  [{_describe(e)}]  (est. {saved / e['size'] * 100:.0f}%)")

    if skip:
        print(f"\n  Within limits, skipped ({len(skip)}):\n")
        for e in skip:
            print(f"    {e['path'].name}  [{_describe(e)}]")

    if failed:
        print(f"\n  [WARN] Could not probe bitrate ({len(failed)}):\n")
        for e in failed:
            print(f"    {e['path'].name}")

    print(f"\n{'─'*72}\n")

# -- Dry run ------------------------------------------------------------------

def _quote(arg: str) -> str:
    """Wrap in quotes if the argument contains spaces or shell-special characters."""
    if any(c in arg for c in (' ', '(', ')', '&', '^', '%', '!')):
        return f'"{arg}"'
    return arg

def print_dry_run(plan: list[dict], handbrake: str):
    to_transcode = [e for e in plan if e["action"] == "transcode"]

    if not to_transcode:
        print("No files need transcoding. Nothing to do.\n")
        return

    is_windows = sys.platform == "win32"
    cont        = " `"  if is_windows else " \\"   # PowerShell vs POSIX
    shell_label = "PowerShell" if is_windows else "shell"

    print(f"\n{'─'*72}")
    print(f"  Dry run -- {len(to_transcode)} command(s) that would be executed ({shell_label}):")
    print(f"  Each writes to a {TEMP_SUFFIX} file, which is verified and then renamed.")
    print(f"{'─'*72}\n")

    for i, entry in enumerate(to_transcode, 1):
        tmp    = temp_path_for(entry["output"])
        cmd    = build_handbrake_cmd(handbrake, entry["path"], tmp,
                                     entry["target_bitrate"], entry["box"])
        quoted = [_quote(arg) for arg in cmd]
        print(f"# [{i}/{len(to_transcode)}] {entry['path'].name}  ({entry['bitrate']:.0f} kbps)")
        if is_windows:
            # Single-line form is most reliable in PowerShell
            print(" ".join(quoted))
        else:
            lines = [quoted[0]] + [f"  {arg}" for arg in quoted[1:]]
            print((cont + "\n").join(lines))
        print()

    print(f"{'─'*72}\n")



def build_handbrake_cmd(
    handbrake: str,
    input_path: Path,
    output_path: Path,
    target_bitrate: float,
    box: tuple[int, int] | None = None,
) -> list[str]:
    """
    HEVC via NVENC at target_bitrate kbps, audio copied. When box is given the
    video is scaled down to fit inside it (aspect ratio kept, never upscaled).
    All other stream properties (framerate, etc.) are passed through.
    """
    cmd = [
        handbrake,
        "--input",  str(input_path),
        "--output", str(output_path),
        # Video
        "--encoder",        "nvenc_h265",
        "--vb",             str(int(target_bitrate)),   # target bitrate (kbps)
        "--pfr",                                     # peak framerate: preserve source
    ]
    if box:
        cmd += [
            "--maxWidth",  str(box[0]),
            "--maxHeight", str(box[1]),
            "--keep-display-aspect",
        ]
    cmd += [
        # Audio: copy all tracks as-is
        "--aencoder",       "copy",
        "--audio",          "1,2,3,4,5,6,7,8",      # pass through up to 8 tracks
        "--audio-fallback", "copy",
        # Subtitles: copy all
        "--subtitle",       "scan",
        "--subtitle-default", "none",
        # Format
        "--format",         "av_mp4",
        "--optimize",                                # fast-start for streaming
    ]
    return cmd

# -- Verification -------------------------------------------------------------

def verify_output(ffprobe: str, produced: Path, src_duration: float | None) -> str | None:
    """
    Check that `produced` is a usable video. Return None if it passes, or a
    human-readable reason describing why it must not replace the source.
    """
    if not produced.exists():
        return "output file was not created"
    if produced.stat().st_size == 0:
        return "output file is empty"

    _, duration = probe_format(ffprobe, produced)
    if duration is None:
        return "output file has no readable duration (probe failed)"
    if src_duration is None:
        return "source duration is unknown, cannot verify output"
    delta = abs(duration - src_duration)
    if delta > DURATION_TOLERANCE_SEC:
        return (f"duration mismatch: source {src_duration:.1f}s vs "
                f"output {duration:.1f}s (off by {delta:.1f}s)")
    return None

def copy_timestamps(src: Path, dst: Path):
    """Carry the source's access/modification times over so archival dates survive."""
    try:
        st = src.stat()
        os.utime(dst, (st.st_atime, st.st_mtime))
    except OSError as e:
        print(f"  [WARN] Could not copy timestamps: {e}")

# -- Execution ----------------------------------------------------------------

def execute_plan(
    plan: list[dict],
    handbrake: str,
    ffprobe: str,
    skip_errors: bool,
    in_place: bool,
    show_log: bool = False,
):
    to_transcode = [e for e in plan if e["action"] == "transcode"]

    if not to_transcode:
        print("Nothing to transcode.")
        return

    print(f"\nTranscoding {len(to_transcode)} file(s)...\n")
    success, skipped, kept = 0, 0, 0
    bytes_before, bytes_after = 0, 0   # totals over successfully transcoded files

    def fail(message: str, tmp: Path) -> int:
        """Report a failure, discard the partial output, and honour --skip-errors."""
        print(f"  [ERROR] {message}")
        if tmp.exists():
            tmp.unlink()
        if not skip_errors:
            print("\nAborted. Use --skip-errors to continue past failures.")
            print("The original file was left untouched.")
            sys.exit(1)
        return 1

    for i, entry in enumerate(to_transcode, 1):
        src = entry["path"]
        dst = entry["output"]
        br  = f"{entry['bitrate']:.0f}"
        target = entry["target_bitrate"]

        line = f"  [{i}/{len(to_transcode)}] {src.name}  ({br} kbps -> {int(target)} kbps"
        if entry["scale_to"]:
            line += f", {entry['width']}x{entry['height']} -> {entry['scale_to'][0]}x{entry['scale_to'][1]}"
        print(line + ")")
        print(f"         -> {dst}" + ("  (in place)" if in_place else ""))

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = temp_path_for(dst)
        cmd = build_handbrake_cmd(handbrake, src, tmp, target, entry["box"])

        try:
            if tmp.exists():
                tmp.unlink()  # leftover from an interrupted run

            result = subprocess.run(cmd, text=True, capture_output=not show_log)
            if result.returncode != 0:
                message = f"HandBrakeCLI exited with code {result.returncode}"
                if not show_log and result.stderr:
                    message += "\n" + "\n".join(
                        f"           {line}" for line in result.stderr.strip().splitlines()[-5:]
                    )
                skipped += fail(message, tmp)
                print()
                continue

            reason = verify_output(ffprobe, tmp, entry["duration"])
            if reason:
                skipped += fail(f"Verification failed -- {reason}", tmp)
                print()
                continue

            src_size = src.stat().st_size
            new_size = tmp.stat().st_size

            # Transcoding a file that is already efficiently encoded can inflate
            # it; replacing the original in that case would make the archive worse.
            if in_place and new_size >= src_size:
                old_mb = src_size / 1024 / 1024
                new_mb = new_size / 1024 / 1024
                print(f"  [KEPT] Output is not smaller "
                      f"({old_mb:.1f} MB -> {new_mb:.1f} MB); original kept.")
                tmp.unlink()
                kept += 1
                print()
                continue

            copy_timestamps(src, tmp)

            # Only now is the original expendable: the replacement is verified on disk.
            if in_place:
                src.unlink()
            tmp.replace(dst)

            bytes_before += src_size
            bytes_after  += new_size
            print(f"  [OK] {fmt_size(src_size)} -> {fmt_size(new_size)} "
                  f"(saved {fmt_size(src_size - new_size)})")
            success += 1

        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Cleaning up partial output...")
            if tmp.exists():
                tmp.unlink()
            sys.exit(130)
        except Exception as e:
            skipped += fail(str(e), tmp)

        print()

    summary = f"Done. Transcoded: {success}, skipped/failed: {skipped}"
    if in_place:
        summary += f", kept (not smaller): {kept}"
    print(summary)
    if success and bytes_before > 0:
        saved = bytes_before - bytes_after
        print(f"Total: {fmt_size(bytes_before)} -> {fmt_size(bytes_after)}  "
              f"(saved {fmt_size(saved)}, -{saved / bytes_before * 100:.0f}%)")
    print()

# -- Entry point --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transcode videos to HEVC (NVENC) when bitrate or resolution exceeds a limit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("folder",          help="Folder containing video files")
    parser.add_argument("--max-bitrate",   type=float, metavar="KBPS",
                        help="Maximum bitrate in kbps; files above this are transcoded")
    parser.add_argument("--max-resolution", metavar="RES",
                        help="Maximum resolution (e.g. 1080p, 4k, 1920x1080); "
                             "aspect ratio kept, orientation follows the video, never upscales")
    parser.add_argument("--min-saving",    type=float, metavar="PERCENT",
                        help="Skip files whose estimated saving is below this percentage")
    parser.add_argument("--output-dir",    metavar="PATH",
                        help="Output directory (default: replace the original files in place)")
    parser.add_argument("--recursive",     action="store_true",
                        help="Process subdirectories recursively")
    parser.add_argument("--execute",       action="store_true",
                        help="Apply transcoding (default: preview only)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print the exact HandBrakeCLI commands without running them")
    parser.add_argument("--skip-errors",   action="store_true",
                        help="Continue past files that fail")
    parser.add_argument("--handbrake",     metavar="PATH",
                        help="Path to HandBrakeCLI binary (default: auto-detect)")
    parser.add_argument("--postfix",       default="", metavar="STRING",
                        help="Postfix string appended to output filenames (default: empty)")
    parser.add_argument("--show-log",      action="store_true",
                        help="Stream HandBrakeCLI output directly to the terminal")
    args = parser.parse_args()

    if args.max_bitrate is None and args.max_resolution is None:
        parser.error("at least one of --max-bitrate / --max-resolution is required")
    if args.max_bitrate is not None and args.max_bitrate <= 0:
        parser.error("--max-bitrate must be positive")
    if args.min_saving is not None and not (0 <= args.min_saving < 100):
        parser.error("--min-saving must be between 0 and 100")

    max_resolution = None
    if args.max_resolution:
        try:
            max_resolution = parse_max_resolution(args.max_resolution)
        except ValueError as e:
            parser.error(str(e))

    # Resolve tools
    handbrake = find_tool("HandBrakeCLI", args.handbrake)
    ffprobe   = find_ffprobe()

    # Resolve paths
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"[ERROR] Folder not found: {folder}")
        sys.exit(1)

    # No --output-dir means in-place replacement of the originals.
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    in_place   = output_dir is None

    # Collect files
    print("Scanning files...")
    files = collect_files(folder, args.recursive, output_dir)
    if not files:
        print("No supported video files found.")
        sys.exit(0)

    # Build plan
    print(f"Probing {len(files)} file(s)...")
    plan = build_plan(files, ffprobe, args.max_bitrate, max_resolution, args.min_saving,
                      output_dir, folder, args.recursive, args.postfix)

    # Show preview
    print_preview(plan, args.max_bitrate, max_resolution, args.min_saving, output_dir)

    # Dry run: print commands and exit
    if args.dry_run:
        print_dry_run(plan, handbrake)
        return

    # Execute or prompt
    to_do = sum(1 for e in plan if e["action"] == "transcode")
    if not args.execute:
        if to_do > 0:
            print("Preview mode. Add --execute to start transcoding.\n")
    else:
        if to_do == 0:
            print("No files need transcoding. Nothing to do.\n")
            return
        if in_place:
            prompt = (f"Transcode {to_do} file(s) IN PLACE? "
                      f"Each original is deleted once its replacement is verified. [y/N] ")
        else:
            prompt = f"Transcode {to_do} file(s)? [y/N] "
        confirm = input(prompt).strip().lower()
        if confirm == "y":
            execute_plan(plan, handbrake, ffprobe,
                         args.skip_errors, in_place, args.show_log)
        else:
            print("Cancelled.\n")


if __name__ == "__main__":
    main()
