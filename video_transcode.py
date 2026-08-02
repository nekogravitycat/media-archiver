#!/usr/bin/env python3
"""
video_transcode.py -- Transcode videos to HEVC using HandBrakeCLI (NVIDIA NVENC).

Encodes only files whose bitrate exceeds max_bitrate. Files at or below the
threshold are skipped. Audio tracks are copied without re-encoding.

Without --output-dir, originals are replaced in place: each file is encoded to
a temporary file alongside it, verified (exit code, non-empty output, and
duration matching the source), and only then does the original get deleted.
If the result is not smaller than the original, the original is kept.

Usage:
    python video_transcode.py <folder> --max-bitrate <kbps> [options]

Options:
    --max-bitrate <kbps>    Target maximum bitrate in kbps (required)
    --output-dir <path>     Output directory (default: replace originals in place)
    --recursive             Process subdirectories recursively
    --execute               Apply transcoding (default: preview only)
    --dry-run               Print the exact HandBrakeCLI commands without running them
    --skip-errors           Continue past files that fail
    --handbrake <path>      Path to HandBrakeCLI binary (default: auto-detect)

Examples:
    python video_transcode.py ~/Videos --max-bitrate 8000
    python video_transcode.py ~/Videos --max-bitrate 8000 --execute
    python video_transcode.py ~/Videos --max-bitrate 8000 --output-dir ~/Videos/out --execute
"""

import argparse
import json
import os
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

# -- Bitrate probing ----------------------------------------------------------

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
    max_bitrate: float,
    output_dir: Path | None,
    folder: Path,
    recursive: bool,
    postfix: str = "",
) -> list[dict]:
    """
    Return a list of entries:
      {
        "path":      Path,
        "bitrate":   float | None,   # kbps
        "duration":  float | None,   # seconds
        "action":    "transcode" | "skip" | "probe_failed",
        "output":    Path | None,
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

        if bitrate is None:
            plan.append({
                "path": filepath,
                "bitrate": None,
                "duration": duration,
                "action": "probe_failed",
                "output": None,
            })
            continue

        if bitrate <= max_bitrate:
            plan.append({
                "path": filepath,
                "bitrate": bitrate,
                "duration": duration,
                "action": "skip",
                "output": None,
            })
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

        plan.append({
            "path": filepath,
            "bitrate": bitrate,
            "duration": duration,
            "action": "transcode",
            "output": out_path,
        })

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

def print_preview(plan: list[dict], max_bitrate: float, output_dir: Path | None):
    transcode = [e for e in plan if e["action"] == "transcode"]
    skip      = [e for e in plan if e["action"] == "skip"]
    failed    = [e for e in plan if e["action"] == "probe_failed"]

    print(f"\n{'─'*72}")
    print(f"  max-bitrate : {max_bitrate:.0f} kbps")
    if output_dir is None:
        print("  output-dir  : (none -- originals are replaced in place)")
    else:
        print(f"  output-dir  : {output_dir}")
    print(f"  Total files : {len(plan)}")
    print(f"{'─'*72}")

    if transcode:
        print(f"\n  To be transcoded ({len(transcode)}):\n")
        if output_dir is None:
            print("    [WARN] Originals are deleted once the new file is verified.\n")
        for e in transcode:
            br = f"{e['bitrate']:.0f} kbps"
            print(f"    {e['path'].name}  [{br}]")
            if e["output"] == e["path"]:
                print(f"      -> {e['output'].name}  (replaced in place)")
            else:
                print(f"      -> {e['output'].name}")

    if skip:
        print(f"\n  Below threshold, skipped ({len(skip)}):\n")
        for e in skip:
            br = f"{e['bitrate']:.0f} kbps"
            print(f"    {e['path'].name}  [{br}]")

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

def print_dry_run(plan: list[dict], handbrake: str, max_bitrate: float):
    to_transcode = [e for e in plan if e["action"] == "transcode"]

    if not to_transcode:
        print("No files exceed the bitrate threshold. Nothing to transcode.\n")
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
        cmd    = build_handbrake_cmd(handbrake, entry["path"], tmp, max_bitrate)
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
    max_bitrate: float,
) -> list[str]:
    """
    HEVC via NVENC, video bitrate capped at max_bitrate kbps, audio copied.
    All other stream properties (framerate, resolution, etc.) are passed through.
    """
    return [
        handbrake,
        "--input",  str(input_path),
        "--output", str(output_path),
        # Video
        "--encoder",        "nvenc_h265",
        "--vb",             str(int(max_bitrate)),   # target bitrate (kbps)
        "--pfr",                                     # peak framerate: preserve source
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
    max_bitrate: float,
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

        print(f"  [{i}/{len(to_transcode)}] {src.name}  ({br} kbps -> {int(max_bitrate)} kbps)")
        print(f"         -> {dst}" + ("  (in place)" if in_place else ""))

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = temp_path_for(dst)
        cmd = build_handbrake_cmd(handbrake, src, tmp, max_bitrate)

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

            # Transcoding a file that is already efficiently encoded can inflate
            # it; replacing the original in that case would make the archive worse.
            if in_place and tmp.stat().st_size >= src.stat().st_size:
                old_mb = src.stat().st_size / 1024 / 1024
                new_mb = tmp.stat().st_size / 1024 / 1024
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

            print("  [OK]")
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
    print(summary + "\n")

# -- Entry point --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transcode videos to HEVC (NVENC) when bitrate exceeds a threshold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("folder",          help="Folder containing video files")
    parser.add_argument("--max-bitrate",   type=float, required=True, metavar="KBPS",
                        help="Maximum bitrate in kbps; files above this are transcoded")
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
    plan = build_plan(files, ffprobe, args.max_bitrate, output_dir, folder, args.recursive, args.postfix)

    # Show preview
    print_preview(plan, args.max_bitrate, output_dir)

    # Dry run: print commands and exit
    if args.dry_run:
        print_dry_run(plan, handbrake, args.max_bitrate)
        return

    # Execute or prompt
    if not args.execute:
        to_do = sum(1 for e in plan if e["action"] == "transcode")
        if to_do > 0:
            print("Preview mode. Add --execute to start transcoding.\n")
    else:
        to_do = sum(1 for e in plan if e["action"] == "transcode")
        if to_do == 0:
            print("No files exceed the bitrate threshold. Nothing to do.\n")
            return
        if in_place:
            prompt = (f"Transcode {to_do} file(s) IN PLACE? "
                      f"Each original is deleted once its replacement is verified. [y/N] ")
        else:
            prompt = f"Transcode {to_do} file(s)? [y/N] "
        confirm = input(prompt).strip().lower()
        if confirm == "y":
            execute_plan(plan, handbrake, ffprobe, args.max_bitrate,
                         args.skip_errors, in_place, args.show_log)
        else:
            print("Cancelled.\n")


if __name__ == "__main__":
    main()