#!/usr/bin/env python3
"""
video_transcode.py -- Transcode videos to HEVC using HandBrakeCLI (NVIDIA NVENC).

Encodes only files whose bitrate exceeds max_bitrate. Files at or below the
threshold are skipped. Audio tracks are copied without re-encoding.

Usage:
    python video_transcode.py <folder> --max-bitrate <kbps> [options]

Options:
    --max-bitrate <kbps>    Target maximum bitrate in kbps (required)
    --output-dir <path>     Output directory (default: <folder>/converted)
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

OUTPUT_SUBDIR = "converted"

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

def get_bitrate_kbps(ffprobe: str, filepath: Path) -> float | None:
    """Return overall bitrate in kbps using ffprobe, or None on failure."""
    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(filepath),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        bit_rate = data.get("format", {}).get("bit_rate")
        if bit_rate:
            return float(bit_rate) / 1000  # bps -> kbps
    except Exception:
        pass
    return None

# -- File collection ----------------------------------------------------------

def collect_files(folder: Path, recursive: bool, output_dir: Path) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = []
    for p in folder.glob(pattern):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        # Skip files that are already inside the output directory
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
    output_dir: Path,
    folder: Path,
    recursive: bool,
    postfix: str = "",
) -> list[dict]:
    """
    Return a list of entries:
      {
        "path":      Path,
        "bitrate":   float | None,   # kbps
        "action":    "transcode" | "skip" | "probe_failed",
        "output":    Path | None,
      }
    """
    plan = []
    total = len(files)
    for i, filepath in enumerate(files, 1):
        print(f"  Probing {i}/{total}: {filepath.name} ...", end="\r", flush=True)
        bitrate = get_bitrate_kbps(ffprobe, filepath)

        if bitrate is None:
            plan.append({
                "path": filepath,
                "bitrate": None,
                "action": "probe_failed",
                "output": None,
            })
            continue

        if bitrate <= max_bitrate:
            plan.append({
                "path": filepath,
                "bitrate": bitrate,
                "action": "skip",
                "output": None,
            })
            continue

        # Determine output path, mirroring subfolder structure if recursive
        stem = filepath.stem + postfix
        if recursive:
            rel = filepath.relative_to(folder)
            out_path = output_dir / rel.parent / (stem + ".mp4")
        else:
            out_path = output_dir / (stem + ".mp4")

        # Resolve name conflicts
        out_path = resolve_conflict(out_path)

        plan.append({
            "path": filepath,
            "bitrate": bitrate,
            "action": "transcode",
            "output": out_path,
        })

    print(" " * 60, end="\r")  # clear progress line
    return plan

def resolve_conflict(target: Path) -> Path:
    if not target.exists():
        return target
    counter = 2
    while True:
        candidate = target.parent / f"{target.stem}_{counter}{target.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1

# -- Preview ------------------------------------------------------------------

def print_preview(plan: list[dict], max_bitrate: float, output_dir: Path):
    transcode = [e for e in plan if e["action"] == "transcode"]
    skip      = [e for e in plan if e["action"] == "skip"]
    failed    = [e for e in plan if e["action"] == "probe_failed"]

    print(f"\n{'─'*72}")
    print(f"  max-bitrate : {max_bitrate:.0f} kbps")
    print(f"  output-dir  : {output_dir}")
    print(f"  Total files : {len(plan)}")
    print(f"{'─'*72}")

    if transcode:
        print(f"\n  To be transcoded ({len(transcode)}):\n")
        for e in transcode:
            br = f"{e['bitrate']:.0f} kbps"
            print(f"    {e['path'].name}  [{br}]")
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
    print(f"{'─'*72}\n")

    for i, entry in enumerate(to_transcode, 1):
        cmd    = build_handbrake_cmd(handbrake, entry["path"], entry["output"], max_bitrate)
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

# -- Execution ----------------------------------------------------------------

def execute_plan(
    plan: list[dict],
    handbrake: str,
    max_bitrate: float,
    skip_errors: bool,
    show_log: bool = False,
):
    to_transcode = [e for e in plan if e["action"] == "transcode"]

    if not to_transcode:
        print("Nothing to transcode.")
        return

    print(f"\nTranscoding {len(to_transcode)} file(s)...\n")
    success, skipped = 0, 0

    for i, entry in enumerate(to_transcode, 1):
        src = entry["path"]
        dst = entry["output"]
        br  = f"{entry['bitrate']:.0f}"

        print(f"  [{i}/{len(to_transcode)}] {src.name}  ({br} kbps -> {int(max_bitrate)} kbps)")
        print(f"         -> {dst}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_handbrake_cmd(handbrake, src, dst, max_bitrate)

        try:
            result = subprocess.run(cmd, text=True, capture_output=not show_log)
            if result.returncode == 0:
                print(f"  [OK]")
                success += 1
            else:
                print(f"  [ERROR] HandBrakeCLI exited with code {result.returncode}")
                if not show_log and result.stderr:
                    for line in result.stderr.strip().splitlines()[-5:]:
                        print(f"           {line}")
                if dst.exists():
                    dst.unlink()  # remove incomplete output
                if not skip_errors:
                    print("\nAborted. Use --skip-errors to continue past failures.")
                    sys.exit(1)
                skipped += 1
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Cleaning up partial output...")
            if dst.exists():
                dst.unlink()
            sys.exit(130)
        except Exception as e:
            print(f"  [ERROR] {e}")
            if not skip_errors:
                sys.exit(1)
            skipped += 1

        print()

    print(f"Done. Transcoded: {success}, skipped/failed: {skipped}\n")

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
                        help=f"Output directory (default: <folder>/{OUTPUT_SUBDIR})")
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

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir \
                 else folder / OUTPUT_SUBDIR

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
        confirm = input(f"Transcode {to_do} file(s)? [y/N] ").strip().lower()
        if confirm == "y":
            execute_plan(plan, handbrake, args.max_bitrate, args.skip_errors, args.show_log)
        else:
            print("Cancelled.\n")


if __name__ == "__main__":
    main()