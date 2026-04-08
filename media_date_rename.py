#!/usr/bin/env python3
"""
media_archiver.py -- Rename media files using their capture/creation date.

Usage:
    python media_archiver.py <folder> [options]

Options:
    --execute       Apply the renames (default: preview only)
    --recursive     Process subdirectories recursively
    --skip-errors   Skip files that cause errors instead of aborting
"""

import argparse
import os
import sys
import struct
import datetime
from pathlib import Path

# -- Supported extensions -----------------------------------------------------
SUPPORTED_EXTENSIONS = {
    # Images
    ".jpg", ".jpeg", ".heic", ".heif", ".png", ".tiff", ".tif",
    ".webp", ".bmp", ".gif", ".raw", ".cr2", ".cr3", ".nef",
    ".arw", ".dng", ".orf", ".rw2", ".pef",
    # Videos
    ".mov", ".mp4", ".m4v", ".avi", ".mkv", ".wmv", ".flv",
    ".mts", ".m2ts", ".3gp", ".webm",
}

# -- Dependency check ---------------------------------------------------------
def check_dependencies():
    missing = []
    try:
        import piexif  # noqa
    except ImportError:
        missing.append("piexif")
    try:
        from PIL import Image  # noqa
    except ImportError:
        missing.append("Pillow")
    try:
        import pillow_heif  # noqa
    except ImportError:
        missing.append("pillow-heif")

    if missing:
        print("[ERROR] Missing required packages. Install with:")
        print(f"        pip install {' '.join(missing)}")
        sys.exit(1)

# -- Date extraction ----------------------------------------------------------

def parse_exif_date(date_str: str) -> datetime.datetime | None:
    """Parse an EXIF date string in 'YYYY:MM:DD HH:MM:SS' format."""
    if not date_str:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            dt = datetime.datetime.strptime(date_str.strip(), fmt)
            if dt.year > 1970:
                return dt
        except ValueError:
            continue
    return None


def get_exif_date_pil(filepath: Path) -> datetime.datetime | None:
    """Read EXIF DateTimeOriginal via Pillow + piexif."""
    try:
        import piexif
        from PIL import Image
        import pillow_heif

        ext = filepath.suffix.lower()
        if ext in (".heic", ".heif"):
            pillow_heif.register_heif_opener()

        with Image.open(filepath) as img:
            exif_data = img.info.get("exif")
            if not exif_data:
                return None
            exif_dict = piexif.load(exif_data)
            for ifd in ("Exif", "0th", "1st"):
                tag_val = exif_dict.get(ifd, {}).get(
                    piexif.ExifIFD.DateTimeOriginal if ifd == "Exif" else piexif.ImageIFD.DateTime
                )
                if tag_val:
                    s = tag_val.decode("utf-8", errors="ignore") if isinstance(tag_val, bytes) else str(tag_val)
                    dt = parse_exif_date(s)
                    if dt:
                        return dt
    except Exception:
        pass
    return None


def get_mp4_creation_date(filepath: Path) -> datetime.datetime | None:
    """Read creation time from the MP4/MOV mvhd box (32- or 64-bit)."""
    try:
        with open(filepath, "rb") as f:
            data = f.read(min(1024 * 1024, os.path.getsize(filepath)))  # read up to 1 MB

        i = 0
        while i < len(data) - 8:
            try:
                box_size = struct.unpack(">I", data[i:i+4])[0]
                box_type = data[i+4:i+8]

                if box_type == b"mvhd":
                    version = data[i+8]
                    if version == 1:
                        # 64-bit
                        creation_time = struct.unpack(">Q", data[i+12:i+20])[0]
                    else:
                        # 32-bit
                        creation_time = struct.unpack(">I", data[i+12:i+16])[0]

                    # MP4 epoch: 1904-01-01
                    mac_epoch = datetime.datetime(1904, 1, 1)
                    dt = mac_epoch + datetime.timedelta(seconds=creation_time)
                    if dt.year > 1970:
                        return dt

                if box_size < 8:
                    break
                i += box_size
            except Exception:
                break
    except Exception:
        pass
    return None


def get_file_system_date(filepath: Path) -> datetime.datetime:
    """Return the earliest available filesystem timestamp (ctime, mtime, birthtime)."""
    stat = filepath.stat()
    candidates = [stat.st_mtime]
    # st_ctime is creation time on Windows; inode-change time on Unix
    if stat.st_ctime:
        candidates.append(stat.st_ctime)
    # st_birthtime is available on macOS / BSD only
    if hasattr(stat, "st_birthtime") and stat.st_birthtime:
        candidates.append(stat.st_birthtime)

    earliest = min(candidates)
    return datetime.datetime.fromtimestamp(earliest)


def get_best_date(filepath: Path) -> tuple[datetime.datetime, str]:
    """
    Return (best_date, source_label).
    Priority: EXIF DateTimeOriginal -> MP4 mvhd -> earliest filesystem date.
    """
    ext = filepath.suffix.lower()

    # 1. EXIF (images)
    if ext in {".jpg", ".jpeg", ".heic", ".heif", ".tiff", ".tif", ".png",
               ".webp", ".raw", ".cr2", ".cr3", ".nef", ".arw", ".dng",
               ".orf", ".rw2", ".pef"}:
        dt = get_exif_date_pil(filepath)
        if dt:
            return dt, "EXIF"

    # 2. MP4 / MOV mvhd box
    if ext in {".mp4", ".m4v", ".mov", ".3gp", ".m4a"}:
        dt = get_mp4_creation_date(filepath)
        if dt:
            return dt, "MP4 mvhd"

    # 3. Earliest filesystem date
    dt = get_file_system_date(filepath)
    return dt, "filesystem"


# -- Conflict resolution ------------------------------------------------------

def resolve_name_conflict(target: Path, used_names: set) -> Path:
    """Append _2, _3, ... to the stem if the target path is already taken."""
    if str(target) not in used_names and not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    counter = 2
    while True:
        new_target = parent / f"{stem}_{counter}{suffix}"
        if str(new_target) not in used_names and not new_target.exists():
            return new_target
        counter += 1


# -- Core logic ---------------------------------------------------------------

def collect_files(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = []
    for p in folder.glob(pattern):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(p)
    return sorted(files)


def build_plan(files: list[Path], postfix: str = "") -> list[tuple[Path, Path, str]]:
    """
    Return [(original_path, new_path, date_source), ...].
    Guarantees no naming conflicts within the same directory.
    """
    used_names: set[str] = set()
    plan = []

    for filepath in files:
        dt, source = get_best_date(filepath)
        new_stem = dt.strftime("%Y-%m-%d %H-%M-%S")
        if postfix:
            new_stem = f"{new_stem}{postfix}"
        new_suffix = filepath.suffix.lower()
        candidate = filepath.parent / f"{new_stem}{new_suffix}"
        target = resolve_name_conflict(candidate, used_names)
        used_names.add(str(target))
        plan.append((filepath, target, source))

    return plan


def print_preview(plan: list[tuple[Path, Path, str]], folder: Path):
    print(f"\n{'─'*70}")
    print(f"  Folder : {folder}")
    print(f"  Files  : {len(plan)}")
    print(f"{'─'*70}")

    unchanged = [(o, n, s) for o, n, s in plan if o == n]
    changed   = [(o, n, s) for o, n, s in plan if o != n]

    if changed:
        print(f"\n  To be renamed ({len(changed)}):\n")
        for orig, new, source in changed:
            print(f"    {orig.name}")
            print(f"      -> {new.name}  [{source}]")
    else:
        print("\n  [OK] All filenames already match the target format.")

    if unchanged:
        print(f"\n  No change needed ({len(unchanged)}):")
        for orig, _, source in unchanged:
            print(f"    {orig.name}  [{source}]")

    print(f"\n{'─'*70}\n")


def execute_plan(plan: list[tuple[Path, Path, str]], skip_errors: bool):
    success, skipped = 0, 0
    changed = [(o, n, s) for o, n, s in plan if o != n]

    if not changed:
        print("Nothing to rename.")
        return

    print(f"\nRenaming {len(changed)} file(s)...\n")
    for orig, new, _ in changed:
        try:
            orig.rename(new)
            print(f"  [OK]    {orig.name} -> {new.name}")
            success += 1
        except Exception as e:
            print(f"  [ERROR] {orig.name}: {e}")
            if not skip_errors:
                print("Aborted. Use --skip-errors to continue past failures.")
                sys.exit(1)
            skipped += 1

    print(f"\nDone. Renamed: {success}, skipped: {skipped}\n")


# -- Entry point --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rename media files using their capture/creation date.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("folder", help="Path to the folder to process")
    parser.add_argument("--execute", action="store_true", help="Apply renames (default: preview only)")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively")
    parser.add_argument("--skip-errors", action="store_true", help="Skip files that cause errors")
    parser.add_argument("--postfix", default="", help="String to append to output filenames before the extension")
    args = parser.parse_args()

    check_dependencies()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"[ERROR] Folder not found: {folder}")
        sys.exit(1)

    print("Scanning files...")
    files = collect_files(folder, args.recursive)

    if not files:
        print("No supported media files found.")
        sys.exit(0)

    print("Reading date information...")
    plan = build_plan(files, args.postfix)

    print_preview(plan, folder)

    if not args.execute:
        changed_count = sum(1 for o, n, _ in plan if o != n)
        if changed_count > 0:
            print("Preview mode. Add --execute to apply the renames.\n")
    else:
        confirm = input("Confirm rename? [y/N] ").strip().lower()
        if confirm == "y":
            execute_plan(plan, args.skip_errors)
        else:
            print("Cancelled.\n")


if __name__ == "__main__":
    main()