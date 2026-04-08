#!/usr/bin/env python3
"""
find_duplicates.py -- Detect duplicate images and videos in a folder.

Images : perceptual hash (pHash) -- tolerates compression, resizing, etc.
Videos : duration comparison (delta < threshold) + SHA-256 confirmation.

Usage:
    python find_duplicates.py <folder>
    python find_duplicates.py <folder> --video-threshold 2.0
    python find_duplicates.py <folder> --image-threshold 5 --output report.txt
    python find_duplicates.py <folder> --no-recurse
    python find_duplicates.py <folder> --video-sha256
"""

import os
import sys
import bisect
import hashlib
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

# -- Dependency check ---------------------------------------------------------

def check_dependencies():
    missing = []
    try:
        import imagehash  # noqa: F401
    except ImportError:
        missing.append("imagehash")
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        missing.append("Pillow")
    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append("opencv-python")
    try:
        import pybktree  # noqa: F401
    except ImportError:
        missing.append("pybktree")
    try:
        import pillow_heif  # noqa: F401
    except ImportError:
        missing.append("pillow-heif")
    try:
        import tqdm  # noqa: F401
    except ImportError:
        missing.append("tqdm")
    if missing:
        print("[ERROR] Missing dependencies. Please run:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

import imagehash
from PIL import Image
import cv2
import pybktree
import pillow_heif
from tqdm import tqdm
pillow_heif.register_heif_opener()  # enables Image.open() for .heic / .heif

# -- Constants ----------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif", ".heic", ".heif"
}

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm",
    ".m4v", ".3gp", ".ts", ".mts", ".m2ts", ".rmvb", ".rm"
}

# -- Data structures ----------------------------------------------------------

@dataclass
class MediaFile:
    path: Path
    kind: str           # "image" | "video"
    size: int           # bytes
    phash: object = None        # imagehash.ImageHash (images only)
    duration: float = None      # seconds (videos only)
    sha256: str = None          # used for exact matching


@dataclass
class DuplicateGroup:
    kind: str
    reason: str         # "exact" | "phash" | "duration+exact" | "duration"
    files: list = field(default_factory=list)


# -- Utilities ----------------------------------------------------------------

def file_sha256(path: Path, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def image_phash(path: Path):
    try:
        with Image.open(path) as img:
            return imagehash.phash(img)
    except Exception as e:
        print(f"  [WARN] Cannot read image {path.name}: {e}")
        return None


def video_duration(path: Path) -> float | None:
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps and fps > 0 and frames > 0:
            return frames / fps
        return None
    except Exception as e:
        print(f"  [WARN] Cannot read video {path.name}: {e}")
        return None


# -- Scanning -----------------------------------------------------------------

def _scan_one(path: Path) -> MediaFile | None:
    """Compute the hash/duration for a single file. Runs in a worker thread."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        kind = "image"
    elif ext in VIDEO_EXTENSIONS:
        kind = "video"
    else:
        return None

    mf = MediaFile(path=path, kind=kind, size=path.stat().st_size)

    if kind == "image":
        mf.phash = image_phash(path)
        if mf.phash is None:
            return None
    else:
        mf.duration = video_duration(path)
        if mf.duration is None:
            return None

    return mf


def scan_folder(folder: Path, recurse: bool = True, workers: int = 4) -> list[MediaFile]:
    walker = folder.rglob("*") if recurse else folder.iterdir()
    all_paths = sorted([p for p in walker if p.is_file()])
    total = len(all_paths)

    print(f"\nScanning: {folder}")
    print(f"Found {total} file(s). Analysing with {workers} worker(s)...\n")

    results: list[MediaFile] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_path = {pool.submit(_scan_one, p): p for p in all_paths}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            completed += 1
            try:
                mf = future.result()
            except Exception as e:
                print(f"  [WARN] {path.name}: {e}")
                continue

            if mf is None:
                continue

            if mf.kind == "image":
                print(f"  [{completed}/{total}] {path.name} pHash={str(mf.phash)[:8]}...")
            else:
                print(f"  [{completed}/{total}] {path.name} duration={mf.duration:.2f}s")

            results.append(mf)

    # Restore a stable order (as_completed arrives out of order)
    order = {p: i for i, p in enumerate(all_paths)}
    results.sort(key=lambda mf: order[mf.path])
    return results


# -- Matching: images ---------------------------------------------------------

def find_image_duplicates(
    images: list[MediaFile],
    phash_threshold: int,
    workers: int = 4,
) -> list[DuplicateGroup]:
    """
    Strategy:
    1. SHA-256 exact match (bit-identical files).
    2. pHash similarity match on the remainder (compressed variants).
    """
    groups = []
    used = set()

    # -- Exact duplicates (parallel SHA-256) --
    def _hash(mf: MediaFile) -> tuple[MediaFile, str]:
        return mf, file_sha256(mf.path)

    sha_map: defaultdict[str, list[MediaFile]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for mf, digest in tqdm(
            pool.map(_hash, images),
            total=len(images),
            desc="  Hashing images",
            unit="file",
        ):
            mf.sha256 = digest
            sha_map[digest].append(mf)

    for digest, group in sha_map.items():
        if len(group) >= 2:
            groups.append(DuplicateGroup(kind="image", reason="exact", files=group))
            for mf in group:
                used.add(mf.path)

    # -- pHash near-duplicates (excluding exact matches) --
    remaining = [mf for mf in images if mf.path not in used]

    # BK-tree keyed on pHash Hamming distance: O(n log n) average,
    # substantially faster than brute-force for large libraries.
    if remaining:
        tree = pybktree.BKTree(lambda a, b: a.phash - b.phash, remaining)
        visited = set()
        for mf in tqdm(remaining, desc="  pHash matching", unit="file"):
            if id(mf) in visited:
                continue
            # find() returns (distance, item) pairs within the threshold
            neighbours = [
                item for _, item in tree.find(mf, phash_threshold)
                if item is not mf and id(item) not in visited
            ]
            if neighbours:
                group = [mf] + neighbours
                groups.append(DuplicateGroup(kind="image", reason="phash", files=group))
                for m in group:
                    visited.add(id(m))
            else:
                visited.add(id(mf))

    return groups


# -- Matching: videos ---------------------------------------------------------

def find_video_duplicates(
    videos: list[MediaFile],
    duration_threshold: float,
    workers: int = 4,
    sha256: bool = False,
) -> list[DuplicateGroup]:
    """
    Strategy:
    1. (Optional, sha256=True) SHA-256 exact match.
    2. Group remaining files whose durations differ by <= threshold, then
       subdivide by SHA-256 when sha256=True:
         - same SHA-256  -> "duration+exact"
         - different     -> "duration" (likely re-encoded copies)
       When sha256=False, all duration matches are reported as "duration".
    """
    groups = []
    used = set()

    if sha256:
        # -- Exact duplicates (parallel SHA-256) --
        def _hash(mf: MediaFile) -> tuple[MediaFile, str]:
            return mf, file_sha256(mf.path)

        sha_map: defaultdict[str, list[MediaFile]] = defaultdict(list)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for mf, digest in tqdm(
                pool.map(_hash, videos),
                total=len(videos),
                desc="  Hashing videos",
                unit="file",
            ):
                mf.sha256 = digest
                sha_map[digest].append(mf)

        for digest, group in sha_map.items():
            if len(group) >= 2:
                groups.append(DuplicateGroup(kind="video", reason="exact", files=group))
                for mf in group:
                    used.add(mf.path)

    # -- Duration near-duplicates (excluding exact matches) --
    # After sorting by duration, use bisect to find the right-side boundary
    # of each window in O(log n), making the overall scan O(n log n) worst case.

    remaining = [mf for mf in videos if mf.path not in used]
    remaining.sort(key=lambda x: x.duration)
    durations = [mf.duration for mf in remaining]

    visited = set()
    for i, a in enumerate(remaining):
        if i in visited:
            continue
        # bisect_right finds the first index beyond duration + threshold
        hi = bisect.bisect_right(durations, a.duration + duration_threshold)
        group_members = [
            remaining[j] for j in range(i, hi) if j not in visited
        ]

        if len(group_members) >= 2:
            if sha256:
                # Subdivide: same SHA-256 -> exact; different -> duration-only
                sub_sha: defaultdict[str, list[MediaFile]] = defaultdict(list)
                for mf in group_members:
                    sub_sha[mf.sha256].append(mf)

                exact_groups = [v for v in sub_sha.values() if len(v) >= 2]
                for eg in exact_groups:
                    groups.append(DuplicateGroup(kind="video", reason="duration+exact", files=eg))
                    for mf in eg:
                        used.add(mf.path)

                duration_only = [mf for mf in group_members if mf.path not in used]
                if len(duration_only) >= 2:
                    groups.append(DuplicateGroup(kind="video", reason="duration", files=duration_only))
            else:
                groups.append(DuplicateGroup(kind="video", reason="duration", files=group_members))

        for j in range(i, hi):
            visited.add(j)

    return groups


# -- Reporting ----------------------------------------------------------------

REASON_LABELS = {
    "exact":          "Exact duplicate (bit-identical)",
    "phash":          "Visual near-duplicate (pHash)",
    "duration+exact": "Exact duplicate (same duration + same SHA-256)",
    "duration":       "Same duration (likely re-encoded copy)",
}

def format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

def format_duration(sec: float) -> str:
    if sec is None:
        return "?"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def print_report(groups: list[DuplicateGroup], show_paths: bool = True) -> str:
    if not groups:
        return "No duplicate media found.\n"

    lines = []
    img_groups = [g for g in groups if g.kind == "image"]
    vid_groups = [g for g in groups if g.kind == "video"]

    total_files = sum(len(g.files) for g in groups)
    total_groups = len(groups)

    lines.append("=" * 60)
    lines.append(f"  Results: {total_groups} duplicate group(s), {total_files} file(s) involved")
    lines.append("=" * 60)

    for section_label, section_groups in [("Images", img_groups), ("Videos", vid_groups)]:
        if not section_groups:
            continue
        lines.append(f"\n-- {section_label} ({len(section_groups)} group(s)) " + "-" * 30)

        for idx, g in enumerate(section_groups, 1):
            reason = REASON_LABELS.get(g.reason, g.reason)
            lines.append(f"\n  Group {idx}  [{reason}]")
            for mf in g.files:
                size_str = format_size(mf.size)
                if mf.kind == "video" and mf.duration is not None:
                    detail = f"{format_duration(mf.duration)}  {size_str}"
                else:
                    detail = size_str
                path_str = str(mf.path) if show_paths else mf.path.name
                lines.append(f"    {path_str}  ({detail})")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def save_json(groups: list[DuplicateGroup], output_path: Path):
    data = []
    for g in groups:
        data.append({
            "kind": g.kind,
            "reason": g.reason,
            "reason_label": REASON_LABELS.get(g.reason, g.reason),
            "files": [
                {
                    "path": str(mf.path),
                    "size": mf.size,
                    "duration": mf.duration,
                    "sha256": mf.sha256,
                }
                for mf in g.files
            ]
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


# -- Entry point --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect duplicate images and videos in a folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python find_duplicates.py ~/Photos
  python find_duplicates.py ~/Videos --video-threshold 2.0
  python find_duplicates.py ~/Media --image-threshold 5 --output report.json
  python find_duplicates.py ~/Media --no-recurse
        """
    )
    parser.add_argument("folder", type=Path, help="Folder to scan")
    parser.add_argument(
        "--video-threshold", type=float, default=1.0, metavar="SEC",
        help="Max duration difference (seconds) to consider videos identical (default: 1.0)"
    )
    parser.add_argument(
        "--image-threshold", type=int, default=8, metavar="DIST",
        help="Max pHash Hamming distance to consider images similar (0=identical, default: 8)"
    )
    parser.add_argument(
        "--output", type=Path, default=None, metavar="PATH",
        help="Write results to file (.txt plain text, .json structured)"
    )
    parser.add_argument(
        "--no-recurse", action="store_true",
        help="Scan top-level folder only, do not recurse into subdirectories"
    )
    parser.add_argument(
        "--images-only", action="store_true", help="Scan images only"
    )
    parser.add_argument(
        "--videos-only", action="store_true", help="Scan videos only"
    )
    parser.add_argument(
        "--video-sha256", action="store_true", default=False,
        help="Enable SHA-256 hashing for videos to detect exact duplicates (default: off)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, metavar="N",
        help="Number of threads for hashing (default: 4)"
    )

    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"[ERROR] Folder not found: {args.folder}")
        sys.exit(1)

    # Scan
    all_files = scan_folder(args.folder, recurse=not args.no_recurse, workers=args.workers)
    images = [f for f in all_files if f.kind == "image"] if not args.videos_only else []
    videos = [f for f in all_files if f.kind == "video"] if not args.images_only else []

    print(f"\nImages: {len(images)}  Videos: {len(videos)}")

    # Match
    print("\nMatching...")
    groups: list[DuplicateGroup] = []

    if images:
        print(f"  Image pHash (threshold={args.image_threshold})...")
        groups += find_image_duplicates(images, args.image_threshold, workers=args.workers)

    if videos:
        print(f"  Video duration (threshold={args.video_threshold}s){', SHA-256 enabled' if args.video_sha256 else ''}...")
        groups += find_video_duplicates(videos, args.video_threshold, workers=args.workers, sha256=args.video_sha256)

    # Output
    report = print_report(groups)
    print("\n" + report)

    if args.output:
        if args.output.suffix.lower() == ".json":
            save_json(groups, args.output)
        else:
            args.output.write_text(report, encoding="utf-8")
            print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
    