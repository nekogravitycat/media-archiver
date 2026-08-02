# media-archiver

A collection of Python scripts for organizing, compressing, and deduplicating media archives.

- **`media_date_rename.py`** — Rename photos and videos to their capture date/time.
- **`video_transcode.py`** — Transcode videos to HEVC using HandBrakeCLI (NVIDIA NVENC).
- **`find_duplicates.py`** — Detect duplicate images and videos using perceptual hashing and duration matching.

---

## media_date_rename.py

Renames media files using their capture or creation date, producing filenames like `2024-06-15 14-32-01.jpg`.

**Date source priority:**

1. EXIF `DateTimeOriginal` (images)
2. MP4/MOV `mvhd` box creation time (videos)
3. Earliest filesystem timestamp (fallback)

### Requirements

```
pip install piexif Pillow pillow-heif
```

### Supported formats

Images: `.jpg` `.jpeg` `.heic` `.heif` `.png` `.tiff` `.tif` `.webp` `.bmp` `.gif` `.raw` `.cr2` `.cr3` `.nef` `.arw` `.dng` `.orf` `.rw2` `.pef`

Videos: `.mov` `.mp4` `.m4v` `.avi` `.mkv` `.wmv` `.flv` `.mts` `.m2ts` `.3gp` `.webm`

### Usage

```
python media_date_rename.py <folder> [options]
```

| Option               | Description                                              |
| -------------------- | -------------------------------------------------------- |
| `--execute`          | Apply the renames (default: preview only)                |
| `--recursive`        | Process subdirectories recursively                       |
| `--skip-errors`      | Skip files that cause errors instead of aborting         |
| `--postfix <string>` | Append a string to output filenames before the extension |

### Examples

```bash
# Preview what would be renamed
python media_date_rename.py ~/Pictures

# Apply renames
python media_date_rename.py ~/Pictures --execute

# Rename recursively, appending a postfix to each filename
python media_date_rename.py ~/Pictures --recursive --postfix " - vacation" --execute
```

---

## video_transcode.py

Transcodes videos to HEVC (H.265) using HandBrakeCLI with NVIDIA NVENC hardware encoding. Only files whose bitrate exceeds `--max-bitrate` are transcoded; others are skipped. Audio tracks are copied without re-encoding.

By default the originals are **replaced in place**, and each one is deleted only after its replacement has been verified — see [In-place replacement](#in-place-replacement-no---output-dir). Pass `--output-dir` to write elsewhere and leave the originals untouched.

### Requirements

- [HandBrake](https://handbrake.fr/) (provides `HandBrakeCLI`)
- [FFmpeg](https://ffmpeg.org/) (provides `ffprobe`)
- An NVIDIA GPU with NVENC support

Both `HandBrakeCLI` and `ffprobe` must be available in `PATH`, or their paths supplied via `--handbrake`.

### Supported formats

`.mp4` `.mov` `.mkv` `.avi` `.m4v` `.wmv` `.flv` `.mts` `.m2ts` `.3gp` `.webm`

### Usage

```
python video_transcode.py <folder> --max-bitrate <kbps> [options]
```

| Option                 | Description                                                |
| ---------------------- | ---------------------------------------------------------- |
| `--max-bitrate <kbps>` | Files above this bitrate are transcoded **(required)**     |
| `--output-dir <path>`  | Output directory (default: replace the originals in place) |
| `--recursive`          | Process subdirectories recursively                         |
| `--execute`            | Apply transcoding (default: preview only)                  |
| `--dry-run`            | Print the exact HandBrakeCLI commands without running them |
| `--skip-errors`        | Continue past files that fail                              |
| `--handbrake <path>`   | Path to HandBrakeCLI binary (default: auto-detect)         |
| `--postfix <string>`   | Append a string to output filenames (default: empty)       |
| `--show-log`           | Stream HandBrakeCLI output directly to the terminal        |

### Examples

```bash
# Preview which files exceed 8000 kbps
python video_transcode.py ~/Videos --max-bitrate 8000

# Replace the originals in place
python video_transcode.py ~/Videos --max-bitrate 8000 --execute

# Write to a separate output directory instead, leaving the originals alone
python video_transcode.py ~/Videos --max-bitrate 8000 --output-dir ~/Videos/out --execute

# Print the exact HandBrakeCLI commands that would run (no transcoding)
python video_transcode.py ~/Videos --max-bitrate 8000 --dry-run
```

Output files mirror the source folder structure when `--recursive` is used. Naming conflicts are resolved automatically by appending `_2`, `_3`, etc.

### In-place replacement (no `--output-dir`)

Without `--output-dir`, each original is replaced by its transcoded version. The original is never deleted before its replacement exists and has been checked:

1. HandBrake encodes to a temporary `*.transcoding.tmp.mp4` file next to the destination.
2. The result is verified — HandBrakeCLI exited `0`, the file exists and is non-empty, and its duration is within 1 second of the source.
3. If the result is **not smaller** than the original, it is discarded and the original kept (reported as `kept (not smaller)`).
4. Only then is the original deleted and the temporary file renamed into place.

Any failure at steps 1–2 discards the temporary file and leaves the original untouched; interrupting with `Ctrl-C` does the same. The new file inherits the original's modification and access times, so date-based tooling such as `media_date_rename.py` keeps working.

Because HandBrake writes an MP4 container, a non-MP4 source changes extension: `foo.mkv` becomes `foo.mp4` and `foo.mkv` is removed. If `foo.mp4` already exists, the output becomes `foo_2.mp4`.

---

## find_duplicates.py

Detects duplicate images and videos in a folder.

- **Images**: perceptual hash (pHash) — tolerates compression, resizing, and re-encoding.
- **Videos**: duration comparison, with optional SHA-256 confirmation (`--video-sha256`).

### Requirements

```
pip install imagehash Pillow opencv-python pybktree pillow-heif
```

### Supported formats

Images: `.jpg` `.jpeg` `.png` `.webp` `.bmp` `.gif` `.tiff` `.tif` `.heic` `.heif`

Videos: `.mp4` `.mov` `.avi` `.mkv` `.wmv` `.flv` `.webm` `.m4v` `.3gp` `.ts` `.mts` `.m2ts` `.rmvb` `.rm`

### Detection strategy

| Media  | Step 1                | Step 2                                           |
| ------ | --------------------- | ------------------------------------------------ |
| Images | SHA-256 exact match   | pHash Hamming distance (near-duplicates)         |
| Videos | Duration window match | SHA-256 subdivision (only with `--video-sha256`) |

Duplicate reasons reported:

| Reason           | Meaning                                              |
| ---------------- | ---------------------------------------------------- |
| `exact`          | Bit-identical files                                  |
| `phash`          | Visually similar (different compression/resize)      |
| `duration+exact` | Same duration and identical SHA-256                  |
| `duration`       | Same duration, different content (likely re-encoded) |

### Usage

```
python find_duplicates.py <folder> [options]
```

| Option                     | Description                                                                         |
| -------------------------- | ----------------------------------------------------------------------------------- |
| `--image-threshold <dist>` | Max pHash Hamming distance to consider images similar (0 = identical, default: `8`) |
| `--video-threshold <sec>`  | Max duration difference (seconds) to consider videos identical (default: `1.0`)     |
| `--output <path>`          | Write results to file (`.txt` plain text or `.json` structured)                     |
| `--no-recurse`             | Scan top-level folder only, do not recurse into subdirectories                      |
| `--images-only`            | Scan images only                                                                    |
| `--videos-only`            | Scan videos only                                                                    |
| `--video-sha256`           | Enable SHA-256 hashing for videos to detect exact duplicates (default: off)         |
| `--workers <n>`            | Number of threads for hashing (default: `4`)                                        |

### Examples

```bash
# Scan a folder for all duplicate media
python find_duplicates.py ~/Photos

# Use a stricter pHash threshold (exact visual matches only)
python find_duplicates.py ~/Photos --image-threshold 0

# Allow a 2-second window for video duration matching
python find_duplicates.py ~/Videos --video-threshold 2.0

# Save results as JSON
python find_duplicates.py ~/Media --output report.json

# Scan top-level only, images only
python find_duplicates.py ~/Media --no-recurse --images-only

# Enable SHA-256 for video exact-duplicate detection
python find_duplicates.py ~/Videos --video-sha256
```
