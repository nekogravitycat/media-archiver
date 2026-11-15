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

Transcodes videos to HEVC (H.265) using HandBrakeCLI with NVIDIA NVENC hardware encoding. Only files whose bitrate exceeds their bitrate limit and/or whose resolution exceeds `--max-resolution` are transcoded; others are skipped. The limit can be fixed (`--max-bitrate`) or scaled to each file's resolution (`--ref-bitrate`, or a table in a [profile](#profiles)), so 1080p and 1440p files in the same run each get an appropriate budget. Audio tracks are copied without re-encoding. The preview shows the estimated output size and saving for each file, and `--min-saving` can skip files that would not shrink enough.

By default the originals are **replaced in place**, and each one is deleted only after its replacement has been verified — see [In-place replacement](#in-place-replacement-no---output-dir). Pass `--output-dir` to write elsewhere and leave the originals untouched.

### Requirements

- [HandBrake](https://handbrake.fr/) (provides `HandBrakeCLI`)
- [FFmpeg](https://ffmpeg.org/) (provides `ffprobe`)
- An NVIDIA GPU with NVENC support

Both `HandBrakeCLI` and `ffprobe` must be available in `PATH`, or their paths supplied via `--handbrake`. Profiles (`-p`) need Python 3.11+.

### Supported formats

`.mp4` `.mov` `.mkv` `.avi` `.m4v` `.wmv` `.flv` `.mts` `.m2ts` `.3gp` `.webm`

### Usage

```
python video_transcode.py <folder> [-p <profile>] [--max-bitrate <kbps>] [--ref-bitrate <kbps@res>] [--max-resolution <res>] [options]
```

At least one of `--max-bitrate` / `--ref-bitrate` / `--max-resolution` is required, either as a flag or from a profile.

| Option                   | Description                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `-p, --profile <name>`   | Load settings from a TOML profile (see [Profiles](#profiles)); flags override it                             |
| `--max-bitrate <kbps>`   | Absolute bitrate limit at every resolution; files above it are transcoded                                    |
| `--ref-bitrate <kbps@res>` | Reference bitrate, e.g. `6000@1080p`; each file's limit is scaled to its output resolution (see [below](#resolution-scaled-bitrate)) |
| `--max-resolution <res>` | Files larger than this are scaled down, e.g. `1080p`, `4k`, `1920x1080` (see [below](#resolution-limiting)) |
| `--min-saving <percent>` | Skip files whose estimated saving is below this percentage (0–100)                                           |
| `--output-dir <path>`    | Output directory (default: replace the originals in place)                                                   |
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

# Limit both bitrate and resolution
python video_transcode.py ~/Videos --max-bitrate 8000 --max-resolution 1080p --execute

# Resolution only, skipping files that would save less than 20%
python video_transcode.py ~/Videos --max-resolution 1280x720 --min-saving 20

# Mixed FHD / 2K / 4K library capped at 1440p; 1080p files get ~6000 kbps, 1440p ~9200 kbps
python video_transcode.py ~/Videos --ref-bitrate 6000@1080p --max-resolution 1440p

# Use profiles/archive.toml, overriding one of its values
python video_transcode.py ~/Videos -p archive --min-saving 25
```

Output files mirror the source folder structure when `--recursive` is used. Naming conflicts are resolved automatically by appending `_2`, `_3`, etc.

### Resolution limiting

`--max-resolution` accepts `<N>p` (16:9 box with N as the short side, e.g. `1080p` = 1920x1080), `4k` / `uhd` (2160p), `8k` (4320p), or an explicit `<W>x<H>`.

- The aspect ratio is kept, and the box follows the video's orientation (1920x1080 for landscape, 1080x1920 for portrait). Rotation metadata is taken into account.
- Videos are **never upscaled**; those already within the limit are skipped (unless they exceed their bitrate limit).
- Output dimensions are rounded down to even numbers.

### Resolution-scaled bitrate

At the same quality, a 1080p video needs less bitrate than a 1440p one, so a single fixed `--max-bitrate` is either too generous for the smaller files or too tight for the larger ones. `--ref-bitrate <kbps>@<res>` sets the limit at one reference resolution and scales it to each file:

```
limit = kbps × (output pixels / reference pixels) ^ 0.75
```

The exponent is below 1 because the bitrate needed for equal quality grows more slowly than the pixel count. With `6000@1080p` that gives 720p ≈ 3300, 1080p = 6000, 1440p ≈ 9200, 4K ≈ 17000 kbps.

- The limit uses the **output** resolution: a 4K file downscaled by `--max-resolution 1440p` gets the 1440p budget.
- `--max-bitrate` still applies as an absolute ceiling on top.
- A profile can instead give an explicit table of resolution → kbps (interpolated between points), change the exponent, and add a frame-rate adjustment. See [Profiles](#profiles).

### Target bitrate and size estimate

- The encoder targets `min(limit, source bitrate)`, so the bitrate is never raised above the source's. When a file is downscaled, the source bitrate is first scaled to the smaller frame along the same `pixels ^ exponent` curve.
- With `--max-resolution` only (no bitrate limit), the target is that scaled source bitrate.
- The preview estimates each output as `duration × (target video bitrate + audio bitrate)`. This is an approximation; NVENC often lands at or below the target. Audio streams reporting no bitrate are assumed to be 128 kbps.
- `--min-saving <percent>` skips files whose estimated saving falls below the threshold; they are listed separately in the preview.

### HDR

HDR sources (HLG or HDR10/PQ, detected from the stream's transfer characteristics) are encoded with `nvenc_h265_10bit` as 10-bit Main10, so the BT.2020 color and HLG/PQ transfer signaling carry over. SDR sources keep the 8-bit `nvenc_h265` encoder.

- HDR dynamic metadata is requested with `--hdr-dynamic-metadata all`, but as of HandBrake 1.11 the NVENC encoder drops Dolby Vision. iPhone Dolby Vision clips (profile 8.4) still play back as HLG HDR without it.
- The preview shows the HDR type next to each HDR file.
- HDR needs more bitrate than SDR for the same quality. A profile can set `bitrate.hdr_factor` (e.g. `1.25`, the HDR/SDR ratio in YouTube's recommendations) to raise the limit for HDR files. It needs a scaled limit, and `bitrate.max` still caps the result.

### Profiles

A profile is a TOML file that holds transcoding settings, so everyday runs can stay short while detailed tuning lives in a file. Start from the documented template:

```bash
cp profiles/example.toml profiles/archive.toml
python video_transcode.py ~/Videos -p archive
```

`-p` accepts a path, or a bare name looked up as `<name>.toml` in `profiles/` next to the script, then in `~/.config/media-archiver/profiles/`. Only `profiles/example.toml` is tracked by git; your own profiles are ignored.

```toml
max_resolution = "1440p"     # --max-resolution
min_saving     = 15          # --min-saving

[bitrate]
max       = 20000            # --max-bitrate (absolute ceiling)
reference = "6000@1080p"     # --ref-bitrate
exponent  = 0.75             # scaling exponent (default 0.75)
hdr_factor = 1.25            # limit multiplier for HDR sources (default 1)

# Instead of `reference`, an explicit table, interpolated on a log-log curve:
# [bitrate.table]
# "720p"  = 3500
# "1080p" = 6000
# "1440p" = 10000

# Optional frame-rate adjustment: limit *= (fps / reference) ^ exponent
# [bitrate.fps]
# reference = 30
# exponent  = 0.5
```

- Precedence is built-in defaults < profile < command-line flags. `--ref-bitrate` on the command line also replaces a profile's `[bitrate.table]`.
- Unknown keys and wrong types are rejected, so a typo fails instead of silently falling back to a default.
- Per-run choices (`--execute`, `--output-dir`, `--recursive`, `--postfix`, …) cannot be set in a profile.
- The preview shows the effective settings and marks the ones that came from the profile with `[profile]`. Each file's line shows its own computed limit.

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
