#!/usr/bin/env python3
"""
video_transcode.py -- Transcode videos to HEVC using HandBrakeCLI (NVIDIA NVENC).

Encodes files whose bitrate exceeds their bitrate limit and/or whose resolution
exceeds --max-resolution. Other files are skipped. Audio tracks are copied
without re-encoding.

The bitrate limit is either fixed (--max-bitrate) or scaled to each file's
output resolution (--ref-bitrate, or a [bitrate.table] in a profile), so a
1080p file gets a smaller budget than a 1440p one at the same quality level.
--max-bitrate always acts as an absolute ceiling.

Without --output-dir, originals are replaced in place: each file is encoded to
a temporary file alongside it, verified (exit code, non-empty output, and
duration matching the source), and only then does the original get deleted.
If the result is not smaller than the original, the original is kept.

Usage:
    python video_transcode.py <folder> [--max-bitrate <kbps>] [--max-resolution <res>] [options]

Options:
    -p, --profile <name|path>
                            Load settings from a TOML profile (see profiles/example.toml).
                            A bare name is looked up in ./profiles next to this script,
                            then ~/.config/media-archiver/profiles. CLI flags override it.
    --max-bitrate <kbps>    Absolute maximum bitrate in kbps, for every resolution
    --ref-bitrate <kbps@res>
                            Reference bitrate at a resolution, e.g. 6000@1080p. Each
                            file's limit is scaled by (output pixels / reference pixels)^0.75
    --max-resolution <res>  Maximum resolution, e.g. 1080p, 4k, or 1920x1080.
                            The aspect ratio is kept and the box follows the
                            video's orientation (1920x1080 box for landscape,
                            1080x1920 for portrait). Never upscales.
                            (at least one bitrate limit or --max-resolution is required)
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
    python video_transcode.py ~/Videos --ref-bitrate 6000@1080p --max-resolution 1440p
    python video_transcode.py ~/Videos -p archive --min-saving 25
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
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

# Bitrate needed for equal quality grows slower than pixel count: 4x the pixels
# needs well under 4x the bitrate. 0.7-0.8 is the usual rule of thumb for HEVC.
DEFAULT_BITRATE_EXPONENT = 0.75
DEFAULT_FPS_EXPONENT = 0.5

# Where a bare --profile name is looked up, in order.
PROFILE_DIRS = [
    Path(__file__).resolve().parent / "profiles",
    Path.home() / ".config" / "media-archiver" / "profiles",
]

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

def _parse_rate(text: str | None) -> float | None:
    """'30000/1001' -> 29.97; None for missing or degenerate values like '0/0'."""
    try:
        num, _, den = (text or "").partition("/")
        value = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None

def probe_streams(ffprobe: str, filepath: Path) -> dict:
    """
    Return {"width", "height", "fps", "audio_kbps"}. width/height are the *display*
    dimensions (rotation applied) of the first video stream, or None on failure.
    audio_kbps is the summed bitrate of all audio streams.
    """
    info = {"width": None, "height": None, "fps": None, "audio_kbps": 0.0}
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
                info["fps"] = (_parse_rate(s.get("avg_frame_rate"))
                               or _parse_rate(s.get("r_frame_rate")))
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

# -- Bitrate limit ------------------------------------------------------------

@dataclass
class BitratePolicy:
    """
    Per-file bitrate limit. With no points the limit is simply `cap` (fixed mode).
    With points (pixels, kbps, label) it is scaled to the output pixel count:
      - one point:  kbps * (pixels / ref_pixels) ^ exponent
      - a table:    interpolated in log-log space (a piecewise power curve),
                    extrapolated past either end with the nearest segment's slope
    then multiplied by (fps / fps_ref) ^ fps_exponent when fps_ref is set, and
    finally clamped to `cap`.
    """
    cap: float | None = None
    points: list[tuple[int, float, str]] = field(default_factory=list)
    exponent: float = DEFAULT_BITRATE_EXPONENT
    fps_ref: float | None = None
    fps_exponent: float = DEFAULT_FPS_EXPONENT

    @property
    def scaled(self) -> bool:
        return bool(self.points)

    def limit_for(self, pixels: int | None, fps: float | None) -> float | None:
        if not self.points or not pixels:
            return self.cap
        if len(self.points) == 1:
            p0, k0, _ = self.points[0]
            kbps = k0 * (pixels / p0) ** self.exponent
        else:
            pts = self.points
            i = 0
            while i < len(pts) - 2 and pixels > pts[i + 1][0]:
                i += 1
            (pa, ka, _), (pb, kb, _) = pts[i], pts[i + 1]
            slope = math.log(kb / ka) / math.log(pb / pa)
            kbps = ka * (pixels / pa) ** slope
        if self.fps_ref and fps:
            kbps *= (fps / self.fps_ref) ** self.fps_exponent
        if self.cap is not None:
            kbps = min(kbps, self.cap)
        return kbps

def _resolution_point(res: str, kbps: float) -> tuple[int, float, str]:
    long_side, short_side = parse_max_resolution(res)
    if kbps <= 0:
        raise ValueError(f"bitrate for {res} must be positive")
    label = f"{long_side}x{short_side}"
    if res.strip().lower() != label:
        label = f"{res.strip()} ({label})"
    return long_side * short_side, float(kbps), label

def parse_ref_bitrate(text: str) -> tuple[int, float, str]:
    """'6000@1080p' -> (pixels, 6000.0, '1920x1080')."""
    kbps, sep, res = text.partition("@")
    try:
        if not sep:
            raise ValueError
        value = float(kbps)
    except ValueError:
        raise ValueError(f"invalid reference bitrate '{text}' (use <kbps>@<res>, e.g. 6000@1080p)")
    return _resolution_point(res, value)

def parse_bitrate_table(table: dict) -> list[tuple[int, float, str]]:
    """{'720p': 3500, '1080p': 6000, ...} -> points sorted by pixel count."""
    points = []
    for res, kbps in table.items():
        if isinstance(kbps, bool) or not isinstance(kbps, (int, float)):
            raise ValueError(f"bitrate for '{res}' must be a number")
        points.append(_resolution_point(res, kbps))
    points.sort()
    for (pa, ka, la), (pb, kb, lb) in zip(points, points[1:]):
        if pa == pb:
            raise ValueError(f"'{la}' and '{lb}' have the same pixel count")
        if kb < ka:
            raise ValueError(f"bitrate must not decrease with resolution ({la}: {ka:g} > {lb}: {kb:g})")
    return points

# -- Profiles -----------------------------------------------------------------

class ProfileError(ValueError):
    pass

_NUMBER = (int, float)
_TABLE = "table"   # free-form resolution -> kbps mapping, validated separately

# Every key a profile may contain. Unknown keys are rejected so that a typo
# never silently falls back to a default -- in place mode deletes originals.
PROFILE_SCHEMA = {
    "max_resolution": str,
    "min_saving": _NUMBER,
    "bitrate": {
        "max": _NUMBER,
        "reference": str,
        "exponent": _NUMBER,
        "table": _TABLE,
        "fps": {"reference": _NUMBER, "exponent": _NUMBER},
    },
}

def _check_schema(data: dict, schema: dict, where: str = ""):
    for key, value in data.items():
        path = f"{where}.{key}" if where else key
        if key not in schema:
            raise ProfileError(f"unknown key '{path}' (allowed here: {', '.join(schema)})")
        expected = schema[key]
        if isinstance(expected, dict) or expected == _TABLE:
            if not isinstance(value, dict):
                raise ProfileError(f"'{path}' must be a table")
            if isinstance(expected, dict):
                _check_schema(value, expected, path)
        elif isinstance(value, bool) or not isinstance(value, expected):
            kind = "a number" if expected is _NUMBER else "a string"
            raise ProfileError(f"'{path}' must be {kind}")

def find_profile(name: str) -> Path:
    """A path to an existing file, or a bare name looked up in PROFILE_DIRS."""
    direct = Path(name).expanduser()
    if direct.is_file():
        return direct
    if direct.parent == Path("."):
        filename = name if name.endswith(".toml") else f"{name}.toml"
        for d in PROFILE_DIRS:
            if (d / filename).is_file():
                return d / filename
    searched = "\n".join(f"          {d}" for d in PROFILE_DIRS)
    print(f"[ERROR] Profile not found: {name}\n        Searched:\n{searched}")
    sys.exit(1)

def load_profile(path: Path) -> dict:
    """Read and validate a TOML profile; return its settings keyed by argparse dest."""
    try:
        import tomllib
    except ModuleNotFoundError:
        raise ProfileError("profiles require Python 3.11+ (tomllib)")
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as e:
        raise ProfileError(f"invalid TOML: {e}")

    _check_schema(data, PROFILE_SCHEMA)
    bitrate = data.get("bitrate", {})
    fps = bitrate.get("fps", {})
    if "reference" in bitrate and "table" in bitrate:
        raise ProfileError("use either 'bitrate.reference' or 'bitrate.table', not both")
    if fps and "reference" not in fps:
        raise ProfileError("'bitrate.fps.reference' is required when [bitrate.fps] is present")

    table = None
    if "table" in bitrate:
        try:
            table = parse_bitrate_table(bitrate["table"])
        except ValueError as e:
            raise ProfileError(f"bitrate.table: {e}")
        if not table:
            raise ProfileError("'bitrate.table' is empty")

    settings = {
        "max_resolution":   data.get("max_resolution"),
        "min_saving":       data.get("min_saving"),
        "max_bitrate":      bitrate.get("max"),
        "ref_bitrate":      bitrate.get("reference"),
        "bitrate_exponent": bitrate.get("exponent"),
        "bitrate_table":    table,
        "fps_ref":          fps.get("reference"),
        "fps_exponent":     fps.get("exponent"),
    }
    return {k: v for k, v in settings.items() if v is not None}

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
    policy: BitratePolicy,
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
        "fps":            float | None,
        "action":         "transcode" | "skip" | "low_saving" | "probe_failed",
        "output":         Path | None,
        "limit":          float | None,        # this file's bitrate limit (kbps)
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
            "fps": None,
            "action": "skip",
            "output": None,
            "limit": None,
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
        w, h, fps = streams["width"], streams["height"], streams["fps"]
        entry["width"], entry["height"], entry["fps"] = w, h, fps

        # A resolution-scaled limit is meaningless without the dimensions.
        if policy.scaled and not (w and h):
            entry["action"] = "probe_failed"
            plan.append(entry)
            continue

        box = scale_to = None
        if max_resolution and w and h:
            box = box_for(w, h, max_resolution)
            scale_to = fit_within(w, h, box)

        # The limit follows the *output* resolution: a 4K file downscaled to
        # 1440p gets the 1440p budget.
        out_w, out_h = scale_to or (w, h)
        limit = policy.limit_for(out_w * out_h if w and h else None, fps)
        entry["limit"] = limit

        over_bitrate = limit is not None and bitrate > limit
        if not over_bitrate and scale_to is None:
            plan.append(entry)   # action stays "skip"
            continue

        # Target: the limit, but never more than the source carries. When
        # downscaling, the source bitrate is first adjusted to the smaller
        # frame along the same pixels^exponent curve.
        source_equiv = bitrate
        if scale_to:
            source_equiv *= ((scale_to[0] * scale_to[1]) / (w * h)) ** policy.exponent
        target = source_equiv if limit is None else min(limit, source_equiv)
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
    if e["fps"]:
        parts.append(f"{e['fps']:.3g} fps")
    if e["size"] is not None:
        parts.append(fmt_size(e["size"]))
    if e["limit"] is not None:
        parts.append(f"limit {e['limit']:.0f} kbps")
    return ", ".join(parts)

def describe_policy(policy: BitratePolicy, sources: dict[str, str]) -> list[str]:
    """Preview lines for the bitrate settings, tagged with where each came from."""
    def tag(*keys: str) -> str:
        return "  [profile]" if any(sources.get(k) == "profile" for k in keys) else ""

    cap = f"{policy.cap:.0f} kbps{tag('max_bitrate')}" if policy.cap is not None else "(none)"
    if not policy.scaled:
        return [f"  max-bitrate    : {cap}"]

    if len(policy.points) == 1:
        _, kbps, label = policy.points[0]
        lines = [f"  ref-bitrate    : {kbps:.0f} kbps @ {label}{tag('ref_bitrate')}",
                 f"  scaling        : (pixels / reference)^{policy.exponent:g}"
                 f"{tag('bitrate_exponent')}"]
    else:
        lines = [f"  bitrate table  : (log-log interpolated){tag('bitrate_table')}"]
        width = max(len(label) for _, _, label in policy.points)
        lines += [f"                     {label:<{width}} : {kbps:.0f} kbps"
                  for _, kbps, label in policy.points]
    if policy.fps_ref:
        lines.append(f"  fps scaling    : (fps / {policy.fps_ref:g})^{policy.fps_exponent:g}"
                     f"{tag('fps_ref', 'fps_exponent')}")
    lines.append(f"  max-bitrate    : {cap}")
    return lines

def print_preview(
    plan: list[dict],
    policy: BitratePolicy,
    max_resolution: tuple[int, int] | None,
    min_saving: float | None,
    output_dir: Path | None,
    sources: dict[str, str] | None = None,
    profile_path: Path | None = None,
):
    sources = sources or {}
    transcode  = [e for e in plan if e["action"] == "transcode"]
    skip       = [e for e in plan if e["action"] == "skip"]
    low_saving = [e for e in plan if e["action"] == "low_saving"]
    failed     = [e for e in plan if e["action"] == "probe_failed"]

    def tag(key: str) -> str:
        return "  [profile]" if sources.get(key) == "profile" else ""

    print(f"\n{'─'*72}")
    if profile_path:
        print(f"  profile        : {profile_path}")
    for line in describe_policy(policy, sources):
        print(line)
    if max_resolution:
        print(f"  max-resolution : {max_resolution[0]}x{max_resolution[1]} "
              f"(orientation follows the video){tag('max_resolution')}")
    else:
        print("  max-resolution : (none)")
    if min_saving is not None:
        print(f"  min-saving     : {min_saving:g}%{tag('min_saving')}")
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
        print(f"\n  [WARN] Could not probe bitrate or resolution ({len(failed)}):\n")
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
    parser.add_argument("-p", "--profile", metavar="NAME|PATH",
                        help="TOML profile with transcoding settings (see profiles/example.toml); "
                             "CLI flags override its values")
    parser.add_argument("--max-bitrate",   type=float, metavar="KBPS",
                        help="Absolute maximum bitrate in kbps, applied at every resolution")
    parser.add_argument("--ref-bitrate",   metavar="KBPS@RES",
                        help="Reference bitrate at a resolution (e.g. 6000@1080p); each file's "
                             "limit is scaled to its output resolution")
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
    # Profile-only settings (no CLI flag)
    parser.set_defaults(bitrate_exponent=None, bitrate_table=None,
                        fps_ref=None, fps_exponent=None)
    args = parser.parse_args()

    # Precedence: built-in defaults < profile < CLI flags.
    sources: dict[str, str] = {}
    profile_path = None
    if args.profile:
        profile_path = find_profile(args.profile)
        try:
            profile = load_profile(profile_path)
        except ProfileError as e:
            parser.error(f"profile {profile_path}: {e}")
        for key, value in profile.items():
            # --ref-bitrate on the CLI replaces the profile's table as well.
            if key == "bitrate_table" and args.ref_bitrate is not None:
                continue
            if getattr(args, key) is None:
                setattr(args, key, value)
                sources[key] = "profile"

    def origin(key: str, flag: str, profile_key: str) -> str:
        """Name a setting the way the user wrote it, for error messages."""
        return f"'{profile_key}' in profile" if sources.get(key) == "profile" else flag

    if (args.max_bitrate is None and args.ref_bitrate is None
            and args.bitrate_table is None and args.max_resolution is None):
        parser.error("at least one of --max-bitrate / --ref-bitrate / --max-resolution "
                     "(or a profile setting them) is required")
    if args.max_bitrate is not None and args.max_bitrate <= 0:
        parser.error(f"{origin('max_bitrate', '--max-bitrate', 'bitrate.max')} must be positive")
    if args.min_saving is not None and not (0 <= args.min_saving < 100):
        parser.error(f"{origin('min_saving', '--min-saving', 'min_saving')} must be between 0 and 100")
    if args.bitrate_exponent is not None and args.bitrate_exponent <= 0:
        parser.error("bitrate.exponent must be positive")
    if args.fps_ref is not None and args.fps_ref <= 0:
        parser.error("bitrate.fps.reference must be positive")
    if args.fps_exponent is not None and args.fps_exponent < 0:
        parser.error("bitrate.fps.exponent must not be negative")

    policy = BitratePolicy(cap=args.max_bitrate, fps_ref=args.fps_ref)
    if args.bitrate_exponent is not None:
        policy.exponent = args.bitrate_exponent
    if args.fps_exponent is not None:
        policy.fps_exponent = args.fps_exponent
    if args.ref_bitrate is not None:
        try:
            policy.points = [parse_ref_bitrate(args.ref_bitrate)]
        except ValueError as e:
            parser.error(f"{origin('ref_bitrate', '--ref-bitrate', 'bitrate.reference')}: {e}")
    elif args.bitrate_table is not None:
        policy.points = args.bitrate_table
    if policy.fps_ref and not policy.scaled:
        parser.error("bitrate.fps needs a scaled limit (--ref-bitrate, bitrate.reference or bitrate.table)")

    max_resolution = None
    if args.max_resolution:
        try:
            max_resolution = parse_max_resolution(args.max_resolution)
        except ValueError as e:
            parser.error(f"{origin('max_resolution', '--max-resolution', 'max_resolution')}: {e}")

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
    plan = build_plan(files, ffprobe, policy, max_resolution, args.min_saving,
                      output_dir, folder, args.recursive, args.postfix)

    # Show preview
    print_preview(plan, policy, max_resolution, args.min_saving, output_dir,
                  sources, profile_path)

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
