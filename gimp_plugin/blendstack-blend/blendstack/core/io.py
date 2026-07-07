"""Image loading, saving and bit-depth handling (brief §2 formats, §4.4).

Loading — everything becomes float32 RGB, shape (H, W, 3), values 0–1
(brief §3 design rule 3):

* TIFF/JPEG/PNG/GIF (first frame only)/BMP/WebP via Pillow — except 16-bit
  TIFF, which goes through imageio's tifffile backend because Pillow
  silently truncates 16-bit RGB to 8-bit on read (verified against Pillow
  12.x; this would break the brief §8 lossless round-trip criterion).
* RAW (.cr2/.cr3/.nef/.arw/.dng, …) via rawpy with LibRaw **default**
  postprocessing — default demosaic, white balance and tone mapping, no
  RAW controls in v1 (brief §2).
* Alpha channels are flattened against black on load.
* 16-bit sources are divided by 65535, 8-bit by 255.

Saving (brief §4.4) — TIFF and PNG write 16-bit, JPEG writes 8-bit quality
95 with 4:4:4 chroma (subsampling=0).  The float accumulator is clipped to
0–1 then quantised with rounding.  An sRGB profile is tagged on every
output.  Because Pillow cannot *write* 16-bit RGB either, TIFF and PNG are
emitted by small self-contained encoders in this module (uncompressed
little-endian TIFF with an ICC tag; zlib-compressed PNG with sRGB/gAMA/cHRM
chunks) — both verified readable by imageio/tifffile, GIMP and macOS.

Default output filename pattern: ``blend_<mode>_<YYYYMMDD-HHMMSS>.<ext>``.

Note: this module intentionally shadows the stdlib ``io`` *inside the
package namespace only*; import it as ``from blendstack.core import io as
bs_io`` (absolute imports keep the stdlib module available everywhere).
"""

from __future__ import annotations

import struct
import zlib
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

__all__ = [
    "RAW_EXTENSIONS",
    "PIL_EXTENSIONS",
    "SUPPORTED_INPUT_EXTENSIONS",
    "OUTPUT_FORMATS",
    "load_image",
    "probe_size",
    "save_image",
    "default_filename",
]

PathLike = Union[str, Path]

#: RAW formats handed to LibRaw (rawpy). Canon .CR2/.CR3 plus common others.
RAW_EXTENSIONS = frozenset({
    ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2",
    ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw", ".raw", ".rwl",
    ".3fr", ".kdc", ".mrw", ".x3f", ".iiq", ".erf", ".mef", ".mos",
})

#: Formats decoded by Pillow (16-bit TIFF is rerouted to tifffile, see above).
PIL_EXTENSIONS = frozenset({
    ".tif", ".tiff", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
})

SUPPORTED_INPUT_EXTENSIONS = RAW_EXTENSIONS | PIL_EXTENSIONS

#: Output format name -> default file extension (brief §4.4).
OUTPUT_FORMATS = {"tiff": "tif", "png": "png", "jpeg": "jpg"}

_SUFFIX_TO_FORMAT = {
    ".tif": "tiff", ".tiff": "tiff",
    ".png": "png",
    ".jpg": "jpeg", ".jpeg": "jpeg",
}


# ==========================================================================
# Loading
# ==========================================================================

def load_image(path: PathLike) -> np.ndarray:
    """Load any supported file as float32 RGB (H, W, 3), values 0–1.

    GIF (and any other animated format) yields frame 0 only; alpha is
    flattened against black; RAW files go through LibRaw defaults.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in RAW_EXTENSIONS:
        return _load_raw(path)
    if suffix in (".tif", ".tiff"):
        return _load_tiff(path)
    return _load_pil(path)


def _load_raw(path: Path) -> np.ndarray:
    """RAW via rawpy/LibRaw with default postprocessing (brief §2)."""
    import rawpy  # lazy: keeps core importable where LibRaw is absent

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess()  # LibRaw defaults: demosaic, WB, tone, 8-bit
    return _normalise_array(rgb)


def _load_tiff(path: Path) -> np.ndarray:
    """TIFF via imageio/tifffile — preserves 16-bit data (see module doc)."""
    import imageio.v3 as iio  # lazy

    try:
        arr = iio.imread(path, index=0)  # first page of multi-page files
    except (TypeError, ValueError, IndexError):
        arr = iio.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 4:  # stacked pages
        arr = arr[0]
    return _normalise_array(arr)


def _load_pil(path: Path) -> np.ndarray:
    """Everything else via Pillow; frame 0 of animated files."""
    with Image.open(path) as im:
        if getattr(im, "is_animated", False):
            im.seek(0)  # GIF/WebP: first frame only (brief §2)

        mode = im.mode
        if mode in ("I;16", "I;16L", "I;16B", "I;16N"):
            arr = np.asarray(im, dtype=np.uint16)
        elif mode == "I":
            # Pillow loads 16-bit greyscale PNG as 32-bit "I" with 0–65535 data.
            arr = np.asarray(im, dtype=np.int32).astype(np.float32) / 65535.0
            return _grey_to_rgb(np.clip(arr, 0.0, 1.0))
        elif mode == "F":
            arr = np.asarray(im, dtype=np.float32)
            return _grey_to_rgb(np.clip(arr, 0.0, 1.0))
        else:
            if mode == "P":
                im = im.convert("RGBA" if "transparency" in im.info else "RGB")
            elif mode == "LA":
                im = im.convert("RGBA")
            elif mode not in ("RGB", "RGBA", "L"):
                im = im.convert("RGB")
            arr = np.asarray(im)
    return _normalise_array(arr)


def _grey_to_rgb(grey: np.ndarray) -> np.ndarray:
    return np.repeat(grey[..., np.newaxis].astype(np.float32, copy=False), 3, axis=-1)


def _normalise_array(arr: np.ndarray) -> np.ndarray:
    """Any decoded array -> float32 RGB (H, W, 3), 0–1, alpha over black."""
    if arr.ndim == 2:
        return _grey_to_rgb(_to_unit_float(arr))
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise ValueError(f"Unsupported image array shape {arr.shape}")
    unit = _to_unit_float(arr)
    if unit.shape[2] == 1:
        return _grey_to_rgb(unit[..., 0])
    if unit.shape[2] == 4:
        # Flatten alpha against BLACK (brief §3): rgb * a + 0 * (1 - a).
        return (unit[..., :3] * unit[..., 3:4]).astype(np.float32, copy=False)
    return unit


def _to_unit_float(arr: np.ndarray) -> np.ndarray:
    """Scale by bit depth: uint8/255, uint16/65535, floats clipped 0–1."""
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return arr.astype(np.float32) / float(info.max)
    if np.issubdtype(arr.dtype, np.floating):
        return np.clip(arr.astype(np.float32, copy=False), 0.0, 1.0)
    raise ValueError(f"Unsupported image dtype {arr.dtype}")


def probe_size(path: PathLike) -> Tuple[int, int]:
    """Cheaply read the post-load (width, height) of a file without a full
    decode — used by the engine to pick target dimensions when streaming."""
    path = Path(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        import rawpy  # lazy

        with rawpy.imread(str(path)) as raw:
            sizes = raw.sizes
            width, height = sizes.width, sizes.height
            if sizes.flip in (5, 6):  # LibRaw applies 90° rotation on output
                width, height = height, width
        return width, height
    with Image.open(path) as im:
        return im.size


# ==========================================================================
# Saving (brief §4.4)
# ==========================================================================

def save_image(
    image: np.ndarray,
    path: PathLike,
    format: Optional[str] = None,
) -> Path:
    """Clip, quantise (with rounding) and write ``image`` to ``path``.

    ``format`` is "tiff", "png" or "jpeg"; if omitted it is inferred from
    the file suffix.  TIFF/PNG are 16-bit, JPEG is 8-bit quality 95 with
    4:4:4 chroma.  All outputs are tagged with an sRGB profile.
    """
    path = Path(path)
    if format is None:
        try:
            format = _SUFFIX_TO_FORMAT[path.suffix.lower()]
        except KeyError:
            raise ValueError(
                f"Cannot infer output format from suffix '{path.suffix}'; "
                f"pass format= one of {sorted(OUTPUT_FORMATS)}"
            ) from None
    format = format.lower()
    if format not in OUTPUT_FORMATS:
        raise ValueError(f"Unknown output format '{format}'; expected one of "
                         f"{sorted(OUTPUT_FORMATS)}")

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"save_image expects (H, W, 3), got {arr.shape}")
    clipped = np.clip(arr, 0.0, 1.0)

    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "jpeg":
        data8 = np.round(clipped * 255.0).astype(np.uint8)
        _write_jpeg(path, data8)
    else:
        data16 = np.round(clipped * 65535.0).astype(np.uint16)
        if format == "tiff":
            _write_tiff16(path, data16)
        else:
            _write_png16(path, data16)
    return path


def default_filename(mode: str, format: str = "tiff") -> str:
    """``blend_<mode>_<YYYYMMDD-HHMMSS>.<ext>`` (brief §4.4)."""
    try:
        ext = OUTPUT_FORMATS[format.lower()]
    except KeyError:
        raise ValueError(f"Unknown output format '{format}'") from None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"blend_{mode}_{stamp}.{ext}"


@lru_cache(maxsize=1)
def _srgb_profile_bytes() -> Optional[bytes]:
    """ICC bytes for an sRGB profile (via Pillow's littleCMS), or None."""
    try:
        from PIL import ImageCms

        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:
        return None


def _write_jpeg(path: Path, data: np.ndarray) -> None:
    """8-bit JPEG, quality 95, 4:4:4 (subsampling=0), sRGB-tagged."""
    im = Image.fromarray(data, mode="RGB")
    kwargs: dict = {"quality": 95, "subsampling": 0}
    icc = _srgb_profile_bytes()
    if icc:
        kwargs["icc_profile"] = icc
    im.save(path, format="JPEG", **kwargs)


# -- minimal 16-bit RGB TIFF writer ----------------------------------------

_TIFF_SHORT, _TIFF_LONG, _TIFF_UNDEFINED = 3, 4, 7


def _tiff_entry(tag: int, ftype: int, count: int, value_bytes: bytes) -> bytes:
    value_bytes = value_bytes.ljust(4, b"\x00")
    return struct.pack("<HHI", tag, ftype, count) + value_bytes[:4]


def _write_tiff16(path: Path, data: np.ndarray) -> None:
    """Uncompressed little-endian 16-bit RGB TIFF with an ICC tag."""
    height, width = data.shape[:2]
    pixels = data.astype("<u2", copy=False).tobytes()
    icc = _srgb_profile_bytes() or b""

    n_entries = 10 if icc else 9
    ifd_offset = 8
    ifd_size = 2 + n_entries * 12 + 4
    bps_offset = ifd_offset + ifd_size          # BitsPerSample (3 SHORTs)
    icc_offset = bps_offset + 6
    data_offset = icc_offset + len(icc)
    if data_offset % 2:                          # word-align the strip
        data_offset += 1
    byte_count = len(pixels)

    entries = [
        _tiff_entry(256, _TIFF_LONG, 1, struct.pack("<I", width)),        # ImageWidth
        _tiff_entry(257, _TIFF_LONG, 1, struct.pack("<I", height)),       # ImageLength
        _tiff_entry(258, _TIFF_SHORT, 3, struct.pack("<I", bps_offset)),  # BitsPerSample
        _tiff_entry(259, _TIFF_SHORT, 1, struct.pack("<HH", 1, 0)),       # Compression=none
        _tiff_entry(262, _TIFF_SHORT, 1, struct.pack("<HH", 2, 0)),       # Photometric=RGB
        _tiff_entry(273, _TIFF_LONG, 1, struct.pack("<I", data_offset)),  # StripOffsets
        _tiff_entry(277, _TIFF_SHORT, 1, struct.pack("<HH", 3, 0)),       # SamplesPerPixel
        _tiff_entry(278, _TIFF_LONG, 1, struct.pack("<I", height)),       # RowsPerStrip
        _tiff_entry(279, _TIFF_LONG, 1, struct.pack("<I", byte_count)),   # StripByteCounts
    ]
    if icc:
        entries.append(
            _tiff_entry(34675, _TIFF_UNDEFINED, len(icc),
                        struct.pack("<I", icc_offset))                    # ICC profile
        )

    with open(path, "wb") as fh:
        fh.write(b"II*\x00" + struct.pack("<I", ifd_offset))
        fh.write(struct.pack("<H", n_entries))
        fh.write(b"".join(entries))
        fh.write(struct.pack("<I", 0))                    # no next IFD
        fh.write(struct.pack("<HHH", 16, 16, 16))         # BitsPerSample values
        fh.write(icc)
        fh.write(b"\x00" * (data_offset - icc_offset - len(icc)))
        fh.write(pixels)


# -- minimal 16-bit RGB PNG writer -----------------------------------------

def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    body = ctype + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def _write_png16(path: Path, data: np.ndarray) -> None:
    """16-bit RGB PNG (colour type 2, bit depth 16), tagged sRGB.

    Scanlines use filter type 0 (None); zlib default compression.  Writes
    sRGB + gAMA + cHRM chunks per the PNG spec's recommended sRGB tagging.
    """
    height, width = data.shape[:2]
    big_endian = data.astype(">u2", copy=False)
    rows = np.empty((height, 1 + width * 6), dtype=np.uint8)
    rows[:, 0] = 0  # filter type 0 on every scanline
    rows[:, 1:] = np.frombuffer(big_endian.tobytes(), dtype=np.uint8).reshape(
        height, width * 6
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)
    chrm = struct.pack(">8I", 31270, 32900, 64000, 33000,
                       30000, 60000, 15000, 6000)
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(_png_chunk(b"IHDR", ihdr))
        fh.write(_png_chunk(b"sRGB", b"\x00"))            # perceptual intent
        fh.write(_png_chunk(b"gAMA", struct.pack(">I", 45455)))
        fh.write(_png_chunk(b"cHRM", chrm))
        fh.write(_png_chunk(b"IDAT", zlib.compress(rows.tobytes(), 6)))
        fh.write(_png_chunk(b"IEND", b""))
