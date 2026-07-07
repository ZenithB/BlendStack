"""Preset save/load (project brief §5, "Presets").

A **preset** (user-facing term — never "recipe") is a plain **JSON**
document saved with the ``.bsp`` extension ("BlendStack preset"; the file
format is JSON, only the extension is custom).  Schema version 1 contents:

.. code-block:: json

    {
      "schema": 1,
      "app": "BlendStack",
      "mode": "canon_bright",
      "params": {"softness": 0.0, "bias": 0.0, "basis": "per_channel"},
      "output": {"format": "tiff"},
      "images": [
        {"path": "/abs/path/one.tif",
         "adjustments": {"exposure": 0.0, "contrast": 0.0, "saturation": 0.0,
                         "sharpen_radius": 1.0, "sharpen_amount": 0.0,
                         "opacity": 100.0}}
      ]
    }

``images`` is the ordered fold list (first = base image) with **absolute**
paths and every per-image adjustment.  Loading a preset whose files have
moved warns per-file and loads what it can — the missing-path check lives
in the main window; this module only (de)serialises and validates.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence, Union

from blendstack.core import engine
from blendstack.core import io as bs_io
from blendstack.core.adjustments import Adjustments

__all__ = [
    "PRESET_EXTENSION",
    "PRESET_SCHEMA_VERSION",
    "PresetError",
    "save_preset",
    "load_preset",
]

PRESET_EXTENSION = ".bsp"
PRESET_SCHEMA_VERSION = 1

PathLike = Union[str, Path]


class PresetError(ValueError):
    """A preset file could not be parsed or failed validation."""


def save_preset(
    path: PathLike,
    mode: str,
    params: dict[str, Any],
    output_format: str,
    images: Sequence[tuple[Path, Adjustments]],
) -> Path:
    """Write a schema-1 ``.bsp`` (JSON) preset.  Returns the written path."""
    document = {
        "schema": PRESET_SCHEMA_VERSION,
        "app": "BlendStack",
        "mode": mode,
        "params": dict(params),
        "output": {"format": output_format},
        "images": [
            {
                "path": str(Path(image_path).resolve()),
                "adjustments": asdict(adjustments),
            }
            for image_path, adjustments in images
        ],
    }
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(PRESET_EXTENSION)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def load_preset(path: PathLike) -> dict[str, Any]:
    """Parse and validate a ``.bsp`` preset.

    Returns ``{"mode": str, "params": dict, "output_format": str,
    "images": list[tuple[Path, Adjustments]]}`` (ordered, absolute paths).
    Raises :class:`PresetError` on malformed files; missing image *files*
    are NOT an error here (the caller warns per-file and loads the rest).
    """
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PresetError(f"Cannot read preset '{path}': {exc}") from exc
    if not isinstance(document, dict):
        raise PresetError(f"Preset '{path}' is not a JSON object")

    schema = document.get("schema")
    if schema != PRESET_SCHEMA_VERSION:
        raise PresetError(
            f"Unsupported preset schema {schema!r} in '{path}' "
            f"(this build reads schema {PRESET_SCHEMA_VERSION})"
        )

    mode = document.get("mode")
    if mode not in engine.mode_names():
        raise PresetError(f"Preset '{path}' names unknown blend mode {mode!r}")

    raw_params = document.get("params") or {}
    try:
        params = engine.get_mode(mode).resolve_params(raw_params)
    except ValueError as exc:
        raise PresetError(f"Preset '{path}': {exc}") from exc

    output = document.get("output") or {}
    output_format = str(output.get("format", "tiff")).lower()
    if output_format not in bs_io.OUTPUT_FORMATS:
        raise PresetError(
            f"Preset '{path}' names unknown output format '{output_format}'"
        )

    raw_images = document.get("images")
    if not isinstance(raw_images, list):
        raise PresetError(f"Preset '{path}' has no image list")
    images: list[tuple[Path, Adjustments]] = []
    for i, item in enumerate(raw_images):
        if not isinstance(item, dict) or "path" not in item:
            raise PresetError(f"Preset '{path}': image entry {i} is malformed")
        try:
            adjustments = Adjustments.from_mapping(item.get("adjustments"))
        except (TypeError, ValueError) as exc:
            raise PresetError(
                f"Preset '{path}': image entry {i} adjustments: {exc}"
            ) from exc
        images.append((Path(item["path"]), adjustments))

    return {
        "mode": mode,
        "params": params,
        "output_format": output_format,
        "images": images,
    }
