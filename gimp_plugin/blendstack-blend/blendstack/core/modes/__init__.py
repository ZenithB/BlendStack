"""Blend-mode registry and built-in modes (project brief §3).

Importing this package registers the two v1 modes (Canon Bright, Canon
Dark).  New modes are added by dropping a module here that defines a
:class:`~blendstack.core.modes.registry.BlendMode` subclass decorated with
:func:`~blendstack.core.modes.registry.register_mode`, and importing it
below.
"""

from . import canon_bright, canon_dark  # noqa: F401  (side effect: registration)
from .registry import (
    BlendMode,
    ModeParameter,
    all_modes,
    get_mode,
    mode_names,
    register_mode,
)

__all__ = [
    "BlendMode",
    "ModeParameter",
    "all_modes",
    "get_mode",
    "mode_names",
    "register_mode",
]
