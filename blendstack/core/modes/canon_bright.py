"""Canon Bright blend mode (project brief §4.2).

Comparative Bright: per pixel (per channel by default) the *brighter* value
of accumulator and incoming image is retained.  With softness 0, bias 0 and
per-channel basis the result equals ``np.maximum(A, B)`` exactly — the
pixel-exact Canon EOS R5 "Comparative Bright" behaviour, and the fold is
order-independent (max is commutative and associative).
"""

from __future__ import annotations

from typing import ClassVar

from ._comparative import ComparativeMode
from .registry import register_mode

__all__ = ["CanonBright"]


@register_mode
class CanonBright(ComparativeMode):
    """Comparative Bright — select the brighter comparison value."""

    name: ClassVar[str] = "canon_bright"
    label: ClassVar[str] = "Canon Bright"
    _direction: ClassVar[float] = 1.0
