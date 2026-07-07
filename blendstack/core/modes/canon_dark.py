"""Canon Dark blend mode (project brief §4.2).

Comparative Dark: per pixel (per channel by default) the *darker* value of
accumulator and incoming image is retained.  With softness 0, bias 0 and
per-channel basis the result equals ``np.minimum(A, B)`` exactly — the
pixel-exact Canon EOS R5 "Comparative Dark" behaviour, and the fold is
order-independent (min is commutative and associative).
"""

from __future__ import annotations

from typing import ClassVar

from ._comparative import ComparativeMode
from .registry import register_mode

__all__ = ["CanonDark"]


@register_mode
class CanonDark(ComparativeMode):
    """Comparative Dark — select the darker comparison value."""

    name: ClassVar[str] = "canon_dark"
    label: ClassVar[str] = "Canon Dark"
    _direction: ClassVar[float] = -1.0
