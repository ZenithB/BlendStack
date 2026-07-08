"""Blend-mode registry (project brief §3, design rule 2).

Blend modes are pluggable: each mode is a class registered by name that
declares its parameters (name, range, default), a fold function
``blend(accumulator, incoming, params) -> ndarray`` and whether it requires
linear-light input (``needs_linear``, False for both v1 modes).  The engine
(:mod:`blendstack.core.engine`) contains the scaffolding that linearises via
the sRGB EOTF before the fold and re-encodes after for any mode that sets
``needs_linear = True`` — the door for future Additive/Average modes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Type

import numpy as np

__all__ = [
    "ModeParameter",
    "BlendMode",
    "register_mode",
    "get_mode",
    "mode_names",
    "all_modes",
]


@dataclass(frozen=True)
class ModeParameter:
    """Declaration of a single blend-mode parameter.

    Numeric parameters set ``min_value``/``max_value`` (UI-space range);
    enumerated parameters set ``choices`` instead.
    """

    name: str
    default: float | str
    min_value: float | None = None
    max_value: float | None = None
    choices: tuple[str, ...] | None = None
    label: str = ""


class BlendMode(abc.ABC):
    """Base class for all blend modes (brief §3, design rule 2).

    Subclasses must define ``name``, ``label``, ``parameters`` and
    implement :meth:`blend`.  ``needs_linear`` defaults to False; a mode
    that sets it True receives linear-light arrays from the engine.
    """

    #: Registry key, e.g. ``"canon_bright"``.
    name: ClassVar[str]
    #: Human-readable UI label, e.g. ``"Canon Bright"``.
    label: ClassVar[str]
    #: True if the fold must run in linear light (sRGB-linearised) data.
    needs_linear: ClassVar[bool] = False
    #: Declared parameters (UI-space names, ranges, defaults).
    parameters: ClassVar[tuple[ModeParameter, ...]] = ()

    @abc.abstractmethod
    def blend(
        self,
        accumulator: np.ndarray,
        incoming: np.ndarray,
        params: Mapping[str, Any] | None = None,
        count: int = 1,
    ) -> np.ndarray:
        """Fold one incoming image into the accumulator.

        Both arrays are float32 RGB, shape (H, W, 3), values nominally 0–1.
        Must return a new array; must not mutate its inputs.

        ``count`` is the number of source images already represented by
        ``accumulator`` (so ``incoming`` is image number ``count + 1``).
        Most modes ignore it; running-aggregate modes such as Average need
        it to weight the incoming image correctly (e.g. a true mean must
        weight the new frame by ``1 / (count + 1)``).
        """

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        """Return a dict of every declared parameter at its default."""
        return {p.name: p.default for p in cls.parameters}

    @classmethod
    def resolve_params(cls, params: Mapping[str, Any] | None) -> dict[str, Any]:
        """Merge ``params`` over the declared defaults.

        Raises ``ValueError`` for parameter names this mode does not declare
        (catches typos early rather than silently ignoring a control).
        """
        resolved = cls.default_params()
        if params:
            unknown = set(params) - set(resolved)
            if unknown:
                raise ValueError(
                    f"Unknown parameter(s) for mode '{cls.name}': {sorted(unknown)}; "
                    f"declared: {sorted(resolved)}"
                )
            resolved.update(params)
        return resolved

    @classmethod
    def pick_params(cls, params: Mapping[str, Any] | None) -> dict[str, Any]:
        """Like :meth:`resolve_params` but *lenient*: keep only the keys this
        mode declares and silently ignore any others.

        Frontends carry one global parameter dict (e.g. softness/bias/basis)
        and reuse it across modes; a mode that declares none of those keys
        (Multiply, Screen, …) must not choke on them. Use this at the fold
        call site; use :meth:`resolve_params` where an unexpected key really
        is a bug worth surfacing.
        """
        resolved = cls.default_params()
        if params:
            for name in resolved:
                if name in params:
                    resolved[name] = params[name]
        return resolved


_MODES: dict[str, Type[BlendMode]] = {}


def register_mode(cls: Type[BlendMode]) -> Type[BlendMode]:
    """Class decorator: register a :class:`BlendMode` subclass by its name."""
    if not (isinstance(getattr(cls, "name", None), str) and cls.name):
        raise TypeError(f"{cls!r} must define a non-empty class attribute 'name'")
    if cls.name in _MODES:
        raise ValueError(f"Blend mode '{cls.name}' is already registered")
    _MODES[cls.name] = cls
    return cls


def get_mode(name: str) -> BlendMode:
    """Look up a registered mode by name and return an instance of it."""
    try:
        return _MODES[name]()
    except KeyError:
        raise ValueError(
            f"Unknown blend mode '{name}'; available: {mode_names()}"
        ) from None


def mode_names() -> list[str]:
    """Names of all registered modes, in registration order."""
    return list(_MODES)


def all_modes() -> dict[str, Type[BlendMode]]:
    """Mapping of name -> mode class (a copy; mutating it is harmless)."""
    return dict(_MODES)
