"""Geometry tests — project brief §8 mismatched-sizes criterion and §4.3."""

from __future__ import annotations

import numpy as np

from blendstack.core import engine, geometry


def _circle_image(w: int, h: int, radius: int) -> np.ndarray:
    """Black (H, W, 3) image with a white anti-alias-free centred circle."""
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 <= radius**2
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[mask] = 1.0
    return img


def _bbox(mask: np.ndarray) -> tuple[int, int]:
    """(width, height) of the bounding box of True pixels."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1


class TestTargetDimensions:
    def test_smallest_by_area_wins(self) -> None:
        # 500x300 (150k) is smaller by area than 400x400 (160k) despite
        # having the larger long edge.
        assert geometry.target_dimensions([(400, 400), (500, 300)]) == (500, 300)

    def test_identical_sizes_pass_through_untouched(self) -> None:
        a = np.zeros((10, 20, 3), dtype=np.float32)
        b = np.ones((10, 20, 3), dtype=np.float32)
        out = geometry.conform_stack([a, b])
        assert out[0] is a and out[1] is b  # same objects, no copies


class TestMismatchedSizes:
    def test_output_matches_smallest_dims(self) -> None:
        small = np.zeros((300, 400, 3), dtype=np.float32)   # 400x300, smallest
        large = np.full((500, 800, 3), 0.5, dtype=np.float32)
        wide = np.full((350, 900, 3), 0.25, dtype=np.float32)
        out = engine.blend_arrays([small, large, wide], mode="canon_bright")
        assert out.shape == (300, 400, 3)

    def test_circle_stays_circular(self) -> None:
        """Cover-scale + centre-crop must not distort: a circle in the larger
        image keeps a square bounding box after processing (brief §8)."""
        small = np.zeros((300, 400, 3), dtype=np.float32)
        large = _circle_image(800, 500, radius=100)  # scale 0.6 -> radius 60

        out = engine.blend_arrays([small, large], mode="canon_bright")
        assert out.shape == (300, 400, 3)

        mask = out[..., 0] > 0.5
        assert mask.any(), "circle must survive the centre-crop"
        bbox_w, bbox_h = _bbox(mask)
        assert abs(bbox_w - bbox_h) <= 2, (
            f"circle distorted: bbox {bbox_w}x{bbox_h} (aspect must be preserved)"
        )
        # Expected diameter after 0.6x cover scale: ~120 px.
        assert abs(bbox_w - 120) <= 3

    def test_cover_crops_never_letterbox(self) -> None:
        """The scaled image always covers the target fully: blending a black
        target-size image with a solid white larger image of a different
        aspect must yield pure white everywhere (no black borders)."""
        small = np.zeros((100, 200, 3), dtype=np.float32)
        big = np.ones((900, 500, 3), dtype=np.float32)  # portrait vs landscape
        out = engine.blend_arrays([small, big], mode="canon_bright")
        assert out.shape == (100, 200, 3)
        assert np.all(out > 0.99)
