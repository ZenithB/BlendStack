"""Mute: a muted image is removed from the blend but still sizes the canvas."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from PIL import Image

from blendstack.core import engine
from blendstack.core import io as bs_io
from blendstack.core.adjustments import Adjustments


def _imgs(sizes=((40, 60), (40, 60), (40, 60)), seed=0):
    rng = np.random.default_rng(seed)
    return [rng.random((h, w, 3)).astype(np.float32) for h, w in sizes]


@pytest.mark.parametrize("mode", ["canon_bright", "canon_dark", "average", "screen",
                                  "multiply", "grain_merge", "overlay"])
def test_muted_image_equals_blend_without_it(mode):
    a, b, c = _imgs()
    muted = [Adjustments(), Adjustments(mute=True), Adjustments()]
    with_mute = engine.blend_arrays([a, b, c], adjustments=muted, mode=mode)
    without = engine.blend_arrays([a, c], mode=mode)
    assert np.array_equal(with_mute, without)


def test_muted_image_still_sizes_the_canvas():
    # The smallest-by-area image is muted; the canvas must still be ITS size, so
    # layer positions and the crop don't shift when toggling mute.
    big1, big2, small = _imgs(sizes=((80, 120), (80, 120), (20, 30)))
    out = engine.blend_arrays([big1, big2, small],
                              adjustments=[Adjustments(), Adjustments(), Adjustments(mute=True)])
    assert out.shape[:2] == (20, 30)


def test_need_two_unmuted():
    a, b, c = _imgs()
    with pytest.raises(ValueError, match="un-muted"):
        engine.blend_arrays([a, b, c], adjustments=[Adjustments(mute=True), Adjustments(mute=True), Adjustments()])
    with pytest.raises(ValueError, match="un-muted"):
        engine.blend_arrays([a, b], adjustments=[Adjustments(), Adjustments(mute=True)])


def test_toggling_mute_does_not_change_cache_key_or_identity():
    adj = Adjustments(exposure=0.5)
    assert adj.adjust_key() == dataclasses.replace(adj, mute=True).adjust_key()
    assert Adjustments(mute=True).is_identity


def test_mute_json_round_trip_and_old_presets():
    adj = Adjustments(mute=True, exposure=0.2)
    assert Adjustments.from_mapping(json.loads(json.dumps(dataclasses.asdict(adj)))) == adj
    assert Adjustments.from_mapping({"exposure": 0.1}).mute is False      # old preset


def test_blend_files_never_loads_muted_images(tmp_path, monkeypatch):
    paths = []
    for i, v in enumerate((50, 120, 200)):
        p = tmp_path / f"{i}.png"
        Image.fromarray(np.full((10, 12, 3), v, np.uint8)).save(p)
        paths.append(p)
    loaded = []
    real = bs_io.load_image
    monkeypatch.setattr(bs_io, "load_image", lambda p: (loaded.append(p.name if hasattr(p, "name") else str(p)), real(p))[1])
    out = engine.blend_files(paths, mode="canon_bright", out_path=tmp_path / "o.png",
                             adjustments=[Adjustments(), Adjustments(mute=True), Adjustments()])
    assert sorted(loaded) == ["0.png", "2.png"]            # the muted one was skipped entirely
    assert out.exists()


def test_blend_files_matches_blend_arrays_with_mute(tmp_path):
    arrays = _imgs(sizes=((16, 24),) * 3, seed=3)
    paths = []
    for i, a in enumerate(arrays):
        p = tmp_path / f"{i}.tif"
        bs_io.save_image(a, p)
        paths.append(p)
    adjs = [Adjustments(), Adjustments(mute=True), Adjustments(exposure=0.3)]
    out = engine.blend_files(paths, mode="screen", adjustments=adjs, out_path=tmp_path / "o.tif")
    got = bs_io.load_image(out)
    exp = engine.blend_arrays([bs_io.load_image(p) for p in paths], adjustments=adjs, mode="screen")
    assert np.abs(got - exp).max() < 2e-4                   # 16-bit quantisation only
