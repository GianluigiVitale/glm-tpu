from __future__ import annotations

import numpy as np
import pytest

from scripts.greenfield.pack_ws32_strategy_nd_layer0 import (
    _model_rank_values,
    _model_ranks_for_expert,
)


def test_ws32_strategy_nd_expert_ranks_cover_model_axis_once() -> None:
    groups = tuple(_model_ranks_for_expert(expert) for expert in range(8))
    assert groups == tuple(
        tuple(range(expert * 4, expert * 4 + 4)) for expert in range(8)
    )
    assert tuple(value for group in groups for value in group) == tuple(range(32))
    with pytest.raises(ValueError, match="expert coordinate"):
        _model_ranks_for_expert(8)


def test_ws32_strategy_nd_rank_selection_preserves_model_order() -> None:
    name = "tensor"
    sources = {
        (owner, name): np.arange(owner * 80, owner * 80 + 80).reshape(8, 10)
        for owner in range(4)
    }
    ranks = (7, 8, 15, 16, 31)
    selected = _model_rank_values(sources, name, ranks)
    assert selected.shape == (5, 10)
    assert selected[:, 0].tolist() == [70, 80, 150, 160, 310]
