"""Tests of :mod:`glm_tpu.model_loader.sharded_state.verify` (its checks run in ``test_format.py`` and
``test_manifest.py``): the names the verification helpers answer to."""

from __future__ import annotations

import pytest

from glm_tpu.model_loader.sharded_state import manifest, verify, writer


@pytest.mark.parametrize("name", ["verify_runtime_value", "verify_runtime_files"])
def test_the_verification_helpers_answer_to_their_public_names_only(name):
    # The helpers were private (a leading underscore) until they became public; no alias keeps the old name in the
    # module or in the modules that call them.
    assert callable(getattr(verify, name))
    for module in (verify, writer, manifest):
        assert not hasattr(module, "_" + name), module.__name__
