"""``record`` keeps an unchanged characterization file, with its provenance and re-baseline marker."""

from __future__ import annotations

import json

from tools.equivalence import gates


def test_record_keeps_an_unchanged_characterization_file(tmp_path, monkeypatch):
    functions = ["glm_tpu.x:f", "glm_tpu.x:g"]
    old = dict(gate="G7", format=1, environment={}, source={}, recorded_unix=1, functions=functions, count=2,
               modules=["glm_tpu.x"], digest="d", rebaseline=dict(kind="reviewed", reason="S2c"))
    path = tmp_path / gates.DATA_FILES["G7"]
    path.write_text(json.dumps(old))
    monkeypatch.setattr(gates, "DATA", tmp_path)
    monkeypatch.setattr(gates, "source_record",
                        lambda: dict(head="h", production_paths_equal_baseline=False, baseline="b"))
    monkeypatch.setattr(gates, "static_environment", lambda *args: {})
    fresh = dict(functions=list(functions), count=2, modules=["glm_tpu.x"], digest="d")
    monkeypatch.setattr(gates, "produce", lambda gate, **kwargs: fresh)

    [report] = gates.record(["G7"], reason="S2d")
    assert (report["status"], report["written"], report["unchanged"]) == ("unchanged", [], ["trace_closure.json"])
    assert json.loads(path.read_text()) == old  # provenance and the S2c marker are kept

    fresh.update(functions=[*functions, "glm_tpu.x:h"], count=3, digest="e")
    [report] = gates.record(["G7"], reason="S2d")
    assert (report["status"], report["written"], report["unchanged"]) == ("recorded", ["trace_closure.json"], [])
    new = json.loads(path.read_text())
    assert new["functions"][-1] == "glm_tpu.x:h" and new["rebaseline"] == dict(kind="reviewed", reason="S2d")
    assert report["diff"]["G7"]["lines"] == ["<root>: + glm_tpu.x:h"]
