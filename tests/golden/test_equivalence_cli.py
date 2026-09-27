"""The harness command line: ``site-check`` exits 1 on a difference, ``authenticity`` lowers under the
kernel-name mode it is given (``--kernel-names recorded|public``), and ``record`` refuses a changed tree without one
reason token. Light: no child process and no 32-device mesh; the production build is faked here, and G14 case
``j-public`` proves the mode on the real build."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from glm_tpu.kernels import names
from tools.equivalence import __main__ as cli
from tools.equivalence import authenticity, common, driver, fixture, gates, identities, lowering, normalize, programs


@pytest.mark.parametrize(("argv", "status", "code"), [([], "pass", 0), (["--record"], "recorded", 0), ([], "fail", 1)])
def test_site_check_exits_1_only_when_the_facts_differ(monkeypatch, capsys, argv, status, code):
    seen = {}

    def site_check(*, record: bool, requests_dir: Path | None) -> dict[str, Any]:
        seen.update(record=record, requests_dir=requests_dir)
        return dict(gate="G5", status=status, differing=["topology"] if status == "fail" else [])

    monkeypatch.setattr(identities, "site_check", site_check)
    assert cli.main(["site-check", *argv]) == code
    assert json.loads(capsys.readouterr().out)["status"] == status
    assert seen == dict(record=bool(argv), requests_dir=None)


@pytest.mark.parametrize(("argv", "mode"), [([], "recorded"), (["--kernel-names", "public"], "public")])
@pytest.mark.parametrize("status", ["pass", "fail"])
def test_authenticity_forwards_the_kernel_name_mode_to_its_child(monkeypatch, capsys, tmp_path, argv, mode, status):
    calls = []

    def run_child(module: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((module, args))
        return dict(gate="adapter-authenticity", status=status, kernel_names=mode, environment={}, programs={})

    monkeypatch.setattr(common, "run_child", run_child)
    monkeypatch.setattr(gates, "refuse_if_live", lambda requested: requested)
    out = tmp_path / "report.json"
    assert cli.main(["authenticity", str(tmp_path), *argv, "--out", str(out)]) == (0 if status == "pass" else 1)
    assert calls == [
        ("tools.equivalence.authenticity", (str(tmp_path), "--kernel-names", mode, "--out", str(out.resolve())))
    ]
    assert json.loads(capsys.readouterr().out)["kernel_names"] == mode


def test_authenticity_refuses_an_unknown_kernel_name_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "run_child", lambda *args, **kwargs: pytest.fail("the child must not start"))
    with pytest.raises(SystemExit) as refused:
        cli.main(["authenticity", str(tmp_path), "--kernel-names", "legacy"])
    assert refused.value.code == 2
    with pytest.raises(ValueError, match="kernel-name mode"), lowering.kernel_names("legacy"):
        pass


@pytest.mark.parametrize("reason", [[], ["--reason", "S9 governance"], ["--reason", "60b68b4a"]])
def test_record_refuses_a_changed_tree_without_one_reason_token(monkeypatch, reason):
    """The refusal comes before anything runs or is written, and it names the accepted tokens by example only: it
    points to no document outside the repository."""
    monkeypatch.setattr(gates, "refuse_if_live", lambda requested: requested)
    monkeypatch.setattr(gates, "source_record", lambda: dict(production_paths_equal_baseline=False))
    monkeypatch.setattr(gates, "run_child", lambda *args, **kwargs: pytest.fail("no gate may run"))
    monkeypatch.setattr(gates, "write_json", lambda *args, **kwargs: pytest.fail("no record may be written"))
    with pytest.raises(SystemExit) as refused:
        cli.main(["record", "--gates", "G7", *reason])
    assert refused.value.code == (
        "this re-baseline needs --reason set to exactly one H number (e.g. H11), stage token (e.g. S2d, S4.2b) or "
        "work unit (e.g. WU-E)"
    )


def _current() -> dict[str, str]:
    return dict(names.KERNEL_NAMES)


def test_the_kernel_name_mode_chooses_what_location_free_patches_back():
    public = dict(names.KERNEL_NAMES)
    table = lowering.kernel_renames()
    recorded = {key: table["names"][value] for key, value in public.items()}
    with lowering.location_free():
        assert _current() == recorded  # the default: the records' 181c013e names
    with lowering.kernel_names("public"):
        with lowering.location_free():
            assert _current() == public
        with lowering.location_free(table):  # an explicit table still wins (G14 j-mapped)
            assert _current() == recorded
        with lowering.kernel_names("recorded"), lowering.location_free():
            assert _current() == recorded
        with lowering.location_free():  # the inner mode ended
            assert _current() == public
    with lowering.kernel_names("recorded"), lowering.location_free():
        assert _current() == recorded
    with lowering.location_free():
        assert _current() == recorded
    assert _current() == public


# --------------------------------------------------------------------------- compare() over a faked production build
def _program() -> Any:
    """A jitted program with one Pallas kernel whose name is read when the program is traced, as production's
    kernels read ``KERNEL_NAMES`` (so the name is the one of the ``location_free()`` it is lowered in)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    def kernel(x_ref: Any, o_ref: Any) -> None:
        o_ref[...] = x_ref[...] * 2.0

    def body(x: Any) -> Any:
        name = names.KERNEL_NAMES["sparse_mla"] + "_h2_k128"
        return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32), name=name)(x) + 1.0

    return jax.jit(body)


def _abstract() -> tuple[Any, ...]:
    import jax
    import jax.numpy as jnp

    return (jax.ShapeDtypeStruct((8, 128), jnp.float32),)


def _original(root: Path, mode: str) -> Path:
    """A run's ``native.rank0`` holding ``decode.stablehlo.mlir`` lowered under ``mode``."""
    directory = root / f"run-{mode}" / "native.rank0"
    directory.mkdir(parents=True)
    with lowering.kernel_names(mode), lowering.location_free():
        text = normalize.stablehlo_text(lowering.lower_for_tpu(_program(), _abstract()))
    (directory / "decode.stablehlo.mlir").write_text(text)
    return directory


@pytest.fixture
def faked_build(monkeypatch):
    """``programs.program_specs`` builds the one program inside ``location_free()``, as ``driver.build_runtime``
    does, and keeps its ``Lowered``: the kernel name is the one of the mode active around the build."""
    built = []

    def program_specs(tier: str, mesh: Any, *, only: set[str], keep_lowered: bool) -> dict[str, Any]:
        assert (tier, mesh, only, keep_lowered) == ("production", None, {authenticity.KEYS["decode"]}, True)
        with lowering.location_free():
            lowered = lowering.lower_for_tpu(_program(), _abstract())
        built.append(tier)
        return {authenticity.KEYS["decode"]: driver.ProgramSpec("decode", _program(), _abstract(), lowered=lowered)}

    monkeypatch.setattr(programs, "program_specs", program_specs)
    monkeypatch.setattr(fixture, "cpu_mesh", lambda: None)
    return built


@pytest.mark.parametrize("original", ["recorded", "public"])
@pytest.mark.parametrize("mode", ["recorded", "public"])
def test_authenticity_builds_and_compares_under_the_kernel_name_mode(
    monkeypatch, tmp_path, faked_build, original, mode
):
    directory = _original(tmp_path, original)
    reports = []
    monkeypatch.setattr(authenticity, "emit", reports.append)
    out = tmp_path / "kernels.json"
    argv = [str(directory), "--out", str(out)] + (["--kernel-names", mode] if mode != "recorded" else [])
    assert authenticity.main(argv) == 0
    (report,) = reports
    row = report["programs"]["decode"]
    assert report["kernel_names"] == mode and json.loads(out.read_text())["kernel_names"] == mode
    assert row["kernel_calls"] == [1, 1] and len(faked_build) == 1
    if mode == original:
        assert report["status"] == "pass"
        assert row["raw_equal"] and row["masked_equal"] and row["decoded_equal"] and row["decoded_kernels_equal"] == 1
    else:  # the name differs in the custom call and, as the Mosaic function symbol, in the decoded body
        assert report["status"] == "fail"
        assert not row["kernel_names_equal"] and not row["masked_equal"] and not row["decoded_equal"]
        assert row["decoded_kernels_equal"] == 0
