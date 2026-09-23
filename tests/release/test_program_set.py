"""S2c: ``glm_tpu.runner.programs.build_program_set`` is the one production program builder.

A CPU child with 32 forced devices builds the set for the frozen fixture config (no lowering, no
compilation) and checks names, compile order, the donation rule, the admission flags and the
builder arguments; a static check proves the runtime and ``compile_batch`` build no program
themselves. That the runtime compiles exactly these functions and that every spec lowers to the
program the runtime compiled is proven by G1/G2 (the ProgramSet cross-check).
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]

CHILD = r'''
import json
from glm_tpu.runner import programs
from tools.equivalence import fixture
from glm_tpu.greenfield.runtime.ws32_decoder import ws32_decoder_state_specs

mesh = fixture.cpu_mesh()
calls = []
for name in ("build_prefill_program", "build_packed_decoder_program", "build_batched_decoder_program"):
    real = getattr(programs, name)
    def record(*args, _real=real, _name=name, **kwargs):
        calls.append([_name, {k: v for k, v in sorted(kwargs.items())}])
        return _real(*args, **kwargs)
    setattr(programs, name, record)

def describe(program_set):
    return dict(
        specs=[[s.name, list(s.donate_argnums), s.model, callable(s.fn)] for s in program_set.specs()],
        decode=program_set.decode is not None, batch=program_set.batch is not None,
        prefill_rows=list(program_set.prefill),
        bank_specs=None if program_set.batch is None else sorted({str(s.spec[:1]) for s in
            __import__("jax").tree.leaves(program_set.batch.state_shardings)}),
        bank_leaves=None if program_set.batch is None else len(__import__("jax").tree.leaves(
            program_set.batch.state_shardings)))

out = {}
for label, capacity, lanes, interpret in (("plain", 1536, 0, False), ("donated", 8704, 0, False),
                                          ("concurrent", 1536, 4, False), ("interpret", 1536, 4, True)):
    calls.clear()
    out[label] = dict(describe(programs.build_program_set(mesh, fixture.decoder_config(capacity=capacity),
                                                          concurrent_size=lanes, interpret=interpret)),
                      calls=list(calls))
out["state_leaves"] = len(__import__("jax").tree.leaves(ws32_decoder_state_specs()))
print(json.dumps(out))
'''


def _child() -> dict:
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1",
               XLA_FLAGS="--xla_force_host_platform_device_count=32")
    output = subprocess.check_output([sys.executable, "-c", CHILD], cwd=REPO, env=env, text=True, timeout=600)
    return json.loads(output.strip().splitlines()[-1])


def test_program_set_names_order_donation_and_builder_arguments():
    from glm_tpu.runner.programs import INTERPRET

    out = _child()
    head = [["wk_decode", [], False, True], ["wk_promote", [], False, True], ["cache_init", [], False, True]]
    assert out["plain"]["specs"] == head + [["prefill_128", [], True, True], ["prefill_114", [], True, True],
                                            ["decode", [], True, True]]
    # capacity > 8,192: the prefill (state = argument 2) and decode (state = argument 1) donate
    assert out["donated"]["specs"] == head + [["prefill_128", [2], True, True], ["prefill_114", [2], True, True],
                                              ["decode", [1], True, True]]
    batch = [["batch_cache_init", [], False, True], ["batch_insert", [0], False, True],
             ["batch_decode", [1], True, True]]
    assert out["concurrent"]["specs"] == head + [["prefill_128", [], True, True], ["prefill_114", [], True, True],
                                                 *batch]
    assert (out["plain"]["decode"], out["plain"]["batch"]) == (True, False)
    assert (out["concurrent"]["decode"], out["concurrent"]["batch"]) == (False, True)
    assert out["plain"]["prefill_rows"] == [128, 114]
    # the bank shardings: one per decoder-state leaf, the lane axis unsharded
    assert out["concurrent"]["bank_leaves"] == out["state_leaves"] and out["concurrent"]["bank_specs"] == ["(None,)"]
    # the admitted prefill profile is hard-wired in the builder: only the block rows are passed
    prefill = [["build_prefill_program", {"block_rows": rows}] for rows in (128, 114)]
    assert out["plain"]["calls"] == prefill + [["build_packed_decoder_program", {}]]
    assert out["concurrent"]["calls"] == prefill + [["build_batched_decoder_program", {"batch_size": 4}]]
    interpret = dict(INTERPRET)
    assert out["interpret"]["calls"] == [
        ["build_prefill_program", dict(block_rows=rows, **interpret)] for rows in (128, 114)
    ] + [["build_batched_decoder_program", dict(interpret, batch_size=4)]]


def test_runtime_and_compile_batch_build_no_program_themselves():
    builders = {"build_prefill_program", "build_ws32_prefill_challenger_program", "build_packed_decoder_program",
                "build_batched_decoder_program", "build_cache_initializer", "build_wk_programs", "jit"}
    for relative in ("glm_tpu/optimized/runtime.py", "glm_tpu/optimized/batched_runtime.py"):
        tree = ast.parse((REPO / relative).read_text())
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        names |= {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                  for alias in node.names}
        assert not names & builders, (relative, sorted(names & builders))
