"""Seal accepted legacy long-context results (spec §23.1) as token oracles.

Two workloads, both rebuilt deterministically from provenance rather than
re-captured on the pod:

* ``passkey``: one legacy protected passkey item (DB run 403, 128K, four depths).
  The stored ``items.prompt`` field is truncated (head + tail + sha256) for long
  prompts, so the prompt text is rebuilt with the legacy bench's own
  ``build_trial(tok, L, depth, seed)`` and accepted only if its SHA-256 equals the
  digest embedded in the stored field and the stored field itself is reproduced.
* ``e0``: the legacy E0 synthetic prompt (DB run 402, 256K) rebuilt with the legacy
  bench's ``build_prompt_ids(ctx, seed, vocab_size)`` and accepted only if the
  int32 stream SHA-256 equals the digest in the stored descriptor.

The legacy bench modules are loaded as pinned utilities (their file SHA-256 is
recorded); no legacy model-execution path is imported.  Tokenizer files are
pinned the same way as the short-context oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any, Mapping

import numpy as np

from .short_context_oracle import (
    MODEL_ID,
    MODEL_VOCAB_SIZE,
    PROMPT_PREFIX_IDS,
    _array_sha256,
    _canonical_json,
    _file_record,
    _manifest_hash,
    _sha256_file,
    _validate_digest,
    _write_text_once,
)

FORMAT_VERSION = 1
ARTIFACT_KIND = "greenfield_long_context_legacy_oracle"
KINDS = ("passkey", "e0")

# Spec §23.5 L7/L8 profiles. The context label alone fixes which sealed oracle a
# run may bind: the wrapper supplies a path, this table supplies the identity, so
# a 128K seal cannot be produced from a different depth's capture or from a 2k/8k
# artifact. ``prompt_token_count`` is the length the eight host records must all
# report, and ``generated_token_count`` the number of greedy tokens the legacy
# capture holds (20 for a passkey item, 256 for E0).
WS32_LONG_CONTEXT_PROFILES: Mapping[str, Mapping[str, Any]] = {
    "128k_d0_0": {
        "profile": "128k_d0.0",
        "kind": "passkey",
        "depth": 0.0,
        "prompt_token_count": 127363,
        "generated_token_count": 20,
        "source_run_id": 403,
        "item_row_id": 1517,
        "manifest_sha256": "f311214501093190226264a6821d5a9f331e91c76e4a1d812a92be34943e913b",
        "success_sha256": "6af757be0da515960d6ae870ea6c9b1edd7932233106bdda1c14c99f513dfee1",
    },
    "128k_d0_05": {
        "profile": "128k_d0.05",
        "kind": "passkey",
        "depth": 0.05,
        "prompt_token_count": 127363,
        "generated_token_count": 20,
        "source_run_id": 403,
        "item_row_id": 1518,
        "manifest_sha256": "71a94209f75e3e5602a07c66ab07641f34163a4c9cdfb07e60017f8de1e51c30",
        "success_sha256": "baa482838744ab3ee6672dc17988356fa4a45a436cdc6fe3a916b7830b63cd17",
    },
    "128k_d0_95": {
        "profile": "128k_d0.95",
        "kind": "passkey",
        "depth": 0.95,
        "prompt_token_count": 127363,
        "generated_token_count": 20,
        "source_run_id": 403,
        "item_row_id": 1519,
        "manifest_sha256": "bb71f3faf7ea972114818ea81e94a8084bd874c39bf96f72bafb750f20218764",
        "success_sha256": "47fa11cfd36b0b6ad8e6f3d9a3e6fa0542a426481cf2f7b7209c4a1684dd39ff",
    },
    "128k_d1_0": {
        "profile": "128k_d1.0",
        "kind": "passkey",
        "depth": 1.0,
        "prompt_token_count": 127363,
        "generated_token_count": 20,
        "source_run_id": 403,
        "item_row_id": 1520,
        "manifest_sha256": "c8771512c25fa1faf46ac336b40ae3605daaf2b682dbd142a8937d1c09313415",
        "success_sha256": "24f1bc4eff8b63038629cb87daa982d51d38b5be6dc63f3a3ef41363e9fb2749",
    },
    "256k_e0": {
        "profile": "256k_e0",
        "kind": "e0",
        "depth": None,
        "prompt_token_count": 262144,
        "generated_token_count": 256,
        "source_run_id": 402,
        "item_row_id": 1516,
        "manifest_sha256": "9dd17e69ca28d73bee09c175595039fdea6eed22499d2d9a547e341c5fbc0440",
        "success_sha256": "827421edae3fbe8db050a6d6d1fb42b8da601cbbd060fb0cb02bfce5e07eb236",
    },
}

# The short-context prompts are the sealed 2k/8k oracle lengths (§21).
WS32_PROMPT_LENGTHS: Mapping[str, int] = {
    "2k": 2034,
    "8k": 8155,
    **{
        label: int(entry["prompt_token_count"])
        for label, entry in WS32_LONG_CONTEXT_PROFILES.items()
    },
}
WS32_CONTEXT_LABELS = ("2k", "8k", *WS32_LONG_CONTEXT_PROFILES)


def require_ws32_long_context_profile(
    label: str, oracle: "Ws32LongContextOracle", *, success_sha256: str
) -> Mapping[str, Any]:
    """Refuse a long-context seal whose oracle is not the label's sealed capture."""

    entry = WS32_LONG_CONTEXT_PROFILES.get(label)
    if entry is None:
        raise ValueError(f"WS32 context label {label!r} is not a long-context profile")
    observed = {
        "kind": oracle.kind,
        "depth": oracle.depth,
        "prompt_token_count": int(oracle.prompt_token_ids.size),
        "generated_token_count": int(oracle.generated_token_ids.size),
        "source_run_id": int(oracle.source_run_id),
        "item_row_id": int(oracle.item_row_id),
        "manifest_sha256": oracle.manifest_sha256,
        "success_sha256": success_sha256,
    }
    expected = {key: entry[key] for key in observed}
    if observed != expected:
        raise ValueError(
            f"WS32 long-context oracle does not match profile {label!r}: "
            f"{observed} != {expected}"
        )
    return entry
_LEGACY_BENCH_MODULES = ("glm_longctx.py", "dsa_throughput.py", "extract.py", "provenance.py", "engine.py")
_PROMPT_SHA_PATTERN = re.compile(r"sha256=([0-9a-f]{64})")


@dataclass(frozen=True, slots=True)
class LongContextOracleConfig:
    kind: str
    results_db: Path
    tokenizer_root: Path
    legacy_bench_root: Path
    output_dir: Path
    capture_code_hash: str
    legacy_repository_pin: str
    run_id: int
    item_row_id: int
    expected_harness_git: str
    expected_fork_git: str
    expected_benchmark: str
    expected_model_uri: str
    expected_prompt_tokens: int
    expected_generated_tokens: int
    expected_seed: int
    expected_context_length: int
    expected_depth: float | None = None
    expected_gold: str | None = None

    def __post_init__(self) -> None:
        for name in ("results_db", "tokenizer_root", "legacy_bench_root", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.kind not in KINDS:
            raise ValueError(f"long-context oracle kind must be one of {KINDS}")
        for name in (
            "run_id",
            "item_row_id",
            "expected_prompt_tokens",
            "expected_generated_tokens",
            "expected_context_length",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.expected_seed, int) or isinstance(self.expected_seed, bool):
            raise ValueError("expected_seed must be an integer")
        for name in (
            "expected_harness_git",
            "expected_fork_git",
            "expected_benchmark",
            "expected_model_uri",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        _validate_digest(self.capture_code_hash, "capture_code_hash", (40, 64))
        _validate_digest(self.legacy_repository_pin, "legacy_repository_pin", (40, 64))
        _validate_digest(self.expected_harness_git, "expected_harness_git", tuple(range(7, 65)))
        _validate_digest(self.expected_fork_git, "expected_fork_git", tuple(range(7, 65)))
        if not self.expected_model_uri.startswith("gs://driftbench-dsv4-uc/"):
            raise ValueError("expected_model_uri must use the approved bucket")
        if self.kind == "passkey":
            if not isinstance(self.expected_depth, float) or not 0.0 <= self.expected_depth <= 1.0:
                raise ValueError("passkey oracle requires a depth in [0, 1]")
            if not self.expected_gold or not self.expected_gold.isdigit():
                raise ValueError("passkey oracle requires a digit-string gold key")
            if self.expected_benchmark != f"passkey_L{self.expected_context_length}_d{self.expected_depth}":
                raise ValueError("passkey benchmark name does not match length/depth")
        else:
            if self.expected_depth is not None or self.expected_gold is not None:
                raise ValueError("e0 oracle carries no depth or gold")
            if self.expected_benchmark != f"dsa_throughput_ctx{self.expected_context_length}":
                raise ValueError("e0 benchmark name does not match the context length")
            if self.expected_prompt_tokens != self.expected_context_length:
                raise ValueError("e0 prompt token count must equal the context length")


_FORBIDDEN_IMPORTS = ("vllm", "tpu_inference", "torch", "jax", "ray")
_LEGACY_MODULE_CACHE: dict[tuple[str, str, str], Any] = {}


def load_legacy_bench_module(
    name: str,
    *,
    pinned_files: Mapping[str, Mapping[str, Any]],
    repository_root: Path | None = None,
) -> Any:
    """Load one legacy bench module from its FILE, under a private module name.

    ``bench/`` is a namespace package whose modules put their own directory on
    ``sys.path`` when imported, and this file already loads two of them under
    private names for provenance. A plain ``import bench.<name>`` elsewhere can
    therefore be answered from whatever a previous loader left in
    ``sys.modules``. The §23.5 L7 criterion runs the legacy extractor, so it
    loads the file itself rather than trusting the import system's cache.
    """

    if f"{name}.py" not in _LEGACY_BENCH_MODULES:
        raise ValueError(f"{name!r} is not one of the pinned legacy bench modules")
    root = (
        Path(__file__).resolve().parents[3]
        if repository_root is None
        else Path(repository_root)
    ) / "bench"
    # The capture recorded the sha256 of every legacy bench file it ran. The
    # module loaded here imports its siblings, so ALL of them are bound, not
    # just the named one: a criterion computed from different bytes than the
    # capture used is not the legacy criterion.
    if not pinned_files:
        raise ValueError("legacy bench modules need the capture's pinned digests")
    digests: dict[str, str] = {}
    for filename in sorted(_LEGACY_BENCH_MODULES):
        record = pinned_files.get(filename)
        if record is None:
            raise ValueError(f"the capture pins no digest for {filename}")
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path.read_bytes()).hexdigest()
        if observed != record.get("sha256"):
            raise ValueError(
                f"legacy bench module {filename} is not the pinned capture's bytes"
            )
        digests[filename] = observed
    key = (str(root), name, digests[f"{name}.py"])
    cached = _LEGACY_MODULE_CACHE.get(key)
    if cached is not None:
        return cached
    path = root / f"{name}.py"
    before = set(sys.modules)
    inserted = str(root) not in sys.path
    if inserted:
        sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location(
            f"greenfield_legacy_bench_{name}", path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(str(root))
    offenders = sorted(
        loaded
        for loaded in set(sys.modules) - before
        if loaded.split(".")[0] in _FORBIDDEN_IMPORTS
    )
    if offenders:
        raise ImportError(
            f"legacy bench import crossed the model-execution boundary: {offenders}"
        )
    _LEGACY_MODULE_CACHE[key] = module
    return module


def _legacy_bench_git_records(root: Path, harness_git: str) -> dict[str, Any]:
    """Bind the bench files to the legacy harness pin recorded in the DB row.

    ``harness_git`` is the (abbreviated) commit the legacy run recorded; every
    bench module used here must be byte-identical to that commit's blob, and the
    repository HEAD is recorded so the capture names the code it ran.
    """
    import subprocess

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments], check=True, capture_output=True, text=True
        )
        return completed.stdout.strip()

    head = git("rev-parse", "HEAD")
    pinned = git("rev-parse", f"{harness_git}^{{commit}}")
    for filename in _LEGACY_BENCH_MODULES:
        blob = git("rev-parse", f"{pinned}:bench/{filename}")
        local = git("hash-object", str(root / "bench" / filename))
        if blob != local:
            raise ValueError(
                f"legacy bench module {filename} differs from harness pin {harness_git}"
            )
    return {"repository_head": head, "harness_git": harness_git, "harness_commit": pinned}


def _load_legacy_bench(root: Path) -> tuple[Any, Any, dict[str, dict[str, Any]]]:
    """Import the legacy bench prompt builders as pinned, namespaced utilities.

    Only the CPU prompt builders are used.  ``glm_longctx``/``dsa_throughput``
    import ``engine``, ``extract`` and ``provenance`` at module scope, so all
    five files are content-hashed; the import boundary is asserted afterwards:
    no model-execution package may have been loaded.
    """
    records = {}
    for filename in _LEGACY_BENCH_MODULES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        records[filename] = _file_record(path)
    before = set(sys.modules)
    inserted = str(root) not in sys.path
    if inserted:
        sys.path.insert(0, str(root))
    modules: dict[str, Any] = {}
    try:
        for name in ("glm_longctx", "dsa_throughput"):
            namespaced = f"greenfield_legacy_bench_{name}"
            spec = importlib.util.spec_from_file_location(namespaced, root / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[namespaced] = module
            spec.loader.exec_module(module)
            modules[name] = module
    finally:
        if inserted:
            sys.path.remove(str(root))
    loaded = set(sys.modules) - before
    offenders = sorted(
        name for name in loaded if name.split(".")[0] in _FORBIDDEN_IMPORTS
    )
    if offenders:
        raise ImportError(
            f"legacy bench import crossed the model-execution boundary: {offenders}"
        )
    longctx, throughput = modules["glm_longctx"], modules["dsa_throughput"]
    if list(longctx.PROMPT_PREFIX_IDS) != list(PROMPT_PREFIX_IDS):
        raise ValueError("legacy prompt prefix drifted")
    return longctx, throughput, records


def _read_source_row(config: LongContextOracleConfig) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{config.results_db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        integrity = [row[0] for row in connection.execute("pragma integrity_check")]
        if integrity != ["ok"]:
            raise ValueError(f"legacy results DB integrity failed: {integrity}")
        row = connection.execute(
            """
            SELECT i.id AS item_row_id, i.run_id, i.benchmark, i.item_id, i.asked_utc,
                   i.prompt, i.gold, i.raw_output, i.extracted, i.correct, i.score,
                   i.n_prompt_tokens, i.n_gen_tokens, i.latency_ms, i.seed,
                   i.finish_reason, i.truncated, r.created_utc AS run_created_utc,
                   r.model, r.model_revision, r.harness_git, r.fork_git, r.env_json,
                   r.pod, r.note
            FROM items AS i JOIN runs AS r ON r.run_id = i.run_id
            WHERE i.run_id = ? AND i.id = ?
            """,
            (config.run_id, config.item_row_id),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError("legacy long-context oracle row is missing")
    source = dict(row)
    expected: dict[str, Any] = {
        "benchmark": config.expected_benchmark,
        "fork_git": config.expected_fork_git,
        "harness_git": config.expected_harness_git,
        "item_row_id": config.item_row_id,
        "model": config.expected_model_uri,
        "n_gen_tokens": config.expected_generated_tokens,
        "n_prompt_tokens": config.expected_prompt_tokens,
        "run_id": config.run_id,
        "seed": config.expected_seed,
    }
    if config.kind == "passkey":
        expected.update({"correct": 1, "gold": config.expected_gold})
    mismatches = {
        name: {"expected": value, "observed": source.get(name)}
        for name, value in expected.items()
        if source.get(name) != value
    }
    if mismatches:
        raise ValueError(f"legacy long-context row identity drifted: {mismatches}")
    if not source.get("prompt") or not source.get("raw_output"):
        raise ValueError("legacy long-context row lacks prompt or raw output")
    environment = json.loads(source["env_json"])
    environment_expected: dict[str, Any] = {
        "model": config.expected_model_uri,
        "prompt_prefix_ids": list(PROMPT_PREFIX_IDS),
        "temperature": 0.0,
    }
    if config.kind == "passkey":
        environment_expected.update({"protocol": "raw", "max_new": config.expected_generated_tokens})
    else:
        environment_expected.update(
            {
                "prompt_generator": "dsa_throughput.build_prompt_ids/v1",
                "measure_tokens": config.expected_generated_tokens,
                "ignore_eos": True,
            }
        )
    environment_mismatches = {
        name: {"expected": value, "observed": environment.get(name)}
        for name, value in environment_expected.items()
        if environment.get(name) != value
    }
    if environment_mismatches:
        raise ValueError(f"legacy long-context generation protocol drifted: {environment_mismatches}")
    source["env_json"] = environment
    return source


def _rebuild_passkey(config: LongContextOracleConfig, source: Mapping[str, Any], tokenizer: Any, longctx: Any) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    stored_prompt = str(source["prompt"])
    match = _PROMPT_SHA_PATTERN.search(stored_prompt)
    if match is None:
        raise ValueError("stored passkey prompt carries no embedded sha256 (not a truncated long prompt)")
    stored_sha = match.group(1)
    text, key = longctx.build_trial(
        tokenizer, config.expected_context_length, config.expected_depth, config.expected_seed
    )
    if key != config.expected_gold:
        raise ValueError("rebuilt passkey differs from the legacy gold key")
    rebuilt_sha = sha256(text.encode("utf-8")).hexdigest()
    if rebuilt_sha != stored_sha:
        raise ValueError("rebuilt passkey prompt text differs from the legacy sha256")
    if longctx._prompt_for_db(text, 65536) != stored_prompt:
        raise ValueError("rebuilt passkey prompt does not reproduce the stored DB field")
    prompt_ids = np.asarray(longctx._prompt_ids(tokenizer, text), dtype=np.int32)
    generated_ids = np.asarray(
        tokenizer(source["raw_output"], add_special_tokens=False)["input_ids"], dtype=np.int32
    )
    roundtrip = tokenizer.decode(
        generated_ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if roundtrip != source["raw_output"]:
        raise ValueError("legacy output token roundtrip is not exact")
    if longctx.extract_passkey(source["raw_output"]) != config.expected_gold:
        raise ValueError("legacy raw output does not extract to the gold key")
    rebuild = {
        "builder": "glm_longctx.build_trial(tok, L, depth, seed)",
        "context_length": config.expected_context_length,
        "depth": config.expected_depth,
        "prompt_chars": len(text),
        "prompt_text_sha256": rebuilt_sha,
        "seed": config.expected_seed,
        "stored_prompt_field_reproduced": True,
        "prompt_ids_provenance": (
            "client re-tokenization of the sha256-verified prompt text with the pinned "
            "tokenizer ([gMASK]<sop> prefix ids + BPE stream, add_special_tokens=False); "
            "count-matched to the legacy server-side count; id-level identity to the "
            "legacy server-side ids is not verified"
        ),
        "generated_ids_provenance": (
            "re-tokenized from the legacy raw_output text (the legacy stored no ids); "
            "exact text roundtrip verified; diagnostic reference only, never a "
            "raw-tokens-exact claim"
        ),
    }
    return prompt_ids, generated_ids, text, rebuild


def _rebuild_e0(config: LongContextOracleConfig, source: Mapping[str, Any], throughput: Any) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    descriptor = json.loads(source["prompt"])
    if descriptor.get("generator") != "dsa_throughput.build_prompt_ids/v1":
        raise ValueError("stored E0 prompt descriptor generator drifted")
    if descriptor.get("seed") != config.expected_seed or descriptor.get("ctx") != config.expected_context_length:
        raise ValueError("stored E0 prompt descriptor seed/ctx drifted")
    vocab_size = int(descriptor["vocab_size"])
    ids = throughput.build_prompt_ids(config.expected_context_length, config.expected_seed, vocab_size)
    prompt_ids = np.asarray(ids, dtype=np.int32)
    if prompt_ids.shape != (config.expected_context_length,):
        raise ValueError("rebuilt E0 prompt length drifted")
    if throughput.prompt_sha256(ids) != descriptor.get("sha256"):
        raise ValueError("rebuilt E0 prompt ids differ from the legacy descriptor sha256")
    if (descriptor.get("sample_low"), descriptor.get("sample_high")) != tuple(throughput.sample_range(vocab_size)):
        raise ValueError("stored E0 sample range drifted")
    output = json.loads(source["raw_output"])
    generated_ids = np.asarray(output["token_ids"], dtype=np.int32)
    if generated_ids.shape != (config.expected_generated_tokens,):
        raise ValueError("legacy E0 generated token count drifted")
    rebuild = {
        "builder": "dsa_throughput.build_prompt_ids(ctx, seed, vocab_size)",
        "context_length": config.expected_context_length,
        "prompt_ids_sha256_int32_le": descriptor["sha256"],
        "sample_high": descriptor["sample_high"],
        "sample_low": descriptor["sample_low"],
        "seed": config.expected_seed,
        "vocab_size": vocab_size,
        "prompt_ids_provenance": "rebuilt by the legacy generator; int32-stream sha256 verified against the stored descriptor (exact ids)",
        "generated_ids_provenance": "legacy E0 greedy ids as stored; diagnostic only, E0 has no correctness oracle",
    }
    return prompt_ids, generated_ids, str(source["prompt"]), rebuild


def capture_long_context_oracle(config: LongContextOracleConfig) -> dict[str, Any]:
    """Seal one accepted legacy long-context row as an append-only token oracle."""
    if config.output_dir.exists():
        raise FileExistsError(f"append-only long-context oracle exists: {config.output_dir}")
    if not config.results_db.is_file():
        raise FileNotFoundError(config.results_db)
    required_tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    for filename in required_tokenizer_files:
        if not (config.tokenizer_root / filename).is_file():
            raise FileNotFoundError(config.tokenizer_root / filename)
    from safetensors.numpy import save_file
    from transformers import AutoTokenizer

    longctx, throughput, legacy_records = _load_legacy_bench(config.legacy_bench_root)
    source = _read_source_row(config)
    legacy_git = _legacy_bench_git_records(config.legacy_bench_root.parent, str(source["harness_git"]))
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_root, local_files_only=True, trust_remote_code=True
    )
    if config.kind == "passkey":
        prompt_ids, generated_ids, prompt_text, rebuild = _rebuild_passkey(config, source, tokenizer, longctx)
    else:
        prompt_ids, generated_ids, prompt_text, rebuild = _rebuild_e0(config, source, throughput)
    if prompt_ids.shape != (config.expected_prompt_tokens,):
        raise ValueError("rebuilt prompt length differs from the legacy row")
    if not np.array_equal(prompt_ids[:2], np.asarray(PROMPT_PREFIX_IDS)):
        raise ValueError("rebuilt prompt prefix drifted")
    for name, value in (("prompt", prompt_ids), ("generated", generated_ids)):
        if value.ndim != 1 or not value.size or np.any(value < 0) or np.any(value >= MODEL_VOCAB_SIZE):
            raise ValueError(f"legacy {name} token IDs escaped model vocabulary")
    config.output_dir.mkdir(parents=True)
    tensor_path = config.output_dir / "tokens.safetensors"
    partial = tensor_path.with_suffix(".safetensors.partial")
    save_file(
        {"generated_token_ids": generated_ids, "prompt_token_ids": prompt_ids},
        partial,
        metadata={"artifact_kind": ARTIFACT_KIND, "format_version": str(FORMAT_VERSION), "model_id": MODEL_ID},
    )
    partial.replace(tensor_path)
    prompt_path = config.output_dir / "prompt.txt"
    output_path = config.output_dir / "raw_output.txt"
    source_path = config.output_dir / "source_row.json"
    rebuild_path = config.output_dir / "rebuild.json"
    _write_text_once(prompt_path, prompt_text)
    _write_text_once(output_path, str(source["raw_output"]))
    _write_text_once(source_path, json.dumps(source, indent=2, sort_keys=True) + "\n")
    _write_text_once(rebuild_path, json.dumps(rebuild, indent=2, sort_keys=True) + "\n")
    manifest: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "capture_code_hash": config.capture_code_hash,
        "files": {
            "prompt": _file_record(prompt_path),
            "raw_output": _file_record(output_path),
            "rebuild": _file_record(rebuild_path),
            "source_row": _file_record(source_path),
            "tokens": _file_record(tensor_path),
        },
        "format_version": FORMAT_VERSION,
        "generated_token_count": int(generated_ids.size),
        "generated_token_ids_sha256": _array_sha256(generated_ids),
        "kind": config.kind,
        "legacy_bench_files": legacy_records,
        "legacy_bench_git": legacy_git,
        "legacy_repository_pin_at_capture": config.legacy_repository_pin,
        "model_id": MODEL_ID,
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "prompt_prefix_ids": list(PROMPT_PREFIX_IDS),
        "prompt_token_count": int(prompt_ids.size),
        "prompt_token_ids_sha256": _array_sha256(prompt_ids),
        "rebuild": rebuild,
        "source": {
            "benchmark": source["benchmark"],
            "context_length": config.expected_context_length,
            "depth": config.expected_depth,
            "fork_git": source["fork_git"],
            "gold": source["gold"],
            "harness_git": source["harness_git"],
            "item_id": source["item_id"],
            "item_row_id": source["item_row_id"],
            "latency_ms": source["latency_ms"],
            "model_uri": source["model"],
            "run_id": source["run_id"],
            "seed": source["seed"],
            "source_row_sha256": sha256(_canonical_json(source).encode("utf-8")).hexdigest(),
        },
        "tokenizer_files": {
            filename: _file_record(config.tokenizer_root / filename) for filename in required_tokenizer_files
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    _write_text_once(config.output_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


@dataclass(frozen=True)
class Ws32LongContextOracle:
    """A token-only long-context oracle (spec §23.1/§23.5).

    There is deliberately no DSA field: the legacy harness captured no DSA
    events at these lengths and §23.5 forbids inventing one. ``generated_token_ids``
    are the legacy 20 (L7) or 256 (L8) ids and are a DIAGNOSTIC reference only —
    §23.5 states that nothing may be labelled "raw tokens exact" at these
    lengths. For L7 the pass criterion is ``gold``.
    """

    kind: str
    manifest: Mapping[str, Any]
    manifest_sha256: str
    prompt_token_ids: np.ndarray
    generated_token_ids: np.ndarray
    gold: str | None
    depth: float | None
    source_run_id: int
    item_row_id: int


def load_ws32_long_context_oracle(
    oracle_dir: Path,
    *,
    expected_manifest_sha256: str,
    expected_success_sha256: str,
    expected_kind: str,
) -> Ws32LongContextOracle:
    """Load a fully inspected long-context oracle, bound by manifest and SUCCESS."""

    from safetensors import safe_open

    oracle_dir = Path(oracle_dir)
    manifest = inspect_long_context_oracle(oracle_dir)
    if manifest.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("WS32 long-context oracle identity drifted")
    if manifest.get("kind") != expected_kind:
        raise ValueError(
            f"WS32 long-context oracle kind is {manifest.get('kind')}, not {expected_kind}"
        )

    success = oracle_dir.parent / "SUCCESS"
    raw = success.read_bytes()
    if sha256(raw).hexdigest() != expected_success_sha256:
        raise ValueError("WS32 long-context oracle terminal SUCCESS identity drifted")
    fields: dict[str, str] = {}
    for line in raw.decode("utf-8").splitlines():
        if line.count("=") != 1:
            raise ValueError("WS32 long-context oracle SUCCESS schema drifted")
        key, value = line.split("=", 1)
        if not key or key in fields:
            raise ValueError("WS32 long-context oracle SUCCESS schema drifted")
        fields[key] = value
    if (
        fields.get("artifact_kind") != ARTIFACT_KIND
        or fields.get("manifest_sha256") != expected_manifest_sha256
        or not fields.get("remote_prefix", "").startswith("gs://driftbench-dsv4-uc/")
    ):
        raise ValueError("WS32 long-context oracle SUCCESS binding drifted")

    with safe_open(oracle_dir / "tokens.safetensors", framework="np") as handle:
        prompt = np.asarray(handle.get_tensor("prompt_token_ids"), dtype=np.int32)
        generated = np.asarray(handle.get_tensor("generated_token_ids"), dtype=np.int32)
    source = manifest["source"]
    gold = source.get("gold") if manifest["kind"] == "passkey" else None
    if manifest["kind"] == "passkey" and not (isinstance(gold, str) and gold.isdigit()):
        raise ValueError("WS32 long-context passkey oracle has no gold key")
    return Ws32LongContextOracle(
        kind=str(manifest["kind"]),
        manifest=manifest,
        manifest_sha256=str(manifest["manifest_sha256"]),
        prompt_token_ids=prompt,
        generated_token_ids=generated,
        gold=gold,
        depth=source.get("depth"),
        source_run_id=int(source["run_id"]),
        item_row_id=int(source["item_row_id"]),
    )


def inspect_long_context_oracle(output_dir: Path) -> dict[str, Any]:
    """Verify every long-context oracle identity without opening the source DB."""
    from safetensors import safe_open

    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    if manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("not a greenfield long-context legacy oracle")
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("kind") not in KINDS:
        raise ValueError("long-context oracle format drifted")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("long-context oracle manifest checksum mismatch")
    if manifest.get("model_id") != MODEL_ID or manifest.get("model_vocab_size") != MODEL_VOCAB_SIZE:
        raise ValueError("long-context oracle model identity drifted")
    if manifest.get("prompt_prefix_ids") != list(PROMPT_PREFIX_IDS):
        raise ValueError("long-context oracle prompt prefix drifted")
    if set(manifest.get("files", {})) != {"prompt", "raw_output", "rebuild", "source_row", "tokens"}:
        raise ValueError("long-context oracle file set drifted")
    for record in manifest["files"].values():
        path = output_dir / record["filename"]
        if path.stat().st_size != record["byte_count"] or _sha256_file(path) != record["sha256"]:
            raise ValueError(f"long-context oracle file drifted: {record['filename']}")
    source = json.loads((output_dir / manifest["files"]["source_row"]["filename"]).read_text())
    if sha256(_canonical_json(source).encode("utf-8")).hexdigest() != manifest["source"]["source_row_sha256"]:
        raise ValueError("long-context oracle source row drifted")
    with safe_open(output_dir / "tokens.safetensors", framework="np") as handle:
        tokens = {name: handle.get_tensor(name) for name in handle.keys()}
    prompt = np.asarray(tokens["prompt_token_ids"], dtype=np.int32)
    generated = np.asarray(tokens["generated_token_ids"], dtype=np.int32)
    if (
        prompt.size != manifest["prompt_token_count"]
        or generated.size != manifest["generated_token_count"]
        or _array_sha256(prompt) != manifest["prompt_token_ids_sha256"]
        or _array_sha256(generated) != manifest["generated_token_ids_sha256"]
        or not np.array_equal(prompt[:2], np.asarray(PROMPT_PREFIX_IDS))
    ):
        raise ValueError("long-context oracle token arrays drifted")
    if manifest["kind"] == "passkey":
        gold = manifest["source"]["gold"]
        text = (output_dir / manifest["files"]["prompt"]["filename"]).read_text(encoding="utf-8")
        if sha256(text.encode("utf-8")).hexdigest() != manifest["rebuild"]["prompt_text_sha256"]:
            raise ValueError("long-context oracle prompt text drifted")
        if not isinstance(gold, str) or not gold.isdigit() or f"passcode is {gold}." not in text:
            raise ValueError("long-context oracle passkey is not present in the prompt")
    return manifest
