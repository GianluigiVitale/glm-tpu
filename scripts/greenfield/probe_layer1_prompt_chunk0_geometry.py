#!/usr/bin/env python3
"""Bounded probe: layer-1 prompt keys of chunk 0 in the legacy prefill geometry.

Produces the accepted layer-0 input normalization for prompt rows ``[0, 2048)``
in one executable, synchronizes that completed device buffer, and passes it to
a separate executable containing the full layer-0 consumer.  The consumer has
no input-normalization weight or recomputation path.  It executes every
projection, attention, norm and dense MLP in the legacy 2,048-row geometry,
forms the layer-1 key, and compares it bitwise with the sealed legacy cache.
Row 0 is decisive (attention over one key is exact by construction); rows >= 1
use a plain FP32 causal softmax and are diagnostic only.

Provenance (the accepted acquisition drivers' boundary): this file verifies
that its own bytes equal the committed blob at the approved pin; the project
modules are imported from a sealed in-memory zip archive built from exact
committed Git blobs (no worktree bytes); the interpreter is the root-owned
immutable Gate-D Python under ``-I -S -B``; the sealed JAX/libtpu sites are
verified by tree digest; every input is read once through a no-follow
descriptor and digest-bound.  The separately compiled key control must equal
the DB518 layer-0 cache bitwise.  No decoder, Gate-D, DB or performance claim.
"""

from __future__ import annotations

import argparse
import fcntl
from functools import partial
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any
import zipfile

PROBE_REPOSITORY_PATH = "scripts/greenfield/probe_layer1_prompt_chunk0_geometry.py"
INSTALL_PATH = Path(
    "/usr/local/libexec/glm-tpu/gate-d-layer1-prompt-chunk0-geometry-v7/"
    "probe_layer1_prompt_chunk0_geometry.py"
)
RUN_ROOT = Path("/home/gianl/gate-d-runs")
TAG_PATTERN = re.compile(
    r"greenfield_layer1_prompt_chunk0_geometry_[0-9]{8}T[0-9]{15}Z"
)
JAX_SITE_ROOT = "/opt/glm-tpu/gate-d-jax-site-55233c63939e"
LIBTPU_SITE_ROOT = "/opt/glm-tpu/gate-d-libtpu-site-db7598c867f3"
PYTHON_RUNTIME_ROOT = "/opt/glm-tpu/gate-d-python-3.12.13-021044895e95"
EXPECTED_RUNTIME_PATH = (
    f"{PYTHON_RUNTIME_ROOT}/lib/python312.zip",
    f"{PYTHON_RUNTIME_ROOT}/lib/python3.12",
    f"{PYTHON_RUNTIME_ROOT}/lib/python3.12/lib-dynload",
)
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_MEMFD_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008
GIT_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "HOME": "/nonexistent",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_SSH_COMMAND": "/bin/false",
}


def _git_bytes(repo: Path, *arguments: str) -> bytes:
    return subprocess.check_output(
        [
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.attributesFile=/dev/null",
            "-C",
            str(repo),
            *arguments,
        ],
        env=GIT_ENVIRONMENT,
    )


def _git_text(repo: Path, *arguments: str) -> str:
    return _git_bytes(repo, *arguments).decode("ascii", errors="strict").strip()


def _snapshot_regular(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"file identity is unsafe: {path}")
        chunks = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(block)
        final = os.fstat(descriptor)
        if (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        ) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ):
            raise RuntimeError(f"file changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _require_root_boundary(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeError("probe installation path is unsafe")
    for parent in (path.parent, *path.parent.parents):
        metadata = os.stat(parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise RuntimeError(f"probe installation parent is mutable: {parent}")


def _verify_running_source(
    repo: Path, code_pin: str, expected_source_sha256: str
) -> str:
    """This file's bytes must equal the committed blob at the approved pin."""

    if _git_text(repo, "rev-parse", f"{code_pin}^{{commit}}") != code_pin:
        raise RuntimeError("code pin is not a commit in the repository")
    if _git_bytes(repo, "for-each-ref", "--format=%(refname)", "refs/replace").strip():
        raise RuntimeError("repository has forbidden replacement refs")
    if Path(__file__) != INSTALL_PATH:
        raise RuntimeError("probe is not executing from the immutable capsule")
    _require_root_boundary(INSTALL_PATH)
    metadata = os.stat(INSTALL_PATH, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o555
        or os.listxattr(INSTALL_PATH, follow_symlinks=False)
    ):
        raise RuntimeError("probe installation identity is unsafe")
    raw = _snapshot_regular(INSTALL_PATH)
    committed = _git_bytes(repo, "show", f"{code_pin}:{PROBE_REPOSITORY_PATH}")
    observed_sha256 = sha256(raw).hexdigest()
    if raw != committed or observed_sha256 != expected_source_sha256:
        raise RuntimeError("probe is not the committed blob at the approved pin")
    return observed_sha256


def _open_inherited_run_dir(run_dir: Path, inherited_fd: int) -> int:
    if (
        not run_dir.is_absolute()
        or run_dir.parent != RUN_ROOT
        or TAG_PATTERN.fullmatch(run_dir.name) is None
        or inherited_fd != 7
    ):
        raise RuntimeError("probe run-directory invocation drifted")
    root_fd = os.open(
        RUN_ROOT, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    descriptor = -1
    try:
        descriptor = os.dup(inherited_fd)
        held = os.fstat(descriptor)
        named = os.stat(run_dir.name, dir_fd=root_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(held.st_mode)
            or stat.S_IMODE(held.st_mode) != 0o700
            or held.st_uid != os.geteuid()
            or held.st_gid != os.getegid()
            or os.listxattr(descriptor)
            or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise RuntimeError("probe run-directory authority drifted")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        os.close(root_fd)


def _write_run_member_exclusive(run_fd: int, relative: str, raw: bytes) -> None:
    parts = Path(relative).parts
    if (
        not parts
        or relative.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or len(parts) > 2
        or (len(parts) == 2 and parts[0] != "hlo")
    ):
        raise RuntimeError(f"unsafe probe output member: {relative}")
    parent_fd = run_fd
    owned_parent = False
    try:
        if len(parts) == 2:
            parent_fd = os.open(
                parts[0],
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=run_fd,
            )
            owned_parent = True
        descriptor = os.open(
            parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o400,
            dir_fd=parent_fd,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise RuntimeError(f"unsafe probe output identity: {relative}")
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RuntimeError(f"probe output write stalled: {relative}")
                view = view[written:]
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            named = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            if (
                (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
                or not stat.S_ISREG(named.st_mode)
                or named.st_nlink != 1
                or named.st_size != len(raw)
            ):
                raise RuntimeError(f"probe output identity changed: {relative}")
            os.fsync(parent_fd)
        finally:
            os.close(descriptor)
    finally:
        if owned_parent:
            os.close(parent_fd)


def _sealed_git_source_archive(repo: Path, code_pin: str) -> tuple[str, dict[str, Any]]:
    """Sealed zipimport archive of ``glm_tpu/`` built from exact committed blobs."""

    tree = _git_bytes(repo, "ls-tree", "-r", "-z", code_pin, "--", "glm_tpu")
    records = []
    buffer = BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for raw_entry in tree.split(b"\0"):
            if not raw_entry:
                continue
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, kind, object_id = metadata.split(b" ", 2)
            path = raw_path.decode("utf-8", errors="strict")
            if (
                kind != b"blob"
                or mode not in {b"100644", b"100755"}
                or not path.startswith("glm_tpu/")
            ):
                raise RuntimeError(f"unsupported committed source entry: {path}")
            payload = _git_bytes(repo, "cat-file", "blob", object_id.decode("ascii"))
            records.append(
                {
                    "git_object_id": object_id.decode("ascii"),
                    "path": path,
                    "sha256": sha256(payload).hexdigest(),
                    "size": len(payload),
                }
            )
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100555 if mode == b"100755" else 0o100444) << 16
            archive.writestr(info, payload)
    if not records or records != sorted(records, key=lambda item: item["path"]):
        raise RuntimeError("committed source tree order drifted")
    raw_archive = buffer.getvalue()
    descriptor = os.memfd_create(
        "gate-d-chunk0-probe-source.zip", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC
    )
    view = memoryview(raw_archive)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RuntimeError("committed source archive write stalled")
        view = view[written:]
    fcntl.fcntl(descriptor, _F_ADD_SEALS, _MEMFD_SEALS)
    if fcntl.fcntl(descriptor, _F_GET_SEALS) != _MEMFD_SEALS:
        raise RuntimeError("committed source archive seal drifted")
    if os.pread(descriptor, len(raw_archive) + 1, 0) != raw_archive:
        raise RuntimeError("committed source archive revalidation drifted")
    manifest = json.dumps(
        records,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"/proc/self/fd/{descriptor}", {
        "archive_sha256": sha256(raw_archive).hexdigest(),
        "file_manifest_count": len(records),
        "file_manifest_sha256": sha256(manifest).hexdigest(),
    }


def _install_sealed_path(archive_path: str) -> None:
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
    ):
        raise RuntimeError("probe must run under python -I -S")
    sys.path[:] = [
        archive_path,
        JAX_SITE_ROOT,
        LIBTPU_SITE_ROOT,
        *EXPECTED_RUNTIME_PATH,
    ]


ARTIFACT_KIND = "greenfield_layer1_prompt_chunk0_legacy_geometry_probe"
CHUNK_ROWS = 2048
PROMPT_ROWS = 8155
MAIN_ROPE_CAPACITY = 8192
MAIN_ROPE_THETA = 8_000_000.0
LAYER0_NAMES = {
    "q_b": "model.layers.0.self_attn.q_b_proj.weight",
    "kv_b": "model.layers.0.self_attn.kv_b_proj.weight",
    "o": "model.layers.0.self_attn.o_proj.weight",
    "gate": "model.layers.0.mlp.gate_proj.weight",
    "up": "model.layers.0.mlp.up_proj.weight",
    "down": "model.layers.0.mlp.down_proj.weight",
}
LAYER0_NORMS = {
    "kv_a_norm": "model.layers.0.self_attn.kv_a_layernorm.weight",
    "post_norm": "model.layers.0.post_attention_layernorm.weight",
}
LAYER1_NAMES = {"wk1": "model.layers.1.self_attn.indexer.wk.weight"}
LAYER1_NORMS = {
    "input_norm1": "model.layers.1.input_layernorm.weight",
    "k_norm1_weight": "model.layers.1.self_attn.indexer.k_norm.weight",
    "k_norm1_bias": "model.layers.1.self_attn.indexer.k_norm.bias",
}
CHECKPOINT_TENSOR_NAMES = {
    **LAYER0_NAMES,
    **LAYER0_NORMS,
    **LAYER1_NAMES,
    **LAYER1_NORMS,
    **{
        f"{k}_scale": f"{n}_scale_inv"
        for k, n in {**LAYER0_NAMES, **LAYER1_NAMES}.items()
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--code-pin", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-index-sha256", required=True)
    parser.add_argument("--weight-digests", type=Path, required=True)
    parser.add_argument("--weight-digests-sha256", required=True)
    parser.add_argument("--legacy-layer1-cache-dir", type=Path, required=True)
    parser.add_argument("--legacy-layer1-manifest-sha256", required=True)
    parser.add_argument("--db518-result", type=Path, required=True)
    parser.add_argument("--db518-result-sha256", required=True)
    parser.add_argument("--softmax-scale", type=float, default=256**-0.5)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-dir-fd", type=int, choices=(7,), required=True)
    return parser.parse_args()


def _cache_rows(owner_bits: Any, rows: int) -> Any:
    """DB518 capture layout: owner=(p%512)//256, page=p//512, row=p%256."""

    import numpy as np

    positions = np.arange(rows)
    local = positions % 512
    return np.ascontiguousarray(owner_bits[local // 256, positions // 512, local % 256])


def main() -> int:
    args = _parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.code_pin) is None:
        raise RuntimeError("code pin must be a 40-hex commit")
    if re.fullmatch(r"[0-9a-f]{64}", args.expected_source_sha256) is None:
        raise RuntimeError("expected probe source digest is invalid")
    if args.run_tag != args.run_dir.name:
        raise RuntimeError("probe run tag and run-directory identity differ")
    probe_sha256 = _verify_running_source(
        args.repository, args.code_pin, args.expected_source_sha256
    )
    run_fd = _open_inherited_run_dir(args.run_dir, args.run_dir_fd)
    archive_path, archive_identity = _sealed_git_source_archive(
        args.repository, args.code_pin
    )
    _install_sealed_path(archive_path)

    # Everything below imports only from the sealed archive and sealed sites.
    from glm_tpu.greenfield.benchmarking import sealed_runtime

    provenance = {
        "probe_sha256": probe_sha256,
        "source_archive": archive_identity,
        "python": sealed_runtime.validate_python_runtime(),
        "sites": sealed_runtime.validate_dependency_sites(),
        "sys_path": list(sys.path),
    }
    if tuple(sys.path) != (
        archive_path,
        JAX_SITE_ROOT,
        LIBTPU_SITE_ROOT,
        *EXPECTED_RUNTIME_PATH,
    ):
        raise RuntimeError("sealed import path drifted")

    import numpy as np
    import jax
    import jax.numpy as jnp
    import ml_dtypes

    from glm_tpu.greenfield.benchmarking import legacy_prefill_owner_packing as pk
    from glm_tpu.greenfield.benchmarking import numpy_safetensors as ns
    from glm_tpu.greenfield.benchmarking.legacy_prefill_chunk_probe import (
        CONSUMER_WEIGHT_KEYS,
        legacy_geometry_chunk_consumer_gather_from_normalized,
    )
    from glm_tpu.greenfield.kernels.reference.dsa_association import (
        Layer0DsaProbeGeometry,
        layer0_prompt_index_key_from_normalized_boundary_chunk,
        layer0_prompt_normalized_hidden_boundary_chunk,
    )
    from glm_tpu.greenfield.kernels.reference.dsa import DsaNumericalContract
    from glm_tpu.greenfield.kernels.reference.prefill_index import (
        decode_stage_local_prefill_index_wk_bf16,
        promote_stage_local_prefill_index_wk,
    )
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host
    from glm_tpu.greenfield.validation.chunk0_real_layer_consumer_hlo import (
        require_real_layer_consumer_hlo,
    )
    from glm_tpu.greenfield.validation.original_db518_normalized_boundary_hlo import (
        require_completed_normalization_boundary_hlo,
        require_normalized_key_control_boundary_hlo,
    )

    if jax.default_backend() != "tpu":
        raise RuntimeError(f"probe requires TPU, got {jax.default_backend()}")
    if jax.local_device_count() != 4 or jax.device_count() != 4:
        raise RuntimeError("probe requires one isolated four-chip TPU-v4 host")
    device = jax.local_devices()[0]

    started = time.time()
    input_manifest, arrays = ns.load_layer0_dsa_association_input(
        args.input_dir, expected_manifest_sha256=args.input_manifest_sha256
    )
    legacy_manifest, legacy_layer1_bits = ns.load_legacy_prompt_index_cache(
        args.legacy_layer1_cache_dir,
        expected_manifest_sha256=args.legacy_layer1_manifest_sha256,
    )
    if int(legacy_manifest.get("layer_id", 0)) != 1:
        raise RuntimeError("legacy cache artifact is not the layer-1 prompt cache")
    db518_raw = _snapshot_regular(args.db518_result)
    if sha256(db518_raw).hexdigest() != args.db518_result_sha256:
        raise RuntimeError("DB518 result.npz identity drifted")
    db518 = np.load(BytesIO(db518_raw))
    greenfield_layer0 = _cache_rows(
        db518["layer0_index_cache_owners_bfloat16_bits"], CHUNK_ROWS
    )
    greenfield_layer1 = _cache_rows(
        db518["layer1_index_cache_owners_bfloat16_bits"], CHUNK_ROWS
    )
    legacy_layer1 = legacy_layer1_bits[:CHUNK_ROWS]

    digests_raw = _snapshot_regular(args.weight_digests)
    if sha256(digests_raw).hexdigest() != args.weight_digests_sha256:
        raise RuntimeError("weight digest record identity drifted")
    digest_record = json.loads(digests_raw)
    if digest_record.get("artifact_kind") != "gate_d_chunk0_probe_weight_digests" or (
        digest_record.get("checkpoint_index_sha256") != args.checkpoint_index_sha256
        or digest_record.get("diagnostic_only") is not True
    ):
        raise RuntimeError("weight digest record contract drifted")
    tensors = ns.load_checkpoint_tensors(
        args.checkpoint_root,
        CHECKPOINT_TENSOR_NAMES,
        index_sha256=args.checkpoint_index_sha256,
        shard_digests={
            digest_record["shard"]["filename"]: digest_record["shard"]["sha256"]
        },
    )
    expected_tensors = digest_record["tensors"]
    if set(expected_tensors) != set(tensors):
        raise RuntimeError("weight digest record covers a different tensor set")
    for key, array in tensors.items():
        entry = expected_tensors[key]
        if (
            entry["shape"] != list(array.shape)
            or entry["dtype"] != str(array.dtype)
            or entry["sha256"]
            != sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        ):
            raise RuntimeError(
                f"checkpoint tensor drifted from the pinned digest: {key}"
            )

    prompt_ids = arrays["prompt_token_ids"]
    if prompt_ids.shape != (PROMPT_ROWS,):
        raise RuntimeError("prompt token geometry drifted")
    unique_token_ids = arrays["unique_token_ids"]
    unique_embedding_bits = arrays["unique_embedding_bfloat16_bits"]
    embedding_lookup = {
        int(token): index for index, token in enumerate(unique_token_ids.tolist())
    }
    embedding_rows = np.asarray(
        [embedding_lookup[int(token)] for token in prompt_ids[:CHUNK_ROWS]],
        dtype=np.int32,
    )
    if not np.array_equal(unique_token_ids[embedding_rows], prompt_ids[:CHUNK_ROWS]):
        raise RuntimeError("prompt embedding-row association drifted")
    qkv_bits, qkv_scale = pk.pack_fused_qkv_a_owners(
        arrays["self_attn__q_a_proj__weight"],
        arrays["self_attn__q_a_proj__weight_scale_inv"],
        arrays["self_attn__kv_a_proj_with_mqa__weight"],
        arrays["self_attn__kv_a_proj_with_mqa__weight_scale_inv"],
    )
    qb_bits, qb_scale = pk.pack_q_b_owners(tensors["q_b"], tensors["q_b_scale"])
    w_uk_t, w_uv = pk.absorbed_kv_b_owners(tensors["kv_b"], tensors["kv_b_scale"])
    o_bits, o_scale = pk.pack_o_proj_owners(tensors["o"], tensors["o_scale"])
    gu_bits, gu_scale = pk.pack_gate_up_owners(
        tensors["gate"], tensors["gate_scale"], tensors["up"], tensors["up_scale"]
    )
    down_bits, down_scale = pk.pack_down_owners(tensors["down"], tensors["down_scale"])
    rope_table = build_rotary_table_host(
        MAIN_ROPE_CAPACITY, rotary_dim=64, theta=MAIN_ROPE_THETA
    )
    contract = DsaNumericalContract()

    def bf16(bits: Any) -> Any:
        return jax.device_put(
            jnp.asarray(
                np.ascontiguousarray(bits).astype(np.uint16).view(ml_dtypes.bfloat16)
            ),
            device,
        )

    def put(value: Any) -> Any:
        return jax.device_put(jnp.asarray(value), device)

    # DB518/DB519 require each half of wk adaptation to finish as its own
    # executable.  Keeping raw FP8 wk inside the large chunk computation
    # changes TPU lowering and was already rejected by the protected DB519
    # control.  The main pipeline therefore receives only adapted FP32 wk.
    wk0_bits_device = put(arrays["self_attn__indexer__wk__weight"])
    wk0_scale_device = put(arrays["self_attn__indexer__wk__weight_scale_inv"])
    wk1_bits_device = put(tensors["wk1"])
    wk1_scale_device = put(tensors["wk1_scale"])

    decode_lowered = jax.jit(
        lambda bits, scale: decode_stage_local_prefill_index_wk_bf16(
            bits, scale, contract=contract
        )
    ).lower(wk0_bits_device, wk0_scale_device)
    decode_compiled = decode_lowered.compile()
    decode_hlo = decode_compiled.as_text()
    decode_stablehlo = decode_lowered.as_text()
    _write_run_member_exclusive(
        run_fd, "hlo/wk_decode.optimized_hlo.txt", decode_hlo.encode("utf-8")
    )
    _write_run_member_exclusive(
        run_fd, "hlo/wk_decode.stablehlo.mlir", decode_stablehlo.encode("utf-8")
    )
    wk0_bf16 = decode_compiled(wk0_bits_device, wk0_scale_device)
    wk0_bf16.block_until_ready()
    wk1_bf16 = decode_compiled(wk1_bits_device, wk1_scale_device)
    wk1_bf16.block_until_ready()

    promote_lowered = jax.jit(
        lambda value: promote_stage_local_prefill_index_wk(value, contract=contract)
    ).lower(wk0_bf16)
    promote_compiled = promote_lowered.compile()
    promote_hlo = promote_compiled.as_text()
    promote_stablehlo = promote_lowered.as_text()
    _write_run_member_exclusive(
        run_fd, "hlo/wk_promote.optimized_hlo.txt", promote_hlo.encode("utf-8")
    )
    _write_run_member_exclusive(
        run_fd, "hlo/wk_promote.stablehlo.mlir", promote_stablehlo.encode("utf-8")
    )
    wk0 = promote_compiled(wk0_bf16)
    wk0.block_until_ready()
    wk1 = promote_compiled(wk1_bf16)
    wk1.block_until_ready()

    input_norm0 = bf16(arrays["input_layernorm__weight"])
    k_norm0_weight = bf16(arrays["self_attn__indexer__k_norm__weight"])
    k_norm0_bias = bf16(arrays["self_attn__indexer__k_norm__bias"])
    consumer_weights = {
        "q_a_norm": bf16(arrays["self_attn__q_a_layernorm__weight"]),
        "kv_a_norm": bf16(tensors["kv_a_norm"]),
        "post_norm": bf16(tensors["post_norm"]),
        "input_norm1": bf16(tensors["input_norm1"]),
        "k_norm1_weight": bf16(tensors["k_norm1_weight"]),
        "k_norm1_bias": bf16(tensors["k_norm1_bias"]),
        "qkv_bits": put(qkv_bits),
        "qkv_scale": put(qkv_scale),
        "qb_bits": put(qb_bits),
        "qb_scale": put(qb_scale),
        "w_uk_t": put(w_uk_t),
        "w_uv": put(w_uv),
        "o_bits": put(o_bits),
        "o_scale": put(o_scale),
        "gu_bits": put(gu_bits),
        "gu_scale": put(gu_scale),
        "down_bits": put(down_bits),
        "down_scale": put(down_scale),
        "wk1": wk1,
        "rope_table": bf16(rope_table.view(np.uint16)),
    }
    if set(consumer_weights) != set(CONSUMER_WEIGHT_KEYS):
        raise RuntimeError("real layer consumer weight boundary drifted")

    positions_np = np.arange(CHUNK_ROWS, dtype=np.int32)
    unique_embeddings = bf16(unique_embedding_bits)
    embedding_rows_full = put(embedding_rows)
    positions_full = put(positions_np)

    geometry = Layer0DsaProbeGeometry()
    normalizer = partial(
        layer0_prompt_normalized_hidden_boundary_chunk,
        geometry=geometry,
    )
    normalized_lowered = jax.jit(normalizer).lower(
        unique_embeddings, embedding_rows_full, input_norm0
    )
    normalized_compiled = normalized_lowered.compile()
    normalized_hlo = normalized_compiled.as_text()
    normalized_stablehlo = normalized_lowered.as_text()
    _write_run_member_exclusive(
        run_fd,
        "hlo/normalized_boundary.optimized_hlo.txt",
        normalized_hlo.encode("utf-8"),
    )
    _write_run_member_exclusive(
        run_fd,
        "hlo/normalized_boundary.stablehlo.mlir",
        normalized_stablehlo.encode("utf-8"),
    )

    key_control = partial(
        layer0_prompt_index_key_from_normalized_boundary_chunk,
        geometry=geometry,
    )
    normalized_abstract = jax.ShapeDtypeStruct(
        (CHUNK_ROWS, geometry.hidden_size), jnp.bfloat16
    )
    key_lowered = jax.jit(key_control).lower(
        normalized_abstract,
        positions_full,
        wk0,
        k_norm0_weight,
        k_norm0_bias,
    )
    key_compiled = key_lowered.compile()
    key_hlo = key_compiled.as_text()
    key_stablehlo = key_lowered.as_text()
    _write_run_member_exclusive(
        run_fd,
        "hlo/normalized_key_control.optimized_hlo.txt",
        key_hlo.encode("utf-8"),
    )
    _write_run_member_exclusive(
        run_fd,
        "hlo/normalized_key_control.stablehlo.mlir",
        key_stablehlo.encode("utf-8"),
    )

    consumer = partial(
        legacy_geometry_chunk_consumer_gather_from_normalized,
        softmax_scale=args.softmax_scale,
        contract=contract,
    )
    consumer_lowered = jax.jit(consumer).lower(
        unique_embeddings,
        embedding_rows_full,
        normalized_abstract,
        positions_full,
        consumer_weights,
    )
    consumer_compiled = consumer_lowered.compile()
    consumer_hlo = consumer_compiled.as_text()
    consumer_stablehlo = consumer_lowered.as_text()
    _write_run_member_exclusive(
        run_fd,
        "hlo/real_layer_consumer.optimized_hlo.txt",
        consumer_hlo.encode("utf-8"),
    )
    _write_run_member_exclusive(
        run_fd,
        "hlo/real_layer_consumer.stablehlo.mlir",
        consumer_stablehlo.encode("utf-8"),
    )

    # Persist all compiler products, then fail closed on every executable
    # before invoking any numerical work.
    normalization_contract = require_completed_normalization_boundary_hlo(
        normalized_hlo, normalized_stablehlo
    )
    key_contract = require_normalized_key_control_boundary_hlo(key_hlo, key_stablehlo)
    consumer_contract = require_real_layer_consumer_hlo(
        consumer_hlo, consumer_stablehlo
    )

    # The completed normalization is synchronized once and supplied directly
    # to two separately compiled device consumers.  It is never transferred
    # to the host.  One final device_get occurs only after both consumers have
    # completed.
    normalized_device = normalized_compiled(
        unique_embeddings, embedding_rows_full, input_norm0
    )
    normalized_device.block_until_ready()
    boundary_keys_device = key_compiled(
        normalized_device,
        positions_full,
        wk0,
        k_norm0_weight,
        k_norm0_bias,
    )
    result_full_device = consumer_compiled(
        unique_embeddings,
        embedding_rows_full,
        normalized_device,
        positions_full,
        consumer_weights,
    )
    boundary_keys_device.block_until_ready()
    jax.tree.map(lambda value: value.block_until_ready(), result_full_device)
    boundary_keys_host, result_full = jax.device_get(
        (boundary_keys_device, result_full_device)
    )
    import_closure = sealed_runtime.verify_import_closure(Path(archive_path))

    def bits(value: Any) -> Any:
        return np.ascontiguousarray(np.asarray(value)).view(np.uint16)

    keys1_bits = bits(result_full["keys1"])
    keys0_bits = bits(boundary_keys_host)
    control_mismatch_rows = int(
        np.count_nonzero(np.any(keys0_bits != greenfield_layer0, axis=1))
    )
    control_mismatch_lanes = int(np.count_nonzero(keys0_bits != greenfield_layer0))
    legacy_lane_mismatch = keys1_bits != legacy_layer1
    greenfield_lane_mismatch = keys1_bits != greenfield_layer1
    per_row_legacy = legacy_lane_mismatch.sum(axis=1)
    per_row_greenfield = greenfield_lane_mismatch.sum(axis=1)
    all_hlo_text = "\n".join(
        (
            normalized_hlo,
            normalized_stablehlo,
            key_hlo,
            key_stablehlo,
            consumer_hlo,
            consumer_stablehlo,
            decode_hlo,
            decode_stablehlo,
            promote_hlo,
            promote_stablehlo,
        )
    )
    forbidden = [
        token
        for token in ("host_callback", 'CustomCall("xla_python', "python_callback")
        if token in all_hlo_text
    ]
    row0_legacy_mismatch_lanes = int(per_row_legacy[0])
    exact = (
        control_mismatch_rows == 0
        and control_mismatch_lanes == 0
        and row0_legacy_mismatch_lanes == 0
        and not forbidden
        and normalization_contract["passed"]
        and key_contract["passed"]
        and consumer_contract["passed"]
    )
    status = "SUCCESS" if exact else "DIAGNOSTIC"
    summary = {
        "artifact_kind": ARTIFACT_KIND,
        "code_hash": args.code_pin,
        "provenance": provenance,
        "import_closure_module_count": len(import_closure),
        "run_tag": args.run_tag,
        "status": status,
        "control_layer0_keys_vs_db518_mismatched_rows": control_mismatch_rows,
        "control_layer0_keys_vs_db518_mismatched_lanes": control_mismatch_lanes,
        "execution_boundary": {
            "normalizer_invocations": 1,
            "key_control_invocations": 1,
            "real_layer_consumer_invocations": 1,
            "normalization_to_consumer_host_transfers": 0,
            "final_host_transfers": 1,
        },
        "forbidden_hlo_tokens": forbidden,
        "hlo": {
            "normalization_contract": normalization_contract,
            "normalized_boundary_optimized_byte_count": len(
                normalized_hlo.encode("utf-8")
            ),
            "normalized_boundary_optimized_sha256": sha256(
                normalized_hlo.encode("utf-8")
            ).hexdigest(),
            "normalized_boundary_stablehlo_byte_count": len(
                normalized_stablehlo.encode("utf-8")
            ),
            "normalized_boundary_stablehlo_sha256": sha256(
                normalized_stablehlo.encode("utf-8")
            ).hexdigest(),
            "key_contract": key_contract,
            "normalized_key_control_optimized_byte_count": len(key_hlo.encode("utf-8")),
            "normalized_key_control_optimized_sha256": sha256(
                key_hlo.encode("utf-8")
            ).hexdigest(),
            "normalized_key_control_stablehlo_byte_count": len(
                key_stablehlo.encode("utf-8")
            ),
            "normalized_key_control_stablehlo_sha256": sha256(
                key_stablehlo.encode("utf-8")
            ).hexdigest(),
            "real_layer_consumer_contract": consumer_contract,
            "real_layer_consumer_optimized_byte_count": len(
                consumer_hlo.encode("utf-8")
            ),
            "real_layer_consumer_optimized_sha256": sha256(
                consumer_hlo.encode("utf-8")
            ).hexdigest(),
            "real_layer_consumer_stablehlo_byte_count": len(
                consumer_stablehlo.encode("utf-8")
            ),
            "real_layer_consumer_stablehlo_sha256": sha256(
                consumer_stablehlo.encode("utf-8")
            ).hexdigest(),
            "wk_decode_optimized_byte_count": len(decode_hlo.encode("utf-8")),
            "wk_decode_optimized_sha256": sha256(
                decode_hlo.encode("utf-8")
            ).hexdigest(),
            "wk_decode_stablehlo_byte_count": len(decode_stablehlo.encode("utf-8")),
            "wk_decode_stablehlo_sha256": sha256(
                decode_stablehlo.encode("utf-8")
            ).hexdigest(),
            "wk_promote_optimized_byte_count": len(promote_hlo.encode("utf-8")),
            "wk_promote_optimized_sha256": sha256(
                promote_hlo.encode("utf-8")
            ).hexdigest(),
            "wk_promote_stablehlo_byte_count": len(promote_stablehlo.encode("utf-8")),
            "wk_promote_stablehlo_sha256": sha256(
                promote_stablehlo.encode("utf-8")
            ).hexdigest(),
        },
        "inputs": {
            "layer0_input_manifest_sha256": input_manifest["manifest_sha256"],
            "legacy_layer1_manifest_sha256": legacy_manifest["manifest_sha256"],
            "legacy_layer1_bits_sha256": legacy_manifest[
                "prompt_index_key_bfloat16_sha256"
            ],
            "db518_result_sha256": args.db518_result_sha256,
            "checkpoint_index_sha256": args.checkpoint_index_sha256,
            "weight_digests_sha256": args.weight_digests_sha256,
            "checkpoint_shard_sha256": digest_record["shard"]["sha256"],
            "db518_layer0_chunk0_bits_sha256": sha256(
                np.ascontiguousarray(greenfield_layer0).tobytes()
            ).hexdigest(),
            "db518_layer1_chunk0_bits_sha256": sha256(
                np.ascontiguousarray(greenfield_layer1).tobytes()
            ).hexdigest(),
            "legacy_layer1_chunk0_bits_sha256": sha256(
                np.ascontiguousarray(legacy_layer1).tobytes()
            ).hexdigest(),
            "softmax_scale": args.softmax_scale,
        },
        "row0": {
            "legacy_geometry_vs_legacy_lanes": int(per_row_legacy[0]),
            "legacy_geometry_vs_greenfield_db518_lanes": int(per_row_greenfield[0]),
            "attention_row0_db533_vs_pairwise_lanes": int(
                np.count_nonzero(
                    bits(result_full["attention_row0"])
                    != bits(result_full["attention_pairwise_row0"])
                )
            ),
        },
        "chunk0_vs_legacy": {
            "rows_exact": int(np.count_nonzero(per_row_legacy == 0)),
            "rows": CHUNK_ROWS,
            "lanes_mismatched": int(legacy_lane_mismatch.sum()),
            "per_row_first_16": per_row_legacy[:16].tolist(),
            "per_64_row_block_mean": [
                float(per_row_legacy[i : i + 64].mean())
                for i in range(0, CHUNK_ROWS, 64)
            ],
        },
        "chunk0_vs_greenfield_db518": {
            "rows_exact": int(np.count_nonzero(per_row_greenfield == 0)),
            "lanes_mismatched": int(greenfield_lane_mismatch.sum()),
        },
        "elapsed_seconds": round(time.time() - started, 1),
        "claim_scope": "Completed device normalization consumed by a separate full layer-0 path; row 0 is the only decisive legacy comparison, rows >= 1 use a non-legacy softmax and are diagnostic only; no decoder, Gate-D, DB or performance claim.",
    }
    output_buffer = BytesIO()
    np.savez(
        output_buffer,
        keys1_bits=keys1_bits,
        keys0_bits=keys0_bits,
        greenfield_layer0_bits=greenfield_layer0,
        legacy_layer1_bits=legacy_layer1,
        greenfield_layer1_bits=greenfield_layer1,
        normalized1_chunk_bits=bits(result_full["normalized1"]),
        **{
            f"{name}_bits": bits(value)
            for name, value in result_full.items()
            if name.endswith("_row0")
        },
    )
    output_raw = output_buffer.getvalue()
    summary["arrays_sha256"] = sha256(output_raw).hexdigest()
    _write_run_member_exclusive(run_fd, "probe_arrays.npz", output_raw)
    _write_run_member_exclusive(
        run_fd,
        "runner.json",
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "status",
                    "row0",
                    "chunk0_vs_legacy",
                    "control_layer0_keys_vs_db518_mismatched_rows",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
