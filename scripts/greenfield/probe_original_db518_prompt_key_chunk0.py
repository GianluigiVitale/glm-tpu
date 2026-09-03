#!/usr/bin/env python3
"""Test one completed normalization-to-key boundary against original DB518.

This bounded discriminator keeps all 2,048 rows because shrinking the row set
can change TPU lowering.  It completes the BF16 normalization buffer on device,
synchronizes it, and passes that same buffer to a separately compiled M64 key
control.  The normalized buffer never transfers to the host and no layer
consumer or 8K decoder is compiled or invoked.
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

PROBE_REPOSITORY_PATH = (
    "scripts/greenfield/probe_original_db518_prompt_key_chunk0.py")
INSTALL_PATH = Path(
    "/usr/local/libexec/glm-tpu/gate-d-original-db518-prompt-key-v9/"
    "probe_original_db518_prompt_key_chunk0.py")
RUN_ROOT = Path("/home/gianl/gate-d-runs")
TAG_PATTERN = re.compile(
    r"greenfield_original_db518_prompt_key_chunk0_[0-9]{8}T[0-9]{15}Z")
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

CHUNK_ROWS = 2048
PROMPT_ROWS = 8155
UNIQUE_ROWS = 37
EXPECTED_ADAPTED_WK_SHA256 = (
    "d680f7b1c2fed426174c41ee9153d9db47f5db8ec0e803e20d8853ec1f483469")
EXPECTED_ADAPTED_WK_BYTE_SUM = 193_298_069
EXPECTED_CHUNK_BITS_SHA256 = (
    "96d261cbef56bccc20c3d2bee275b995f168e0ff7609713baae21d0a6ea3887c")


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
    return _git_bytes(repo, *arguments).decode("ascii",
                                               errors="strict").strip()


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
        if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != 0
                or metadata.st_gid != 0
                or stat.S_IMODE(metadata.st_mode) & 0o022):
            raise RuntimeError(
                f"probe installation parent is mutable: {parent}")


def _verify_running_source(repo: Path, code_pin: str,
                           expected_source_sha256: str) -> str:
    if _git_text(repo, "rev-parse", f"{code_pin}^{{commit}}") != code_pin:
        raise RuntimeError("code pin is not a commit in the repository")
    if _git_bytes(repo, "for-each-ref", "--format=%(refname)",
                  "refs/replace").strip():
        raise RuntimeError("repository has forbidden replacement refs")
    if Path(__file__) != INSTALL_PATH:
        raise RuntimeError("probe is not executing from the immutable capsule")
    _require_root_boundary(INSTALL_PATH)
    metadata = os.stat(INSTALL_PATH, follow_symlinks=False)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or metadata.st_uid != 0 or metadata.st_gid != 0
            or stat.S_IMODE(metadata.st_mode) != 0o555
            or os.listxattr(INSTALL_PATH, follow_symlinks=False)):
        raise RuntimeError("probe installation identity is unsafe")
    raw = _snapshot_regular(INSTALL_PATH)
    committed = _git_bytes(repo, "show", f"{code_pin}:{PROBE_REPOSITORY_PATH}")
    observed_sha256 = sha256(raw).hexdigest()
    if raw != committed or observed_sha256 != expected_source_sha256:
        raise RuntimeError(
            "probe is not the committed blob at the approved pin")
    return observed_sha256


def _open_inherited_run_dir(run_dir: Path, inherited_fd: int) -> int:
    if (not run_dir.is_absolute() or run_dir.parent != RUN_ROOT
            or TAG_PATTERN.fullmatch(run_dir.name) is None
            or inherited_fd != 7):
        raise RuntimeError("probe run-directory invocation drifted")
    root_fd = os.open(
        RUN_ROOT, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptor = -1
    try:
        descriptor = os.dup(inherited_fd)
        held = os.fstat(descriptor)
        named = os.stat(run_dir.name, dir_fd=root_fd, follow_symlinks=False)
        if (not stat.S_ISDIR(held.st_mode)
                or stat.S_IMODE(held.st_mode) != 0o700
                or held.st_uid != os.geteuid() or held.st_gid != os.getegid()
                or os.listxattr(descriptor)
                or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)):
            raise RuntimeError("probe run-directory authority drifted")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        os.close(root_fd)


def _write_run_member_exclusive(run_fd: int, relative: str,
                                raw: bytes) -> None:
    parts = Path(relative).parts
    if (not parts or relative.startswith("/")
            or any(part in {"", ".", ".."} for part in parts) or len(parts) > 2
            or (len(parts) == 2 and parts[0] != "hlo")):
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
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
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
                    raise RuntimeError(
                        f"probe output write stalled: {relative}")
                view = view[written:]
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            named = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            if ((before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
                    or not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
                    or named.st_size != len(raw)):
                raise RuntimeError(
                    f"probe output identity changed: {relative}")
            os.fsync(parent_fd)
        finally:
            os.close(descriptor)
    finally:
        if owned_parent:
            os.close(parent_fd)


def _sealed_git_source_archive(repo: Path,
                               code_pin: str) -> tuple[str, dict[str, Any]]:
    tree = _git_bytes(repo, "ls-tree", "-r", "-z", code_pin, "--", "glm_tpu")
    records = []
    buffer = BytesIO()
    with zipfile.ZipFile(buffer,
                         "w",
                         compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for raw_entry in tree.split(b"\0"):
            if not raw_entry:
                continue
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, kind, object_id = metadata.split(b" ", 2)
            path = raw_path.decode("utf-8", errors="strict")
            if (kind != b"blob" or mode not in {b"100644", b"100755"}
                    or not path.startswith("glm_tpu/")):
                raise RuntimeError(
                    f"unsupported committed source entry: {path}")
            payload = _git_bytes(repo, "cat-file", "blob",
                                 object_id.decode("ascii"))
            records.append({
                "git_object_id": object_id.decode("ascii"),
                "path": path,
                "sha256": sha256(payload).hexdigest(),
                "size": len(payload),
            })
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100555
                                  if mode == b"100755" else 0o100444) << 16
            archive.writestr(info, payload)
    if not records or records != sorted(records,
                                        key=lambda item: item["path"]):
        raise RuntimeError("committed source tree order drifted")
    raw_archive = buffer.getvalue()
    descriptor = os.memfd_create(
        "gate-d-original-db518-source",
        os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC,
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
    if (sys.flags.isolated != 1 or sys.flags.no_site != 1
            or not sys.flags.ignore_environment):
        raise RuntimeError("probe must run under python -I -S")
    sys.path[:] = [
        archive_path,
        JAX_SITE_ROOT,
        LIBTPU_SITE_ROOT,
        *EXPECTED_RUNTIME_PATH,
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--code-pin", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--accepted-cache-dir", type=Path, required=True)
    parser.add_argument("--accepted-cache-manifest-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-dir-fd", type=int, choices=(7, ), required=True)
    return parser.parse_args()


def _array_sha256(value: Any) -> str:
    import numpy as np

    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _memory_stats(device: Any) -> dict[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def main() -> int:
    args = _parse_args()
    if TAG_PATTERN.fullmatch(args.run_tag) is None:
        raise RuntimeError("invalid original-DB518 discriminator tag")
    if args.run_dir != RUN_ROOT / args.run_tag:
        raise RuntimeError("run directory and tag disagree")
    run_fd = _open_inherited_run_dir(args.run_dir, args.run_dir_fd)
    try:
        source_sha256 = _verify_running_source(args.repository, args.code_pin,
                                               args.expected_source_sha256)
        archive_path, archive_provenance = _sealed_git_source_archive(
            args.repository, args.code_pin)
        _install_sealed_path(archive_path)

        import jax
        import jax.numpy as jnp
        import ml_dtypes
        import numpy as np

        from glm_tpu.greenfield.benchmarking import numpy_safetensors as ns
        from glm_tpu.greenfield.benchmarking import sealed_runtime
        from glm_tpu.greenfield.kernels.reference.dsa import (
            DsaNumericalContract, )
        from glm_tpu.greenfield.kernels.reference.dsa_association import (
            Layer0DsaProbeGeometry,
            layer0_prompt_index_key_from_normalized_boundary_chunk,
            layer0_prompt_normalized_hidden_boundary_chunk,
        )
        from glm_tpu.greenfield.kernels.reference.prefill_index import (
            decode_stage_local_prefill_index_wk_bf16,
            promote_stage_local_prefill_index_wk,
        )
        from glm_tpu.greenfield.validation.original_db518_prompt_key import (
            ORIGINAL_DB518_CACHE_SHA256,
            ORIGINAL_DB518_CODE_HASH,
            ORIGINAL_DB518_COMPARISON_MANIFEST_SHA256,
            ORIGINAL_DB518_TAG,
            require_completed_normalization_boundary_hlo,
            require_normalized_key_control_boundary_hlo,
        )
        if jax.default_backend() != "tpu":
            raise RuntimeError(
                f"original-DB518 discriminator requires TPU, got "
                f"{jax.default_backend()}")
        if jax.local_device_count() != 4 or jax.device_count() != 4:
            raise RuntimeError(
                "discriminator requires one isolated TPU-v4 host")
        device = jax.local_devices()[0]
        if device.device_kind != "TPU v4" or device.process_index != 0:
            raise RuntimeError("unexpected original-DB518 physical device")

        started = time.monotonic()
        input_manifest, arrays = ns.load_layer0_dsa_association_input(
            args.input_dir,
            expected_manifest_sha256=args.input_manifest_sha256,
        )
        cache_manifest, accepted_bits = ns.load_legacy_prompt_index_cache(
            args.accepted_cache_dir,
            expected_manifest_sha256=args.accepted_cache_manifest_sha256,
        )
        if (input_manifest["manifest_sha256"] != args.input_manifest_sha256
                or cache_manifest.get("layer_id", 0) != 0
                or cache_manifest["layer0_input_manifest_sha256"]
                != input_manifest["manifest_sha256"]
                or cache_manifest["prompt_index_key_bfloat16_sha256"]
                != ORIGINAL_DB518_CACHE_SHA256):
            raise RuntimeError("original DB518 input/cache authority drifted")
        if accepted_bits.shape != (PROMPT_ROWS, 128):
            raise RuntimeError(
                "original DB518 accepted-cache geometry drifted")
        accepted_chunk = np.ascontiguousarray(accepted_bits[:CHUNK_ROWS])
        if _array_sha256(accepted_chunk) != EXPECTED_CHUNK_BITS_SHA256:
            raise RuntimeError("original DB518 chunk-0 cache slice drifted")

        prompt_ids = np.asarray(arrays["prompt_token_ids"])
        unique_ids = np.asarray(arrays["unique_token_ids"])
        if prompt_ids.shape != (PROMPT_ROWS, ) or unique_ids.shape != (
                UNIQUE_ROWS, ):
            raise RuntimeError("original DB518 token geometry drifted")
        embedding_rows_host = np.searchsorted(
            unique_ids, prompt_ids[:CHUNK_ROWS]).astype(np.int32)
        if not np.array_equal(unique_ids[embedding_rows_host],
                              prompt_ids[:CHUNK_ROWS]):
            raise RuntimeError("original DB518 row mapping drifted")
        positions_host = np.arange(CHUNK_ROWS, dtype=np.int32)
        def bf16(bits: Any) -> Any:
            value = (np.ascontiguousarray(bits).astype(np.uint16).view(
                ml_dtypes.bfloat16))
            return jax.device_put(jnp.asarray(value), device)

        def put(value: Any) -> Any:
            return jax.device_put(jnp.asarray(value), device)

        contract = DsaNumericalContract()
        geometry = Layer0DsaProbeGeometry()
        unique_embeddings = bf16(arrays["unique_embedding_bfloat16_bits"])
        embedding_rows = put(embedding_rows_host)
        positions = put(positions_host)
        input_norm_weight = bf16(arrays["input_layernorm__weight"])
        raw_wk_bits = put(arrays["self_attn__indexer__wk__weight"])
        raw_wk_scale = put(arrays["self_attn__indexer__wk__weight_scale_inv"])
        key_norm_weight = bf16(arrays["self_attn__indexer__k_norm__weight"])
        key_norm_bias = bf16(arrays["self_attn__indexer__k_norm__bias"])

        decode_lowered = jax.jit(
            lambda bits, scale: decode_stage_local_prefill_index_wk_bf16(
                bits, scale, contract=contract)).lower(raw_wk_bits,
                                                       raw_wk_scale)
        decode_compiled = decode_lowered.compile()
        decode_hlo = decode_compiled.as_text()
        decode_stablehlo = decode_lowered.as_text()
        _write_run_member_exclusive(run_fd, "hlo/wk_decode.optimized_hlo.txt",
                                    decode_hlo.encode())
        _write_run_member_exclusive(run_fd, "hlo/wk_decode.stablehlo.mlir",
                                    decode_stablehlo.encode())
        wk_bf16 = decode_compiled(raw_wk_bits, raw_wk_scale)
        wk_bf16.block_until_ready()

        promote_lowered = jax.jit(
            lambda value: promote_stage_local_prefill_index_wk(
                value, contract=contract)).lower(wk_bf16)
        promote_compiled = promote_lowered.compile()
        promote_hlo = promote_compiled.as_text()
        promote_stablehlo = promote_lowered.as_text()
        _write_run_member_exclusive(run_fd, "hlo/wk_promote.optimized_hlo.txt",
                                    promote_hlo.encode())
        _write_run_member_exclusive(
            run_fd,
            "hlo/wk_promote.stablehlo.mlir",
            promote_stablehlo.encode(),
        )
        wk_fp32 = promote_compiled(wk_bf16)
        wk_fp32.block_until_ready()

        normalizer = partial(
            layer0_prompt_normalized_hidden_boundary_chunk,
            geometry=geometry,
        )
        normalization_arguments = (
            unique_embeddings,
            embedding_rows,
            input_norm_weight,
        )
        before_memory = _memory_stats(device)
        compile_started = time.monotonic()
        normalized_lowered = jax.jit(normalizer).lower(
            *normalization_arguments
        )
        normalized_compiled = normalized_lowered.compile()
        normalized_hlo = normalized_compiled.as_text()
        normalized_stablehlo = normalized_lowered.as_text()
        _write_run_member_exclusive(
            run_fd,
            "hlo/normalized_boundary.optimized_hlo.txt",
            normalized_hlo.encode(),
        )
        _write_run_member_exclusive(
            run_fd,
            "hlo/normalized_boundary.stablehlo.mlir",
            normalized_stablehlo.encode(),
        )
        key_control = partial(
            layer0_prompt_index_key_from_normalized_boundary_chunk,
            geometry=geometry,
        )
        key_lowered = jax.jit(key_control).lower(
            jax.ShapeDtypeStruct(
                (CHUNK_ROWS, geometry.hidden_size), jnp.bfloat16
            ),
            positions,
            wk_fp32,
            key_norm_weight,
            key_norm_bias,
        )
        key_compiled = key_lowered.compile()
        key_hlo = key_compiled.as_text()
        key_stablehlo = key_lowered.as_text()
        _write_run_member_exclusive(
            run_fd,
            "hlo/normalized_key_control.optimized_hlo.txt",
            key_hlo.encode(),
        )
        _write_run_member_exclusive(
            run_fd,
            "hlo/normalized_key_control.stablehlo.mlir",
            key_stablehlo.encode(),
        )
        # Persist both compiler products before either admission check so one
        # bounded failure diagnoses both backend forms without another probe.
        normalization_contract = require_completed_normalization_boundary_hlo(
            normalized_hlo, normalized_stablehlo
        )
        key_contract = require_normalized_key_control_boundary_hlo(
            key_hlo, key_stablehlo
        )
        compile_seconds = time.monotonic() - compile_started

        execute_started = time.monotonic()
        normalized_device = normalized_compiled(*normalization_arguments)
        normalized_device.block_until_ready()
        boundary_keys_device = key_compiled(
            normalized_device,
            positions,
            wk_fp32,
            key_norm_weight,
            key_norm_bias,
        )
        boundary_keys_device.block_until_ready()
        import_closure = sealed_runtime.verify_import_closure(
            Path(archive_path))
        # This is the sole host transfer.  The completed normalization buffer is
        # deliberately absent and cannot feed a host-staged consumer.
        boundary_keys_host, wk_host = jax.device_get(
            (boundary_keys_device, wk_fp32))
        execute_seconds = time.monotonic() - execute_started
        wk_host = np.ascontiguousarray(wk_host, dtype=np.float32)
        wk_sha256 = _array_sha256(wk_host)
        wk_byte_sum = int(wk_host.view(np.uint8).sum(dtype=np.uint64))
        if (wk_host.shape != (128, 6144)
                or wk_sha256 != EXPECTED_ADAPTED_WK_SHA256
                or wk_byte_sum != EXPECTED_ADAPTED_WK_BYTE_SUM):
            raise RuntimeError("accepted adapted-FP32 wk identity drifted")

        boundary_chunk = np.ascontiguousarray(boundary_keys_host).view(
            np.uint16
        )
        mismatch = boundary_chunk != accepted_chunk
        mismatched_rows = int(np.count_nonzero(np.any(mismatch, axis=1)))
        mismatched_lanes = int(np.count_nonzero(mismatch))
        boundary_sha256 = _array_sha256(boundary_chunk)
        exact = (mismatched_rows == 0 and mismatched_lanes == 0
                 and boundary_sha256 == EXPECTED_CHUNK_BITS_SHA256)
        output = BytesIO()
        np.savez(
            output,
            accepted_chunk_bits=accepted_chunk,
            boundary_key_bits=boundary_chunk,
            embedding_rows=embedding_rows_host,
            positions=positions_host,
        )
        output_raw = output.getvalue()
        after_memory = _memory_stats(device)
        if tuple(sys.path) != (
                archive_path,
                JAX_SITE_ROOT,
                LIBTPU_SITE_ROOT,
                *EXPECTED_RUNTIME_PATH,
        ):
            raise RuntimeError("sealed import path drifted")
        provenance = {
            "probe_sha256": source_sha256,
            "source_archive": archive_provenance,
            "python": sealed_runtime.validate_python_runtime(),
            "sites": sealed_runtime.validate_dependency_sites(),
            "sys_path": list(sys.path),
        }
        summary = {
            "artifact_kind":
            "gate_d_original_db518_normalized_boundary_chunk0_discriminator",
            "accepted_chunk_bits_sha256":
            EXPECTED_CHUNK_BITS_SHA256,
            "adapted_wk": {
                "byte_sum": wk_byte_sum,
                "dtype": str(wk_host.dtype),
                "sha256": wk_sha256,
                "shape": list(wk_host.shape),
            },
            "backend":
            jax.default_backend(),
            "boundary_key_bits_sha256":
            boundary_sha256,
            "claim_scope":
            ("One device-resident normalization-to-key composability control; "
             "no independent normalized-tensor, layer consumer, decoder, "
             "Gate-D or performance claim."),
            "code_hash":
            args.code_pin,
            "compile_seconds":
            compile_seconds,
            "device": {
                "device_count": jax.device_count(),
                "device_kind": device.device_kind,
                "device_id": device.id,
                "local_device_count": jax.local_device_count(),
                "process_index": device.process_index,
            },
            "execute_seconds":
            execute_seconds,
            "host_transfer_count_after_completion":
            1,
            "hlo": {
                "normalization_contract":
                normalization_contract,
                "normalization_optimized_sha256":
                sha256(normalized_hlo.encode()).hexdigest(),
                "normalization_stablehlo_sha256":
                sha256(normalized_stablehlo.encode()).hexdigest(),
                "key_control_contract":
                key_contract,
                "key_control_optimized_sha256":
                sha256(key_hlo.encode()).hexdigest(),
                "key_control_stablehlo_sha256":
                sha256(key_stablehlo.encode()).hexdigest(),
                "wk_decode_optimized_sha256":
                sha256(decode_hlo.encode()).hexdigest(),
                "wk_decode_stablehlo_sha256":
                sha256(decode_stablehlo.encode()).hexdigest(),
                "wk_promote_optimized_sha256":
                sha256(promote_hlo.encode()).hexdigest(),
                "wk_promote_stablehlo_sha256":
                sha256(promote_stablehlo.encode()).hexdigest(),
            },
            "import_closure_module_count":
            len(import_closure),
            "input_authority": {
                "accepted_cache_manifest_sha256":
                (args.accepted_cache_manifest_sha256),
                "input_manifest_sha256":
                args.input_manifest_sha256,
                "original_db518_cache_sha256":
                ORIGINAL_DB518_CACHE_SHA256,
                "original_db518_code_hash":
                ORIGINAL_DB518_CODE_HASH,
                "original_db518_comparison_manifest_sha256":
                (ORIGINAL_DB518_COMPARISON_MANIFEST_SHA256),
                "original_db518_tag":
                ORIGINAL_DB518_TAG,
                "row_mapping_sha256":
                _array_sha256(embedding_rows_host),
                "positions_sha256":
                _array_sha256(positions_host),
            },
            "memory_after":
            after_memory,
            "memory_before":
            before_memory,
            "mismatched_lanes":
            mismatched_lanes,
            "mismatched_rows":
            mismatched_rows,
            "normalized_boundary_key_control_exact":
            exact,
            "normalization_device_to_host_transfer_count":
            0,
            "performance_claim":
            False,
            "normalization_invocation_count":
            1,
            "key_control_invocation_count":
            1,
            "provenance":
            provenance,
            "run_tag":
            args.run_tag,
            "source_sha256":
            source_sha256,
            "status":
            "COMPLETED",
        }
        summary_raw = (json.dumps(summary, indent=2, sort_keys=True) +
                       "\n").encode("ascii")
        _write_run_member_exclusive(run_fd, "boundary_arrays.npz", output_raw)
        _write_run_member_exclusive(run_fd, "runner.json", summary_raw)
        print(
            json.dumps(
                {
                    "boundary_key_bits_sha256": boundary_sha256,
                    "mismatched_lanes": mismatched_lanes,
                    "mismatched_rows": mismatched_rows,
                    "status": summary["status"],
                },
                sort_keys=True,
            ))
        # A nonexact key control is a completed discriminator, not
        # infrastructure failure.  Downstream consumer execution remains out
        # of scope regardless of this result.
        return 0
    finally:
        os.close(run_fd)


if __name__ == "__main__":
    raise SystemExit(main())
