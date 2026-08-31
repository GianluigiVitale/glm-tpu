"""Append-only precompile admission v2 for Gate-D mechanisms.

Version 1 remains the authority for candidates that already have an executable
identity.  This separate path keeps the parent auditor stdlib-only and delegates
parsing to a sealed jaxlib/MLIR child.  It represents the earlier, precompile
state honestly: exact source AST, plan authority, causal StableHLO, and one
candidate-coherent typed capsule.  Admission never authorizes compilation or
TPU execution.
"""

from __future__ import annotations

import ast
import base64
from contextlib import contextmanager
import fcntl
from hashlib import sha256
import io
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import struct
import subprocess
from typing import Any, BinaryIO, Iterator, Mapping, Sequence
import zipfile

from .errors import BenchmarkValidationError


__all__ = (
    "GATE_D_PRECOMPILE_ADMISSION_SCHEMA_VERSION",
    "admit_gate_d_precompile_candidates",
    "write_gate_d_precompile_admission_report",
)


GATE_D_PRECOMPILE_ADMISSION_SCHEMA_VERSION = 2
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CODE_PIN = re.compile(r"^[0-9a-f]{40}$")
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_STABLEHLO_BYTES = 32 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_ARRAY_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_PARSER_TOTAL_BYTES = 512 * 1024 * 1024
_F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
_PARSER_MEMFD_SEALS = 1 | 2 | 4 | 8
_FORBIDDEN_RUNTIME_MODE = stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX
_VERIFIED_RUNTIME_TREES: set[tuple[str, str]] = set()
_EXPECTED_JAXLIB_VERSION = "0.10.1"
_EXPECTED_STABLEHLO_VERSION = "1.17.0"
_EXPECTED_LOWERING_CLAIM_SCOPE = (
    "Forced-CPU abstract compiler lowering only; no executable compilation, JAX array/"
    "numerical execution, model, cloud workflow, TPU backend initialization, performance "
    "or Gate-D closure claim. JAX import may perform read-only host TPU PCI discovery."
)
_EXPECTED_LOWERING_ENVIRONMENT = {
    "jax_version": "0.10.1",
    "jaxlib_version": "0.10.1",
    "python_executable": (
        "/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12"
    ),
    "python_runtime_root": "/opt/glm-tpu/gate-d-python-3.12.13-021044895e95",
    "python_runtime_tree_sha256": (
        "308748a9a3c3758a6b4f233aa5c034e8cb419362dbeafe0322448be40170d616"
    ),
    "python_sha256": (
        "021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7"
    ),
    "site_root": "/opt/glm-tpu/gate-d-jax-site-55233c63939e",
    "site_tree_sha256": (
        "55233c63939ea28485cdf2f0fc3d9c1d2ce4d9d93aad828e94498d712a26a0df"
    ),
}
_EXPECTED_LOWERING_INPUTS = {
    "activation_dtype": "bf16",
    "activation_shape": [1, 6144],
    "epsilon": 1e-5,
    "weight_dtype": "bf16",
    "weight_shape": [6144],
}
_EXPECTED_LOWERING_PRODUCER_INSTALLED_PATH = (
    "/opt/glm-tpu/bin/produce_gate_d_tuple_auxiliary_stablehlo.py"
)
_EXPECTED_LOWERING_PRODUCER_SOURCE_PATH = (
    "scripts/greenfield/produce_gate_d_tuple_auxiliary_stablehlo.py"
)
_EXPECTED_LOWERING_DEPENDENCY_MANIFESTS = {
    "native_mappings": {
        "count": 46,
        "sha256": "bf8f246cdec213e86988c6b224a920ad7244f3e232092f97e912c5ab0ec086a6",
    },
    "python_modules": {
        "count": 533,
        "sha256": "a37beeaec7baa7c78acc15efa75395bd3b6071d4519ed7bd22662653b5eb66d8",
    },
}
_EXPECTED_CAPSULE_PRODUCER_CLAIM_SCOPE = (
    "Bounded forced-two-CPU candidate-coherent layer-1/event-1 replay only; "
    "sealed real inputs are explicit, no full decoder or model load runs, and no "
    "TPU, cloud, performance or Gate-D claim is made. Local SUCCESS-last "
    "publication is provisional until an independent protected archive/seal."
)
_EXPECTED_CAPSULE_EXECUTION = {
    "cloud_workflow": False,
    "contract_valid": True,
    "decoder_executed": False,
    "jax_plugins_loaded": False,
    "libtpu_loaded": False,
    "model_loaded": False,
    "scope": "bounded.layer1.event1.candidate.replay",
    "tpus_used": 0,
}
_EXPECTED_CAPSULE_ENVIRONMENT = {
    "JAX_PLATFORMS": "cpu",
    "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
    "jax_version": "0.10.1",
    "jaxlib_version": "0.10.1",
    "ml_dtypes_version": "0.5.4",
    "numpy_version": "2.3.5",
    "python_executable": (
        "/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12"
    ),
    "python_runtime_root": "/opt/glm-tpu/gate-d-python-3.12.13-021044895e95",
    "python_runtime_tree_sha256": (
        "308748a9a3c3758a6b4f233aa5c034e8cb419362dbeafe0322448be40170d616"
    ),
    "python_sha256": (
        "021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7"
    ),
    "site_root": "/opt/glm-tpu/gate-d-jax-site-55233c63939e",
    "site_tree_sha256": (
        "55233c63939ea28485cdf2f0fc3d9c1d2ce4d9d93aad828e94498d712a26a0df"
    ),
}
_EXPECTED_CAPSULE_INSTALLED_PRODUCER = (
    "/opt/glm-tpu/bin/produce_gate_d_tuple_auxiliary_capsule.py"
)
_EXPECTED_CAPSULE_PRODUCER_SOURCE = {
    "git_object_id": "d82cfff69dc5a167a894716182fb0e95c807e5b1",
    "repo_path": "scripts/greenfield/produce_gate_d_tuple_auxiliary_capsule.py",
    "sha256": "a81fd59ea2b32f27a102666e24a35827b251560d4df52dc04276325aafa1794a",
}
_EXPECTED_CAPSULE_REPLAY_SOURCE = {
    "git_object_id": "7476af8c6ace4472d9c39faf1e7155b328d9e230",
    "repo_path": "glm_tpu/greenfield/benchmarking/gate_d_tuple_capsule.py",
    "sha256": "9b9d11d0f312a173a8014a81f9b8fd957b9eefe2148bf327da0d57d82fed03b4",
}
# The admission module is the verifier, not part of the sealed replay import
# closure.  Excluding it avoids a self-referential digest while binding every
# other committed glm_tpu blob, including package initializers and all
# transitive kernel imports available to the replay.
_CAPSULE_EXECUTION_MANIFEST_EXCLUDED_PATHS = frozenset(
    {"glm_tpu/greenfield/gate_d_precompile_admission.py"}
)
_EXPECTED_CAPSULE_EXECUTION_SOURCE_MANIFEST = {
    "count": 139,
    "sha256": "f3683029a4c3eaa12e0673c0e1d84ff13ff2c42fb7a475323e8d099ac156c0b0",
}
_EXPECTED_CAPSULE_TENSOR_NAMES = (
    "attention.slot_01.input_norm",
    "attention.slot_01.qkv_a.weight_bits",
    "attention.slot_01.qkv_a.scale_inv",
    "attention.slot_01.q_a_norm",
    "indexer.slot_01.wq_b.weight_bits",
    "indexer.slot_01.wq_b.scale_inv",
    "indexer.slot_01.wk.weight_bits",
    "indexer.slot_01.wk.scale_inv",
    "indexer.slot_01.key_norm_weight",
    "indexer.slot_01.key_norm_bias",
    "indexer.slot_01.head_weight",
)
_CAPSULE_RUNTIME_INPUT_SOURCES = {
    "head_weight_bf16_bits": ("indexer.slot_01.head_weight", (0, 1)),
    "key_norm_bias_bf16_bits": ("indexer.slot_01.key_norm_bias", (0, 1)),
    "key_norm_weight_bf16_bits": ("indexer.slot_01.key_norm_weight", (0, 1)),
    "q_a_norm_bf16_bits": ("attention.slot_01.q_a_norm", (0,)),
    "qkv_a_scale_inv": ("attention.slot_01.qkv_a.scale_inv", (0,)),
    "qkv_a_weight_bits": ("attention.slot_01.qkv_a.weight_bits", (0,)),
    "rms_weight_bf16_bits": ("attention.slot_01.input_norm", (0,)),
    "wk_scale_inv": ("indexer.slot_01.wk.scale_inv", (0, 1)),
    "wk_weight_bits": ("indexer.slot_01.wk.weight_bits", (0, 1)),
    "wq_b_scale_inv": ("indexer.slot_01.wq_b.scale_inv", (0, 1)),
    "wq_b_weight_bits": ("indexer.slot_01.wq_b.weight_bits", (0, 1)),
}
_EXPECTED_CAPSULE_ACCEPTED_OUTPUTS = {
    "event1_positions_sha256": (
        "e55e66c6dcb35de94b9dce54d8fff602704cf26ae92d4afb333bd2501ab88ad7"
    ),
    "event1_scores_sha256": (
        "a61587a9d18bd169c1697ecb0060b39e8f1aad1caf532bca835b15a426c0b0e7"
    ),
    "event1_valid_count": 2048,
    "rms_hidden_update_sha256": (
        "efde853254c03dd18a5f5f22733630ce0e785dfbb4eba09c41eea9085e47b4fc"
    ),
    "rms_residual_sha256": (
        "a105fdbd429adb1d06a70bf71598a72a91d7b6faa83360005487ce11ce099f8e"
    ),
}
_EXPECTED_CAPSULE_UPSTREAM_INPUTS = {
    "db518_comparison": (
        "/home/gianl/glm-run/greenfield_pp16_feature2_layer0_db518_numerical_"
        "20260829T115022665987633Z/comparison.json",
        "06ee82b9d487e3fdf8f9f19d4e824e33f1f9d453738a0090ace5e2cac7272a4d",
    ),
    "db518_result": (
        "/home/gianl/glm-run/greenfield_pp16_feature2_layer0_db518_numerical_"
        "20260829T115022665987633Z/result.npz",
        "534bacc54d74992f5a8ab4d422f9fa0947523d59325b4bfa272d4fbeb56262f0",
    ),
    "db550_boundary": (
        "/home/gianl/gcs-models/results/greenfield_layer0_dense_partial_capture_"
        "20260813T200736889447458Z/dense_partial_capture.npz",
        "f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298",
    ),
    "plan_authority": (
        "/home/gianl/glm-tpu-topology-rewrite/docs/artifacts/"
        "gate-d-tuple-auxiliary-pp16-plan-authority.json",
        "98b4fa21272e0bb019af1ad99abedee35593f9a1c8067a15983e9023c01f7880",
    ),
    "runtime_manifest": (
        "/home/gianl/glm-run/greenfield_runtime_feature_qkv_direct_pp16_"
        "20260827T164842844148623Z/final/runtime_manifest.json",
        "e13ccefb7341756cd68d85e51209eaa8516ea506eac7ace3b4fd0a1d56828032",
    ),
    "runtime_success": (
        "/home/gianl/glm-run/greenfield_runtime_feature_qkv_direct_pp16_"
        "20260827T164842844148623Z/final/SUCCESS",
        "dbef7e366e2fdf2a4815b0b58d1645667580d55fc5926133d48c838929f7ee7e",
    ),
    "source_authority": (
        "/home/gianl/glm-tpu-topology-rewrite/docs/artifacts/"
        "gate-d-tuple-auxiliary-source-authority.json",
        "c95c8aa188e2eda77270022128d121dc3693a899607b1ab2f9977ddb45def0e1",
    ),
    "stablehlo_authority": (
        "/home/gianl/glm-tpu-topology-rewrite/docs/artifacts/"
        "gate-d-tuple-auxiliary-stablehlo-authority.json",
        "8cc45b81c3a85ad6131790e82eb73f0839cab7eefd13a634db7ff5a697b0338b",
    ),
}
_EXPECTED_CAPSULE_RUNTIME_DATA_ROOT = Path(
    "/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP16_LP2/"
    "greenfield_runtime_feature_qkv_direct_pp16_20260827T164842844148623Z"
)
_EXPECTED_CAPSULE_RUNTIME_MOUNT_POINT = "/home/gianl/gcs-models"
_EXPECTED_CAPSULE_RUNTIME_MOUNT_SOURCE = "driftbench-dsv4-uc"
_EXPECTED_STABLEHLO_CERTIFICATE_CLAIM_SCOPE = (
    "Forced-CPU abstract-lowering structural StableHLO authority only; no executable "
    "compilation, numerical execution, model, TPU, performance or Gate-D closure claim."
)
_EXPECTED_ACCEPTED_PRIMARY_SLICE_SHA256 = (
    "5037b5a7ef21226f8405d0175c83bc7528fadf588811755e719d20a75f295610"
)
_EXPECTED_TOPOLOGY_HASH = (
    "294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559"
)
_EXPECTED_PP16_LP2_HASH = (
    "6383e57c81478ac0d6de4525a4675f2a0d7cbc7aa73bd67bc662dc4e05840f21"
)
_EXPECTED_PP8_LP4_HASH = (
    "d5943ab8d7a074677d82f8e823c8bc983847f8df1deefdee1fbda8da98923c14"
)
_EXPECTED_TOPOLOGY_SOURCE_CONTRACT_HASH = (
    "74e5eafaf264b2d12663f653df469cf3feb8ead1f31be6217797c96e99864ad6"
)
_EXPECTED_ACCEPTED_SOURCE = {
    "ast_sha256": "490175a1e0732b4967b3e53a8b88df081c20855bae009425d6ba81c48422ed59",
    "file_sha256": "d8e4380ca97d2c719836a7e15fb410a73d354b15d79fc54275b573bbb1c06910",
    "git_commit": "a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c",
    "path": "vllm/ir/ops/layernorm.py",
    "symbol": "fused_add_rms_norm",
}
_EXPECTED_CONCRETE_TUPLE_SOURCE = {
    "candidate_ast_sha256": "c94e64312aef9fa24bb5466082f787c579874e6f60ea512a72802ea77e3b4b14",
    "callsite_ast_sha256": "27caa9d6276301a8caacaabd1d5564f5ee7c6b595793a54d915ee0dcfcd2045d",
    "certificate_sha256": "c95c8aa188e2eda77270022128d121dc3693a899607b1ab2f9977ddb45def0e1",
    "code_pin": "c8b220067577004ddfb824667eadc746642baaca",
    "semantic_sha256": "49c2854b8c2e5a1dee760e22a647e45ba1a345fd031b6a1c14f1e732aa048ef3",
    "source_set_sha256": "f1055660a97389ecd4700eff40d679657c8f274153cde5fe19e567b6435323bc",
}
_EXPECTED_VALIDATOR_IMPORTS = {
    "jaxlib/__init__.py": (629, "2a5b37b2bc9802769f45ca43de7cdc1a8b532f0afd2bf9542131ab68f649c4a3"),
    "jaxlib/libjax_common.so": (
        348039992,
        "b836d25d48b50c3e32f8344f19a3cbff25b3e5840d1500d85024ef3dfaf55fbd",
    ),
    "jaxlib/mlir/_mlir_libs/__init__.py": (
        8512,
        "1e053fc5d003af7296a29f55ae8079f10a81c36b35819e03b283fe5dbb8e6dea",
    ),
    "jaxlib/mlir/_mlir_libs/_jax_mlir_ext.so": (
        3632,
        "7a92d593271ab74d921de638bc2ee50dd90d9f191fc42cfe927e5d34321ff9c8",
    ),
    "jaxlib/mlir/_mlir_libs/_chlo.so": (
        3592,
        "9e2e2a1f402c556061570d47cee5d6406b0d1968a35555a7e8a79ec942c2c4f6",
    ),
    "jaxlib/mlir/_mlir_libs/_mlir.so": (
        3592,
        "ee42e2f7d805b9cbed32cf002320092cfa75a95f901694274ce1f41a76c5b929",
    ),
    "jaxlib/mlir/_mlir_libs/_stablehlo.so": (
        3624,
        "e21b420f7a3d58f8594ac4ab23f59ce3bbd9442aea7b775640e33064fe7a0ffd",
    ),
    "jaxlib/mlir/dialects/_ods_common.py": (
        10887,
        "52f38ee6701961f4840d52958feb128f3fcc31a0c41ef586bd70a6e63d95b7a7",
    ),
    "jaxlib/mlir/dialects/_chlo_ops_gen.py": (
        127559,
        "7a728d3b8249532dbdcf8418e3c1dfea21440aa99ab4c79de6c631f6dff4ba43",
    ),
    "jaxlib/mlir/dialects/_stablehlo_ops_gen.py": (
        412837,
        "4a4a1ac5b161a5d65862838fc2bc9d63b5d7db554e82a3ae377313bc95b007dd",
    ),
    "jaxlib/mlir/dialects/stablehlo.py": (
        1133,
        "912840177a1c2945155e0f0841b4fde386315a1bca83d0f63cfc0439a10a21e7",
    ),
    "jaxlib/mlir/dialects/chlo.py": (
        1025,
        "a85a5da8895f401f01f9d0816873cb28fb7d848832d07a19704d4b9be4595071",
    ),
    "jaxlib/mlir/ir.py": (
        12758,
        "d4fb18407e74c6d496eeeaeda99a9be354f5ffbd9edad41b8498525c4ebfdaf9",
    ),
    "jaxlib/version.py": (6848, "6cecd708aacaa994c28a11cfcdcc56e9e67e64e4d1dcfc3ac7b85b87585c1bc0"),
}
_EXPECTED_SOURCE_SEMANTIC_SHA256 = {
    "auxiliary_device_tuple_dependency": (
        "fe303c7aeb51032f23a90ad61fcf3cb72030a376bc6f9bf20f7be794ce3b7534"
    ),
    "compensated_auxiliary_dependency": (
        "46654fcde1c8e9e6e19ef3bd530db3d6e085c0e5951cbb8c34997a9fafec6dc0"
    ),
}
_EXPECTED_AUXILIARY_SLICE_SHA256 = {
    "auxiliary_device_tuple_dependency": (
        "02eee1d7a84fd447396577c21c1d2988f23d9b6312b8fa0df4838a9b41642f7c"
    ),
    "compensated_auxiliary_dependency": (
        "d55e5f45120080dd0a6f81d3f3dddb9c8a738b59e2454ec1f9d87b9bce5d9a43"
    ),
}
_DTYPE_BYTES = {
    "|b1": 1,
    "|u1": 1,
    "<u2": 2,
    "<u4": 4,
    "<i4": 4,
    "<f4": 4,
}
_CAPSULE_INPUT_SCHEMA: dict[str, tuple[str, tuple[int, ...]]] = {
    "head_weight_bf16_bits": ("<u2", (2, 16, 6144)),
    "key_norm_bias_bf16_bits": ("<u2", (2, 128)),
    "key_norm_weight_bf16_bits": ("<u2", (2, 128)),
    "prompt_cache_bf16_bits": ("<u2", (2, 16, 256, 128)),
    "q_a_norm_bf16_bits": ("<u2", (2048,)),
    "qkv_a_scale_inv": ("<f4", (32, 48, 82)),
    "qkv_a_weight_bits": ("|u1", (32, 6144, 82)),
    "rms_hidden_update_bf16_bits": ("<u2", (6144,)),
    "rms_residual_bf16_bits": ("<u2", (6144,)),
    "rms_weight_bf16_bits": ("<u2", (6144,)),
    "wk_scale_inv": ("<f4", (2, 1, 48)),
    "wk_weight_bits": ("|u1", (2, 128, 6144)),
    "wq_b_scale_inv": ("<f4", (2, 16, 16)),
    "wq_b_weight_bits": ("|u1", (2, 2048, 2048)),
}
_CAPSULE_DEVICE_EVIDENCE_SCHEMA: dict[
    str, tuple[str, tuple[int, ...], str, tuple[int, ...]]
] = {
    "contract_valid_owners": ("|u1", (2,), "", ()),
    "current_key_owners": ("<f4", (2, 1, 128), "current_key", (0, 0)),
    "rms_input_fp32_owners": ("<f4", (2, 1, 6144), "rms_input", (0, 0)),
    "selected_positions_owners": (
        "<i4",
        (2, 1, 2048),
        "event1_positions",
        (0,),
    ),
    "selected_scores_owners": (
        "<f4",
        (2, 1, 2048),
        "event1_scores",
        (0,),
    ),
    "valid_counts_owners": ("<i4", (2, 1), "event1_valid_count", (0,)),
}
_SEMANTIC_DTYPES = {
    "bf16_bits",
    "float32",
    "int32",
    "uint32",
}
_FINGERPRINT_KEYS = {
    "association",
    "consumer_boundary",
    "reduction",
    "representation",
    "transport",
}
_FRONTIER_ACTIONS = {"expose", "resolve"}
_EXPECTED_EVIDENCE_IDS = {
    "constructability",
    "direct_shadow_rejection",
    "observability_frontier",
    "shadow_adjudication",
}
_REQUIRED_WATCHPOINT_SCHEMA: dict[str, dict[str, Any]] = {
    "layer1.rms_operands_bf16": {
        "arrays": {
            "hidden_update": {
                "index_prefix": [],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": None,
                "semantic_dtype": "bf16_bits",
                "shape": [6144],
                "storage_dtype": "<u2",
            },
            "residual": {
                "index_prefix": [],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": None,
                "semantic_dtype": "bf16_bits",
                "shape": [6144],
                "storage_dtype": "<u2",
            },
        },
        "layer": 1,
        "position": 8155,
    },
    "layer1.rms_input_fp32": {
        "arrays": {
            "value": {
                "index_prefix": [],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": None,
                "semantic_dtype": "float32",
                "shape": [6144],
                "storage_dtype": "<f4",
            }
        },
        "layer": 1,
        "position": 8155,
    },
    "layer1.normalized": {
        "arrays": {
            "owner0": {
                "index_prefix": [0, 0],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": 0,
                "semantic_dtype": "bf16_bits",
                "shape": [6144],
                "storage_dtype": "<u2",
            },
            "owner1": {
                "index_prefix": [1, 0],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": 1,
                "semantic_dtype": "bf16_bits",
                "shape": [6144],
                "storage_dtype": "<u2",
            },
        },
        "layer": 1,
        "position": 8155,
    },
    "layer1.cache_history": {
        "arrays": {
            "value": {
                "index_prefix": [],
                "owner_axis": 0,
                "owner_axis_slots": [0, 1],
                "owner_slot": None,
                "semantic_dtype": "bf16_bits",
                "shape": [2, 16, 256, 128],
                "storage_dtype": "<u2",
            }
        },
        "layer": 1,
        "position": 8155,
    },
    "layer1.query": {
        "arrays": {
            "owner0": {
                "index_prefix": [0, 0],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": 0,
                "semantic_dtype": "float32",
                "shape": [32, 128],
                "storage_dtype": "<f4",
            },
            "owner1": {
                "index_prefix": [1, 0],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": 1,
                "semantic_dtype": "float32",
                "shape": [32, 128],
                "storage_dtype": "<f4",
            },
        },
        "layer": 1,
        "position": 8155,
    },
    "layer1.head_weights": {
        "arrays": {
            "owner0": {
                "index_prefix": [0, 0],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": 0,
                "semantic_dtype": "float32",
                "shape": [32],
                "storage_dtype": "<f4",
            },
            "owner1": {
                "index_prefix": [1, 0],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": 1,
                "semantic_dtype": "float32",
                "shape": [32],
                "storage_dtype": "<f4",
            },
        },
        "layer": 1,
        "position": 8155,
    },
    "layer1.current_key": {
        "arrays": {
            "value": {
                "index_prefix": [],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": None,
                "semantic_dtype": "float32",
                "shape": [128],
                "storage_dtype": "<f4",
            }
        },
        "layer": 1,
        "position": 8155,
    },
    "layer1.scorer_event1": {
        "arrays": {
            "positions": {
                "index_prefix": [],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": None,
                "semantic_dtype": "int32",
                "shape": [1, 2048],
                "storage_dtype": "<i4",
            },
            "scores": {
                "index_prefix": [],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": None,
                "semantic_dtype": "float32",
                "shape": [1, 2048],
                "storage_dtype": "<f4",
            },
            "valid_count": {
                "index_prefix": [],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": None,
                "semantic_dtype": "int32",
                "shape": [1],
                "storage_dtype": "<i4",
            },
        },
        "layer": 1,
        "position": 8155,
    },
}


def _required_watchpoint_schema(owner_count: int) -> dict[str, dict[str, Any]]:
    """Return the exact capsule schema for one complete LP2/LP4 owner group."""

    if owner_count not in (2, 4):
        raise BenchmarkValidationError("watchpoint owner count is not LP2/LP4")
    schema = json.loads(
        json.dumps(_REQUIRED_WATCHPOINT_SCHEMA, allow_nan=False, ensure_ascii=True)
    )
    for watchpoint_id in (
        "layer1.normalized",
        "layer1.query",
        "layer1.head_weights",
    ):
        prototype = schema[watchpoint_id]["arrays"]["owner0"]
        schema[watchpoint_id]["arrays"] = {
            f"owner{slot}": {
                **prototype,
                "index_prefix": [slot, 0],
                "owner_slot": slot,
            }
            for slot in range(owner_count)
        }
    cache = schema["layer1.cache_history"]["arrays"]["value"]
    cache["owner_axis_slots"] = list(range(owner_count))
    cache["shape"] = [owner_count, 16, 256, 128]
    return schema


def _serialized_watchpoint_schema(owner_count: int) -> list[dict[str, Any]]:
    schema = _required_watchpoint_schema(owner_count)
    return [
        {
            "arrays": [
                {"role": role, **specification}
                for role, specification in sorted(item["arrays"].items())
            ],
            "id": watchpoint_id,
            "layer": item["layer"],
            "position": item["position"],
        }
        for watchpoint_id, item in schema.items()
    ]
_EXPECTED_SURVIVORS: dict[str, dict[str, Any]] = {
    "auxiliary_device_tuple_dependency": {
        "mechanism_fingerprint": {
            "association": "plan.local.shadow",
            "consumer_boundary": "device.auxiliary",
            "reduction": "local.fp32",
            "representation": "tuple.bf16.fp32",
            "transport": "stage.local",
        },
        "normal_form": {
            "compensation": "none",
            "dependency": "device.auxiliary",
            "normalization_input": "persistent.shadow",
            "primary_recurrence": "bf16.rounded",
        },
    },
    "compensated_auxiliary_dependency": {
        "mechanism_fingerprint": {
            "association": "plan.local.compensated",
            "consumer_boundary": "device.auxiliary",
            "reduction": "cancelled.local.fp32",
            "representation": "compensated.bf16.fp32",
            "transport": "stage.local",
        },
        "normal_form": {
            "compensation": "cancelled",
            "dependency": "device.auxiliary",
            "normalization_input": "persistent.shadow",
            "primary_recurrence": "bf16.rounded",
        },
    },
}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise BenchmarkValidationError(f"non-finite JSON value: {value}")


@contextmanager
def _open_directory_no_symlinks(path: Path, label: str) -> Iterator[int]:
    normalized = Path(os.path.abspath(os.fspath(path)))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open("/", flags)
    except OSError as error:
        raise BenchmarkValidationError(
            f"cannot safely open {label}: {normalized}"
        ) from error
    try:
        for component in normalized.parts:
            if component in ("", ".", "/"):
                continue
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise BenchmarkValidationError(
                    f"cannot safely open {label}: {normalized}"
                ) from error
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _open_regular_file(path: Path, label: str) -> Iterator[BinaryIO]:
    normalized = Path(os.path.abspath(os.fspath(path)))
    if not normalized.name or normalized.name in (".", ".."):
        raise BenchmarkValidationError(f"{label} path is invalid: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    with _open_directory_no_symlinks(normalized.parent, f"{label} parent") as parent:
        try:
            descriptor = os.open(normalized.name, flags, dir_fd=parent)
        except OSError as error:
            raise BenchmarkValidationError(
                f"cannot open {label}: {normalized}"
            ) from error
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise BenchmarkValidationError(
                    f"{label} is not a regular file: {normalized}"
                )
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                yield stream
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _require_root_owned_immutable_directory(path: Path, label: str) -> Path:
    """Require an absolute, symlink-free directory chain outside same-UID control."""

    normalized = Path(os.path.abspath(os.fspath(path)))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open("/", flags)
    except OSError as error:
        raise BenchmarkValidationError(
            f"cannot safely open immutable {label}: {normalized}"
        ) from error
    try:
        for component in normalized.parts:
            if component in ("", ".", "/"):
                continue
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise BenchmarkValidationError(
                    f"cannot safely open immutable {label}: {normalized}"
                ) from error
            os.close(descriptor)
            descriptor = child
            metadata = os.fstat(descriptor)
            if metadata.st_uid != 0 or metadata.st_gid != 0 or metadata.st_mode & (
                stat.S_IWGRP | stat.S_IWOTH | _FORBIDDEN_RUNTIME_MODE
            ):
                raise BenchmarkValidationError(
                    f"immutable {label} is not root-owned read-only: {normalized}"
                )
        return normalized
    finally:
        os.close(descriptor)


def _require_root_owned_immutable_file(
    path: Path, label: str, *, beneath: Path
) -> None:
    normalized = Path(os.path.abspath(os.fspath(path)))
    try:
        normalized.relative_to(beneath)
    except ValueError as error:
        raise BenchmarkValidationError(
            f"immutable {label} is outside its runtime root: {normalized}"
        ) from error
    _require_root_owned_immutable_directory(normalized.parent, f"{label} parent")
    with _open_regular_file(normalized, label) as stream:
        metadata = os.fstat(stream.fileno())
        if metadata.st_uid != 0 or metadata.st_gid != 0 or metadata.st_mode & (
            stat.S_IWGRP | stat.S_IWOTH | _FORBIDDEN_RUNTIME_MODE
        ):
            raise BenchmarkValidationError(
                f"immutable {label} is not root-owned read-only: {normalized}"
            )
        try:
            attributes = os.listxattr(normalized, follow_symlinks=False)
        except OSError as error:
            raise BenchmarkValidationError(
                f"cannot inspect immutable {label} attributes: {normalized}"
            ) from error
        if attributes:
            raise BenchmarkValidationError(
                f"immutable {label} has extended attributes: {normalized}"
            )


def _runtime_tree_sha256(root: Path) -> str:
    digest = sha256()
    entries = [root, *sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    )]
    for entry in entries:
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        metadata = entry.lstat()
        if metadata.st_uid != 0 or metadata.st_gid != 0 or (
            not stat.S_ISLNK(metadata.st_mode)
            and metadata.st_mode
            & (stat.S_IWGRP | stat.S_IWOTH | _FORBIDDEN_RUNTIME_MODE)
        ):
            raise BenchmarkValidationError(
                f"immutable Python runtime entry remains writable: {entry}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            kind = b"D"
            payload = b""
        elif stat.S_ISREG(metadata.st_mode):
            kind = b"F"
            file_digest = sha256()
            with _open_regular_file(entry, "immutable Python runtime entry") as stream:
                while block := stream.read(1024 * 1024):
                    file_digest.update(block)
            payload = struct.pack(">Q", metadata.st_size) + file_digest.digest()
        elif stat.S_ISLNK(metadata.st_mode):
            kind = b"L"
            target = os.readlink(entry).encode("utf-8")
            resolved = Path(os.path.realpath(entry))
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise BenchmarkValidationError(
                    f"immutable Python runtime symlink escapes root: {entry}"
                ) from error
            payload = struct.pack(">I", len(target)) + target
        else:
            raise BenchmarkValidationError(
                f"immutable Python runtime entry type is unsupported: {entry}"
            )
        try:
            attributes = os.listxattr(entry, follow_symlinks=False)
        except OSError as error:
            raise BenchmarkValidationError(
                f"cannot inspect immutable Python runtime attributes: {entry}"
            ) from error
        if attributes:
            raise BenchmarkValidationError(
                f"immutable Python runtime entry has extended attributes: {entry}"
            )
        digest.update(kind)
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(
            struct.pack(">I", stat.S_IMODE(metadata.st_mode) & ~0o7022)
        )
        digest.update(payload)
    return digest.hexdigest()


def _snapshot(
    path: Path, expected_sha256: str, label: str, *, limit: int
) -> bytes:
    digest = sha256()
    blocks: list[bytes] = []
    total = 0
    with _open_regular_file(path, label) as stream:
        while block := stream.read(1024 * 1024):
            total += len(block)
            if total > limit:
                raise BenchmarkValidationError(f"{label} is too large: {path}")
            digest.update(block)
            blocks.append(block)
    if digest.hexdigest() != expected_sha256:
        raise BenchmarkValidationError(f"{label} SHA-256 drifted: {path}")
    return b"".join(blocks)


def _regular_file_identity(path: Path, label: str) -> tuple[int, str]:
    digest = sha256()
    total = 0
    with _open_regular_file(path, label) as stream:
        while block := stream.read(8 * 1024 * 1024):
            total += len(block)
            digest.update(block)
    return total, digest.hexdigest()


def _load_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkValidationError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise BenchmarkValidationError(
            f"{label} keys drifted: missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BenchmarkValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _code_pin(value: Any, label: str) -> str:
    if not isinstance(value, str) or _CODE_PIN.fullmatch(value) is None:
        raise BenchmarkValidationError(f"{label} must be a full Git SHA")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise BenchmarkValidationError(f"{label} is not a canonical identifier")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkValidationError(f"{label} must be a non-empty string")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise BenchmarkValidationError(f"{label} must be boolean")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BenchmarkValidationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BenchmarkValidationError(f"{label} must be a non-negative integer")
    return value


def _resolve(base: Path, value: Any, label: str) -> Path:
    raw = _string(value, label)
    path = Path(raw)
    return path if path.is_absolute() else base / path


def _binding(
    base: Path, value: Any, label: str, *, limit: int
) -> tuple[Path, str, bytes]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{label} must be an object")
    _exact_keys(value, {"path", "sha256"}, label)
    expected = _sha(value["sha256"], f"{label} SHA-256")
    path = _resolve(base, value["path"], f"{label} path")
    return path, expected, _snapshot(path, expected, label, limit=limit)


def _mechanism_fingerprint(value: Any, label: str) -> str:
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{label} must be an object")
    _exact_keys(value, _FINGERPRINT_KEYS, label)
    canonical = {
        key: _identifier(value[key], f"{label}.{key}")
        for key in sorted(_FINGERPRINT_KEYS)
    }
    return sha256(_canonical_json(canonical).encode("ascii")).hexdigest()


def _verify_locality(value: Any, expected: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{label} must be an object")
    keys = {
        "logical_rows",
        "max_collective_group_size",
        "no_full_pod_hidden_reconstruction",
        "no_host_effects",
    }
    _exact_keys(value, keys, label)
    if {
        "logical_rows": _positive_int(value["logical_rows"], f"{label} rows"),
        "max_collective_group_size": _positive_int(
            value["max_collective_group_size"], f"{label} group size"
        ),
        "no_full_pod_hidden_reconstruction": _boolean(
            value["no_full_pod_hidden_reconstruction"], f"{label} full-pod rule"
        ),
        "no_host_effects": _boolean(
            value["no_host_effects"], f"{label} host rule"
        ),
    } != dict(expected):
        raise BenchmarkValidationError(f"{label} disagrees with contract")


def _device_groups(value: Any, label: str) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise BenchmarkValidationError(f"{label} must be a non-empty list")
    groups: list[list[int]] = []
    seen: set[int] = set()
    for index, raw_group in enumerate(value):
        if not isinstance(raw_group, list) or not 1 <= len(raw_group) <= 4:
            raise BenchmarkValidationError(f"{label}[{index}] size is invalid")
        group = [
            _nonnegative_int(rank, f"{label}[{index}] rank") for rank in raw_group
        ]
        if (
            len(set(group)) != len(group)
            or any(rank > 31 for rank in group)
            or seen.intersection(group)
        ):
            raise BenchmarkValidationError(f"{label}[{index}] ranks are invalid")
        seen.update(group)
        groups.append(group)
    return groups


def _topology_devices(value: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError("runtime topology must be an object")
    _exact_keys(
        value,
        {"devices", "slice_name", "topology_shape"},
        "runtime topology",
    )
    if value["slice_name"] != "db-v4-64-od" or value["topology_shape"] != [2, 4, 4]:
        raise BenchmarkValidationError("runtime topology identity drifted")
    raw_devices = value["devices"]
    if not isinstance(raw_devices, list) or len(raw_devices) != 32:
        raise BenchmarkValidationError("runtime topology device catalogue drifted")
    devices: dict[int, dict[str, Any]] = {}
    coordinates: set[tuple[int, int, int]] = set()
    process_slots: dict[int, set[int]] = {}
    for index, item in enumerate(raw_devices):
        if not isinstance(item, dict):
            raise BenchmarkValidationError(f"runtime topology device {index} is invalid")
        _exact_keys(
            item,
            {
                "coordinates",
                "core_on_chip",
                "device_id",
                "device_kind",
                "local_device_id",
                "platform",
                "process_index",
            },
            f"runtime topology device {index}",
        )
        device_id = _nonnegative_int(item["device_id"], f"device {index} id")
        process_index = _nonnegative_int(
            item["process_index"], f"device {index} process"
        )
        local_device_id = _nonnegative_int(
            item["local_device_id"], f"device {index} local id"
        )
        raw_coordinates = item["coordinates"]
        if (
            not isinstance(raw_coordinates, list)
            or len(raw_coordinates) != 3
            or any(
                not isinstance(coordinate, int)
                or isinstance(coordinate, bool)
                or coordinate < 0
                or coordinate >= value["topology_shape"][axis]
                for axis, coordinate in enumerate(raw_coordinates)
            )
        ):
            raise BenchmarkValidationError(f"device {index} coordinates are invalid")
        coordinate_tuple = tuple(raw_coordinates)
        if (
            device_id in devices
            or coordinate_tuple in coordinates
            or device_id > 31
            or process_index > 7
            or local_device_id > 3
            or item["core_on_chip"] != 0
            or item["device_kind"] != "TPU v4"
            or item["platform"] != "tpu"
        ):
            raise BenchmarkValidationError(f"runtime topology device {index} drifted")
        coordinates.add(coordinate_tuple)
        process_slots.setdefault(process_index, set()).add(local_device_id)
        devices[device_id] = {
            **item,
            "coordinates": list(raw_coordinates),
        }
    if (
        set(devices) != set(range(32))
        or set(process_slots) != set(range(8))
        or any(slots != set(range(4)) for slots in process_slots.values())
    ):
        raise BenchmarkValidationError("runtime topology process ownership drifted")
    return devices


def _stage_groups(
    value: Any,
    *,
    plan: str,
    group_size: int,
    devices: Mapping[int, Mapping[str, Any]],
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{plan} stage authority must be an object")
    _exact_keys(value, {"groups", "plan"}, f"{plan} stage authority")
    raw_groups = value["groups"]
    expected_count = 32 // group_size
    if value["plan"] != plan or not isinstance(raw_groups, list) or len(raw_groups) != expected_count:
        raise BenchmarkValidationError(f"{plan} stage catalogue drifted")
    records: list[dict[str, Any]] = []
    flat_devices: list[int] = []
    for stage_id, record in enumerate(raw_groups):
        if not isinstance(record, dict):
            raise BenchmarkValidationError(f"{plan} stage {stage_id} is invalid")
        _exact_keys(
            record,
            {"coordinates", "device_ids", "process_index", "stage_id"},
            f"{plan} stage {stage_id}",
        )
        device_ids = record["device_ids"]
        coordinates = record["coordinates"]
        process_index = _nonnegative_int(
            record["process_index"], f"{plan} stage {stage_id} process"
        )
        if (
            record["stage_id"] != stage_id
            or not isinstance(device_ids, list)
            or len(device_ids) != group_size
            or any(
                not isinstance(device_id, int)
                or isinstance(device_id, bool)
                or device_id not in devices
                for device_id in device_ids
            )
            or len(set(device_ids)) != group_size
            or not isinstance(coordinates, list)
            or coordinates != [devices[device_id]["coordinates"] for device_id in device_ids]
            or any(devices[device_id]["process_index"] != process_index for device_id in device_ids)
        ):
            raise BenchmarkValidationError(f"{plan} stage {stage_id} ownership drifted")
        coordinate_tuples = [tuple(item) for item in coordinates]
        if plan == "PP16_LP2":
            first, second = coordinate_tuples
            coordinate_local = (
                first[1:] == second[1:] and {first[0], second[0]} == {0, 1}
            )
        else:
            coordinate_local = (
                len(set(coordinate_tuples)) == 4
                and len({item[2] for item in coordinate_tuples}) == 1
                and {item[0] for item in coordinate_tuples} == {0, 1}
                and len({item[1] for item in coordinate_tuples}) == 2
                and max(item[1] for item in coordinate_tuples)
                - min(item[1] for item in coordinate_tuples)
                == 1
            )
        if not coordinate_local:
            raise BenchmarkValidationError(f"{plan} stage {stage_id} is not coordinate-local")
        flat_devices.extend(device_ids)
        records.append(
            {
                "coordinates": coordinates,
                "device_ids": device_ids,
                "process_index": process_index,
                "stage_id": stage_id,
            }
        )
    if sorted(flat_devices) != list(range(32)):
        raise BenchmarkValidationError(f"{plan} stage ownership is incomplete")
    return [record["device_ids"] for record in records], records


def _verify_physical_locality(value: Any, base: Path) -> dict[str, Any]:
    path, file_sha, raw = _binding(
        base,
        value,
        "runtime physical locality authority",
        limit=_MAX_JSON_BYTES,
    )
    document = _load_json(raw, "runtime physical locality authority")
    _exact_keys(
        document,
        {
            "artifact_kind",
            "claim_scope",
            "db_run_id",
            "pp16_lp2",
            "pp16_lp2_hash",
            "pp8_lp4",
            "pp8_lp4_hash",
            "schema_version",
            "source_archive",
            "source_contract_hash",
            "source_host_record_rank0_sha256",
            "source_summary_sha256",
            "topology",
            "topology_hash",
            "tpu_successor_authorized",
        },
        "runtime physical locality authority",
    )
    topology = document["topology"]
    topology_hash = sha256(_canonical_json(topology).encode("ascii")).hexdigest()
    devices = _topology_devices(topology)
    pp16_groups, pp16_records = _stage_groups(
        document["pp16_lp2"],
        plan="PP16_LP2",
        group_size=2,
        devices=devices,
    )
    pp8_groups, pp8_records = _stage_groups(
        document["pp8_lp4"],
        plan="PP8_LP4",
        group_size=4,
        devices=devices,
    )
    pp16_hash = sha256(
        _canonical_json(document["pp16_lp2"]).encode("ascii")
    ).hexdigest()
    pp8_hash = sha256(
        _canonical_json(document["pp8_lp4"]).encode("ascii")
    ).hexdigest()
    if (
        document["artifact_kind"]
        != "gate_d_runtime_physical_locality_authority"
        or document["schema_version"] != 2
        or document["db_run_id"] != 555
        or topology_hash != _EXPECTED_TOPOLOGY_HASH
        or document["topology_hash"] != topology_hash
        or pp16_hash != _EXPECTED_PP16_LP2_HASH
        or document["pp16_lp2_hash"] != pp16_hash
        or pp8_hash != _EXPECTED_PP8_LP4_HASH
        or document["pp8_lp4_hash"] != pp8_hash
        or document["source_contract_hash"]
        != _EXPECTED_TOPOLOGY_SOURCE_CONTRACT_HASH
        or document["source_host_record_rank0_sha256"]
        != "71a4d01e3d266ba2a080c0634a3878aae40fcf11b1b6bf4f863f0cb1379afc7d"
        or document["source_summary_sha256"]
        != "1631eeb0102036319eef19af4be4f5cc165b889a70b51b36fe41b4996d7462d2"
        or document["tpu_successor_authorized"] is not False
    ):
        raise BenchmarkValidationError("runtime physical locality authority drifted")
    _string(document["claim_scope"], "runtime locality claim scope")
    _string(document["source_archive"], "runtime locality source archive")
    return {
        "authority_path": str(path),
        "authority_sha256": file_sha,
        "plans": {"PP16_LP2": pp16_groups, "PP8_LP4": pp8_groups},
        "plan_records": {
            "PP16_LP2": pp16_records,
            "PP8_LP4": pp8_records,
        },
        "topology_hash": topology_hash,
    }


def _verify_inherited_v1(
    value: Any, base: Path
) -> tuple[dict[str, Any], frozenset[str]]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError("inherited v1 authority must be an object")
    _exact_keys(
        value,
        {"contract", "core", "frontier", "expected_classification"},
        "inherited v1 authority",
    )
    contract_path, contract_sha, contract_raw = _binding(
        base, value["contract"], "inherited v1 contract", limit=_MAX_JSON_BYTES
    )
    core_path, core_sha, _ = _binding(
        base, value["core"], "inherited v1 core", limit=_MAX_SOURCE_BYTES
    )
    _, frontier_sha, frontier_raw = _binding(
        base, value["frontier"], "inherited v1 frontier", limit=_MAX_JSON_BYTES
    )
    contract = _load_json(contract_raw, "inherited v1 contract")
    frontier = _load_json(frontier_raw, "inherited v1 frontier")
    expected_classification = _string(
        value["expected_classification"], "inherited v1 expected classification"
    )
    if (
        frontier.get("classification") != expected_classification
        or frontier.get("admitted_candidate_ids") != []
        or frontier.get("tpu_successor_authorized") is not False
        or frontier.get("jax_or_tpu_work_performed") is not False
    ):
        raise BenchmarkValidationError("inherited v1 frontier claim drifted")
    reproduction = frontier.get("reproduction")
    if (
        not isinstance(reproduction, dict)
        or reproduction.get("contract_sha256") != contract_sha
        or reproduction.get("core_sha256") != core_sha
    ):
        raise BenchmarkValidationError("inherited v1 reproduction binding drifted")
    closed = contract.get("closed_families")
    if not isinstance(closed, list) or not closed:
        raise BenchmarkValidationError("inherited v1 closed families are absent")
    fingerprints: set[str] = set()
    for family in closed:
        if not isinstance(family, dict):
            raise BenchmarkValidationError("inherited v1 closed family is invalid")
        values = family.get("mechanism_fingerprint_sha256s")
        if not isinstance(values, list) or not values:
            raise BenchmarkValidationError(
                "inherited v1 closed fingerprints are absent"
            )
        for item in values:
            fingerprint = _sha(item, "inherited v1 closed fingerprint")
            if fingerprint in fingerprints:
                raise BenchmarkValidationError(
                    "inherited v1 closed fingerprint is duplicated"
                )
            fingerprints.add(fingerprint)
    return (
        {
            "contract_path": str(contract_path),
            "contract_sha256": contract_sha,
            "core_path": str(core_path),
            "core_sha256": core_sha,
            "frontier_sha256": frontier_sha,
            "classification": expected_classification,
            "closed_fingerprint_count": len(fingerprints),
        },
        frozenset(fingerprints),
    )


def _verify_implementation(value: Any, base: Path) -> dict[str, Any]:
    if os.geteuid() == 0:
        raise BenchmarkValidationError(
            "StableHLO admission must run as an unprivileged user"
        )
    if not isinstance(value, dict):
        raise BenchmarkValidationError("precompile implementation must be an object")
    _exact_keys(
        value,
        {
            "cli",
            "core",
            "git",
            "stablehlo_validator",
            "validator_imports",
            "validator_python",
            "validator_python_provisioner",
            "validator_python_provisioner_source",
            "validator_python_runtime_root",
            "validator_python_runtime_sha256",
            "validator_pythonpath",
        },
        "precompile implementation",
    )
    core_path, core_sha, _ = _binding(
        base, value["core"], "precompile admission core", limit=_MAX_SOURCE_BYTES
    )
    cli_path, cli_sha, _ = _binding(
        base, value["cli"], "precompile admission CLI", limit=_MAX_SOURCE_BYTES
    )
    git_path, git_sha, _ = _binding(
        base, value["git"], "Git executable", limit=_MAX_SOURCE_BYTES
    )
    validator_path, validator_sha, _ = _binding(
        base,
        value["stablehlo_validator"],
        "StableHLO validator",
        limit=_MAX_SOURCE_BYTES,
    )
    python_path, python_sha, _ = _binding(
        base,
        value["validator_python"],
        "StableHLO validator Python",
        limit=64 * 1024 * 1024,
    )
    provisioner_path, provisioner_sha, _ = _binding(
        base,
        value["validator_python_provisioner"],
        "StableHLO validator Python runtime provisioner",
        limit=_MAX_SOURCE_BYTES,
    )
    _require_root_owned_immutable_file(
        provisioner_path,
        "StableHLO validator Python runtime provisioner",
        beneath=Path("/opt/glm-tpu"),
    )
    with _open_regular_file(
        provisioner_path, "StableHLO validator Python runtime provisioner"
    ) as stream:
        if stat.S_IMODE(os.fstat(stream.fileno()).st_mode) != 0o555:
            raise BenchmarkValidationError(
                "installed Python runtime provisioner mode is not 0555"
            )
    provisioner_source_path, provisioner_source_sha, _ = _binding(
        base,
        value["validator_python_provisioner_source"],
        "StableHLO validator Python runtime provisioner source",
        limit=_MAX_SOURCE_BYTES,
    )
    if provisioner_source_sha != provisioner_sha:
        raise BenchmarkValidationError(
            "installed Python runtime provisioner differs from reviewed source"
        )
    python_runtime_root = _require_root_owned_immutable_directory(
        _resolve(
            base,
            value["validator_python_runtime_root"],
            "StableHLO validator Python runtime root",
        ),
        "StableHLO validator Python runtime root",
    )
    _require_root_owned_immutable_file(
        python_path,
        "StableHLO validator Python",
        beneath=python_runtime_root,
    )
    python_runtime_sha = _sha(
        value["validator_python_runtime_sha256"],
        "StableHLO validator Python runtime SHA-256",
    )
    runtime_key = (str(python_runtime_root), python_runtime_sha)
    if runtime_key not in _VERIFIED_RUNTIME_TREES:
        if _runtime_tree_sha256(python_runtime_root) != python_runtime_sha:
            raise BenchmarkValidationError(
                "StableHLO validator Python runtime tree SHA-256 drifted"
            )
        # Every entry is root-owned and non-writable to this process, so the
        # authenticated tree cannot change within the same admission process.
        _VERIFIED_RUNTIME_TREES.add(runtime_key)
    pythonpath = _resolve(
        base, value["validator_pythonpath"], "StableHLO validator PYTHONPATH"
    )
    with _open_directory_no_symlinks(
        pythonpath, "StableHLO validator PYTHONPATH"
    ):
        pass
    raw_imports = value["validator_imports"]
    if not isinstance(raw_imports, list) or not raw_imports:
        raise BenchmarkValidationError("StableHLO validator imports are absent")
    parser_imports: list[dict[str, Any]] = []
    import_paths: set[str] = set()
    total_bytes = 0
    for index, item in enumerate(raw_imports):
        if not isinstance(item, dict):
            raise BenchmarkValidationError(
                f"StableHLO validator import {index} is invalid"
            )
        _exact_keys(
            item,
            {"bytes", "path", "sha256"},
            f"StableHLO validator import {index}",
        )
        relative = _canonical_repo_path(
            item["path"], f"StableHLO validator import {index} path"
        )
        if relative in import_paths or not relative.startswith("jaxlib/"):
            raise BenchmarkValidationError(
                "StableHLO validator import path is duplicated or outside jaxlib"
            )
        import_paths.add(relative)
        size = _positive_int(item["bytes"], f"parser import {relative} bytes")
        total_bytes += size
        if total_bytes > _MAX_PARSER_TOTAL_BYTES:
            raise BenchmarkValidationError("StableHLO validator imports exceed limit")
        parser_imports.append(
            {"bytes": size, "path": relative, "sha256": _sha(item["sha256"], relative)}
        )
    observed_imports = {
        item["path"]: (item["bytes"], item["sha256"]) for item in parser_imports
    }
    if observed_imports != _EXPECTED_VALIDATOR_IMPORTS:
        raise BenchmarkValidationError("StableHLO validator import manifest drifted")
    if Path(os.path.abspath(os.fspath(core_path))) != Path(
        os.path.abspath(__file__)
    ):
        raise BenchmarkValidationError(
            "precompile core binding is not the active implementation"
        )
    return {
        "cli_path": str(cli_path),
        "cli_sha256": cli_sha,
        "core_path": str(core_path),
        "core_sha256": core_sha,
        "git_path": str(git_path),
        "git_sha256": git_sha,
        "stablehlo_validator_path": str(validator_path),
        "stablehlo_validator_sha256": validator_sha,
        "validator_imports": sorted(parser_imports, key=lambda item: item["path"]),
        "validator_python_path": str(python_path),
        "validator_python_provisioner_path": str(provisioner_path),
        "validator_python_provisioner_sha256": provisioner_sha,
        "validator_python_provisioner_source_path": str(provisioner_source_path),
        "validator_python_provisioner_source_sha256": provisioner_source_sha,
        "validator_python_runtime_root": str(python_runtime_root),
        "validator_python_runtime_sha256": python_runtime_sha,
        "validator_python_sha256": python_sha,
        "validator_pythonpath": str(pythonpath),
    }


def _qualified_ast_node(tree: ast.Module, qualified_name: str) -> ast.AST:
    components = qualified_name.split(".")
    if not components or any(not component.isidentifier() for component in components):
        raise BenchmarkValidationError(
            f"source symbol is not a qualified Python name: {qualified_name}"
        )
    body: Sequence[ast.stmt] = tree.body
    node: ast.AST | None = None
    for index, component in enumerate(components):
        matches = [
            item
            for item in body
            if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == component
        ]
        if len(matches) != 1:
            raise BenchmarkValidationError(
                f"source symbol is absent or ambiguous: {qualified_name}"
            )
        node = matches[0]
        if index + 1 < len(components):
            if not isinstance(node, ast.ClassDef):
                raise BenchmarkValidationError(
                    f"source symbol parent is not a class: {qualified_name}"
                )
            body = node.body
    assert node is not None
    return node


def _call_of(value: ast.AST, name: str, arguments: Sequence[ast.AST]) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == name
        and not value.keywords
        and len(value.args) == len(arguments)
        and all(ast.dump(left) == ast.dump(right) for left, right in zip(value.args, arguments))
    )


def _named_assignment(statement: ast.stmt, name: str) -> ast.AST | None:
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == name
    ):
        return statement.value
    return None


def _candidate_source_text(candidate_id: str) -> str:
    """Return the declarative semantic DSL used by hostile fixture validation.

    The names in this snippet are intentionally abstract operators.  This is not
    executable candidate source authority: a future candidate must separately
    bind concrete operator definitions and its real integration caller.
    """
    auxiliary = "    auxiliary = transient\n"
    if candidate_id == "compensated_auxiliary_dependency":
        auxiliary = (
            "    rounded = float32(carried_residual)\n"
            "    correction = transient - rounded\n"
            "    restored = rounded + correction\n"
            "    auxiliary = restored - correction\n"
        )
    return (
        "def candidate_dependency(hidden_update, residual, weight):\n"
        "    transient = float32(hidden_update) + float32(residual)\n"
        "    carried_residual = round_to_bf16(transient)\n"
        "    variance = mean(square(transient))\n"
        "    normalized = transient * rsqrt(variance + epsilon())\n"
        "    weighted_output = round_to_bf16(round_to_bf16(normalized) * weight)\n"
        f"{auxiliary}"
        "    return weighted_output, carried_residual, auxiliary\n"
        "\n"
        "def candidate_callsite(hidden_update, residual, weight):\n"
        "    return candidate_dependency(hidden_update, residual, weight)\n"
    )


def _candidate_source_semantics(
    node: ast.AST,
    callsite: ast.AST,
    candidate_id: str,
) -> dict[str, Any]:
    if candidate_id not in _EXPECTED_SURVIVORS:
        raise BenchmarkValidationError("candidate source identity is unsupported")
    expected = ast.parse(_candidate_source_text(candidate_id))
    expected_node, expected_callsite = expected.body
    if ast.dump(node) != ast.dump(expected_node):
        raise BenchmarkValidationError("candidate source RMS semantics drifted")
    if ast.dump(callsite) != ast.dump(expected_callsite):
        raise BenchmarkValidationError("candidate source callsite semantics drifted")
    return {
        "authority_scope": "declarative.semantic.dsl.only",
        "auxiliary": (
            "transient.fp32.sum"
            if candidate_id == "auxiliary_device_tuple_dependency"
            else "graph.identity.only;numeric.cancellation.unproven"
        ),
        "callsite": "declarative.consumer(candidate_dependency(hidden_update,residual,weight))",
        "carried_residual": "bf16.round(transient.fp32.sum)",
        "frontier": "layer1.rms_input_fp32",
        "normalization_input": "transient.fp32.sum",
        "weighted_output": (
            "bf16.round(bf16.round(transient*rsqrt(mean(square(transient))+epsilon))*weight)"
        ),
    }


def _module_imports(tree: ast.Module) -> set[str]:
    return {
        ast.dump(node, annotate_fields=True, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }


def _assignment_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_assignment_names(item) for item in target.elts))
    return set()


def _top_level_bindings(tree: ast.Module) -> dict[str, list[str]]:
    bindings: dict[str, list[str]] = {}
    for statement in tree.body:
        names: set[str] = set()
        if isinstance(statement, ast.Import):
            names = {
                alias.asname or alias.name.split(".", 1)[0]
                for alias in statement.names
            }
        elif isinstance(statement, ast.ImportFrom):
            names = {alias.asname or alias.name for alias in statement.names}
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = {statement.name}
        elif isinstance(statement, ast.Assign):
            names = set().union(
                *(_assignment_names(target) for target in statement.targets)
            )
        elif isinstance(statement, ast.AnnAssign):
            names = _assignment_names(statement.target)
        dump = ast.dump(statement, annotate_fields=True, include_attributes=False)
        for name in names:
            bindings.setdefault(name, []).append(dump)
    return bindings


def _refuse_sensitive_rebinding(
    tree: ast.Module,
    names: set[str],
    *,
    label: str,
) -> None:
    top_level_imports = {
        id(node)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in names and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            raise BenchmarkValidationError(f"{label} was rebound")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (
            node.name in names
        ):
            raise BenchmarkValidationError(f"{label} was rebound")
        if isinstance(node, (ast.Global, ast.Nonlocal)) and names.intersection(node.names):
            raise BenchmarkValidationError(f"{label} was rebound")
        if isinstance(node, ast.Import):
            bound = {
                alias.asname or alias.name.split(".", 1)[0]
                for alias in node.names
            }
            if names.intersection(bound) and id(node) not in top_level_imports:
                raise BenchmarkValidationError(f"{label} was rebound")
        if isinstance(node, ast.ImportFrom):
            bound = {alias.asname or alias.name for alias in node.names}
            if names.intersection(bound) and id(node) not in top_level_imports:
                raise BenchmarkValidationError(f"{label} was rebound")


def _simple_assignments(function: ast.FunctionDef) -> dict[str, ast.AST]:
    return {
        target.id: statement.value
        for statement in function.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance((target := statement.targets[0]), ast.Name)
    }


class _AstNameRename(ast.NodeTransformer):
    def __init__(self, old: str, new: str) -> None:
        self.old = old
        self.new = new

    def visit_Name(self, node: ast.Name) -> ast.Name:  # noqa: N802
        if node.id == self.old:
            return ast.copy_location(ast.Name(id=self.new, ctx=node.ctx), node)
        return node


def _normalized_expression(value: ast.AST, old: str, new: str) -> str:
    clone = ast.parse(ast.unparse(value), mode="eval").body
    clone = _AstNameRename(old, new).visit(clone)
    ast.fix_missing_locations(clone)
    return ast.dump(clone, annotate_fields=True, include_attributes=False)


def _validation_guards(function: ast.FunctionDef) -> list[ast.If]:
    if (
        not function.body
        or not isinstance(function.body[0], ast.Expr)
        or not isinstance(function.body[0].value, ast.Constant)
        or not isinstance(function.body[0].value.value, str)
    ):
        raise BenchmarkValidationError("concrete source function docstring drifted")
    guards: list[ast.If] = []
    index = 1
    while index < len(function.body) and isinstance(function.body[index], ast.If):
        guard = function.body[index]
        if (
            guard.orelse
            or len(guard.body) != 1
            or not isinstance(guard.body[0], ast.Raise)
            or not isinstance(guard.body[0].exc, ast.Call)
            or ast.unparse(guard.body[0].exc.func) != "ValueError"
            or len(guard.body[0].exc.args) != 1
            or not isinstance(guard.body[0].exc.args[0], ast.Constant)
            or not isinstance(guard.body[0].exc.args[0].value, str)
            or guard.body[0].exc.keywords
            or guard.body[0].cause is not None
        ):
            raise BenchmarkValidationError("concrete source validation guard drifted")
        guards.append(guard)
        index += 1
    if len(guards) != 7:
        raise BenchmarkValidationError("concrete source validation guards drifted")
    return guards


def _named_tuple_fields(node: ast.AST, label: str) -> list[tuple[str, str]]:
    if (
        not isinstance(node, ast.ClassDef)
        or [ast.unparse(base) for base in node.bases] != ["NamedTuple"]
        or node.decorator_list
        or node.keywords
        or not node.body
        or not isinstance(node.body[0], ast.Expr)
        or not isinstance(node.body[0].value, ast.Constant)
        or not isinstance(node.body[0].value.value, str)
        or any(not isinstance(statement, ast.AnnAssign) for statement in node.body[1:])
        or any(
            statement.value is not None or statement.simple != 1
            for statement in node.body[1:]
            if isinstance(statement, ast.AnnAssign)
        )
    ):
        raise BenchmarkValidationError(f"{label} is not one exact NamedTuple")
    fields = [
        (statement.target.id, ast.unparse(statement.annotation))
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
    ]
    if len(fields) != len(node.body) - 1:
        raise BenchmarkValidationError(f"{label} fields are not canonical")
    return fields


def _concrete_tuple_source_semantics(
    symbols: Mapping[str, ast.AST],
    modules: Mapping[str, ast.Module],
) -> dict[str, Any]:
    required = {
        "candidate.source:FusedAddRmsNormAuxiliaryResult",
        "candidate.source:fused_add_rms_norm",
        "candidate.source:fused_add_rms_norm_with_auxiliary",
        "candidate.callsite:StageLocalSplitLayerFp8AuxiliaryResult",
        "candidate.callsite:stage_local_transformer_layer_fp8_split_mapped",
    }
    if not required.issubset(symbols):
        raise BenchmarkValidationError("concrete source symbol set is incomplete")
    rms_module = modules.get("candidate.source")
    layer_module = modules.get("candidate.callsite")
    if not isinstance(rms_module, ast.Module) or not isinstance(layer_module, ast.Module):
        raise BenchmarkValidationError("concrete source modules are absent")
    rms_imports = _module_imports(rms_module)
    layer_imports = _module_imports(layer_module)
    expected_rms_imports = _module_imports(
        ast.parse(
            "from __future__ import annotations\n"
            "from typing import NamedTuple\n"
            "import jax\n"
            "from jax import lax\n"
            "import jax.numpy as jnp\n"
        )
    )
    expected_layer_import = ast.dump(
        ast.parse(
            "from .reference.rmsnorm import (fused_add_rms_norm, "
            "fused_add_rms_norm_with_auxiliary, rms_norm)\n"
        ).body[0],
        annotate_fields=True,
        include_attributes=False,
    )
    if not expected_rms_imports.issubset(rms_imports) or expected_layer_import not in layer_imports:
        raise BenchmarkValidationError("concrete source imports are not bound")
    rms_bindings = _top_level_bindings(rms_module)
    layer_bindings = _top_level_bindings(layer_module)
    expected_rms_tree = ast.parse(
        "from typing import NamedTuple\n"
        "import jax\n"
        "from jax import lax\n"
        "import jax.numpy as jnp\n"
    )
    expected_binding = _top_level_bindings(expected_rms_tree)
    for name in ("NamedTuple", "jax", "lax", "jnp"):
        if rms_bindings.get(name) != expected_binding[name]:
            raise BenchmarkValidationError("concrete source global import was rebound")
    if any(name in rms_bindings for name in ("ValueError", "isinstance", "int", "float", "bool")):
        raise BenchmarkValidationError("concrete source builtin was rebound")
    layer_import_node = ast.parse(
        "from .reference.rmsnorm import (fused_add_rms_norm, "
        "fused_add_rms_norm_with_auxiliary, rms_norm)\n"
    ).body[0]
    layer_import_dump = ast.dump(
        layer_import_node, annotate_fields=True, include_attributes=False
    )
    for name in ("fused_add_rms_norm", "fused_add_rms_norm_with_auxiliary"):
        if layer_bindings.get(name) != [layer_import_dump]:
            raise BenchmarkValidationError("concrete source caller import was rebound")
    _refuse_sensitive_rebinding(
        rms_module,
        {"NamedTuple", "ValueError", "bool", "float", "int", "isinstance", "jax", "jnp", "lax"},
        label="concrete source global import",
    )
    _refuse_sensitive_rebinding(
        layer_module,
        {"fused_add_rms_norm", "fused_add_rms_norm_with_auxiliary"},
        label="concrete source caller import",
    )
    if _named_tuple_fields(
        symbols["candidate.source:FusedAddRmsNormAuxiliaryResult"],
        "candidate RMS result",
    ) != [
        ("output", "jax.Array"),
        ("carried_residual", "jax.Array"),
        ("rms_input_fp32", "jax.Array"),
    ]:
        raise BenchmarkValidationError("candidate RMS result fields drifted")
    if _named_tuple_fields(
        symbols["candidate.callsite:StageLocalSplitLayerFp8AuxiliaryResult"],
        "candidate layer result",
    ) != [
        ("result", "StageLocalSplitLayerFp8Result"),
        ("input_rms_fp32", "Any"),
    ]:
        raise BenchmarkValidationError("candidate layer result fields drifted")
    accepted = symbols["candidate.source:fused_add_rms_norm"]
    candidate = symbols["candidate.source:fused_add_rms_norm_with_auxiliary"]
    caller = symbols["candidate.callsite:stage_local_transformer_layer_fp8_split_mapped"]
    if not all(isinstance(item, ast.FunctionDef) for item in (accepted, candidate, caller)):
        raise BenchmarkValidationError("concrete source functions are invalid")
    if (
        accepted.decorator_list
        or candidate.decorator_list
        or ast.dump(accepted.args) != ast.dump(candidate.args)
    ):
        raise BenchmarkValidationError("concrete source function signature drifted")
    expected_args = ast.parse(
        "def expected(hidden_states: jax.Array, residual: jax.Array, "
        "weight: jax.Array, *, epsilon: float):\n    pass\n"
    ).body[0]
    if not isinstance(expected_args, ast.FunctionDef) or ast.dump(
        accepted.args
    ) != ast.dump(expected_args.args):
        raise BenchmarkValidationError("accepted source function signature drifted")
    accepted_guards = _validation_guards(accepted)
    candidate_guards = _validation_guards(candidate)
    expected_guard_tests = [
        ast.parse(f"if {expression}:\n    pass\n").body[0].test
        for expression in (
            "hidden_states.shape != residual.shape",
            "hidden_states.dtype != residual.dtype",
            "hidden_states.ndim < 1",
            "weight.shape != (hidden_states.shape[-1],)",
            "not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or epsilon <= 0",
            "not jnp.issubdtype(hidden_states.dtype, jnp.inexact)",
            "not jnp.issubdtype(weight.dtype, jnp.inexact)",
        )
    ]
    if [ast.dump(item.test) for item in accepted_guards] != [
        ast.dump(item) for item in expected_guard_tests
    ] or [ast.dump(item.test) for item in candidate_guards] != [
        ast.dump(item) for item in expected_guard_tests
    ]:
        raise BenchmarkValidationError("concrete source validation conditions drifted")
    accepted_assignments = _simple_assignments(accepted)
    candidate_assignments = _simple_assignments(candidate)
    pairs = (
        ("activation_dtype", "activation_dtype"),
        ("summed", "rms_input_fp32"),
        ("carried_residual", "carried_residual"),
        ("variance", "variance"),
        ("normalized", "normalized"),
        ("output", "output"),
    )
    if set(accepted_assignments) != {left for left, _ in pairs} or set(
        candidate_assignments
    ) != {right for _, right in pairs}:
        raise BenchmarkValidationError("concrete source arithmetic assignments drifted")
    if [type(item) for item in accepted.body] != (
        [ast.Expr] + [ast.If] * 7 + [ast.Assign] * 6 + [ast.Return]
    ) or [type(item) for item in candidate.body] != (
        [ast.Expr] + [ast.If] * 7 + [ast.Assign] * 6 + [ast.Return]
    ):
        raise BenchmarkValidationError("concrete source executable body drifted")
    expected_expressions = {
        "activation_dtype": "hidden_states.dtype",
        "summed": "hidden_states.astype(jnp.float32) + residual.astype(jnp.float32)",
        "carried_residual": "summed.astype(activation_dtype)",
        "variance": "jnp.mean(lax.square(summed), axis=-1, keepdims=True)",
        "normalized": "summed * lax.rsqrt(variance + jnp.float32(epsilon))",
        "output": "(normalized.astype(weight.dtype) * weight).astype(activation_dtype)",
    }
    for name, expression in expected_expressions.items():
        if _normalized_expression(
            accepted_assignments[name], "summed", "summed"
        ) != _normalized_expression(ast.parse(expression, mode="eval").body, "summed", "summed"):
            raise BenchmarkValidationError("accepted source primary arithmetic drifted")
    for accepted_name, candidate_name in pairs:
        if _normalized_expression(
            accepted_assignments[accepted_name], "summed", "summed"
        ) != _normalized_expression(
            candidate_assignments[candidate_name], "rms_input_fp32", "summed"
        ):
            raise BenchmarkValidationError("concrete source primary arithmetic drifted")
    accepted_return = accepted.body[-1]
    if (
        not isinstance(accepted_return, ast.Return)
        or not isinstance(accepted_return.value, ast.Tuple)
        or [ast.unparse(item) for item in accepted_return.value.elts]
        != ["output", "carried_residual"]
    ):
        raise BenchmarkValidationError("accepted source return drifted")
    returned = candidate.body[-1]
    if (
        not isinstance(returned, ast.Return)
        or not isinstance(returned.value, ast.Call)
        or ast.unparse(returned.value.func) != "FusedAddRmsNormAuxiliaryResult"
        or [ast.unparse(item) for item in returned.value.args]
        != ["output", "carried_residual", "rms_input_fp32"]
        or returned.value.keywords
    ):
        raise BenchmarkValidationError("concrete source tuple return drifted")
    keyword_defaults = dict(
        zip(
            (argument.arg for argument in caller.args.kwonlyargs),
            caller.args.kw_defaults,
            strict=True,
        )
    )
    default = keyword_defaults.get("retain_input_rms_auxiliary")
    if not isinstance(default, ast.Constant) or default.value is not False:
        raise BenchmarkValidationError("concrete source flag is not default off")
    branches = [
        node
        for node in caller.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "retain_input_rms_auxiliary"
    ]
    if len(branches) != 2:
        raise BenchmarkValidationError("concrete source caller branches drifted")
    expected_enabled = ast.parse(
        "rms_candidate = fused_add_rms_norm_with_auxiliary("
        "hidden_states, residual, input_norm_weight, epsilon=rms_norm_epsilon)\n"
        "normalized_input = rms_candidate.output\n"
        "combined_residual = rms_candidate.carried_residual\n"
        "input_rms_fp32 = rms_candidate.rms_input_fp32\n"
    ).body
    expected_default = ast.parse(
        "normalized_input, combined_residual = fused_add_rms_norm("
        "hidden_states, residual, input_norm_weight, epsilon=rms_norm_epsilon)\n"
        "input_rms_fp32 = None\n"
    ).body
    enabled = next((item for item in branches if item.orelse), None)
    if (
        enabled is None
        or ast.dump(ast.Module(body=enabled.body, type_ignores=[]))
        != ast.dump(ast.Module(body=expected_enabled, type_ignores=[]))
        or ast.dump(ast.Module(body=enabled.orelse, type_ignores=[]))
        != ast.dump(ast.Module(body=expected_default, type_ignores=[]))
    ):
        raise BenchmarkValidationError("concrete source caller selection drifted")
    return_branch = next((item for item in branches if not item.orelse), None)
    wrapper_returns = [
        node
        for node in (return_branch.body if return_branch is not None else [])
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "StageLocalSplitLayerFp8AuxiliaryResult"
    ]
    if (
        return_branch is None
        or len(return_branch.body) != 2
        or not isinstance(return_branch.body[0], ast.Assert)
        or ast.unparse(return_branch.body[0].test) != "input_rms_fp32 is not None"
        or len(wrapper_returns) != 1
        or [ast.unparse(item) for item in wrapper_returns[0].value.args]
        != ["result", "input_rms_fp32"]
        or wrapper_returns[0].value.keywords
    ):
        raise BenchmarkValidationError("concrete source device return drifted")
    text = ast.unparse(candidate) + ast.unparse(caller)
    if any(
        marker in text
        for marker in (
            "debug.callback",
            "debug.print",
            "device_get",
            "host_callback",
            "io_callback",
            "np.asarray",
            "pure_callback",
        )
    ):
        raise BenchmarkValidationError("concrete source contains a host effect")
    return {
        "authority_scope": "concrete.committed.jax.source",
        "auxiliary": "transient.fp32.sum",
        "callsite": "stage_local_transformer_layer_fp8_split_mapped.default_off_device_tuple",
        "carried_residual": "bf16.round(transient.fp32.sum)",
        "frontier": "layer1.rms_input_fp32",
        "normalization_input": "transient.fp32.sum",
        "weighted_output": (
            "bf16.round(bf16.round(transient*rsqrt(mean(square(transient))+epsilon))*weight)"
        ),
    }


def _accepted_semantics_from_evidence(document: Mapping[str, Any]) -> dict[str, Any]:
    authority = document.get("accepted_authority")
    logical_hlo = document.get("accepted_logical_hlo_authority")
    semantic_contract = document.get("sealed_semantic_contract")
    if (
        not isinstance(authority, dict)
        or not isinstance(logical_hlo, dict)
        or not isinstance(semantic_contract, dict)
        or authority.get("git_commit") != _EXPECTED_ACCEPTED_SOURCE["git_commit"]
        or logical_hlo.get("sha256")
        != "a8c9577d63909e647cd1b1a6d63210a818c51fc7efe7b305e385bad59189ba21"
        or semantic_contract.get("boundary_sum")
        != "float32(hidden_update) + float32(carried_residual)"
        or semantic_contract.get("normalization_input")
        != "the unrounded boundary_sum"
        or semantic_contract.get("greenfield_activation_dtype") != "bfloat16"
    ):
        raise BenchmarkValidationError("sealed accepted source semantics drifted")
    files = authority.get("files")
    if not isinstance(files, list):
        raise BenchmarkValidationError("sealed accepted source files are absent")
    matching = [
        item
        for item in files
        if isinstance(item, dict)
        and item.get("path") == _EXPECTED_ACCEPTED_SOURCE["path"]
        and item.get("symbol") == _EXPECTED_ACCEPTED_SOURCE["symbol"]
    ]
    if len(matching) != 1 or any(
        matching[0].get(key) != _EXPECTED_ACCEPTED_SOURCE[key]
        for key in ("ast_sha256", "file_sha256")
    ):
        raise BenchmarkValidationError("sealed accepted source symbol drifted")
    payload = {
        "accepted_source": _EXPECTED_ACCEPTED_SOURCE,
        "logical_hlo_sha256": logical_hlo["sha256"],
        "semantic_contract": semantic_contract,
    }
    return {
        **payload,
        "authority_sha256": sha256(
            _canonical_json(payload).encode("ascii")
        ).hexdigest(),
    }


def _git_output(
    repository_descriptor: int,
    git_path: Path,
    git_sha256: str,
    arguments: Sequence[str],
    label: str,
    *,
    limit: int,
) -> bytes:
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "LANG": "C",
        "LC_ALL": "C",
    }
    try:
        with _open_regular_file(git_path, "Git executable") as git_stream:
            digest = sha256()
            while block := git_stream.read(1024 * 1024):
                digest.update(block)
            if digest.hexdigest() != git_sha256:
                raise BenchmarkValidationError("Git executable SHA-256 drifted")
            completed = subprocess.run(
                [
                    "git",
                    "--no-pager",
                    "--no-replace-objects",
                    "-c",
                    "core.attributesfile=/dev/null",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-C",
                    f"/proc/self/fd/{repository_descriptor}",
                    *arguments,
                ],
                check=False,
                capture_output=True,
                env=environment,
                executable=f"/proc/self/fd/{git_stream.fileno()}",
                pass_fds=(repository_descriptor, git_stream.fileno()),
                timeout=30,
            )
    except (OSError, subprocess.SubprocessError) as error:
        raise BenchmarkValidationError(f"cannot query {label}") from error
    if completed.returncode != 0:
        raise BenchmarkValidationError(f"{label} is not available from Git")
    if len(completed.stdout) > limit:
        raise BenchmarkValidationError(f"{label} exceeds the size limit")
    return completed.stdout


def _canonical_repo_path(value: Any, label: str) -> str:
    raw = _string(value, label)
    path = PurePosixPath(raw)
    if path.is_absolute() or raw != path.as_posix() or any(
        component in ("", ".", "..") for component in path.parts
    ):
        raise BenchmarkValidationError(f"{label} is not a canonical repo path")
    return raw


def _read_source_records(
    repository_root: Path,
    code_pin: str,
    files: Any,
    *,
    git_path: Path,
    git_sha256: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, ast.AST],
    dict[str, ast.Module],
]:
    if not isinstance(files, list) or not files:
        raise BenchmarkValidationError("source authority files must be non-empty")
    records: list[dict[str, Any]] = []
    file_ids: set[str] = set()
    symbol_ids: set[str] = set()
    symbol_nodes: dict[str, ast.AST] = {}
    file_modules: dict[str, ast.Module] = {}
    with _open_directory_no_symlinks(repository_root, "source repository") as repository:
        resolved_commit = _git_output(
            repository,
            git_path,
            git_sha256,
            ["rev-parse", "--verify", f"{code_pin}^{{commit}}"],
            "source commit",
            limit=1024,
        ).decode("ascii", errors="strict").strip()
        if resolved_commit != code_pin:
            raise BenchmarkValidationError("source commit authority drifted")
        for index, item in enumerate(files):
            if not isinstance(item, dict):
                raise BenchmarkValidationError(f"source file {index} must be an object")
            _exact_keys(
                item,
                {"id", "repo_path", "sha256", "symbols"},
                f"source file {index}",
            )
            source_id = _identifier(item["id"], f"source file {index} id")
            if source_id in file_ids:
                raise BenchmarkValidationError("source file id is duplicated")
            file_ids.add(source_id)
            expected_sha = _sha(item["sha256"], f"source file {source_id} SHA-256")
            repo_path = _canonical_repo_path(
                item["repo_path"], f"source file {source_id} repo path"
            )
            tree_entry = _git_output(
                repository,
                git_path,
                git_sha256,
                ["ls-tree", "-z", code_pin, "--", repo_path],
                f"source file {source_id} tree entry",
                limit=4096,
            )
            try:
                metadata, listed_path = tree_entry[:-1].split(b"\t", 1)
                mode, object_type, raw_object_id = metadata.split(b" ", 2)
                listed = listed_path.decode("utf-8", errors="strict")
                object_id = raw_object_id.decode("ascii", errors="strict")
            except (UnicodeDecodeError, ValueError) as error:
                raise BenchmarkValidationError(
                    f"source file tree entry is invalid: {source_id}"
                ) from error
            if (
                not tree_entry.endswith(b"\0")
                or tree_entry.count(b"\0") != 1
                or listed != repo_path
                or mode not in {b"100644", b"100755"}
                or object_type != b"blob"
                or _GIT_OBJECT_ID.fullmatch(object_id) is None
            ):
                raise BenchmarkValidationError(
                    f"source file is not a committed regular blob: {source_id}"
                )
            raw_size = _git_output(
                repository,
                git_path,
                git_sha256,
                ["cat-file", "-s", object_id],
                f"source file {source_id} size",
                limit=1024,
            )
            try:
                size = int(raw_size.decode("ascii", errors="strict").strip())
            except (UnicodeDecodeError, ValueError) as error:
                raise BenchmarkValidationError(
                    f"source file size is invalid: {source_id}"
                ) from error
            if size < 0 or size > _MAX_SOURCE_BYTES:
                raise BenchmarkValidationError(f"source file is too large: {source_id}")
            raw = _git_output(
                repository,
                git_path,
                git_sha256,
                ["cat-file", "blob", object_id],
                f"source file {source_id}",
                limit=_MAX_SOURCE_BYTES,
            )
            if len(raw) != size or sha256(raw).hexdigest() != expected_sha:
                raise BenchmarkValidationError(
                    f"source file authority drifted: {source_id}"
                )
            try:
                tree = ast.parse(raw.decode("utf-8"))
            except (UnicodeDecodeError, SyntaxError) as error:
                raise BenchmarkValidationError(
                    f"source file is not valid Python: {source_id}"
                ) from error
            file_modules[source_id] = tree
            symbols = item["symbols"]
            if not isinstance(symbols, list) or not symbols:
                raise BenchmarkValidationError(f"source symbols are absent: {source_id}")
            symbol_records: list[dict[str, str]] = []
            for symbol_index, symbol in enumerate(symbols):
                if not isinstance(symbol, dict):
                    raise BenchmarkValidationError("source symbol must be an object")
                _exact_keys(
                    symbol,
                    {"ast_sha256", "qualified_name"},
                    f"source symbol {source_id}[{symbol_index}]",
                )
                qualified_name = _string(
                    symbol["qualified_name"],
                    f"source symbol {source_id}[{symbol_index}] name",
                )
                canonical_id = f"{source_id}:{qualified_name}"
                if canonical_id in symbol_ids:
                    raise BenchmarkValidationError("source symbol is duplicated")
                symbol_ids.add(canonical_id)
                expected_ast = _sha(
                    symbol["ast_sha256"],
                    f"source symbol {canonical_id} AST SHA-256",
                )
                node = _qualified_ast_node(tree, qualified_name)
                observed_ast = sha256(
                    ast.dump(
                        node, annotate_fields=True, include_attributes=False
                    ).encode("utf-8")
                ).hexdigest()
                if observed_ast != expected_ast:
                    raise BenchmarkValidationError(
                        f"source symbol AST drifted: {canonical_id}"
                    )
                symbol_nodes[canonical_id] = node
                symbol_records.append(
                    {"ast_sha256": observed_ast, "qualified_name": qualified_name}
                )
            records.append(
                {
                    "git_object_id": object_id,
                    "id": source_id,
                    "repo_path": repo_path,
                    "sha256": expected_sha,
                    "symbols": sorted(
                        symbol_records, key=lambda item: item["qualified_name"]
                    ),
                }
            )
    return (
        sorted(records, key=lambda item: item["id"]),
        symbol_nodes,
        file_modules,
    )


def _verify_source_authority(
    value: Any,
    base: Path,
    *,
    candidate_id: str,
    frontier_action: str,
    mechanism_fingerprint_sha256: str,
    normal_form: Mapping[str, Any],
    locality: Mapping[str, Any],
    implementation: Mapping[str, Any],
    accepted_semantics: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError("source authority must be an object")
    _exact_keys(
        value,
        {"files", "repository", "semantics_certificate"},
        "source authority",
    )
    repository = value["repository"]
    if not isinstance(repository, dict):
        raise BenchmarkValidationError("source repository authority must be an object")
    _exact_keys(repository, {"commit", "root"}, "source repository authority")
    code_pin = _code_pin(repository["commit"], "source authority code pin")
    repository_root = _resolve(base, repository["root"], "source repository root")
    files = value["files"]
    records, symbol_nodes, file_modules = _read_source_records(
        repository_root,
        code_pin,
        files,
        git_path=Path(implementation["git_path"]),
        git_sha256=implementation["git_sha256"],
    )
    source_set_payload = {"code_pin": code_pin, "files": records}
    source_set_sha = sha256(
        _canonical_json(source_set_payload).encode("ascii")
    ).hexdigest()

    certificate_path, certificate_sha, certificate_raw = _binding(
        base,
        value["semantics_certificate"],
        "source semantics certificate",
        limit=_MAX_JSON_BYTES,
    )
    certificate = _load_json(certificate_raw, "source semantics certificate")
    _exact_keys(
        certificate,
        {
            "accepted_authority_sha256",
            "arithmetic_contract",
            "authority_kind",
            "candidate_id",
            "candidate_callsite_symbol",
            "candidate_semantic_sha256",
            "candidate_symbol",
            "causal_frontier_action",
            "claim_scope",
            "code_pin",
            "compensation_claim",
            "locality",
            "mechanism_fingerprint_sha256",
            "normal_form",
            "schema_version",
            "source_set_sha256",
        },
        "source semantics certificate",
    )
    if certificate["schema_version"] != GATE_D_PRECOMPILE_ADMISSION_SCHEMA_VERSION:
        raise BenchmarkValidationError("source semantics certificate schema drifted")
    authority_kind = _identifier(
        certificate["authority_kind"], "source authority kind"
    )
    if authority_kind not in {
        "declarative.semantic.dsl",
        "concrete.committed.jax.source",
    }:
        raise BenchmarkValidationError("source authority kind is unsupported")
    if (
        certificate["candidate_id"] != candidate_id
        or certificate["causal_frontier_action"] != frontier_action
        or certificate["code_pin"] != code_pin
        or certificate["mechanism_fingerprint_sha256"]
        != mechanism_fingerprint_sha256
        or certificate["source_set_sha256"] != source_set_sha
        or certificate["normal_form"] != dict(normal_form)
        or certificate["accepted_authority_sha256"]
        != accepted_semantics["authority_sha256"]
    ):
        raise BenchmarkValidationError("source semantics certificate authority drifted")
    available_symbols = {
        f"{record['id']}:{symbol['qualified_name']}": symbol["ast_sha256"]
        for record in records
        for symbol in record["symbols"]
    }
    candidate_symbol = _string(
        certificate["candidate_symbol"], "candidate source symbol"
    )
    callsite_symbol = _string(
        certificate["candidate_callsite_symbol"], "candidate callsite symbol"
    )
    if (
        candidate_symbol not in available_symbols
        or callsite_symbol not in available_symbols
        or callsite_symbol == candidate_symbol
    ):
        raise BenchmarkValidationError("source symbol authority is incomplete")
    if authority_kind == "concrete.committed.jax.source":
        if (
            candidate_id != "auxiliary_device_tuple_dependency"
            or candidate_symbol
            != "candidate.source:fused_add_rms_norm_with_auxiliary"
            or callsite_symbol
            != "candidate.callsite:stage_local_transformer_layer_fp8_split_mapped"
        ):
            raise BenchmarkValidationError("concrete source identity drifted")
        if (
            code_pin != _EXPECTED_CONCRETE_TUPLE_SOURCE["code_pin"]
            or source_set_sha
            != _EXPECTED_CONCRETE_TUPLE_SOURCE["source_set_sha256"]
            or certificate_sha
            != _EXPECTED_CONCRETE_TUPLE_SOURCE["certificate_sha256"]
            or available_symbols[candidate_symbol]
            != _EXPECTED_CONCRETE_TUPLE_SOURCE["candidate_ast_sha256"]
            or available_symbols[callsite_symbol]
            != _EXPECTED_CONCRETE_TUPLE_SOURCE["callsite_ast_sha256"]
        ):
            raise BenchmarkValidationError("concrete source reviewed authority drifted")
        source_semantics = _concrete_tuple_source_semantics(
            symbol_nodes, file_modules
        )
    else:
        source_semantics = _candidate_source_semantics(
            symbol_nodes[candidate_symbol], symbol_nodes[callsite_symbol], candidate_id
        )
    source_semantic_sha = sha256(
        _canonical_json(source_semantics).encode("ascii")
    ).hexdigest()
    if certificate["candidate_semantic_sha256"] != source_semantic_sha:
        raise BenchmarkValidationError("candidate source/HLO semantics binding drifted")
    if (
        authority_kind == "concrete.committed.jax.source"
        and source_semantic_sha
        != _EXPECTED_CONCRETE_TUPLE_SOURCE["semantic_sha256"]
    ):
        raise BenchmarkValidationError("concrete source reviewed semantics drifted")
    claim_scope = _string(
        certificate["claim_scope"], "source semantics certificate claim scope"
    )
    expected_compensation_claim = (
        "graph.identity.only;numeric.cancellation.unproven"
        if candidate_id == "compensated_auxiliary_dependency"
        else "none"
    )
    if certificate["compensation_claim"] != expected_compensation_claim:
        raise BenchmarkValidationError("source compensation claim is overstated")
    _verify_locality(certificate["locality"], locality, "source semantics locality")
    arithmetic = certificate["arithmetic_contract"]
    if not isinstance(arithmetic, dict):
        raise BenchmarkValidationError("source arithmetic contract must be an object")
    _exact_keys(
        arithmetic,
        {
            "auxiliary_affects_primary_arithmetic",
            "normalization_input",
            "primary_outputs_bitwise_identical_by_construction",
            "primary_recurrence",
        },
        "source arithmetic contract",
    )
    if (
        _boolean(
            arithmetic["auxiliary_affects_primary_arithmetic"],
            "auxiliary arithmetic effect",
        )
        is not False
        or _boolean(
            arithmetic["primary_outputs_bitwise_identical_by_construction"],
            "primary bitwise construction",
        )
        is not True
        or arithmetic["normalization_input"] != "transient.fp32.sum"
        or arithmetic["primary_recurrence"] != "bf16.rounded"
    ):
        raise BenchmarkValidationError("source arithmetic contract is not accepted-preserving")
    report = {
        "certificate_path": str(certificate_path),
        "certificate_sha256": certificate_sha,
        "code_pin": code_pin,
        "accepted_authority_sha256": accepted_semantics["authority_sha256"],
        "authority_kind": authority_kind,
        "candidate_symbol": {
            "ast_sha256": available_symbols[candidate_symbol],
            "id": candidate_symbol,
        },
        "callsite_symbol": {
            "ast_sha256": available_symbols[callsite_symbol],
            "id": callsite_symbol,
        },
        "certificate_claim_scope": claim_scope,
        "compensation_claim": expected_compensation_claim,
        "executable_source_authority": (
            authority_kind == "concrete.committed.jax.source"
        ),
        "files": records,
        "mechanism_fingerprint_sha256": mechanism_fingerprint_sha256,
        "repository_root": str(repository_root),
        "source_semantic_sha256": source_semantic_sha,
        "source_set_sha256": source_set_sha,
        "validation_scope": (
            "concrete committed JAX operators and real default-off device caller"
            if authority_kind == "concrete.committed.jax.source"
            else "declarative semantic DSL and consumer-edge fixture only; "
            "abstract operators and the real integration caller are not bound"
        ),
    }
    report["authority_sha256"] = sha256(
        _canonical_json(
            {
                "certificate_sha256": certificate_sha,
                "code_pin": code_pin,
                "accepted_authority_sha256": accepted_semantics[
                    "authority_sha256"
                ],
                "authority_kind": authority_kind,
                "candidate_ast_sha256": available_symbols[candidate_symbol],
                "callsite_ast_sha256": available_symbols[callsite_symbol],
                "compensation_claim": expected_compensation_claim,
                "mechanism_fingerprint_sha256": mechanism_fingerprint_sha256,
                "source_semantic_sha256": source_semantic_sha,
                "source_set_sha256": source_set_sha,
            }
        ).encode("ascii")
    ).hexdigest()
    return report


def _verify_plan_authority(
    value: Any,
    base: Path,
    physical_locality: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError("plan authority must be an object")
    path, expected_file, raw = _binding(
        base, value, "plan authority", limit=_MAX_JSON_BYTES
    )
    document = _load_json(
        raw, "plan authority"
    )
    _exact_keys(
        document,
        {
            "local_device_groups",
            "plan",
            "plan_sha256",
            "schema_version",
            "topology_hash",
            "watchpoints",
        },
        "plan authority document",
    )
    if document["schema_version"] != GATE_D_PRECOMPILE_ADMISSION_SCHEMA_VERSION:
        raise BenchmarkValidationError("plan authority schema drifted")
    expected_plan = _sha(document["plan_sha256"], "plan SHA-256")
    plan_name = _string(document["plan"], "plan name")
    local_groups = _device_groups(document["local_device_groups"], "plan local groups")
    if (
        document["topology_hash"] != physical_locality["topology_hash"]
        or plan_name not in physical_locality["plans"]
        or local_groups != physical_locality["plans"][plan_name]
    ):
        raise BenchmarkValidationError(
            "plan groups are not the sealed runtime physical allowlist"
        )
    local_group_size = 2 if plan_name == "PP16_LP2" else 4
    expected_layout = "lp2.local" if plan_name == "PP16_LP2" else "lp4.local"
    if any(len(group) != local_group_size for group in local_groups):
        raise BenchmarkValidationError("plan local group size drifted")
    watchpoints = document["watchpoints"]
    if not isinstance(watchpoints, dict) or set(watchpoints) != set(
        _REQUIRED_WATCHPOINT_SCHEMA
    ):
        raise BenchmarkValidationError("plan watchpoint catalogue drifted")
    normalized_watchpoints: dict[str, dict[str, Any]] = {}
    sealed_owner_group: list[int] | None = None
    for watchpoint_id, item in watchpoints.items():
        if not isinstance(item, dict):
            raise BenchmarkValidationError(
                f"plan watchpoint is invalid: {watchpoint_id}"
            )
        _exact_keys(item, {"layout", "owner_ids"}, f"plan watchpoint {watchpoint_id}")
        layout = _identifier(item["layout"], f"plan watchpoint {watchpoint_id} layout")
        if layout != expected_layout:
            raise BenchmarkValidationError(
                f"plan watchpoint layout is not exact {plan_name} authority: {watchpoint_id}"
            )
        owner_ids = item["owner_ids"]
        if (
            not isinstance(owner_ids, list)
            or len(owner_ids) != local_group_size
            or any(
                not isinstance(owner, int)
                or isinstance(owner, bool)
                or owner < 0
                or owner > 31
                for owner in owner_ids
            )
            or len(set(owner_ids)) != local_group_size
            or owner_ids not in local_groups
        ):
            raise BenchmarkValidationError(
                f"plan watchpoint owners are not one exact {plan_name} group: {watchpoint_id}"
            )
        if sealed_owner_group is None:
            sealed_owner_group = owner_ids
        elif owner_ids != sealed_owner_group:
            raise BenchmarkValidationError(
                "plan watchpoints do not share one exact local owner group"
            )
        normalized_watchpoints[watchpoint_id] = {
            "layout": layout,
            "owner_ids": owner_ids,
        }
    if sealed_owner_group != local_groups[0]:
        raise BenchmarkValidationError(
            f"plan watchpoints are not the exact {plan_name} stage-zero owner group"
        )
    payload = {
        "local_device_groups": local_groups,
        "plan": plan_name,
        "topology_hash": physical_locality["topology_hash"],
        "watchpoints": normalized_watchpoints,
    }
    if sha256(_canonical_json(payload).encode("ascii")).hexdigest() != expected_plan:
        raise BenchmarkValidationError("plan SHA-256 is not content-derived")
    return {
        "authority_file_sha256": expected_file,
        "authority_path": str(path),
        "local_device_groups": local_groups,
        "local_group_size": local_group_size,
        "owner_group": sealed_owner_group,
        "plan": plan_name,
        "plan_sha256": expected_plan,
        "topology_hash": physical_locality["topology_hash"],
        "watchpoints": normalized_watchpoints,
        "watchpoint_schema": _required_watchpoint_schema(local_group_size),
    }


def _sealed_parser_memfd(
    source: Path,
    *,
    logical_path: str,
    expected_bytes: int,
    expected_sha256: str,
) -> int:
    if not hasattr(os, "memfd_create") or not hasattr(os, "MFD_ALLOW_SEALING"):
        raise BenchmarkValidationError("sealed parser memfd is unavailable")
    descriptor = os.memfd_create(
        f"glm-gate-d-parser:{logical_path}",
        os.MFD_ALLOW_SEALING | getattr(os, "MFD_CLOEXEC", 0),
    )
    digest = sha256()
    observed = 0
    try:
        with _open_regular_file(source, f"parser import {source.name}") as reader:
            while block := reader.read(1024 * 1024):
                observed += len(block)
                if observed > expected_bytes:
                    raise BenchmarkValidationError("parser import byte count drifted")
                digest.update(block)
                view = memoryview(block)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
        if observed != expected_bytes or digest.hexdigest() != expected_sha256:
            raise BenchmarkValidationError("parser import manifest drifted")
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(descriptor, _F_ADD_SEALS, _PARSER_MEMFD_SEALS)
        if fcntl.fcntl(descriptor, _F_GET_SEALS) != _PARSER_MEMFD_SEALS:
            raise BenchmarkValidationError("parser memfd seal set drifted")
        sealed_digest = sha256()
        sealed_bytes = 0
        while sealed_bytes < expected_bytes:
            block = os.pread(
                descriptor,
                min(1024 * 1024, expected_bytes - sealed_bytes),
                sealed_bytes,
            )
            if not block:
                break
            sealed_digest.update(block)
            sealed_bytes += len(block)
        if (
            sealed_bytes != expected_bytes
            or os.pread(descriptor, 1, sealed_bytes)
            or sealed_digest.hexdigest() != expected_sha256
        ):
            raise BenchmarkValidationError("sealed parser memfd revalidation drifted")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def _sealed_parser_memfds(
    implementation: Mapping[str, Any],
) -> Iterator[tuple[dict[str, dict[str, Any]], tuple[int, ...]]]:
    source_root = Path(implementation["validator_pythonpath"])
    bindings: dict[str, dict[str, Any]] = {}
    descriptors: list[int] = []
    try:
        for item in implementation["validator_imports"]:
            relative = PurePosixPath(item["path"])
            logical_path = relative.as_posix()
            descriptor = _sealed_parser_memfd(
                source_root.joinpath(*relative.parts),
                logical_path=logical_path,
                expected_bytes=item["bytes"],
                expected_sha256=item["sha256"],
            )
            descriptors.append(descriptor)
            bindings[logical_path] = {
                "bytes": item["bytes"],
                "fd": descriptor,
                "sha256": item["sha256"],
            }
        yield bindings, tuple(descriptors)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _verify_immutable_runtime_file_record(
    value: Any,
    *,
    label: str,
    beneath: Path,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{label} record is invalid")
    _exact_keys(
        value,
        {
            "bytes",
            "device_major",
            "device_minor",
            "inode",
            "path",
            "sha256",
        },
        f"{label} record",
    )
    path = Path(value["path"])
    expected_bytes = _nonnegative_int(value["bytes"], f"{label} bytes")
    expected_major = _nonnegative_int(
        value["device_major"], f"{label} device major"
    )
    expected_minor = _nonnegative_int(
        value["device_minor"], f"{label} device minor"
    )
    expected_inode = _positive_int(value["inode"], f"{label} inode")
    expected_sha = _sha(value["sha256"], f"{label} SHA-256")
    _require_root_owned_immutable_file(path, label, beneath=beneath)
    digest = sha256()
    with _open_regular_file(path, label) as stream:
        metadata = os.fstat(stream.fileno())
        while block := stream.read(1024 * 1024):
            digest.update(block)
    if (
        metadata.st_size != expected_bytes
        or os.major(metadata.st_dev) != expected_major
        or os.minor(metadata.st_dev) != expected_minor
        or metadata.st_ino != expected_inode
        or digest.hexdigest() != expected_sha
    ):
        raise BenchmarkValidationError(f"{label} identity drifted")
    return {
        "bytes": expected_bytes,
        "path": str(path),
        "sha256": expected_sha,
    }


def _verify_parser_runtime_authority(
    value: Any, implementation: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError("parser runtime authority is invalid")
    _exact_keys(
        value,
        {
            "immutable_native_dependencies",
            "memfd_seal_mask",
            "native_mapped_paths",
            "new_native_mapping_count",
            "python_executable",
            "python_loaded_files",
            "process_uid",
            "python_runtime_root",
            "python_runtime_tree_sha256",
            "python_search_path",
            "sealed_native_mapping_count",
            "source_loaded_paths",
        },
        "parser runtime authority",
    )
    runtime_root = Path(implementation["validator_python_runtime_root"])
    if (
        value["memfd_seal_mask"] != _PARSER_MEMFD_SEALS
        or value["python_runtime_root"] != str(runtime_root)
        or value["python_runtime_tree_sha256"]
        != implementation["validator_python_runtime_sha256"]
        or value["native_mapped_paths"]
        != sorted(
            item["path"]
            for item in implementation["validator_imports"]
            if item["path"].endswith(".so")
        )
        or value["source_loaded_paths"]
        != sorted(
            item["path"]
            for item in implementation["validator_imports"]
            if item["path"].endswith(".py")
        )
    ):
        raise BenchmarkValidationError("parser runtime authority drifted")
    process_identity = _normalize_bound_nonroot_uid(
        value["process_uid"], os.geteuid()
    )
    search_path = value["python_search_path"]
    if not isinstance(search_path, list) or not search_path:
        raise BenchmarkValidationError("isolated Python search path is absent")
    for raw_path in search_path:
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise BenchmarkValidationError("isolated Python search path is invalid")
        candidate = Path(os.path.realpath(raw_path))
        parent = candidate if candidate.is_dir() else candidate.parent
        try:
            parent.relative_to(runtime_root)
        except ValueError as error:
            raise BenchmarkValidationError(
                "isolated Python search path escapes runtime root"
            ) from error
    executable = _verify_immutable_runtime_file_record(
        value["python_executable"],
        label="parser Python executable",
        beneath=runtime_root,
    )
    if (
        executable["path"] != implementation["validator_python_path"]
        or executable["sha256"] != implementation["validator_python_sha256"]
    ):
        raise BenchmarkValidationError("parser Python executable binding drifted")
    loaded = value["python_loaded_files"]
    if not isinstance(loaded, list) or not loaded:
        raise BenchmarkValidationError("parser Python loaded-file authority is absent")
    loaded_paths: set[str] = set()
    normalized_loaded: list[dict[str, Any]] = []
    for index, item in enumerate(loaded):
        record = _verify_immutable_runtime_file_record(
            item,
            label=f"parser Python loaded file {index}",
            beneath=runtime_root,
        )
        if record["path"] in loaded_paths:
            raise BenchmarkValidationError("parser Python loaded file is duplicated")
        loaded_paths.add(record["path"])
        normalized_loaded.append(record)
    dependencies = value["immutable_native_dependencies"]
    if not isinstance(dependencies, list) or not dependencies:
        raise BenchmarkValidationError("immutable native dependencies are absent")
    dependency_paths: set[str] = set()
    normalized_dependencies: list[dict[str, Any]] = []
    for index, item in enumerate(dependencies):
        record = _verify_immutable_runtime_file_record(
            item,
            label=f"immutable native dependency {index}",
            beneath=Path("/"),
        )
        if record["path"] in dependency_paths:
            raise BenchmarkValidationError("immutable native dependency is duplicated")
        dependency_paths.add(record["path"])
        normalized_dependencies.append(record)
    new_mapping_count = value["new_native_mapping_count"]
    if (
        not isinstance(new_mapping_count, int)
        or isinstance(new_mapping_count, bool)
        or new_mapping_count <= 0
        or value["sealed_native_mapping_count"]
        != sum(
            item["path"].endswith(".so")
            for item in implementation["validator_imports"]
        )
    ):
        raise BenchmarkValidationError("native mapping counts are invalid")
    return {
        "immutable_native_dependencies": normalized_dependencies,
        "memfd_seal_mask": value["memfd_seal_mask"],
        "native_mapped_paths": list(value["native_mapped_paths"]),
        "new_native_mapping_count": new_mapping_count,
        "process_identity": process_identity,
        "python_executable": executable,
        "python_loaded_files": normalized_loaded,
        "python_runtime_root": value["python_runtime_root"],
        "python_runtime_tree_sha256": value["python_runtime_tree_sha256"],
        "python_search_path": list(search_path),
        "sealed_native_mapping_count": value["sealed_native_mapping_count"],
        "source_loaded_paths": list(value["source_loaded_paths"]),
    }


def _normalize_bound_nonroot_uid(value: Any, expected: int) -> str:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        or not isinstance(expected, int)
        or isinstance(expected, bool)
        or expected <= 0
        or value != expected
    ):
        raise BenchmarkValidationError("parser process UID binding drifted")
    return "bound_nonroot"


def _run_stablehlo_validator(
    request: Mapping[str, Any], implementation: Mapping[str, Any]
) -> dict[str, Any]:
    validator_raw = _snapshot(
        Path(implementation["stablehlo_validator_path"]),
        implementation["stablehlo_validator_sha256"],
        "StableHLO validator",
        limit=_MAX_SOURCE_BYTES,
    )
    try:
        validator_source = validator_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BenchmarkValidationError("StableHLO validator is not UTF-8") from error
    payload = (_canonical_json(request) + "\n").encode("ascii")
    python_path = Path(implementation["validator_python_path"])
    try:
        with _sealed_parser_memfds(implementation) as (
            parser_bindings,
            parser_descriptors,
        ), _open_regular_file(
            python_path, "StableHLO validator Python"
        ) as stream:
            digest = sha256()
            while block := stream.read(1024 * 1024):
                digest.update(block)
            if digest.hexdigest() != implementation["validator_python_sha256"]:
                raise BenchmarkValidationError(
                    "StableHLO validator Python SHA-256 drifted"
                )
            environment = {
                "GATE_D_VALIDATOR_FDS": _canonical_json(parser_bindings),
                "GATE_D_VALIDATOR_PYTHON_FD": str(stream.fileno()),
                "GATE_D_VALIDATOR_PYTHON_SHA256": implementation[
                    "validator_python_sha256"
                ],
                "GATE_D_VALIDATOR_RUNTIME_ROOT": implementation[
                    "validator_python_runtime_root"
                ],
                "GATE_D_VALIDATOR_RUNTIME_SHA256": implementation[
                    "validator_python_runtime_sha256"
                ],
                "GATE_D_VALIDATOR_UID": str(os.geteuid()),
                "LANG": "C",
                "LC_ALL": "C",
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            completed = subprocess.run(
                [str(python_path), "-I", "-S", "-c", validator_source],
                check=False,
                capture_output=True,
                cwd="/",
                env=environment,
                executable=f"/proc/self/fd/{stream.fileno()}",
                input=payload,
                pass_fds=(
                    *parser_descriptors,
                    stream.fileno(),
                ),
                timeout=30,
            )
    except (OSError, subprocess.SubprocessError) as error:
        raise BenchmarkValidationError("cannot run StableHLO validator") from error
    if len(completed.stdout) > _MAX_JSON_BYTES or len(completed.stderr) > _MAX_JSON_BYTES:
        raise BenchmarkValidationError("StableHLO validator output is too large")
    if completed.returncode != 0:
        refusal = completed.stderr.decode("utf-8", errors="replace").strip()
        raise BenchmarkValidationError(
            f"StableHLO parser/causal validation failed: {refusal}"
        )
    report = _load_json(completed.stdout, "StableHLO validator output")
    if (
        report.get("parser") != "jaxlib.mlir.ir"
        or report.get("jaxlib_version") != _EXPECTED_JAXLIB_VERSION
        or report.get("stablehlo_version") != _EXPECTED_STABLEHLO_VERSION
        or report.get("jax_imported") is not False
        or report.get("loaded_parser_files")
        != {
            item["path"]: item["sha256"]
            for item in implementation["validator_imports"]
        }
        or report.get("immutable_parser_authority") is not True
        or report.get("parser_authority_scope")
        != "root-owned isolated Python plus sealed-memfd exact-fd parser and "
        "mapped-inode native authority"
    ):
        raise BenchmarkValidationError("StableHLO parser authority drifted")
    report["parser_runtime_authority"] = _verify_parser_runtime_authority(
        report.get("parser_runtime_authority"), implementation
    )
    return report


def _annotate_lowered_candidate(
    raw: bytes, source: Mapping[str, Any]
) -> tuple[bytes, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BenchmarkValidationError("raw candidate StableHLO is not UTF-8") from error
    metadata = {
        "gate_d.callsite_ast_sha256": source["callsite_symbol"]["ast_sha256"],
        "gate_d.candidate_ast_sha256": source["candidate_symbol"]["ast_sha256"],
        "gate_d.source_set_sha256": source["source_set_sha256"],
    }
    fragment = ", ".join(
        f'{key} = "{value}"' for key, value in sorted(metadata.items())
    )
    first_line, separator, remainder = text.partition("\n")
    if not separator or not first_line.startswith("module"):
        raise BenchmarkValidationError("raw candidate module header is not canonical")
    marker = " attributes {"
    if marker in first_line:
        annotated_first = first_line.replace(marker, f"{marker}{fragment}, ", 1)
        inverse_first = annotated_first.replace(f"{marker}{fragment}, ", marker, 1)
    else:
        brace = first_line.find(" {")
        if brace < 0:
            raise BenchmarkValidationError("raw candidate module header is unsupported")
        annotated_first = (
            first_line[:brace]
            + f" attributes {{{fragment}}}"
            + first_line[brace:]
        )
        inverse_first = annotated_first.replace(
            f" attributes {{{fragment}}}", "", 1
        )
    if inverse_first != first_line:
        raise BenchmarkValidationError("candidate metadata transform is not reversible")
    return (
        (annotated_first + separator + remainder).encode("utf-8"),
        sha256(fragment.encode("ascii")).hexdigest(),
    )


def _receipt_dependency_records(
    value: Any,
    *,
    label: str,
    allowed_roots: Sequence[str],
    producer_path: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise BenchmarkValidationError(f"{label} must be a non-empty list")
    records: list[dict[str, Any]] = []
    paths: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise BenchmarkValidationError(f"{label}[{index}] must be an object")
        _exact_keys(item, {"bytes", "path", "sha256"}, f"{label}[{index}]")
        path = _string(item["path"], f"{label}[{index}] path")
        if (
            not path.startswith("/")
            or path.startswith("//")
            or os.path.normpath(path) != path
            or any(part in {".", ".."} for part in Path(path).parts)
            or path.endswith(" (deleted)")
            or path in paths
            or "libtpu" in path.lower()
            or "/jax_plugins/" in path.lower()
        ):
            raise BenchmarkValidationError(f"{label} path is unsafe or duplicated")
        if path != producer_path and not any(
            path == root or path.startswith(f"{root}/") for root in allowed_roots
        ):
            raise BenchmarkValidationError(f"{label} path escaped immutable roots")
        paths.add(path)
        records.append(
            {
                "bytes": _nonnegative_int(item["bytes"], f"{label}[{index}] bytes"),
                "path": path,
                "sha256": _sha(item["sha256"], f"{label}[{index}] SHA-256"),
            }
        )
    return records


def _verify_lowering_receipt(
    *,
    accepted_raw: bytes,
    accepted_sha: str,
    base: Path,
    candidate_annotated_raw: bytes,
    candidate_annotated_sha: str,
    candidate_raw: bytes,
    candidate_raw_sha: str,
    implementation: Mapping[str, Any],
    plan: Mapping[str, Any],
    producer_receipt_raw: bytes,
    producer_receipt_sha: str,
    producer_repository: Any,
    producer_source_path: Path,
    producer_source_raw: bytes,
    producer_source_sha: str,
    source: Mapping[str, Any],
    success_raw: bytes,
    success_sha: str,
) -> dict[str, Any]:
    receipt = _load_json(producer_receipt_raw, "StableHLO producer receipt")
    if producer_receipt_raw != (_canonical_json(receipt) + "\n").encode("ascii"):
        raise BenchmarkValidationError("StableHLO producer receipt is not canonical JSON")
    _exact_keys(
        receipt,
        {
            "artifacts",
            "backend",
            "claim_scope",
            "environment",
            "inputs",
            "loaded_dependencies",
            "metadata_annotation_sha256",
            "plan_authority_sha256",
            "producer",
            "schema_version",
            "source",
        },
        "StableHLO producer receipt",
    )
    if receipt["schema_version"] != 1:
        raise BenchmarkValidationError("StableHLO producer receipt schema drifted")
    artifacts = receipt["artifacts"]
    if not isinstance(artifacts, dict):
        raise BenchmarkValidationError("StableHLO producer artifacts must be an object")
    _exact_keys(
        artifacts,
        {
            "accepted.raw.stablehlo",
            "candidate.raw.stablehlo",
            "candidate.stablehlo",
        },
        "StableHLO producer artifacts",
    )
    expected_artifacts = {
        "accepted.raw.stablehlo": (accepted_raw, accepted_sha),
        "candidate.raw.stablehlo": (candidate_raw, candidate_raw_sha),
        "candidate.stablehlo": (
            candidate_annotated_raw,
            candidate_annotated_sha,
        ),
    }
    for name, (raw, digest) in expected_artifacts.items():
        record = artifacts[name]
        if not isinstance(record, dict):
            raise BenchmarkValidationError(f"producer artifact {name} is invalid")
        _exact_keys(record, {"bytes", "sha256"}, f"producer artifact {name}")
        if (
            _positive_int(record["bytes"], f"producer artifact {name} bytes")
            != len(raw)
            or _sha(record["sha256"], f"producer artifact {name} SHA-256")
            != digest
        ):
            raise BenchmarkValidationError(f"producer artifact {name} drifted")
    backend = receipt["backend"]
    if backend != {"device_count": 1, "platform": "cpu"}:
        raise BenchmarkValidationError("StableHLO lowering backend was not forced CPU")
    if receipt["claim_scope"] != _EXPECTED_LOWERING_CLAIM_SCOPE:
        raise BenchmarkValidationError("StableHLO producer claim scope drifted")
    environment = receipt["environment"]
    if not isinstance(environment, dict):
        raise BenchmarkValidationError("StableHLO producer environment is invalid")
    _exact_keys(
        environment,
        set(_EXPECTED_LOWERING_ENVIRONMENT) | {"python_version"},
        "StableHLO producer environment",
    )
    if any(
        environment[key] != expected
        for key, expected in _EXPECTED_LOWERING_ENVIRONMENT.items()
    ):
        raise BenchmarkValidationError("StableHLO producer environment drifted")
    _string(environment["python_version"], "StableHLO producer Python version")
    if receipt["inputs"] != _EXPECTED_LOWERING_INPUTS:
        raise BenchmarkValidationError("StableHLO lowering inputs drifted")
    annotated, annotation_sha = _annotate_lowered_candidate(candidate_raw, source)
    if (
        annotated != candidate_annotated_raw
        or receipt["metadata_annotation_sha256"] != annotation_sha
    ):
        raise BenchmarkValidationError("candidate StableHLO annotation drifted")
    producer = receipt["producer"]
    if not isinstance(producer, dict):
        raise BenchmarkValidationError("StableHLO producer identity is invalid")
    _exact_keys(
        producer,
        {"code_pin", "installed_path", "sha256", "source_path"},
        "StableHLO producer identity",
    )
    producer_code_pin = _code_pin(producer["code_pin"], "producer code pin")
    if (
        producer["installed_path"] != _EXPECTED_LOWERING_PRODUCER_INSTALLED_PATH
        or producer["source_path"] != _EXPECTED_LOWERING_PRODUCER_SOURCE_PATH
        or _sha(producer["sha256"], "producer SHA-256") != producer_source_sha
    ):
        raise BenchmarkValidationError("StableHLO producer identity drifted")
    if not isinstance(producer_repository, dict):
        raise BenchmarkValidationError("producer repository authority must be an object")
    _exact_keys(
        producer_repository,
        {"commit", "root"},
        "producer repository authority",
    )
    if (
        _code_pin(producer_repository["commit"], "producer repository commit")
        != producer_code_pin
    ):
        raise BenchmarkValidationError("producer repository commit drifted")
    repository = _resolve(
        base, producer_repository["root"], "producer repository root"
    )
    expected_source_path = Path(
        os.path.abspath(repository / _EXPECTED_LOWERING_PRODUCER_SOURCE_PATH)
    )
    if Path(os.path.abspath(producer_source_path)) != expected_source_path:
        raise BenchmarkValidationError("producer source path drifted")
    with _open_directory_no_symlinks(repository, "producer repository") as repository_fd:
        commit = _git_output(
            repository_fd,
            Path(implementation["git_path"]),
            implementation["git_sha256"],
            ["rev-parse", "--verify", f"{producer_code_pin}^{{commit}}"],
            "producer commit",
            limit=1024,
        ).decode("ascii", errors="strict").strip()
        object_id = _git_output(
            repository_fd,
            Path(implementation["git_path"]),
            implementation["git_sha256"],
            [
                "rev-parse",
                "--verify",
                f"{producer_code_pin}:{_EXPECTED_LOWERING_PRODUCER_SOURCE_PATH}",
            ],
            "producer source object",
            limit=1024,
        ).decode("ascii", errors="strict").strip()
        object_type = _git_output(
            repository_fd,
            Path(implementation["git_path"]),
            implementation["git_sha256"],
            ["cat-file", "-t", object_id],
            "producer source object type",
            limit=1024,
        ).decode("ascii", errors="strict").strip()
        committed_source = _git_output(
            repository_fd,
            Path(implementation["git_path"]),
            implementation["git_sha256"],
            ["cat-file", "blob", object_id],
            "producer source blob",
            limit=_MAX_SOURCE_BYTES,
        )
    if (
        commit != producer_code_pin
        or object_type != "blob"
        or committed_source != producer_source_raw
        or sha256(committed_source).hexdigest() != producer_source_sha
    ):
        raise BenchmarkValidationError("committed producer source drifted")
    receipt_source = receipt["source"]
    if not isinstance(receipt_source, dict):
        raise BenchmarkValidationError("producer source tuple is invalid")
    _exact_keys(
        receipt_source,
        {
            "callsite_ast_sha256",
            "candidate_ast_sha256",
            "certificate_sha256",
            "code_pin",
            "files",
            "source_set_sha256",
        },
        "producer source tuple",
    )
    source_files = {
        item["repo_path"]: item["sha256"] for item in source["files"]
    }
    if (
        receipt_source["callsite_ast_sha256"]
        != source["callsite_symbol"]["ast_sha256"]
        or receipt_source["candidate_ast_sha256"]
        != source["candidate_symbol"]["ast_sha256"]
        or receipt_source["certificate_sha256"] != source["certificate_sha256"]
        or receipt_source["code_pin"] != source["code_pin"]
        or receipt_source["files"] != source_files
        or receipt_source["source_set_sha256"] != source["source_set_sha256"]
    ):
        raise BenchmarkValidationError("producer source tuple drifted")
    if receipt["plan_authority_sha256"] != plan["authority_file_sha256"]:
        raise BenchmarkValidationError("producer plan authority drifted")
    dependencies = receipt["loaded_dependencies"]
    if not isinstance(dependencies, dict):
        raise BenchmarkValidationError("producer dependencies must be an object")
    _exact_keys(
        dependencies,
        {"native_mappings", "python_modules"},
        "producer dependencies",
    )
    python_records = _receipt_dependency_records(
        dependencies["python_modules"],
        label="producer Python dependencies",
        allowed_roots=(
            environment["python_runtime_root"],
            environment["site_root"],
        ),
        producer_path=producer["installed_path"],
    )
    native_records = _receipt_dependency_records(
        dependencies["native_mappings"],
        label="producer native dependencies",
        allowed_roots=(
            environment["python_runtime_root"],
            environment["site_root"],
            "/usr/lib",
            "/lib",
        ),
        producer_path=producer["installed_path"],
    )
    if any(item["path"] == producer["installed_path"] for item in native_records):
        raise BenchmarkValidationError("producer source appeared as a native dependency")
    dependency_records = {
        "native_mappings": native_records,
        "python_modules": python_records,
    }
    dependency_manifest_sha256s: dict[str, str] = {}
    for name, records in dependency_records.items():
        digest = sha256(_canonical_json(records).encode("ascii")).hexdigest()
        expected = _EXPECTED_LOWERING_DEPENDENCY_MANIFESTS[name]
        if len(records) != expected["count"] or digest != expected["sha256"]:
            raise BenchmarkValidationError(
                f"producer {name.replace('_', ' ')} manifest drifted"
            )
        dependency_manifest_sha256s[name] = digest
    producer_records = [
        item for item in python_records if item["path"] == producer["installed_path"]
    ]
    if len(producer_records) != 1 or producer_records[0]["sha256"] != producer_source_sha:
        raise BenchmarkValidationError("loaded producer dependency identity drifted")
    success = _load_json(success_raw, "StableHLO producer SUCCESS")
    if success_raw != (_canonical_json(success) + "\n").encode("ascii"):
        raise BenchmarkValidationError("StableHLO producer SUCCESS is not canonical JSON")
    _exact_keys(
        success,
        {"producer_receipt_sha256", "schema_version"},
        "StableHLO producer SUCCESS",
    )
    if success != {
        "producer_receipt_sha256": producer_receipt_sha,
        "schema_version": 1,
    }:
        raise BenchmarkValidationError("StableHLO producer SUCCESS drifted")
    return {
        "backend": dict(backend),
        "candidate_raw_sha256": candidate_raw_sha,
        "dependency_manifest_sha256s": dependency_manifest_sha256s,
        "native_mapping_count": len(native_records),
        "producer_code_pin": producer_code_pin,
        "producer_receipt_sha256": producer_receipt_sha,
        "producer_source_sha256": producer_source_sha,
        "python_module_count": len(python_records),
        "success_sha256": success_sha,
    }


def _verify_stablehlo_authority(
    value: Any,
    base: Path,
    *,
    candidate_id: str,
    mechanism_fingerprint_sha256: str,
    normal_form: Mapping[str, Any],
    source: Mapping[str, Any],
    plan: Mapping[str, Any],
    frontier_id: str,
    frontier_action: str,
    locality: Mapping[str, Any],
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError("StableHLO authority must be an object")
    _exact_keys(
        value,
        {
            "accepted_primary",
            "candidate",
            "candidate_raw",
            "certificate",
            "producer_receipt",
            "producer_repository",
            "producer_source",
            "success",
        },
        "StableHLO authority",
    )
    candidate_path, candidate_sha, candidate_annotated_raw = _binding(
        base,
        value["candidate"],
        "candidate StableHLO",
        limit=_MAX_STABLEHLO_BYTES,
    )
    accepted_path, accepted_sha, accepted_raw = _binding(
        base,
        value["accepted_primary"],
        "accepted-primary StableHLO",
        limit=_MAX_STABLEHLO_BYTES,
    )
    candidate_raw_path, candidate_raw_sha, candidate_raw = _binding(
        base,
        value["candidate_raw"],
        "raw candidate StableHLO",
        limit=_MAX_STABLEHLO_BYTES,
    )
    producer_receipt_path, producer_receipt_sha, producer_receipt_raw = _binding(
        base,
        value["producer_receipt"],
        "StableHLO producer receipt",
        limit=_MAX_JSON_BYTES,
    )
    producer_source_path, producer_source_sha, producer_source_raw = _binding(
        base,
        value["producer_source"],
        "StableHLO producer source",
        limit=_MAX_SOURCE_BYTES,
    )
    success_path, success_sha, success_raw = _binding(
        base,
        value["success"],
        "StableHLO producer SUCCESS",
        limit=_MAX_JSON_BYTES,
    )
    certificate_path, certificate_sha, certificate_raw = _binding(
        base,
        value["certificate"],
        "StableHLO causal certificate",
        limit=_MAX_JSON_BYTES,
    )
    certificate = _load_json(certificate_raw, "StableHLO causal certificate")
    _exact_keys(
        certificate,
        {
            "accepted_primary_stablehlo_sha256",
            "candidate_id",
            "candidate_raw_stablehlo_sha256",
            "candidate_source_semantic_sha256",
            "candidate_stablehlo_sha256",
            "causal_frontier",
            "claim_scope",
            "code_pin",
            "locality",
            "mechanism_fingerprint_sha256",
            "normal_form",
            "plan_sha256",
            "producer_code_pin",
            "producer_receipt_sha256",
            "producer_source_sha256",
            "schema_version",
            "source_set_sha256",
            "success_sha256",
            "validator_contract",
        },
        "StableHLO causal certificate",
    )
    if certificate["schema_version"] != GATE_D_PRECOMPILE_ADMISSION_SCHEMA_VERSION:
        raise BenchmarkValidationError("StableHLO causal certificate schema drifted")
    if (
        certificate["accepted_primary_stablehlo_sha256"] != accepted_sha
        or certificate["candidate_id"] != candidate_id
        or certificate["candidate_raw_stablehlo_sha256"] != candidate_raw_sha
        or certificate["candidate_stablehlo_sha256"] != candidate_sha
        or certificate["candidate_source_semantic_sha256"]
        != source["source_semantic_sha256"]
        or certificate["code_pin"] != source["code_pin"]
        or certificate["mechanism_fingerprint_sha256"]
        != mechanism_fingerprint_sha256
        or certificate["normal_form"] != dict(normal_form)
        or certificate["plan_sha256"] != plan["plan_sha256"]
        or certificate["producer_receipt_sha256"] != producer_receipt_sha
        or certificate["producer_source_sha256"] != producer_source_sha
        or certificate["source_set_sha256"] != source["source_set_sha256"]
        or certificate["success_sha256"] != success_sha
    ):
        raise BenchmarkValidationError("StableHLO authority tuple drifted")
    lowering_receipt = _verify_lowering_receipt(
        accepted_raw=accepted_raw,
        accepted_sha=accepted_sha,
        base=base,
        candidate_annotated_raw=candidate_annotated_raw,
        candidate_annotated_sha=candidate_sha,
        candidate_raw=candidate_raw,
        candidate_raw_sha=candidate_raw_sha,
        implementation=implementation,
        plan=plan,
        producer_receipt_raw=producer_receipt_raw,
        producer_receipt_sha=producer_receipt_sha,
        producer_repository=value["producer_repository"],
        producer_source_path=producer_source_path,
        producer_source_raw=producer_source_raw,
        producer_source_sha=producer_source_sha,
        source=source,
        success_raw=success_raw,
        success_sha=success_sha,
    )
    if certificate["producer_code_pin"] != lowering_receipt["producer_code_pin"]:
        raise BenchmarkValidationError("StableHLO producer code pin drifted")
    if certificate["claim_scope"] != _EXPECTED_STABLEHLO_CERTIFICATE_CLAIM_SCOPE:
        raise BenchmarkValidationError("StableHLO certificate claim scope drifted")
    _verify_locality(certificate["locality"], locality, "StableHLO locality")
    frontier = certificate["causal_frontier"]
    if not isinstance(frontier, dict):
        raise BenchmarkValidationError("StableHLO causal frontier must be an object")
    _exact_keys(frontier, {"action", "id"}, "StableHLO causal frontier")
    if frontier != {"action": frontier_action, "id": frontier_id}:
        raise BenchmarkValidationError("StableHLO causal frontier drifted")
    validator_contract = certificate["validator_contract"]
    if not isinstance(validator_contract, dict):
        raise BenchmarkValidationError("StableHLO validator contract must be an object")
    _exact_keys(
        validator_contract,
        {
            "auxiliary_result_index",
            "carried_residual_result_index",
            "weighted_output_result_index",
        },
        "StableHLO validator contract",
    )
    indices = {
        key: _nonnegative_int(value, f"StableHLO {key}")
        for key, value in validator_contract.items()
    }
    request = {
        "accepted_stablehlo_base64": base64.b64encode(accepted_raw).decode("ascii"),
        "accepted_stablehlo_sha256": accepted_sha,
        "auxiliary_result_index": indices["auxiliary_result_index"],
        "callsite_ast_sha256": source["callsite_symbol"]["ast_sha256"],
        "candidate_ast_sha256": source["candidate_symbol"]["ast_sha256"],
        "candidate_stablehlo_base64": base64.b64encode(candidate_annotated_raw).decode("ascii"),
        "candidate_stablehlo_sha256": candidate_sha,
        "carried_residual_result_index": indices[
            "carried_residual_result_index"
        ],
        "expected_parser_files": {
            item["path"]: item["sha256"]
            for item in implementation["validator_imports"]
        },
        "local_device_groups": plan["local_device_groups"],
        "source_set_sha256": source["source_set_sha256"],
        "weighted_output_result_index": indices["weighted_output_result_index"],
    }
    validator = _run_stablehlo_validator(request, implementation)
    auxiliary_operations = validator.get("auxiliary_path_operations")
    expected_source_semantic_sha256 = (
        _EXPECTED_CONCRETE_TUPLE_SOURCE["semantic_sha256"]
        if candidate_id == "auxiliary_device_tuple_dependency"
        and source.get("executable_source_authority") is True
        else _EXPECTED_SOURCE_SEMANTIC_SHA256[candidate_id]
    )
    if (
        validator.get("accepted_stablehlo_sha256") != accepted_sha
        or validator.get("candidate_stablehlo_sha256") != candidate_sha
        or validator.get("local_device_groups") != plan["local_device_groups"]
        or validator.get("weighted_output_result_index")
        != indices["weighted_output_result_index"]
        or validator.get("carried_residual_result_index")
        != indices["carried_residual_result_index"]
        or validator.get("auxiliary_result_index")
        != indices["auxiliary_result_index"]
        or validator.get("accepted_primary_slice_sha256")
        != validator.get("candidate_primary_slice_sha256")
        or validator.get("accepted_primary_slice_sha256")
        != _EXPECTED_ACCEPTED_PRIMARY_SLICE_SHA256
        or source["source_semantic_sha256"]
        != expected_source_semantic_sha256
        or validator.get("auxiliary_slice_sha256")
        != _EXPECTED_AUXILIARY_SLICE_SHA256[candidate_id]
        or not isinstance(auxiliary_operations, list)
        or any(not isinstance(item, str) for item in auxiliary_operations)
    ):
        raise BenchmarkValidationError("StableHLO validator result drifted")
    validator_sha = sha256(_canonical_json(validator).encode("ascii")).hexdigest()
    report = {
        "accepted_primary_path": str(accepted_path),
        "accepted_primary_sha256": accepted_sha,
        "accepted_primary_slice_sha256": validator[
            "accepted_primary_slice_sha256"
        ],
        "candidate_path": str(candidate_path),
        "candidate_raw_path": str(candidate_raw_path),
        "candidate_raw_sha256": candidate_raw_sha,
        "candidate_sha256": candidate_sha,
        "certificate_path": str(certificate_path),
        "certificate_sha256": certificate_sha,
        "collectives": validator["collectives"],
        "auxiliary_path_operations": auxiliary_operations,
        "auxiliary_slice_sha256": validator["auxiliary_slice_sha256"],
        "local_device_groups": plan["local_device_groups"],
        "immutable_parser_authority": validator["immutable_parser_authority"],
        "parser_authority_scope": validator["parser_authority_scope"],
        "producer_receipt_path": str(producer_receipt_path),
        "producer_receipt_sha256": producer_receipt_sha,
        "producer_source_path": str(producer_source_path),
        "producer_source_sha256": producer_source_sha,
        "lowering_receipt": lowering_receipt,
        "success_path": str(success_path),
        "success_sha256": success_sha,
        "validator_report_sha256": validator_sha,
    }
    report["authority_sha256"] = sha256(
        _canonical_json(
            {
                "accepted_primary_sha256": accepted_sha,
                "accepted_primary_slice_sha256": report[
                    "accepted_primary_slice_sha256"
                ],
                "candidate_sha256": candidate_sha,
                "candidate_raw_sha256": candidate_raw_sha,
                "certificate_sha256": certificate_sha,
                "mechanism_fingerprint_sha256": mechanism_fingerprint_sha256,
                "plan_sha256": plan["plan_sha256"],
                "producer_receipt_sha256": producer_receipt_sha,
                "producer_source_sha256": producer_source_sha,
                "source_semantic_sha256": source["source_semantic_sha256"],
                "source_set_sha256": source["source_set_sha256"],
                "success_sha256": success_sha,
                "validator_report_sha256": validator_sha,
            }
        ).encode("ascii")
    ).hexdigest()
    return report


def _product(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _read_npy_header(stream: BinaryIO, label: str) -> tuple[str, tuple[int, ...]]:
    if stream.read(6) != b"\x93NUMPY":
        raise BenchmarkValidationError(f"{label} has invalid NPY magic")
    version = stream.read(2)
    if version not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        raise BenchmarkValidationError(f"{label} has unsupported NPY version")
    length_size = 2 if version[0] == 1 else 4
    raw_length = stream.read(length_size)
    if len(raw_length) != length_size:
        raise BenchmarkValidationError(f"{label} has truncated NPY header")
    header_size = struct.unpack("<H" if length_size == 2 else "<I", raw_length)[0]
    header_raw = stream.read(header_size)
    if len(header_raw) != header_size:
        raise BenchmarkValidationError(f"{label} has truncated NPY metadata")
    try:
        header = ast.literal_eval(header_raw.decode("latin1").strip())
    except (SyntaxError, ValueError) as error:
        raise BenchmarkValidationError(f"{label} has invalid NPY metadata") from error
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise BenchmarkValidationError(f"{label} NPY metadata schema drifted")
    dtype = header["descr"]
    shape = header["shape"]
    if (
        header["fortran_order"] is not False
        or not isinstance(dtype, str)
        or dtype not in _DTYPE_BYTES
    ):
        raise BenchmarkValidationError(f"{label} dtype/layout is unsupported")
    if not isinstance(shape, tuple) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in shape
    ):
        raise BenchmarkValidationError(f"{label} shape is invalid")
    return dtype, shape


def _inspect_npz(raw: bytes) -> dict[str, dict[str, Any]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except (OSError, zipfile.BadZipFile) as error:
        raise BenchmarkValidationError("candidate capsule artifact is not NPZ") from error
    arrays: dict[str, dict[str, Any]] = {}
    with archive:
        names = archive.namelist()
        if not names or len(names) > 4096 or len(names) != len(set(names)):
            raise BenchmarkValidationError("candidate NPZ member catalogue is invalid")
        try:
            infos = [archive.getinfo(name) for name in names]
        except (KeyError, OSError, zipfile.BadZipFile, RuntimeError) as error:
            raise BenchmarkValidationError("cannot inspect candidate NPZ") from error
        if any(
            info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            for info in infos
        ):
            raise BenchmarkValidationError("candidate NPZ compression is unsupported")
        if sum(info.file_size for info in infos) > _MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise BenchmarkValidationError("candidate NPZ uncompressed bytes exceed limit")
        for info in infos:
            name = info.filename
            if "/" in name or not name.endswith(".npy") or info.flag_bits & 0x1:
                raise BenchmarkValidationError(f"candidate NPZ member is invalid: {name}")
            if info.file_size > _MAX_ARRAY_BYTES + 64 * 1024:
                raise BenchmarkValidationError(f"candidate NPZ member is too large: {name}")
            try:
                with archive.open(info) as stream:
                    dtype, shape = _read_npy_header(stream, name)
                    expected_bytes = _product(shape) * _DTYPE_BYTES[dtype]
                    if expected_bytes > _MAX_ARRAY_BYTES:
                        raise BenchmarkValidationError(
                            f"candidate NPZ array is too large: {name}"
                        )
                    value = stream.read(expected_bytes + 1)
            except BenchmarkValidationError:
                raise
            except (NotImplementedError, OSError, zipfile.BadZipFile, RuntimeError) as error:
                raise BenchmarkValidationError(
                    f"cannot read candidate NPZ member: {name}"
                ) from error
            if len(value) != expected_bytes:
                raise BenchmarkValidationError(
                    f"candidate NPZ byte count drifted: {name}"
                )
            arrays[name[:-4]] = {
                "raw": value,
                "shape": shape,
                "storage_dtype": dtype,
            }
    return arrays


def _selected_array(
    array: Mapping[str, Any], index_prefix: list[int], label: str
) -> tuple[bytes, tuple[int, ...]]:
    shape = array["shape"]
    if not isinstance(index_prefix, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in index_prefix
    ):
        raise BenchmarkValidationError(f"{label} index prefix is invalid")
    if len(index_prefix) > len(shape):
        raise BenchmarkValidationError(f"{label} index prefix rank is invalid")
    flat_offset = 0
    for axis, index in enumerate(index_prefix):
        if index >= shape[axis]:
            raise BenchmarkValidationError(f"{label} index prefix is out of bounds")
        flat_offset += index * _product(shape[axis + 1 :])
    selected_shape = tuple(shape[len(index_prefix) :])
    item_bytes = _DTYPE_BYTES[array["storage_dtype"]]
    byte_offset = flat_offset * item_bytes
    byte_count = _product(selected_shape) * item_bytes
    return array["raw"][byte_offset : byte_offset + byte_count], selected_shape


def _derive_fp32_sum_from_bf16_bits(hidden: bytes, residual: bytes) -> bytes:
    """Derive the unique IEEE binary32 add from two finite BF16 bit rows."""

    expected_bytes = 6144 * 2
    if len(hidden) != expected_bytes or len(residual) != expected_bytes:
        raise BenchmarkValidationError("RMS BF16 operand byte count drifted")
    result = bytearray(6144 * 4)
    for index, (hidden_bits, residual_bits) in enumerate(
        zip(struct.iter_unpack("<H", hidden), struct.iter_unpack("<H", residual))
    ):
        left_bits = hidden_bits[0]
        right_bits = residual_bits[0]
        if (left_bits & 0x7F80) == 0x7F80 or (right_bits & 0x7F80) == 0x7F80:
            raise BenchmarkValidationError("RMS BF16 operands must be finite")
        left = struct.unpack("<f", struct.pack("<I", left_bits << 16))[0]
        right = struct.unpack("<f", struct.pack("<I", right_bits << 16))[0]
        try:
            encoded = struct.pack("<f", left + right)
        except OverflowError as error:
            raise BenchmarkValidationError("RMS FP32 operand sum overflowed") from error
        if struct.unpack("<I", encoded)[0] & 0x7F800000 == 0x7F800000:
            raise BenchmarkValidationError("RMS FP32 operand sum must be finite")
        result[index * 4 : (index + 1) * 4] = encoded
    return bytes(result)


def _bf16_bits_from_fp32(raw: bytes) -> bytes:
    """Round finite IEEE binary32 bytes to BF16 using ties-to-even."""

    if len(raw) % 4:
        raise BenchmarkValidationError("FP32 value byte count is invalid")
    result = bytearray(len(raw) // 2)
    for index, (bits,) in enumerate(struct.iter_unpack("<I", raw)):
        if bits & 0x7F800000 == 0x7F800000:
            raise BenchmarkValidationError("FP32 values must be finite")
        upper = bits >> 16
        lower = bits & 0xFFFF
        if lower > 0x8000 or (lower == 0x8000 and upper & 1):
            upper = (upper + 1) & 0xFFFF
        result[index * 2 : (index + 1) * 2] = struct.pack("<H", upper)
    return bytes(result)


def _verify_capsule_producer_blob(
    value: Any,
    base: Path,
    implementation: Mapping[str, Any],
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError("capsule producer identity must be an object")
    _exact_keys(
        value,
        {"git_object_id", "repo_path", "repository", "sha256"},
        "capsule producer identity",
    )
    repository = value["repository"]
    if not isinstance(repository, dict):
        raise BenchmarkValidationError("capsule producer repository must be an object")
    _exact_keys(repository, {"commit", "root"}, "capsule producer repository")
    code_pin = _code_pin(repository["commit"], "capsule producer code pin")
    repository_root = _resolve(base, repository["root"], "capsule producer root")
    repo_path = _canonical_repo_path(value["repo_path"], "capsule producer repo path")
    expected_sha = _sha(value["sha256"], "capsule producer SHA-256")
    expected_object = _string(value["git_object_id"], "capsule producer Git object")
    if _GIT_OBJECT_ID.fullmatch(expected_object) is None:
        raise BenchmarkValidationError("capsule producer Git object is invalid")
    with _open_directory_no_symlinks(
        repository_root, "capsule producer repository"
    ) as repository_fd:
        commit = _git_output(
            repository_fd,
            Path(implementation["git_path"]),
            implementation["git_sha256"],
            ["rev-parse", "--verify", f"{code_pin}^{{commit}}"],
            "capsule producer commit",
            limit=1024,
        ).decode("ascii", errors="strict").strip()
        tree_entry = _git_output(
            repository_fd,
            Path(implementation["git_path"]),
            implementation["git_sha256"],
            ["ls-tree", "-z", code_pin, "--", repo_path],
            "capsule producer tree entry",
            limit=4096,
        )
        try:
            metadata, listed_path = tree_entry[:-1].split(b"\t", 1)
            mode, object_type, raw_object_id = metadata.split(b" ", 2)
            listed = listed_path.decode("utf-8", errors="strict")
            object_id = raw_object_id.decode("ascii", errors="strict")
        except (UnicodeDecodeError, ValueError) as error:
            raise BenchmarkValidationError(
                "capsule producer tree entry is invalid"
            ) from error
        blob = _git_output(
            repository_fd,
            Path(implementation["git_path"]),
            implementation["git_sha256"],
            ["cat-file", "blob", object_id],
            "capsule producer blob",
            limit=_MAX_SOURCE_BYTES,
        )
    if (
        commit != code_pin
        or not tree_entry.endswith(b"\0")
        or tree_entry.count(b"\0") != 1
        or listed != repo_path
        or mode not in {b"100644", b"100755"}
        or object_type != b"blob"
        or object_id != expected_object
        or sha256(blob).hexdigest() != expected_sha
    ):
        raise BenchmarkValidationError("committed capsule producer drifted")
    return {
        "code_pin": code_pin,
        "git_object_id": object_id,
        "repo_path": repo_path,
        "repository_root": str(repository_root),
        "sha256": expected_sha,
    }


def _verify_real_capsule_execution_source(
    producer: Mapping[str, str],
    source_records: Sequence[Mapping[str, str]],
    replay_blob: bytes,
) -> dict[str, Any]:
    """Bind the real producer and complete replay-visible committed source tree."""

    producer_identity = {
        key: producer.get(key) for key in ("git_object_id", "repo_path", "sha256")
    }
    if producer_identity != _EXPECTED_CAPSULE_PRODUCER_SOURCE:
        raise BenchmarkValidationError(
            "real capsule producer is not the independently reviewed blob"
        )
    execution_records = sorted(
        (
            {
                "git_object_id": _string(
                    record.get("git_object_id"), "capsule execution Git object"
                ),
                "mode": _string(record.get("mode"), "capsule execution source mode"),
                "path": _canonical_repo_path(
                    record.get("path"), "capsule execution source path"
                ),
            }
            for record in source_records
            if record.get("path") not in _CAPSULE_EXECUTION_MANIFEST_EXCLUDED_PATHS
        ),
        key=lambda item: item["path"],
    )
    if any(
        record["mode"] not in {"100644", "100755"}
        or _GIT_OBJECT_ID.fullmatch(record["git_object_id"]) is None
        for record in execution_records
    ):
        raise BenchmarkValidationError("real capsule execution source entry drifted")
    execution_manifest = {
        "count": len(execution_records),
        "sha256": sha256(
            _canonical_json(execution_records).encode("ascii")
        ).hexdigest(),
    }
    replay = _EXPECTED_CAPSULE_REPLAY_SOURCE
    replay_records = [
        record for record in execution_records if record["path"] == replay["repo_path"]
    ]
    if (
        execution_manifest != _EXPECTED_CAPSULE_EXECUTION_SOURCE_MANIFEST
        or len(replay_records) != 1
        or replay_records[0]["git_object_id"] != replay["git_object_id"]
        or sha256(replay_blob).hexdigest() != replay["sha256"]
    ):
        raise BenchmarkValidationError(
            "real capsule replay/import source closure drifted"
        )
    return {
        "execution_manifest": execution_manifest,
        "producer": producer_identity,
        "replay": dict(replay),
    }


def _manifest_summary(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{label} must be an object")
    _exact_keys(value, {"count", "sha256"}, label)
    return {
        "count": _nonnegative_int(value["count"], f"{label} count"),
        "sha256": _sha(value["sha256"], f"{label} SHA-256"),
    }


def _require_immutable_dependency(path: Path, label: str) -> None:
    """Require a pathname and every parent to be outside same-UID mutation."""

    normalized = Path(os.path.abspath(os.fspath(path)))
    _require_root_owned_immutable_directory(normalized.parent, f"{label} parent")
    with _open_regular_file(normalized, label) as stream:
        metadata = os.fstat(stream.fileno())
        if metadata.st_uid != 0 or metadata.st_gid != 0 or metadata.st_mode & (
            stat.S_IWGRP
            | stat.S_IWOTH
            | stat.S_ISUID
            | stat.S_ISGID
            | stat.S_ISVTX
        ):
            raise BenchmarkValidationError(
                f"immutable {label} is not root-owned read-only: {normalized}"
            )
    try:
        attributes = os.listxattr(normalized, follow_symlinks=False)
    except OSError as error:
        raise BenchmarkValidationError(
            f"cannot inspect immutable {label} attributes: {normalized}"
        ) from error
    if attributes:
        raise BenchmarkValidationError(
            f"immutable {label} has extended attributes: {normalized}"
        )


def _require_finite_raw(raw: bytes, storage_dtype: str, label: str) -> None:
    """Reject IEEE BF16/FP32 NaN and infinity without importing NumPy."""

    if storage_dtype == "<u2":
        if len(raw) % 2:
            raise BenchmarkValidationError(f"{label} BF16 byte count drifted")
        if any((value & 0x7F80) == 0x7F80 for (value,) in struct.iter_unpack("<H", raw)):
            raise BenchmarkValidationError(f"{label} contains non-finite BF16 values")
        return
    if storage_dtype == "<f4":
        if len(raw) % 4:
            raise BenchmarkValidationError(f"{label} FP32 byte count drifted")
        if any(
            (value & 0x7F800000) == 0x7F800000
            for (value,) in struct.iter_unpack("<I", raw)
        ):
            raise BenchmarkValidationError(f"{label} contains non-finite FP32 values")
        return
    raise BenchmarkValidationError(f"{label} has no finite-value contract")


def _capsule_mountinfo_path(raw: str, *, require_absolute: bool = True) -> Path:
    value = raw
    for encoded, decoded in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        value = value.replace(encoded, decoded)
    if "\x00" in value or (require_absolute and not value.startswith("/")):
        raise BenchmarkValidationError("capsule runtime mountinfo path is invalid")
    return Path(value)


def _verify_capsule_runtime_data_mount(
    mountinfo_raw: bytes, payload_paths: tuple[Path, ...]
) -> dict[str, Any]:
    """Bind real tensor payloads to the exact read-only same-region GCS mount."""

    normalized_data_root = Path(os.path.abspath(_EXPECTED_CAPSULE_RUNTIME_DATA_ROOT))
    normalized_mount_point = Path(
        os.path.abspath(_EXPECTED_CAPSULE_RUNTIME_MOUNT_POINT)
    )
    if normalized_mount_point not in normalized_data_root.parents:
        raise BenchmarkValidationError(
            "capsule runtime data root escaped its mount authority"
        )
    records: list[dict[str, Any]] = []
    for raw_line in mountinfo_raw.splitlines():
        try:
            left, right = raw_line.decode("utf-8", errors="strict").split(" - ", 1)
        except (UnicodeDecodeError, ValueError) as error:
            raise BenchmarkValidationError("capsule runtime mountinfo is invalid") from error
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 6 or len(right_fields) < 3:
            raise BenchmarkValidationError(
                "capsule runtime mountinfo record is invalid"
            )
        try:
            mount_id = int(left_fields[0])
            major, minor = (int(item) for item in left_fields[2].split(":", 1))
            mount_root = _capsule_mountinfo_path(
                left_fields[3], require_absolute=False
            )
            mount_point = _capsule_mountinfo_path(left_fields[4])
        except (TypeError, ValueError) as error:
            raise BenchmarkValidationError(
                "capsule runtime mountinfo identity is invalid"
            ) from error
        records.append(
            {
                "major": major,
                "minor": minor,
                "mount_id": mount_id,
                "mount_options": set(left_fields[5].split(",")),
                "mount_point": mount_point,
                "mount_root": mount_root,
                "raw_record": raw_line,
                "source": right_fields[1],
                "super_options": set(right_fields[2].split(",")),
                "type": right_fields[0],
            }
        )
    matches = [
        record for record in records if record["mount_point"] == normalized_mount_point
    ]
    if len(matches) != 1:
        raise BenchmarkValidationError(
            "capsule runtime data mount authority is absent or duplicated"
        )
    authority = matches[0]
    if (
        authority["mount_root"] != Path("/")
        or not {"ro", "nosuid", "nodev"} <= authority["mount_options"]
        or authority["type"] != "fuse.gcsfuse"
        or authority["source"] != _EXPECTED_CAPSULE_RUNTIME_MOUNT_SOURCE
        or "ro" not in authority["super_options"]
    ):
        raise BenchmarkValidationError("capsule runtime data mount authority drifted")
    for path in (normalized_data_root, *payload_paths):
        normalized = Path(os.path.abspath(path))
        if normalized != normalized_data_root and normalized_data_root not in normalized.parents:
            raise BenchmarkValidationError(
                "capsule runtime payload escaped its data root"
            )
        covering = [
            record
            for record in records
            if normalized == record["mount_point"]
            or record["mount_point"] in normalized.parents
        ]
        if not covering:
            raise BenchmarkValidationError(
                "capsule runtime payload has no covering mount"
            )
        longest = max(len(record["mount_point"].parts) for record in covering)
        visible = [
            record
            for record in covering
            if len(record["mount_point"].parts) == longest
        ]
        if len(visible) != 1 or visible[0]["mount_id"] != authority["mount_id"]:
            raise BenchmarkValidationError(
                "capsule runtime payload is covered by a nested mount"
            )
    return {
        "major": authority["major"],
        "minor": authority["minor"],
        "mount_id": authority["mount_id"],
        "raw_record": authority["raw_record"],
    }


def _verify_capsule_payload_descriptor_mount(
    descriptor: int,
    metadata: os.stat_result,
    authority: Mapping[str, Any],
) -> None:
    try:
        fields = {}
        for line in Path(f"/proc/self/fdinfo/{descriptor}").read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key] = value.strip()
        mount_id = int(fields["mnt_id"])
    except (KeyError, OSError, ValueError) as error:
        raise BenchmarkValidationError(
            "capsule runtime payload descriptor mount is unavailable"
        ) from error
    if (
        mount_id != authority["mount_id"]
        or os.major(metadata.st_dev) != authority["major"]
        or os.minor(metadata.st_dev) != authority["minor"]
    ):
        raise BenchmarkValidationError(
            "capsule runtime payload descriptor escaped mount authority"
        )


def _capsule_runtime_owner_paths(runtime_manifest_raw: bytes) -> tuple[Path, ...]:
    manifest = _load_json(runtime_manifest_raw, "capsule runtime manifest")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise BenchmarkValidationError("capsule runtime manifest files are invalid")
    records = [
        record
        for record in files
        if isinstance(record, dict)
        and record.get("stage_id") == 0
        and record.get("device_slot") in (0, 1)
    ]
    if len(records) != 2 or {int(record["device_slot"]) for record in records} != {
        0,
        1,
    }:
        raise BenchmarkValidationError("capsule runtime stage-zero owners drifted")
    paths = []
    for record in sorted(records, key=lambda item: int(item["device_slot"])):
        raw = record.get("destination_filename")
        if not isinstance(raw, str):
            raise BenchmarkValidationError(
                "capsule runtime destination filename is invalid"
            )
        relative = PurePosixPath(raw)
        if (
            relative.is_absolute()
            or relative.as_posix() != raw
            or any(part in ("", ".", "..") for part in relative.parts)
        ):
            raise BenchmarkValidationError(
                "capsule runtime destination escaped its data root"
            )
        paths.append(_EXPECTED_CAPSULE_RUNTIME_DATA_ROOT.joinpath(*relative.parts))
    return tuple(paths)


def _verify_capsule_tensor_receipts(
    value: Any,
    *,
    real_source: bool,
    runtime_manifest_raw: bytes | None,
    runtime_paths: tuple[Path, ...] | None = None,
    runtime_mount_authority: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(value, list):
        raise BenchmarkValidationError("capsule tensor receipts must be a list")
    receipts: list[dict[str, Any]] = []
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise BenchmarkValidationError("capsule tensor receipt is invalid")
        label = f"capsule tensor receipt {index}"
        _exact_keys(
            record,
            {
                "byte_count",
                "device_slot",
                "file_bytes",
                "file_path",
                "name",
                "offset",
                "sha256",
                "shape",
            },
            label,
        )
        shape = record["shape"]
        if not isinstance(shape, list) or not shape or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in shape
        ):
            raise BenchmarkValidationError(f"{label} shape is invalid")
        path = Path(_string(record["file_path"], f"{label} file path"))
        if not path.is_absolute():
            raise BenchmarkValidationError(f"{label} file path is not absolute")
        receipt = {
            "byte_count": _positive_int(record["byte_count"], f"{label} bytes"),
            "device_slot": _nonnegative_int(
                record["device_slot"], f"{label} device slot"
            ),
            "file_bytes": _positive_int(record["file_bytes"], f"{label} file bytes"),
            "file_path": str(path),
            "name": _string(record["name"], f"{label} name"),
            "offset": _nonnegative_int(record["offset"], f"{label} offset"),
            "sha256": _sha(record["sha256"], f"{label} SHA-256"),
            "shape": shape,
        }
        receipts.append(receipt)
    if receipts != sorted(
        receipts, key=lambda item: (item["device_slot"], item["name"])
    ) or len({(item["device_slot"], item["name"]) for item in receipts}) != len(
        receipts
    ):
        raise BenchmarkValidationError("capsule tensor receipt catalogue is not canonical")
    if not real_source:
        return receipts, {}
    if runtime_manifest_raw is None:
        raise BenchmarkValidationError("real capsule runtime manifest snapshot is absent")
    if runtime_paths is None or runtime_mount_authority is None:
        raise BenchmarkValidationError("real capsule runtime mount authority is absent")
    expected_pairs = {
        (slot, name) for slot in (0, 1) for name in _EXPECTED_CAPSULE_TENSOR_NAMES
    }
    if {(item["device_slot"], item["name"]) for item in receipts} != expected_pairs:
        raise BenchmarkValidationError("real capsule tensor receipt catalogue drifted")
    manifest = _load_json(runtime_manifest_raw, "capsule runtime manifest")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise BenchmarkValidationError("capsule runtime manifest files are invalid")
    owner_records: dict[int, Mapping[str, Any]] = {}
    for record in files:
        if (
            isinstance(record, dict)
            and record.get("stage_id") == 0
            and record.get("device_slot") in (0, 1)
        ):
            slot = int(record["device_slot"])
            if slot in owner_records:
                raise BenchmarkValidationError("capsule runtime owner is duplicated")
            owner_records[slot] = record
    if set(owner_records) != {0, 1}:
        raise BenchmarkValidationError("capsule runtime stage-zero owners drifted")
    by_slot = {
        slot: [item for item in receipts if item["device_slot"] == slot]
        for slot in (0, 1)
    }
    dtype_bytes = {"BF16": 2, "F32": 4, "U8": 1}
    storage_dtypes = {"BF16": "<u2", "F32": "<f4", "U8": "|u1"}
    payloads: dict[tuple[int, str], bytes] = {}
    payload_shapes: dict[tuple[int, str], list[int]] = {}
    payload_dtypes: dict[tuple[int, str], str] = {}
    for slot in (0, 1):
        owner = owner_records[slot]
        required_keys = {
            "destination_filename",
            "file_bytes",
            "header_bytes",
            "header_sha256",
            "tensors",
        }
        if not required_keys <= set(owner):
            raise BenchmarkValidationError("capsule runtime owner manifest is incomplete")
        expected_path = runtime_paths[slot]
        tensor_records = owner["tensors"]
        if not isinstance(tensor_records, list):
            raise BenchmarkValidationError("capsule runtime tensor manifest is invalid")
        manifest_tensors = {
            item.get("name"): item for item in tensor_records if isinstance(item, dict)
        }
        with _open_regular_file(expected_path, f"capsule runtime owner {slot}") as stream:
            metadata = os.fstat(stream.fileno())
            _verify_capsule_payload_descriptor_mount(
                stream.fileno(), metadata, runtime_mount_authority
            )
            header_bytes = _positive_int(
                owner["header_bytes"], f"capsule runtime owner {slot} header bytes"
            )
            raw_header = stream.read(header_bytes)
            if (
                metadata.st_size != owner["file_bytes"]
                or len(raw_header) != header_bytes
                or sha256(raw_header).hexdigest() != owner["header_sha256"]
                or header_bytes < 8
                or struct.unpack("<Q", raw_header[:8])[0] + 8 != header_bytes
            ):
                raise BenchmarkValidationError("capsule runtime owner/header drifted")
            header = _load_json(raw_header[8:], f"capsule runtime owner {slot} header")
            for receipt in by_slot[slot]:
                tensor = manifest_tensors.get(receipt["name"])
                layout = header.get(receipt["name"])
                if not isinstance(tensor, dict) or not isinstance(layout, dict):
                    raise BenchmarkValidationError("capsule runtime tensor authority is absent")
                offsets = layout.get("data_offsets")
                shape = layout.get("shape")
                item_size = dtype_bytes.get(layout.get("dtype"))
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or not all(isinstance(item, int) for item in offsets)
                    or not isinstance(shape, list)
                    or item_size is None
                ):
                    raise BenchmarkValidationError("capsule runtime tensor layout drifted")
                start, end = offsets
                expected_bytes = end - start
                element_count = 1
                for dimension in shape:
                    if not isinstance(dimension, int) or dimension <= 0:
                        raise BenchmarkValidationError(
                            "capsule runtime tensor shape drifted"
                        )
                    element_count *= dimension
                if (
                    receipt["file_path"] != str(expected_path)
                    or receipt["file_bytes"] != metadata.st_size
                    or receipt["offset"] != header_bytes + start
                    or receipt["byte_count"] != expected_bytes
                    or receipt["byte_count"] != element_count * item_size
                    or receipt["shape"] != shape
                    or tensor.get("byte_count") != expected_bytes
                    or tensor.get("sha256") != receipt["sha256"]
                ):
                    raise BenchmarkValidationError("capsule runtime tensor receipt drifted")
                stream.seek(receipt["offset"])
                payload = stream.read(receipt["byte_count"])
                if (
                    len(payload) != receipt["byte_count"]
                    or sha256(payload).hexdigest() != receipt["sha256"]
                ):
                    raise BenchmarkValidationError(
                        "capsule runtime tensor payload drifted"
                    )
                key = (slot, receipt["name"])
                payloads[key] = payload
                payload_shapes[key] = shape
                payload_dtypes[key] = storage_dtypes[layout["dtype"]]
    runtime_inputs: dict[str, dict[str, Any]] = {}
    for input_name, (tensor_name, slots) in _CAPSULE_RUNTIME_INPUT_SOURCES.items():
        expected_dtype, expected_shape = _CAPSULE_INPUT_SCHEMA[input_name]
        source_keys = [(slot, tensor_name) for slot in slots]
        owner_shape = payload_shapes[source_keys[0]]
        observed_shape = (
            [len(slots), *owner_shape] if len(slots) > 1 else owner_shape
        )
        if (
            observed_shape != list(expected_shape)
            or any(payload_shapes[key] != owner_shape for key in source_keys)
            or any(payload_dtypes[key] != expected_dtype for key in source_keys)
        ):
            raise BenchmarkValidationError(
                f"capsule runtime-derived input layout drifted: {input_name}"
            )
        raw = b"".join(payloads[key] for key in source_keys)
        runtime_inputs[input_name] = {
            "array_sha256": sha256(raw).hexdigest(),
            "shape": observed_shape,
            "storage_dtype": expected_dtype,
        }
    return receipts, runtime_inputs


def _verify_capsule_execution_authority(
    value: Any,
    base: Path,
    *,
    candidate_id: str,
    implementation: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    path, authority_sha, raw = _binding(
        base, value, "capsule execution authority", limit=_MAX_JSON_BYTES
    )
    authority = _load_json(raw, "capsule execution authority")
    if raw != (_canonical_json(authority) + "\n").encode("ascii"):
        raise BenchmarkValidationError("capsule execution authority is not canonical JSON")
    _exact_keys(
        authority,
        {
            "authority_kind",
            "candidate_id",
            "environment",
            "expected_outputs",
            "input_arrays",
            "installed_producer",
            "loaded_dependencies",
            "producer",
            "schema_version",
            "source_snapshot",
            "tensor_receipts",
            "upstream_inputs",
        },
        "capsule execution authority",
    )
    if (
        authority["schema_version"] != 1
        or authority["authority_kind"] != "gate.d.capsule.execution.v1"
        or authority["candidate_id"] != candidate_id
    ):
        raise BenchmarkValidationError("capsule execution authority identity drifted")
    real_source = source.get("executable_source_authority") is True
    producer = _verify_capsule_producer_blob(
        authority["producer"], path.parent, implementation
    )
    installed = authority["installed_producer"]
    if not isinstance(installed, dict):
        raise BenchmarkValidationError("installed capsule producer is invalid")
    _exact_keys(
        installed,
        {"bytes", "gid", "mode", "path", "sha256", "uid"},
        "installed capsule producer",
    )
    installed_path = _resolve(
        path.parent, installed["path"], "installed capsule producer"
    )
    installed_metadata = installed_path.lstat()
    installed_bytes, installed_sha = _regular_file_identity(
        installed_path, "installed capsule producer"
    )
    installed_report = {
        "bytes": installed_bytes,
        "gid": _nonnegative_int(installed["gid"], "installed capsule producer gid"),
        "mode": _nonnegative_int(installed["mode"], "installed capsule producer mode"),
        "path": str(installed_path),
        "sha256": _sha(installed["sha256"], "installed capsule producer SHA-256"),
        "uid": _nonnegative_int(installed["uid"], "installed capsule producer uid"),
    }
    if (
        installed_report["bytes"] != installed_metadata.st_size
        or installed_report["mode"] != stat.S_IMODE(installed_metadata.st_mode)
        or installed_report["uid"] != installed_metadata.st_uid
        or installed_report["gid"] != installed_metadata.st_gid
        or installed_report["sha256"] != installed_sha
        or installed_sha != producer["sha256"]
        or installed_metadata.st_mode
        & (stat.S_IWGRP | stat.S_IWOTH | stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX)
    ):
        raise BenchmarkValidationError("installed capsule producer drifted")
    if real_source:
        if (
            installed_path != Path(_EXPECTED_CAPSULE_INSTALLED_PRODUCER)
            or installed_report["uid"] != 0
            or installed_report["gid"] != 0
            or installed_report["mode"] != 0o555
        ):
            raise BenchmarkValidationError(
                "real installed capsule producer authority drifted"
            )
        _require_immutable_dependency(
            installed_path, "real installed capsule producer"
        )
    environment = authority["environment"]
    if not isinstance(environment, dict):
        raise BenchmarkValidationError("capsule execution environment is invalid")
    _exact_keys(
        environment,
        set(_EXPECTED_CAPSULE_ENVIRONMENT) | {"python_version"},
        "capsule execution environment",
    )
    if any(
        environment[key] != expected
        for key, expected in _EXPECTED_CAPSULE_ENVIRONMENT.items()
    ):
        raise BenchmarkValidationError("capsule execution environment drifted")
    _string(environment["python_version"], "capsule execution Python version")
    python_runtime_root = _require_root_owned_immutable_directory(
        Path(environment["python_runtime_root"]), "capsule Python runtime root"
    )
    site_root = _require_root_owned_immutable_directory(
        Path(environment["site_root"]), "capsule JAX site root"
    )
    python_path = Path(environment["python_executable"])
    _require_root_owned_immutable_file(
        python_path,
        "capsule Python executable",
        beneath=python_runtime_root,
    )
    if _regular_file_identity(python_path, "capsule Python executable")[1] != environment[
        "python_sha256"
    ]:
        raise BenchmarkValidationError("capsule Python executable drifted")
    for runtime_root, expected_tree, label in (
        (
            python_runtime_root,
            environment["python_runtime_tree_sha256"],
            "capsule Python runtime",
        ),
        (site_root, environment["site_tree_sha256"], "capsule JAX site"),
    ):
        runtime_key = (str(runtime_root), expected_tree)
        if runtime_key not in _VERIFIED_RUNTIME_TREES:
            if _runtime_tree_sha256(runtime_root) != expected_tree:
                raise BenchmarkValidationError(f"{label} tree SHA-256 drifted")
            _VERIFIED_RUNTIME_TREES.add(runtime_key)
    source_snapshot = authority["source_snapshot"]
    if not isinstance(source_snapshot, dict):
        raise BenchmarkValidationError("capsule source snapshot is invalid")
    _exact_keys(
        source_snapshot,
        {"archive_sha256", "file_manifest", "repository"},
        "capsule source snapshot",
    )
    repository = source_snapshot["repository"]
    if not isinstance(repository, dict):
        raise BenchmarkValidationError("capsule source repository is invalid")
    _exact_keys(repository, {"commit", "root"}, "capsule source repository")
    source_repository = {
        "commit": _code_pin(repository["commit"], "capsule source snapshot commit"),
        "root": str(_resolve(path.parent, repository["root"], "capsule source root")),
    }
    source_report = {
        "archive_sha256": _sha(
            source_snapshot["archive_sha256"], "capsule source archive SHA-256"
        ),
        "file_manifest": _manifest_summary(
            source_snapshot["file_manifest"], "capsule source file manifest"
        ),
        "repository": source_repository,
    }
    if (
        source_repository["commit"] != producer["code_pin"]
        or source_repository["root"] != producer["repository_root"]
    ):
        raise BenchmarkValidationError(
            "capsule source snapshot is not the producer repository"
        )
    with _open_directory_no_symlinks(
        Path(source_repository["root"]), "capsule source repository"
    ) as repository_fd:
        source_tree = _git_output(
            repository_fd,
            Path(implementation["git_path"]),
            implementation["git_sha256"],
            ["ls-tree", "-r", "-z", source_repository["commit"], "--", "glm_tpu"],
            "capsule source tree",
            limit=_MAX_SOURCE_BYTES,
        )
        source_archive = _git_output(
            repository_fd,
            Path(implementation["git_path"]),
            implementation["git_sha256"],
            ["archive", "--format=zip", source_repository["commit"], "glm_tpu"],
            "capsule source archive",
            limit=_MAX_ARTIFACT_BYTES,
        )
        replay_blob = b""
        if real_source:
            replay_blob = _git_output(
                repository_fd,
                Path(implementation["git_path"]),
                implementation["git_sha256"],
                [
                    "cat-file",
                    "blob",
                    _EXPECTED_CAPSULE_REPLAY_SOURCE["git_object_id"],
                ],
                "capsule replay blob",
                limit=_MAX_SOURCE_BYTES,
            )
    source_records = []
    for raw_entry in source_tree.split(b"\0"):
        if not raw_entry:
            continue
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, kind, object_id = metadata.split(b" ", 2)
            source_path = raw_path.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError) as error:
            raise BenchmarkValidationError("capsule source tree is invalid") from error
        if kind != b"blob" or mode not in {b"100644", b"100755"}:
            raise BenchmarkValidationError(
                f"capsule source tree entry is unsupported: {source_path}"
            )
        source_records.append(
            {
                "git_object_id": object_id.decode("ascii", errors="strict"),
                "mode": mode.decode("ascii", errors="strict"),
                "path": source_path,
            }
        )
    observed_source_manifest = {
        "count": len(source_records),
        "sha256": sha256(_canonical_json(source_records).encode("ascii")).hexdigest(),
    }
    if (
        not source_records
        or source_records != sorted(source_records, key=lambda item: item["path"])
        or observed_source_manifest != source_report["file_manifest"]
        or sha256(source_archive).hexdigest() != source_report["archive_sha256"]
    ):
        raise BenchmarkValidationError("capsule committed source snapshot drifted")
    independently_reviewed_execution = None
    if real_source:
        independently_reviewed_execution = _verify_real_capsule_execution_source(
            producer, source_records, replay_blob
        )
    source_objects = {
        record["path"]: record["git_object_id"] for record in source_records
    }
    if real_source and any(
        source_objects.get(record["repo_path"]) != record["git_object_id"]
        for record in source.get("files", ())
    ):
        raise BenchmarkValidationError(
            "capsule replay source is not the reviewed candidate source"
        )
    loaded = authority["loaded_dependencies"]
    if not isinstance(loaded, dict):
        raise BenchmarkValidationError("capsule loaded dependencies are invalid")
    _exact_keys(
        loaded,
        {"native_mappings", "python_modules"},
        "capsule loaded dependencies",
    )
    loaded_report: dict[str, list[dict[str, Any]]] = {}
    python_roots = (
        Path(environment["python_runtime_root"]),
        Path(environment["site_root"]),
    )
    for category in ("native_mappings", "python_modules"):
        records = loaded[category]
        if not isinstance(records, list):
            raise BenchmarkValidationError(f"capsule loaded {category} must be a list")
        normalized_records = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise BenchmarkValidationError(
                    f"capsule loaded {category} record is invalid"
                )
            _exact_keys(
                record,
                {"bytes", "path", "sha256"},
                f"capsule loaded {category} record {index}",
            )
            dependency_path = _resolve(
                path.parent,
                record["path"],
                f"capsule loaded {category} record {index}",
            )
            dependency_bytes, dependency_sha = _regular_file_identity(
                dependency_path, f"capsule loaded {category} record {index}"
            )
            normalized = {
                "bytes": _nonnegative_int(
                    record["bytes"], f"capsule loaded {category} bytes"
                ),
                "path": str(dependency_path),
                "sha256": _sha(
                    record["sha256"], f"capsule loaded {category} SHA-256"
                ),
            }
            if (
                normalized["bytes"] != dependency_bytes
                or normalized["sha256"] != dependency_sha
            ):
                raise BenchmarkValidationError(
                    f"capsule loaded dependency drifted: {dependency_path}"
                )
            if category == "python_modules" and dependency_path != installed_path:
                if not any(
                    dependency_path == root or root in dependency_path.parents
                    for root in python_roots
                ):
                    raise BenchmarkValidationError(
                        f"capsule Python dependency escaped sealed roots: {dependency_path}"
                    )
            if real_source:
                _require_immutable_dependency(
                    dependency_path, f"real capsule loaded {category}"
                )
            normalized_records.append(normalized)
        if normalized_records != sorted(
            normalized_records, key=lambda item: item["path"]
        ) or len({item["path"] for item in normalized_records}) != len(
            normalized_records
        ):
            raise BenchmarkValidationError(
                f"capsule loaded {category} catalogue is not canonical"
            )
        loaded_report[category] = normalized_records
    raw_tensor_receipts = authority["tensor_receipts"]
    input_arrays = authority["input_arrays"]
    if not isinstance(input_arrays, dict) or set(input_arrays) != set(
        _CAPSULE_INPUT_SCHEMA
    ):
        raise BenchmarkValidationError("capsule authority input catalogue drifted")
    input_report: dict[str, dict[str, Any]] = {}
    for name, (expected_dtype, expected_shape) in _CAPSULE_INPUT_SCHEMA.items():
        record = input_arrays[name]
        if not isinstance(record, dict):
            raise BenchmarkValidationError("capsule authority input record is invalid")
        _exact_keys(
            record,
            {"array_sha256", "shape", "storage_dtype"},
            f"capsule authority input {name}",
        )
        normalized = {
            "array_sha256": _sha(
                record["array_sha256"], f"capsule authority input {name} SHA-256"
            ),
            "shape": record["shape"],
            "storage_dtype": record["storage_dtype"],
        }
        if normalized["shape"] != list(expected_shape) or normalized[
            "storage_dtype"
        ] != expected_dtype:
            raise BenchmarkValidationError(
                f"capsule authority input array drifted: {name}"
            )
        input_report[name] = normalized
    upstream = authority["upstream_inputs"]
    if not isinstance(upstream, dict) or not upstream:
        raise BenchmarkValidationError("capsule upstream input authority is absent")
    upstream_report: dict[str, dict[str, Any]] = {}
    upstream_raw: dict[str, bytes] = {}
    for name, record in sorted(upstream.items()):
        _identifier(name, "capsule upstream input id")
        if not isinstance(record, dict):
            raise BenchmarkValidationError("capsule upstream input record is invalid")
        _exact_keys(record, {"bytes", "path", "sha256"}, f"capsule upstream {name}")
        upstream_path = _resolve(path.parent, record["path"], f"capsule upstream {name}")
        expected_bytes = _nonnegative_int(
            record["bytes"], f"capsule upstream {name} bytes"
        )
        expected_sha = _sha(record["sha256"], f"capsule upstream {name} SHA-256")
        try:
            observed_raw = _snapshot(
                upstream_path,
                expected_sha,
                f"capsule upstream {name}",
                limit=_MAX_ARTIFACT_BYTES,
            )
        except BenchmarkValidationError as error:
            raise BenchmarkValidationError(
                f"capsule upstream input drifted: {name}"
            ) from error
        if len(observed_raw) != expected_bytes:
            raise BenchmarkValidationError(f"capsule upstream input drifted: {name}")
        upstream_raw[name] = observed_raw
        upstream_report[name] = {
            "bytes": len(observed_raw),
            "path": str(upstream_path),
            "sha256": expected_sha,
        }
    if real_source:
        if set(upstream_report) != set(_EXPECTED_CAPSULE_UPSTREAM_INPUTS):
            raise BenchmarkValidationError("real capsule upstream catalogue drifted")
        for name, (expected_path, expected_sha) in _EXPECTED_CAPSULE_UPSTREAM_INPUTS.items():
            if upstream_report[name] != {
                "bytes": len(upstream_raw[name]),
                "path": expected_path,
                "sha256": expected_sha,
            }:
                raise BenchmarkValidationError(
                    f"real capsule upstream authority drifted: {name}"
                )
    runtime_paths = None
    runtime_mount_authority = None
    if real_source:
        runtime_paths = _capsule_runtime_owner_paths(
            upstream_raw["runtime_manifest"]
        )
        runtime_mount_authority = _verify_capsule_runtime_data_mount(
            Path("/proc/self/mountinfo").read_bytes(), runtime_paths
        )
    tensor_receipts, runtime_input_arrays = _verify_capsule_tensor_receipts(
        raw_tensor_receipts,
        real_source=real_source,
        runtime_manifest_raw=upstream_raw.get("runtime_manifest"),
        runtime_paths=runtime_paths,
        runtime_mount_authority=runtime_mount_authority,
    )
    if real_source and _verify_capsule_runtime_data_mount(
        Path("/proc/self/mountinfo").read_bytes(), runtime_paths
    ) != runtime_mount_authority:
        raise BenchmarkValidationError(
            "capsule runtime data mount changed during verification"
        )
    if real_source:
        db518_arrays = _inspect_npz(upstream_raw["db518_result"])
        cache_key = "layer1_index_cache_owners_bfloat16_bits"
        if cache_key not in db518_arrays:
            raise BenchmarkValidationError("real capsule DB518 cache authority is absent")
        cache_array = db518_arrays[cache_key]
        if (
            cache_array["storage_dtype"] != "<u2"
            or cache_array["shape"] != (2, 16, 256, 128)
        ):
            raise BenchmarkValidationError("real capsule DB518 cache geometry drifted")
        expected_cache = bytearray(cache_array["raw"])
        current_offset = (((1 * 16 + 15) * 256 + 219) * 128) * 2
        expected_cache[current_offset : current_offset + 128 * 2] = bytes(128 * 2)
        expected_cache_record = {
            "array_sha256": sha256(expected_cache).hexdigest(),
            "shape": [2, 16, 256, 128],
            "storage_dtype": "<u2",
        }
        if input_report["prompt_cache_bf16_bits"] != expected_cache_record:
            raise BenchmarkValidationError(
                "real capsule prompt cache is not the sealed DB518 history"
            )
    outputs = authority["expected_outputs"]
    if not isinstance(outputs, dict):
        raise BenchmarkValidationError("capsule expected outputs are invalid")
    _exact_keys(
        outputs,
        {
            "event1_positions_sha256",
            "event1_scores_sha256",
            "event1_valid_count",
            "rms_hidden_update_sha256",
            "rms_residual_sha256",
        },
        "capsule expected outputs",
    )
    output_report = {
        "event1_positions_sha256": _sha(
            outputs["event1_positions_sha256"], "accepted event-1 positions SHA-256"
        ),
        "event1_scores_sha256": _sha(
            outputs["event1_scores_sha256"], "accepted event-1 scores SHA-256"
        ),
        "event1_valid_count": _positive_int(
            outputs["event1_valid_count"], "accepted event-1 valid count"
        ),
        "rms_hidden_update_sha256": _sha(
            outputs["rms_hidden_update_sha256"], "accepted RMS hidden update SHA-256"
        ),
        "rms_residual_sha256": _sha(
            outputs["rms_residual_sha256"], "accepted RMS residual SHA-256"
        ),
    }
    if output_report["event1_valid_count"] != 2048:
        raise BenchmarkValidationError("accepted event-1 valid count drifted")
    if real_source and output_report != _EXPECTED_CAPSULE_ACCEPTED_OUTPUTS:
        raise BenchmarkValidationError("real capsule accepted output authority drifted")
    return {
        "authority_path": str(path),
        "authority_sha256": authority_sha,
        "environment": dict(environment),
        "expected_outputs": output_report,
        "input_arrays": input_report,
        "installed_producer": installed_report,
        "loaded_dependencies": loaded_report,
        "producer": producer,
        "independently_reviewed_execution": independently_reviewed_execution,
        "source_snapshot": source_report,
        "tensor_receipts": tensor_receipts,
        "runtime_input_arrays": runtime_input_arrays,
        "upstream_inputs": upstream_report,
    }


def _verify_capsule_producer_receipt(
    *,
    artifact_raw: bytes,
    artifact_sha: str,
    base: Path,
    candidate_id: str,
    capsule: Mapping[str, Any],
    execution_authority: Mapping[str, Any],
    implementation: Mapping[str, Any],
    device_evidence_binding: Any,
    input_binding: Any,
    plan: Mapping[str, Any],
    producer_binding: Any,
    records: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    stablehlo: Mapping[str, Any],
    success_binding: Any,
) -> dict[str, Any]:
    receipt_path, receipt_sha, receipt_raw = _binding(
        base, producer_binding, "capsule producer receipt", limit=_MAX_JSON_BYTES
    )
    receipt = _load_json(receipt_raw, "capsule producer receipt")
    if receipt_raw != (_canonical_json(receipt) + "\n").encode("ascii"):
        raise BenchmarkValidationError("capsule producer receipt is not canonical JSON")
    _exact_keys(
        receipt,
        {
            "artifact",
            "backend",
            "candidate",
            "claim_scope",
            "coherence_id",
            "device_evidence",
            "environment",
            "execution",
            "input_arrays",
            "installed_producer",
            "loaded_dependencies",
            "producer",
            "schema_version",
            "source_snapshot",
            "tensor_receipts",
            "upstream_inputs",
            "watchpoint_manifest_sha256",
        },
        "capsule producer receipt",
    )
    if receipt["schema_version"] != 1:
        raise BenchmarkValidationError("capsule producer receipt schema drifted")
    if receipt["claim_scope"] != _EXPECTED_CAPSULE_PRODUCER_CLAIM_SCOPE:
        raise BenchmarkValidationError("capsule producer claim scope drifted")
    if receipt["coherence_id"] != capsule["coherence_id"]:
        raise BenchmarkValidationError("capsule producer coherence id drifted")
    candidate = receipt["candidate"]
    expected_candidate = {
        "code_pin": source["code_pin"],
        "id": candidate_id,
        "plan_sha256": plan["plan_sha256"],
        "source_authority_sha256": source["authority_sha256"],
        "stablehlo_authority_sha256": stablehlo["authority_sha256"],
    }
    if candidate != expected_candidate:
        raise BenchmarkValidationError("capsule producer candidate tuple drifted")
    if receipt["backend"] != {
        "device_count": 2,
        "device_ids": [0, 1],
        "platform": "cpu",
    }:
        raise BenchmarkValidationError("capsule producer backend was not forced CPU")
    environment = receipt["environment"]
    if not isinstance(environment, dict):
        raise BenchmarkValidationError("capsule producer environment is invalid")
    _exact_keys(
        environment,
        set(_EXPECTED_CAPSULE_ENVIRONMENT) | {"python_version"},
        "capsule producer environment",
    )
    if any(
        environment[key] != expected
        for key, expected in _EXPECTED_CAPSULE_ENVIRONMENT.items()
    ):
        raise BenchmarkValidationError("capsule producer environment drifted")
    _string(environment["python_version"], "capsule producer Python version")
    if environment != execution_authority["environment"]:
        raise BenchmarkValidationError(
            "capsule environment is not the pinned execution authority"
        )
    if receipt["execution"] != _EXPECTED_CAPSULE_EXECUTION:
        raise BenchmarkValidationError("capsule producer execution scope drifted")
    artifact = receipt["artifact"]
    if artifact != {"bytes": len(artifact_raw), "sha256": artifact_sha}:
        raise BenchmarkValidationError("capsule producer output artifact drifted")
    device_path, device_sha, device_raw = _binding(
        receipt_path.parent,
        device_evidence_binding,
        "capsule producer device evidence",
        limit=_MAX_ARTIFACT_BYTES,
    )
    if receipt["device_evidence"] != {
        "bytes": len(device_raw),
        "sha256": device_sha,
    }:
        raise BenchmarkValidationError("capsule producer device evidence drifted")
    device_arrays = _inspect_npz(device_raw)
    if set(device_arrays) != set(_CAPSULE_DEVICE_EVIDENCE_SCHEMA):
        raise BenchmarkValidationError("capsule device evidence catalogue drifted")
    state_arrays = _inspect_npz(artifact_raw)
    device_records: dict[str, dict[str, Any]] = {}
    for name, (expected_dtype, expected_shape, state_key, state_prefix) in (
        _CAPSULE_DEVICE_EVIDENCE_SCHEMA.items()
    ):
        observed = device_arrays[name]
        if (
            observed["storage_dtype"] != expected_dtype
            or observed["shape"] != expected_shape
        ):
            raise BenchmarkValidationError(
                f"capsule device evidence array drifted: {name}"
            )
        if name == "contract_valid_owners":
            if observed["raw"] != b"\x01\x01":
                raise BenchmarkValidationError(
                    "capsule device contracts did not agree on success"
                )
        else:
            owner0, _ = _selected_array(
                observed, [0], f"capsule device evidence {name} owner0"
            )
            owner1, _ = _selected_array(
                observed, [1], f"capsule device evidence {name} owner1"
            )
            if owner0 != owner1:
                raise BenchmarkValidationError(
                    f"capsule replicated device owners disagree: {name}"
                )
            canonical, _ = _selected_array(
                observed,
                list(state_prefix),
                f"capsule device evidence {name} canonical",
            )
            if state_key not in state_arrays or canonical != state_arrays[state_key]["raw"]:
                raise BenchmarkValidationError(
                    f"capsule device evidence is not the canonical state: {name}"
                )
        device_records[name] = {
            "array_sha256": sha256(observed["raw"]).hexdigest(),
            "shape": list(observed["shape"]),
            "storage_dtype": observed["storage_dtype"],
        }
    producer = _verify_capsule_producer_blob(
        receipt["producer"], receipt_path.parent, implementation
    )
    if producer != execution_authority["producer"]:
        raise BenchmarkValidationError(
            "capsule producer is not the pinned execution authority"
        )
    if receipt["installed_producer"] != execution_authority["installed_producer"]:
        raise BenchmarkValidationError(
            "installed capsule producer is not the pinned execution authority"
        )
    if receipt["source_snapshot"] != execution_authority["source_snapshot"]:
        raise BenchmarkValidationError(
            "capsule source snapshot is not the pinned execution authority"
        )
    if receipt["loaded_dependencies"] != execution_authority["loaded_dependencies"]:
        raise BenchmarkValidationError(
            "capsule dependencies are not the pinned execution authority"
        )
    if receipt["tensor_receipts"] != execution_authority["tensor_receipts"]:
        raise BenchmarkValidationError(
            "capsule tensor receipts are not the pinned execution authority"
        )
    if receipt["upstream_inputs"] != execution_authority["upstream_inputs"]:
        raise BenchmarkValidationError(
            "capsule upstream inputs are not the pinned execution authority"
        )
    input_path, input_sha, input_raw = _binding(
        receipt_path.parent,
        input_binding,
        "capsule producer input artifact",
        limit=_MAX_ARTIFACT_BYTES,
    )
    input_arrays = _inspect_npz(input_raw)
    if set(input_arrays) != set(_CAPSULE_INPUT_SCHEMA):
        raise BenchmarkValidationError("capsule producer input array catalogue drifted")
    declared_inputs = receipt["input_arrays"]
    if not isinstance(declared_inputs, dict) or set(declared_inputs) != set(input_arrays):
        raise BenchmarkValidationError("capsule producer input manifest drifted")
    input_records: dict[str, dict[str, Any]] = {}
    for name, (expected_dtype, expected_shape) in _CAPSULE_INPUT_SCHEMA.items():
        observed = input_arrays[name]
        declared = declared_inputs[name]
        if not isinstance(declared, dict):
            raise BenchmarkValidationError("capsule producer input record is invalid")
        _exact_keys(
            declared,
            {"array_sha256", "shape", "storage_dtype"},
            f"capsule producer input {name}",
        )
        record = {
            "array_sha256": sha256(observed["raw"]).hexdigest(),
            "shape": list(observed["shape"]),
            "storage_dtype": observed["storage_dtype"],
        }
        if (
            observed["storage_dtype"] != expected_dtype
            or observed["shape"] != expected_shape
            or declared != record
        ):
            raise BenchmarkValidationError(
                f"capsule producer input array drifted: {name}"
            )
        if expected_dtype in {"<u2", "<f4"}:
            _require_finite_raw(
                observed["raw"], expected_dtype, f"capsule producer input {name}"
            )
        input_records[name] = record
    if input_records != execution_authority["input_arrays"]:
        raise BenchmarkValidationError(
            "capsule inputs are not the pinned execution authority"
        )
    for name, expected in execution_authority.get("runtime_input_arrays", {}).items():
        if input_records.get(name) != expected:
            raise BenchmarkValidationError(
                f"capsule input is not its sealed runtime tensor payload: {name}"
            )
    watchpoint_manifest_sha = sha256(
        _canonical_json({"watchpoints": list(records)}).encode("ascii")
    ).hexdigest()
    if receipt["watchpoint_manifest_sha256"] != watchpoint_manifest_sha:
        raise BenchmarkValidationError("capsule producer watchpoint manifest drifted")
    success_path, success_sha, success_raw = _binding(
        base, success_binding, "capsule producer SUCCESS", limit=_MAX_JSON_BYTES
    )
    success = _load_json(success_raw, "capsule producer SUCCESS")
    if success_raw != (_canonical_json(success) + "\n").encode("ascii"):
        raise BenchmarkValidationError("capsule producer SUCCESS is not canonical JSON")
    expected_success = {
        "artifact_sha256": artifact_sha,
        "device_evidence_sha256": device_sha,
        "input_artifact_sha256": input_sha,
        "producer_receipt_sha256": receipt_sha,
        "schema_version": 1,
    }
    if success != expected_success:
        raise BenchmarkValidationError("capsule producer SUCCESS drifted")
    return {
        "backend": dict(receipt["backend"]),
        "device_evidence_arrays": device_records,
        "device_evidence_path": str(device_path),
        "device_evidence_sha256": device_sha,
        "input_arrays": input_records,
        "input_artifact_path": str(input_path),
        "input_artifact_sha256": input_sha,
        "producer": producer,
        "producer_receipt_path": str(receipt_path),
        "producer_receipt_sha256": receipt_sha,
        "success_path": str(success_path),
        "success_sha256": success_sha,
        "watchpoint_manifest_sha256": watchpoint_manifest_sha,
    }


def _verify_capsule(
    value: Any,
    base: Path,
    *,
    candidate_id: str,
    implementation: Mapping[str, Any],
    execution_authority: Mapping[str, Any],
    source: Mapping[str, Any],
    stablehlo: Mapping[str, Any],
    plan: Mapping[str, Any],
    required_watchpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    path, capsule_sha, raw = _binding(
        base, value, "candidate coherent capsule", limit=_MAX_JSON_BYTES
    )
    capsule = _load_json(raw, "candidate coherent capsule")
    _exact_keys(
        capsule,
        {
            "artifact",
            "candidate_id",
            "claim_scope",
            "code_pin",
            "coherence_id",
            "plan_sha256",
            "producer_input_artifact",
            "producer_device_evidence",
            "producer_receipt",
            "producer_success",
            "schema_version",
            "source_authority_sha256",
            "stablehlo_authority_sha256",
            "watchpoints",
        },
        "candidate coherent capsule",
    )
    if capsule["schema_version"] != GATE_D_PRECOMPILE_ADMISSION_SCHEMA_VERSION:
        raise BenchmarkValidationError("candidate capsule schema drifted")
    if (
        capsule["candidate_id"] != candidate_id
        or capsule["code_pin"] != source["code_pin"]
        or capsule["plan_sha256"] != plan["plan_sha256"]
        or capsule["source_authority_sha256"] != source["authority_sha256"]
        or capsule["stablehlo_authority_sha256"] != stablehlo["authority_sha256"]
    ):
        raise BenchmarkValidationError("candidate capsule authority tuple drifted")
    coherence_id = _identifier(capsule["coherence_id"], "capsule coherence id")
    _string(capsule["claim_scope"], "capsule claim scope")
    artifact_path, artifact_sha, artifact_raw = _binding(
        path.parent,
        capsule["artifact"],
        "candidate capsule artifact",
        limit=_MAX_ARTIFACT_BYTES,
    )
    arrays = _inspect_npz(artifact_raw)
    watchpoints = capsule["watchpoints"]
    if not isinstance(watchpoints, list) or not watchpoints:
        raise BenchmarkValidationError("candidate capsule watchpoints must be non-empty")
    seen: dict[str, set[str]] = {}
    selected_references: set[tuple[str, tuple[int, ...]]] = set()
    selected_values: dict[tuple[str, str], bytes] = {}
    referenced_array_keys: set[str] = set()
    records: list[dict[str, Any]] = []
    for index, watchpoint in enumerate(watchpoints):
        if not isinstance(watchpoint, dict):
            raise BenchmarkValidationError(f"capsule watchpoint {index} is invalid")
        _exact_keys(
            watchpoint,
            {
                "arrays",
                "id",
                "layer",
                "layout",
                "owner_ids",
                "position",
            },
            f"capsule watchpoint {index}",
        )
        watchpoint_id = _identifier(watchpoint["id"], f"capsule watchpoint {index} id")
        if watchpoint_id in seen:
            raise BenchmarkValidationError("candidate capsule watchpoint is duplicated")
        if watchpoint_id not in required_watchpoints:
            raise BenchmarkValidationError(
                f"candidate capsule watchpoint is unexpected: {watchpoint_id}"
            )
        layer = _nonnegative_int(watchpoint["layer"], f"watchpoint {watchpoint_id} layer")
        position = _nonnegative_int(
            watchpoint["position"], f"watchpoint {watchpoint_id} position"
        )
        expected_watchpoint = required_watchpoints[watchpoint_id]
        if (
            layer != expected_watchpoint["layer"]
            or position != expected_watchpoint["position"]
        ):
            raise BenchmarkValidationError(
                f"watchpoint layer/position drifted: {watchpoint_id}"
            )
        layout = _identifier(watchpoint["layout"], f"watchpoint {watchpoint_id} layout")
        owner_ids = watchpoint["owner_ids"]
        if (
            not isinstance(owner_ids, list)
            or not owner_ids
            or len(owner_ids) > 4
            or any(
                not isinstance(owner, int) or isinstance(owner, bool) or owner < 0
                for owner in owner_ids
            )
            or len(set(owner_ids)) != len(owner_ids)
        ):
            raise BenchmarkValidationError(f"watchpoint owners are invalid: {watchpoint_id}")
        plan_watchpoint = plan["watchpoints"].get(watchpoint_id)
        if plan_watchpoint != {"layout": layout, "owner_ids": owner_ids}:
            raise BenchmarkValidationError(
                f"watchpoint layout/owners are not sealed plan authority: {watchpoint_id}"
            )
        array_specs = watchpoint["arrays"]
        if not isinstance(array_specs, list) or not array_specs:
            raise BenchmarkValidationError(f"watchpoint arrays are absent: {watchpoint_id}")
        roles: set[str] = set()
        array_records: list[dict[str, Any]] = []
        for array_index, array_spec in enumerate(array_specs):
            if not isinstance(array_spec, dict):
                raise BenchmarkValidationError("watchpoint array must be an object")
            _exact_keys(
                array_spec,
                {
                    "array_key",
                    "array_sha256",
                    "index_prefix",
                    "owner_axis",
                    "owner_axis_ids",
                    "owner_id",
                    "role",
                    "semantic_dtype",
                    "shape",
                    "storage_dtype",
                },
                f"watchpoint {watchpoint_id} array {array_index}",
            )
            role = _identifier(array_spec["role"], f"watchpoint {watchpoint_id} role")
            if role in roles:
                raise BenchmarkValidationError(
                    f"watchpoint array role is duplicated: {watchpoint_id}"
                )
            roles.add(role)
            expected_array = expected_watchpoint["arrays"].get(role)
            if expected_array is None:
                raise BenchmarkValidationError(
                    f"watchpoint array role is unexpected: {watchpoint_id}.{role}"
                )
            key = _string(array_spec["array_key"], f"watchpoint {watchpoint_id} key")
            if key not in arrays:
                raise BenchmarkValidationError(f"watchpoint array is absent: {key}")
            referenced_array_keys.add(key)
            index_prefix = array_spec["index_prefix"]
            if not isinstance(index_prefix, list) or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in index_prefix
            ):
                raise BenchmarkValidationError(
                    f"watchpoint index prefix is invalid: {watchpoint_id}.{role}"
                )
            if index_prefix != expected_array["index_prefix"]:
                raise BenchmarkValidationError(
                    f"watchpoint owner-axis prefix drifted: {watchpoint_id}.{role}"
                )
            reference = (key, tuple(index_prefix))
            if reference in selected_references:
                raise BenchmarkValidationError(
                    f"watchpoint selected array is reused: {watchpoint_id}.{role}"
                )
            selected_references.add(reference)
            expected_array_sha = _sha(
                array_spec["array_sha256"], f"watchpoint {watchpoint_id} array SHA-256"
            )
            storage_dtype = _string(
                array_spec["storage_dtype"], f"watchpoint {watchpoint_id} storage dtype"
            )
            semantic_dtype = _identifier(
                array_spec["semantic_dtype"], f"watchpoint {watchpoint_id} semantic dtype"
            )
            if semantic_dtype not in _SEMANTIC_DTYPES:
                raise BenchmarkValidationError(
                    f"watchpoint semantic dtype is unsupported: {watchpoint_id}"
                )
            shape = array_spec["shape"]
            if not isinstance(shape, list) or any(
                not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in shape
            ):
                raise BenchmarkValidationError(f"watchpoint shape is invalid: {watchpoint_id}")
            owner_id = array_spec["owner_id"]
            if owner_id is not None and (
                not isinstance(owner_id, int)
                or isinstance(owner_id, bool)
                or owner_id < 0
                or owner_id > 31
            ):
                raise BenchmarkValidationError(
                    f"watchpoint array owner is invalid: {watchpoint_id}.{role}"
                )
            expected_owner_slot = expected_array["owner_slot"]
            expected_owner_id = (
                None if expected_owner_slot is None else owner_ids[expected_owner_slot]
            )
            if owner_id != expected_owner_id:
                raise BenchmarkValidationError(
                    f"watchpoint array owner drifted: {watchpoint_id}.{role}"
                )
            owner_axis = array_spec["owner_axis"]
            owner_axis_ids = array_spec["owner_axis_ids"]
            if not isinstance(owner_axis_ids, list) or any(
                not isinstance(owner, int)
                or isinstance(owner, bool)
                or owner < 0
                or owner > 31
                for owner in owner_axis_ids
            ):
                raise BenchmarkValidationError(
                    f"watchpoint owner-axis ids are invalid: {watchpoint_id}.{role}"
                )
            expected_owner_axis = expected_array["owner_axis"]
            expected_axis_ids = [
                owner_ids[slot] for slot in expected_array["owner_axis_slots"]
            ]
            if (
                owner_axis != expected_owner_axis
                or owner_axis_ids != expected_axis_ids
                or (
                    owner_axis is not None
                    and (
                        not isinstance(owner_axis, int)
                        or isinstance(owner_axis, bool)
                        or owner_axis < 0
                    )
                )
            ):
                raise BenchmarkValidationError(
                    f"watchpoint owner-axis mapping drifted: {watchpoint_id}.{role}"
                )
            selected, selected_shape = _selected_array(
                arrays[key], index_prefix, f"watchpoint {watchpoint_id}"
            )
            if semantic_dtype == "bf16_bits":
                _require_finite_raw(
                    selected, "<u2", f"watchpoint {watchpoint_id}.{role}"
                )
            elif semantic_dtype == "float32":
                _require_finite_raw(
                    selected, "<f4", f"watchpoint {watchpoint_id}.{role}"
                )
            selected_values[(watchpoint_id, role)] = selected
            expected_storage_shape = (
                tuple([len(owner_ids), 1, *shape])
                if index_prefix
                else tuple(shape)
            )
            if arrays[key]["shape"] != expected_storage_shape:
                raise BenchmarkValidationError(
                    f"watchpoint storage shape has unreferenced rows: {watchpoint_id}.{role}"
                )
            if owner_axis is not None and (
                owner_axis >= len(selected_shape)
                or selected_shape[owner_axis] != len(owner_axis_ids)
            ):
                raise BenchmarkValidationError(
                    f"watchpoint owner axis does not match array shape: {watchpoint_id}.{role}"
                )
            if (
                storage_dtype != arrays[key]["storage_dtype"]
                or list(selected_shape) != shape
                or sha256(selected).hexdigest() != expected_array_sha
                or storage_dtype != expected_array["storage_dtype"]
                or semantic_dtype != expected_array["semantic_dtype"]
                or shape != expected_array["shape"]
            ):
                raise BenchmarkValidationError(
                    f"watchpoint array metadata drifted: {watchpoint_id}"
                )
            array_records.append(
                {
                    "array_key": key,
                    "array_sha256": expected_array_sha,
                    "index_prefix": index_prefix,
                    "owner_axis": owner_axis,
                    "owner_axis_ids": owner_axis_ids,
                    "owner_id": owner_id,
                    "role": role,
                    "semantic_dtype": semantic_dtype,
                    "shape": shape,
                    "storage_dtype": storage_dtype,
                }
            )
        if roles != set(expected_watchpoint["arrays"]):
            raise BenchmarkValidationError(f"watchpoint array roles drifted: {watchpoint_id}")
        if watchpoint_id in {
            "layer1.normalized",
            "layer1.query",
            "layer1.head_weights",
        } and len({item["array_key"] for item in array_records}) != 1:
            raise BenchmarkValidationError(
                f"watchpoint owner slices do not share one owner axis: {watchpoint_id}"
            )
        seen[watchpoint_id] = roles
        records.append(
            {
                "arrays": sorted(array_records, key=lambda item: item["role"]),
                "id": watchpoint_id,
                "layer": layer,
                "layout": layout,
                "owner_ids": owner_ids,
                "position": position,
            }
        )
    if set(seen) != set(required_watchpoints):
        missing_watchpoints = sorted(set(required_watchpoints) - set(seen))
        raise BenchmarkValidationError(
            "candidate capsule watchpoint set drifted: "
            f"missing={missing_watchpoints}"
        )
    if referenced_array_keys != set(arrays):
        raise BenchmarkValidationError(
            "candidate NPZ contains unreferenced arrays: "
            f"{sorted(set(arrays) - referenced_array_keys)}"
        )
    derived_rms_input = _derive_fp32_sum_from_bf16_bits(
        selected_values[("layer1.rms_operands_bf16", "hidden_update")],
        selected_values[("layer1.rms_operands_bf16", "residual")],
    )
    if derived_rms_input != selected_values[("layer1.rms_input_fp32", "value")]:
        raise BenchmarkValidationError(
            "candidate RMS FP32 input is not derived from its sealed BF16 operands"
        )
    normalized_roles = sorted(
        role for watchpoint_id, role in selected_values if watchpoint_id == "layer1.normalized"
    )
    normalized_values = [
        selected_values[("layer1.normalized", role)] for role in normalized_roles
    ]
    if not normalized_values or any(
        value != normalized_values[0] for value in normalized_values[1:]
    ):
        raise BenchmarkValidationError("candidate normalized owner values differ")
    current_key = selected_values[("layer1.current_key", "value")]
    rounded_key = _bf16_bits_from_fp32(current_key)
    cache = selected_values[("layer1.cache_history", "value")]
    cache_shape = tuple(
        required_watchpoints["layer1.cache_history"]["arrays"]["value"]["shape"]
    )
    if cache_shape not in {(2, 16, 256, 128), (4, 16, 256, 128)}:
        raise BenchmarkValidationError("candidate cache geometry drifted")
    current_position = 8155
    logical_page_size = cache_shape[0] * cache_shape[2]
    page = current_position // logical_page_size
    page_row = current_position % logical_page_size
    owner_slot = page_row // cache_shape[2]
    local_row = page_row % cache_shape[2]
    cache_key_offset = (
        ((owner_slot * cache_shape[1] + page) * cache_shape[2] + local_row)
        * cache_shape[3]
        * 2
    )
    if cache[cache_key_offset : cache_key_offset + len(rounded_key)] != rounded_key:
        raise BenchmarkValidationError(
            "candidate cache current row is not the BF16 round of its current key"
        )
    positions_raw = selected_values[("layer1.scorer_event1", "positions")]
    scores_raw = selected_values[("layer1.scorer_event1", "scores")]
    valid_count_raw = selected_values[("layer1.scorer_event1", "valid_count")]
    positions = [item[0] for item in struct.iter_unpack("<i", positions_raw)]
    scores = [item[0] for item in struct.iter_unpack("<f", scores_raw)]
    valid_count = struct.unpack("<i", valid_count_raw)[0]
    expected_outputs = execution_authority["expected_outputs"]
    if (
        valid_count != 2048
        or len(positions) != 2048
        or len(set(positions)) != 2048
        or any(position < 0 or position > current_position for position in positions)
        or any(
            struct.unpack("<I", scores_raw[index * 4 : (index + 1) * 4])[0]
            & 0x7F800000
            == 0x7F800000
            for index in range(len(scores))
        )
        or any(scores[index] < scores[index + 1] for index in range(2047))
        or any(
            scores[index] == scores[index + 1]
            and positions[index] >= positions[index + 1]
            for index in range(2047)
        )
    ):
        raise BenchmarkValidationError(
            "candidate scorer event is not one exact ordered cutoff-active set"
        )
    if (
        sha256(positions_raw).hexdigest()
        != expected_outputs["event1_positions_sha256"]
        or sha256(scores_raw).hexdigest()
        != expected_outputs["event1_scores_sha256"]
        or valid_count != expected_outputs["event1_valid_count"]
        or sha256(
            selected_values[("layer1.rms_operands_bf16", "hidden_update")]
        ).hexdigest()
        != expected_outputs["rms_hidden_update_sha256"]
        or sha256(
            selected_values[("layer1.rms_operands_bf16", "residual")]
        ).hexdigest()
        != expected_outputs["rms_residual_sha256"]
    ):
        raise BenchmarkValidationError(
            "candidate output does not match the pinned accepted authority"
        )
    producer = _verify_capsule_producer_receipt(
        artifact_raw=artifact_raw,
        artifact_sha=artifact_sha,
        base=path.parent,
        candidate_id=candidate_id,
        capsule=capsule,
        execution_authority=execution_authority,
        implementation=implementation,
        device_evidence_binding=capsule["producer_device_evidence"],
        input_binding=capsule["producer_input_artifact"],
        plan=plan,
        producer_binding=capsule["producer_receipt"],
        records=sorted(records, key=lambda item: item["id"]),
        source=source,
        stablehlo=stablehlo,
        success_binding=capsule["producer_success"],
    )
    input_arrays = _inspect_npz(
        _binding(
            path.parent,
            capsule["producer_input_artifact"],
            "candidate producer input cross-check",
            limit=_MAX_ARTIFACT_BYTES,
        )[2]
    )
    if (
        input_arrays["rms_hidden_update_bf16_bits"]["raw"]
        != selected_values[("layer1.rms_operands_bf16", "hidden_update")]
        or input_arrays["rms_residual_bf16_bits"]["raw"]
        != selected_values[("layer1.rms_operands_bf16", "residual")]
    ):
        raise BenchmarkValidationError(
            "candidate RMS operands are not the sealed producer inputs"
        )
    return {
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha,
        "capsule_path": str(path),
        "capsule_sha256": capsule_sha,
        "coherence_id": coherence_id,
        "derived_rms_input_sha256": sha256(derived_rms_input).hexdigest(),
        "producer": producer,
        "producer_provenance_verified": True,
        "validation_scope": (
            "typed candidate-coherent bounded CPU replay, pinned Git producer/input/"
            "SUCCESS, exact RMS derivation, cache-current-key and scorer-order invariants; "
            "no TPU, performance or Gate-D claim"
        ),
        "watchpoints": sorted(records, key=lambda item: item["id"]),
    }


def admit_gate_d_precompile_candidates(
    contract_path: Path | str, expected_contract_sha256: str
) -> dict[str, Any]:
    """Authenticate and classify precompile candidates without importing JAX."""

    contract_path = Path(contract_path)
    expected_contract_sha256 = _sha(expected_contract_sha256, "contract SHA-256")
    raw = _snapshot(
        contract_path,
        expected_contract_sha256,
        "Gate-D precompile admission contract",
        limit=_MAX_JSON_BYTES,
    )
    contract = _load_json(raw, "Gate-D precompile admission contract")
    _exact_keys(
        contract,
        {
            "candidates",
            "causal_frontier",
            "claim_scope",
            "contract_id",
            "evidence",
            "implementation",
            "inherited_v1",
            "locality_contract",
            "physical_locality_authority",
            "required_coherent_watchpoints",
            "schema_version",
        },
        "Gate-D precompile admission contract",
    )
    if contract["schema_version"] != GATE_D_PRECOMPILE_ADMISSION_SCHEMA_VERSION:
        raise BenchmarkValidationError("Gate-D precompile admission schema drifted")
    contract_id = _identifier(contract["contract_id"], "contract id")
    claim_scope = _string(contract["claim_scope"], "claim scope")
    base = contract_path.parent
    implementation = _verify_implementation(contract["implementation"], base)
    inherited, closed_fingerprints = _verify_inherited_v1(contract["inherited_v1"], base)
    physical_locality = _verify_physical_locality(
        contract["physical_locality_authority"], base
    )

    frontier = contract["causal_frontier"]
    if not isinstance(frontier, dict):
        raise BenchmarkValidationError("causal frontier must be an object")
    _exact_keys(frontier, {"evidence_id", "id", "order"}, "causal frontier")
    frontier_id = _identifier(frontier["id"], "causal frontier id")
    if frontier["order"] != 0:
        raise BenchmarkValidationError("causal frontier order must remain zero")

    locality = contract["locality_contract"]
    expected_locality = {
        "logical_rows": 1,
        "max_collective_group_size": 4,
        "no_full_pod_hidden_reconstruction": True,
        "no_host_effects": True,
    }
    _verify_locality(locality, expected_locality, "locality contract")

    required_raw = contract["required_coherent_watchpoints"]
    expected_required = {
        "PP16_LP2": _serialized_watchpoint_schema(2),
        "PP8_LP4": _serialized_watchpoint_schema(4),
    }
    if required_raw != expected_required:
        raise BenchmarkValidationError("required coherent watchpoint schema drifted")
    if any(
        frontier_id not in _required_watchpoint_schema(owner_count)
        for owner_count in (2, 4)
    ):
        raise BenchmarkValidationError("causal frontier is absent from required watchpoints")

    evidence = contract["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise BenchmarkValidationError("precompile evidence must be non-empty")
    evidence_shas: dict[str, str] = {}
    evidence_documents: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise BenchmarkValidationError(f"evidence {index} must be an object")
        _exact_keys(item, {"claim_scope", "id", "path", "sha256"}, f"evidence {index}")
        evidence_id = _identifier(item["id"], f"evidence {index} id")
        if evidence_id in evidence_shas:
            raise BenchmarkValidationError("evidence id is duplicated")
        expected = _sha(item["sha256"], f"evidence {evidence_id} SHA-256")
        path = _resolve(base, item["path"], f"evidence {evidence_id} path")
        evidence_raw = _snapshot(
            path,
            expected,
            f"precompile evidence {evidence_id}",
            limit=_MAX_ARTIFACT_BYTES,
        )
        _string(item["claim_scope"], f"evidence {evidence_id} claim scope")
        evidence_shas[evidence_id] = expected
        evidence_documents[evidence_id] = _load_json(
            evidence_raw, f"precompile evidence {evidence_id}"
        )
    if set(evidence_shas) != _EXPECTED_EVIDENCE_IDS:
        raise BenchmarkValidationError("precompile evidence catalogue drifted")
    frontier_evidence = _identifier(frontier["evidence_id"], "frontier evidence id")
    if frontier_evidence not in evidence_shas:
        raise BenchmarkValidationError("causal frontier evidence is absent")
    frontier_document = evidence_documents["observability_frontier"]
    if (
        frontier_document.get("classification")
        != "OBSERVABILITY_GAP;GATE_D_OPEN;NO_TPU_SUCCESSOR"
        or frontier_document.get("causal_frontier")
        != {
            "id": frontier_id,
            "missing_accepted": True,
            "missing_candidate": True,
            "order": 0,
        }
    ):
        raise BenchmarkValidationError("observability frontier evidence drifted")
    constructability = evidence_documents["constructability"]
    if (
        constructability.get("new_variant_capsule_constructable") is not False
        or constructability.get("variant_source_bound") is not False
        or constructability.get("tpu_successor_authorized") is not False
        or "append_only_precompile_admission_v2_with_source_and_stablehlo_authority"
        not in constructability.get("required_next", [])
    ):
        raise BenchmarkValidationError("constructability evidence drifted")
    direct_rejection = evidence_documents["direct_shadow_rejection"]
    if (
        direct_rejection.get("classification")
        != "DIRECT_UNROUNDED_FP32_SHADOW_SUBSTITUTION_REJECTED;"
        "OTHER_SHADOW_FORMS_UNADJUDICATED;NO_TPU_SUCCESSOR"
        or direct_rejection.get("causal_frontier") != frontier_id
        or direct_rejection.get("tpu_successor_authorized") is not False
    ):
        raise BenchmarkValidationError("direct-shadow rejection evidence drifted")
    accepted_semantics = _accepted_semantics_from_evidence(direct_rejection)
    shadow = evidence_documents["shadow_adjudication"]
    unresolved = shadow.get("unresolved_variant_ids")
    variant_results = shadow.get("variant_results")
    if (
        shadow.get("classification")
        != "NO_OFFLINE_COMPLETE_SHADOW_VARIANT;"
        "AUXILIARY_DEVICE_VARIANTS_REMAIN_UNADJUDICATED;"
        "GATE_D_OPEN;NO_TPU_SUCCESSOR"
        or shadow.get("tpu_successor_authorized") is not False
        or not isinstance(unresolved, list)
        or not isinstance(variant_results, list)
    ):
        raise BenchmarkValidationError("shadow adjudication evidence drifted")
    expected_normal_forms = {
        item.get("id"): item.get("normal_form")
        for item in variant_results
        if isinstance(item, dict) and item.get("id") in unresolved
    }
    if set(expected_normal_forms) != set(unresolved):
        raise BenchmarkValidationError("shadow unresolved variant evidence drifted")
    if set(unresolved) != set(_EXPECTED_SURVIVORS) or any(
        expected_normal_forms[candidate_id]
        != _EXPECTED_SURVIVORS[candidate_id]["normal_form"]
        for candidate_id in _EXPECTED_SURVIVORS
    ):
        raise BenchmarkValidationError("sealed survivor identities drifted")

    candidates = contract["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise BenchmarkValidationError("precompile candidates must be non-empty")
    candidate_ids: set[str] = set()
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise BenchmarkValidationError(f"candidate {index} must be an object")
        _exact_keys(
            candidate,
            {
                "causal_frontier_action",
                "capsule_execution_authority",
                "coherent_state_capsule",
                "evidence_ids",
                "host_effect",
                "id",
                "logical_rows",
                "max_collective_group_size",
                "mechanism_fingerprint",
                "normal_form",
                "plan_authority",
                "reconstructs_full_pod_hidden",
                "source_authority",
                "stablehlo_authority",
                "summary",
            },
            f"candidate {index}",
        )
        candidate_id = _identifier(candidate["id"], f"candidate {index} id")
        if candidate_id in candidate_ids:
            raise BenchmarkValidationError("candidate id is duplicated")
        candidate_ids.add(candidate_id)
        _string(candidate["summary"], f"candidate {candidate_id} summary")
        evidence_ids = candidate["evidence_ids"]
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise BenchmarkValidationError(f"candidate evidence is absent: {candidate_id}")
        candidate_evidence = [
            _identifier(item, f"candidate {candidate_id} evidence") for item in evidence_ids
        ]
        if set(candidate_evidence) != _EXPECTED_EVIDENCE_IDS or len(
            candidate_evidence
        ) != len(_EXPECTED_EVIDENCE_IDS):
            raise BenchmarkValidationError(f"candidate evidence drifted: {candidate_id}")
        normal_form = candidate["normal_form"]
        if not isinstance(normal_form, dict):
            raise BenchmarkValidationError(f"candidate normal form is invalid: {candidate_id}")
        _exact_keys(
            normal_form,
            {"compensation", "dependency", "normalization_input", "primary_recurrence"},
            f"candidate {candidate_id} normal form",
        )
        for key, item in normal_form.items():
            _identifier(item, f"candidate {candidate_id} normal form {key}")
        if expected_normal_forms.get(candidate_id) != normal_form:
            raise BenchmarkValidationError(
                f"candidate is not the sealed unresolved normal form: {candidate_id}"
            )
        if candidate["mechanism_fingerprint"] != _EXPECTED_SURVIVORS.get(
            candidate_id, {}
        ).get("mechanism_fingerprint"):
            raise BenchmarkValidationError(
                f"candidate fingerprint is not the sealed survivor: {candidate_id}"
            )
        fingerprint = _mechanism_fingerprint(
            candidate["mechanism_fingerprint"], f"candidate {candidate_id} fingerprint"
        )
        reasons: list[str] = []
        if fingerprint in closed_fingerprints:
            reasons.append("DUPLICATES_V1_CLOSED_FAMILY")
        if _positive_int(candidate["logical_rows"], f"candidate {candidate_id} rows") != 1:
            reasons.append("NOT_TRUE_ONE_ROW")
        if _positive_int(
            candidate["max_collective_group_size"], f"candidate {candidate_id} group size"
        ) > 4:
            reasons.append("NONLOCAL_COLLECTIVE_GROUP")
        if _boolean(
            candidate["reconstructs_full_pod_hidden"],
            f"candidate {candidate_id} full pod",
        ):
            reasons.append("FULL_POD_HIDDEN_RECONSTRUCTION")
        if _boolean(candidate["host_effect"], f"candidate {candidate_id} host effect"):
            reasons.append("HOST_EFFECT_OR_DISPATCH")
        action = candidate["causal_frontier_action"]
        if action not in _FRONTIER_ACTIONS:
            raise BenchmarkValidationError(f"candidate frontier action is invalid: {candidate_id}")

        source_report: dict[str, Any] | None = None
        plan_report: dict[str, Any] | None = None
        stablehlo_report: dict[str, Any] | None = None
        execution_report: dict[str, Any] | None = None
        capsule_report: dict[str, Any] | None = None
        if candidate["source_authority"] is None:
            reasons.append("MISSING_SOURCE_AST_AUTHORITY")
        else:
            try:
                source_report = _verify_source_authority(
                    candidate["source_authority"],
                    base,
                    candidate_id=candidate_id,
                    frontier_action=action,
                    mechanism_fingerprint_sha256=fingerprint,
                    normal_form=normal_form,
                    locality=locality,
                    implementation=implementation,
                    accepted_semantics=accepted_semantics,
                )
                if source_report["executable_source_authority"] is not True:
                    reasons.append("MISSING_EXECUTABLE_SOURCE_AUTHORITY")
            except BenchmarkValidationError as error:
                reasons.append("INVALID_SOURCE_AST_AUTHORITY")
                source_report = {"refusal": str(error)}
        if candidate["plan_authority"] is None:
            reasons.append("MISSING_PLAN_AUTHORITY")
        else:
            try:
                plan_report = _verify_plan_authority(
                    candidate["plan_authority"], base, physical_locality
                )
            except BenchmarkValidationError as error:
                reasons.append("INVALID_PLAN_AUTHORITY")
                plan_report = {"refusal": str(error)}
        if candidate["stablehlo_authority"] is None:
            reasons.append("MISSING_CAUSAL_STABLEHLO_AUTHORITY")
        elif (
            source_report is None
            or "refusal" in source_report
            or plan_report is None
            or "refusal" in plan_report
        ):
            reasons.append("UNVERIFIABLE_CAUSAL_STABLEHLO_AUTHORITY")
        else:
            try:
                stablehlo_report = _verify_stablehlo_authority(
                    candidate["stablehlo_authority"],
                    base,
                    candidate_id=candidate_id,
                    mechanism_fingerprint_sha256=fingerprint,
                    normal_form=normal_form,
                    source=source_report,
                    plan=plan_report,
                    frontier_id=frontier_id,
                    frontier_action=action,
                    locality=locality,
                    implementation=implementation,
                )
                if stablehlo_report["immutable_parser_authority"] is not True:
                    reasons.append("MISSING_IMMUTABLE_STABLEHLO_PARSER_AUTHORITY")
            except BenchmarkValidationError as error:
                reasons.append("INVALID_CAUSAL_STABLEHLO_AUTHORITY")
                stablehlo_report = {"refusal": str(error)}
        if candidate["capsule_execution_authority"] is None:
            reasons.append("MISSING_PINNED_COHERENT_CAPSULE_PRODUCER")
        else:
            try:
                execution_report = _verify_capsule_execution_authority(
                    candidate["capsule_execution_authority"],
                    base,
                    candidate_id=candidate_id,
                    implementation=implementation,
                    source=source_report or {},
                )
            except BenchmarkValidationError as error:
                reasons.append("INVALID_CAPSULE_EXECUTION_AUTHORITY")
                execution_report = {"refusal": str(error)}
        if candidate["coherent_state_capsule"] is None:
            reasons.append("MISSING_CANDIDATE_COHERENT_CAPSULE")
        elif (
            source_report is None
            or "refusal" in source_report
            or plan_report is None
            or "refusal" in plan_report
            or stablehlo_report is None
            or "refusal" in stablehlo_report
            or execution_report is None
            or "refusal" in execution_report
        ):
            reasons.append("UNVERIFIABLE_CANDIDATE_COHERENT_CAPSULE")
        else:
            try:
                capsule_report = _verify_capsule(
                    candidate["coherent_state_capsule"],
                    base,
                    candidate_id=candidate_id,
                    implementation=implementation,
                    execution_authority=execution_report,
                    source=source_report,
                    stablehlo=stablehlo_report,
                    plan=plan_report,
                    required_watchpoints=plan_report["watchpoint_schema"],
                )
                if capsule_report["producer_provenance_verified"] is not True:
                    reasons.append("MISSING_PINNED_COHERENT_CAPSULE_PRODUCER")
            except BenchmarkValidationError as error:
                reasons.append("INVALID_CANDIDATE_COHERENT_CAPSULE")
                capsule_report = {"refusal": str(error)}
        results.append(
            {
                "admitted_precompile": not reasons,
                "capsule": capsule_report,
                "capsule_execution_authority": execution_report,
                "evidence_sha256s": {
                    item: evidence_shas[item] for item in candidate_evidence
                },
                "id": candidate_id,
                "mechanism_fingerprint_sha256": fingerprint,
                "plan_authority": plan_report,
                "reasons": reasons,
                "source_authority": source_report,
                "stablehlo_authority": stablehlo_report,
            }
        )
    if candidate_ids != set(unresolved):
        raise BenchmarkValidationError("precompile candidate catalogue drifted")
    valid_stablehlo_shas = [
        item["stablehlo_authority"]["candidate_sha256"]
        for item in results
        if isinstance(item["stablehlo_authority"], dict)
        and "refusal" not in item["stablehlo_authority"]
    ]
    if len(valid_stablehlo_shas) != len(set(valid_stablehlo_shas)):
        raise BenchmarkValidationError(
            "distinct survivors reuse one candidate StableHLO implementation"
        )
    admitted = [item["id"] for item in results if item["admitted_precompile"]]
    if admitted not in ([], ["auxiliary_device_tuple_dependency"]):
        raise BenchmarkValidationError(
            "unexpected candidate set passed precompile admission v2"
        )
    classification = (
        "PRECOMPILE_CANDIDATE_ADMITTED;COMPILE_ONLY_REVIEW_REQUIRED;"
        "GATE_D_OPEN;NO_TPU_SUCCESSOR"
        if admitted
        else "NO_PRECOMPILE_CANDIDATE_ADMITTED;GATE_D_OPEN;NO_JAX_OR_TPU_SUCCESSOR"
    )
    return {
        "admitted_candidate_ids": admitted,
        "candidate_results": results,
        "claim_scope": claim_scope,
        "classification": classification,
        "compile_only_review_required": True,
        "contract_id": contract_id,
        "contract_sha256": expected_contract_sha256,
        "evidence_sha256s": dict(sorted(evidence_shas.items())),
        "gate_d_closed": False,
        "inherited_v1": inherited,
        "implementation": implementation,
        "jax_compile_or_tpu_work_performed": False,
        "physical_locality_authority": physical_locality,
        "required_coherent_watchpoints": expected_required,
        "schema_version": GATE_D_PRECOMPILE_ADMISSION_SCHEMA_VERSION,
        "tpu_successor_authorized": False,
    }


def write_gate_d_precompile_admission_report(
    path: Path | str, report: Mapping[str, Any]
) -> None:
    """Write one canonical append-only report without following symlinks."""

    path = Path(path)
    if not path.name or path.name in (".", ".."):
        raise BenchmarkValidationError(f"output path is invalid: {path}")
    payload = (_canonical_json(report) + "\n").encode("ascii")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    with _open_directory_no_symlinks(path.parent, "output parent") as parent:
        try:
            os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise BenchmarkValidationError(f"cannot inspect output path: {path}") from error
        else:
            raise BenchmarkValidationError(f"output path is occupied: {path}")
        try:
            descriptor = os.open(path.name, flags, 0o644, dir_fd=parent)
        except OSError as error:
            raise BenchmarkValidationError(f"cannot create output: {path}") from error
        created = os.fstat(descriptor)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(parent)
        except BaseException:
            try:
                current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            except OSError:
                current = None
            if current is not None and (current.st_dev, current.st_ino) == (
                created.st_dev,
                created.st_ino,
            ):
                os.unlink(path.name, dir_fd=parent)
                os.fsync(parent)
            raise
