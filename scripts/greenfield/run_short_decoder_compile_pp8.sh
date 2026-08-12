#!/usr/bin/env bash
# Protected real 78-layer PP8 body or complete-token execution.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly GREENFIELD_ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly ORACLE_REPO=/home/gianl/tpu-inference
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly PROFILE=${GLM_GREENFIELD_SHORT_DECODER_PROFILE:-2k}
readonly WARMUP=${GLM_GREENFIELD_SHORT_DECODER_WARMUP:-2}
readonly ITERATIONS=${GLM_GREENFIELD_SHORT_DECODER_ITERATIONS:-10}
readonly TRACE_STEPS=${GLM_GREENFIELD_SHORT_DECODER_TRACE_STEPS:-0}
readonly RUNTIME_KIND=${GLM_GREENFIELD_DECODER_RUNTIME_KIND:-pallas_feature_linear}
readonly VERIFY_DEVICE_ROUNDTRIP=${GLM_GREENFIELD_RUNTIME_DEVICE_ROUNDTRIP:-0}
readonly FEATURE_FUSE_ROUTE_WEIGHTING=${GLM_GREENFIELD_FEATURE_FUSE_ROUTE_WEIGHTING:-0}
readonly FEATURE_RECONSTRUCT_DOWN_FP32=${GLM_GREENFIELD_FEATURE_RECONSTRUCT_DOWN_FP32:-0}
readonly COMPLETE_TOKEN_PATH=${GLM_GREENFIELD_COMPLETE_TOKEN_PATH:-0}
readonly SPLIT_RESIDUAL_STATE=${GLM_GREENFIELD_SPLIT_RESIDUAL_STATE:-0}
readonly PREFILL_INDEX_REPAIR=${GLM_GREENFIELD_PREFILL_INDEX_REPAIR:-0}
readonly DSA_QUERY_EXACT_ASSOCIATION=${GLM_GREENFIELD_DSA_QUERY_EXACT_ASSOCIATION:-0}
readonly DSA_HEAD_KEY_EXACT_ASSOCIATION=${GLM_GREENFIELD_DSA_HEAD_KEY_EXACT_ASSOCIATION:-0}
readonly DSA_SCORE_DEFAULT_PRECISION=${GLM_GREENFIELD_DSA_SCORE_DEFAULT_PRECISION:-0}
readonly MAIN_ROPE_TABLE=${GLM_GREENFIELD_MAIN_ROPE_TABLE:-0}
readonly PREGATHERED_B512_ATTENTION=${GLM_GREENFIELD_PREGATHERED_B512_ATTENTION:-0}
readonly STRATEGY_ND_ATTENTION_PROJECTION=${GLM_GREENFIELD_STRATEGY_ND_ATTENTION_PROJECTION:-0}
readonly SHORT_CONTEXT_ORACLE=${GLM_GREENFIELD_SHORT_CONTEXT_ORACLE:-0}
readonly SHORT_CONTEXT_DSA_ORACLE=${GLM_GREENFIELD_SHORT_CONTEXT_DSA_ORACLE:-0}
readonly DSA_INTERNAL_OBSERVER=${GLM_GREENFIELD_DSA_INTERNAL_OBSERVER:-0}
readonly DSA_INTERNAL_BASELINE_NPZ=${GLM_GREENFIELD_DSA_INTERNAL_BASELINE_NPZ:-/home/gianl/gcs-models/results/greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_oracle_dsa_trace2_20260808T041407656729112Z/dsa_observer/step_00_position_8155.npz}
readonly DSA_INTERNAL_BASELINE_SHA=${GLM_GREENFIELD_DSA_INTERNAL_BASELINE_SHA:-3e54254cfb48fbcca62f48484242196d1b1d3ffc6e2482377a569439fd65b053}
readonly DSA_INTERNAL_LAYER0_REFERENCE_NPZ=${GLM_GREENFIELD_DSA_INTERNAL_LAYER0_REFERENCE_NPZ:-/home/gianl/gcs-models/oracles/greenfield/glm52/dsa_internals/8k/recovery/greenfield_legacy_layer0_dsa_internals_recovery_20260808T030853085141329Z/internal_comparison/internals.npz}
readonly DSA_INTERNAL_LAYER0_REFERENCE_SHA=${GLM_GREENFIELD_DSA_INTERNAL_LAYER0_REFERENCE_SHA:-0a724bada77d93ddc524368ac3b6e3c7f70442aba59ed96da5abde3dbe75fa39}
readonly LAYER0_RESIDUAL_VARIANTS=${GLM_GREENFIELD_LAYER0_RESIDUAL_VARIANTS:-0}
readonly LAYER0_SUBSHARD_VARIANTS=${GLM_GREENFIELD_LAYER0_SUBSHARD_VARIANTS:-0}
readonly LAYER0_ATTENTION_VARIANTS=${GLM_GREENFIELD_LAYER0_ATTENTION_VARIANTS:-0}
readonly LAYER0_ATTENTION_OUTPUT_VARIANTS=${GLM_GREENFIELD_LAYER0_ATTENTION_OUTPUT_VARIANTS:-0}
readonly LAYER0_STRATEGY_ND_ROW0=${GLM_GREENFIELD_LAYER0_STRATEGY_ND_ROW0:-0}
readonly STRATEGY_ND_CANARY_ROOT=/home/gianl/gcs-models/results/greenfield_collective_association_20260811T213152133863450Z/association
readonly STRATEGY_ND_CANARY_INPUT_BITS=$STRATEGY_ND_CANARY_ROOT/input_bits.npy
readonly STRATEGY_ND_CANARY_INPUT_SHA=1a6c2fb5c8c04b904daad45f751e0b5308919b85cb1f3cf63879c5febf3ea0e9
readonly STRATEGY_ND_CANARY_OUTPUT_BITS=$STRATEGY_ND_CANARY_ROOT/output_bits.npy
readonly STRATEGY_ND_CANARY_OUTPUT_SHA=fda0228123392c29db87e1d6dd01cc6830041196409f49e0144628054cc64bd9
readonly LAYER0_INGREDIENTS=${GLM_GREENFIELD_LAYER0_INGREDIENTS:-0}
readonly LAYER1_INTERNAL_REFERENCE_NPZ=${GLM_GREENFIELD_LAYER1_INTERNAL_REFERENCE_NPZ:-/home/gianl/gcs-models/oracles/greenfield/glm52/dsa_internals/8k/layer1/greenfield_layer1_dsa_internal_comparison_20260808T115135394251231Z/internals.npz}
readonly LAYER1_INTERNAL_REFERENCE_SHA=${GLM_GREENFIELD_LAYER1_INTERNAL_REFERENCE_SHA:-79b813daa8e194b6c9a9ad883a0199f4a938ca4d4ab7277d20a291b480349054}
case "$PROFILE" in
  2k)
    CONTEXT_LABEL=2k
    CONTEXT_NAME=2K
    CONTEXT_CAPACITY=2048
    PROMPT_TOKEN_COUNT=2034
    CONTEXT_TAG_SUFFIX=
    SHORT_CONTEXT_ORACLE_TAG=greenfield_short_context_oracle_20260806T202544155912103Z
    SHORT_CONTEXT_ORACLE_MANIFEST_SHA=f580c14954bcbd0d973b6fe8158520992a18a1375ed88cff9cceb8e01c7efe19
    SHORT_CONTEXT_DSA_ORACLE_TAG=greenfield_short_context_dsa_oracle_recovery_20260806T231905802593249Z
    SHORT_CONTEXT_DSA_ORACLE_MANIFEST_SHA=71224832652ce61024786d39d43dcbfdc6cde76bf2eff0350531b272f38f4f57
    ;;
  8k)
    CONTEXT_LABEL=8k
    CONTEXT_NAME=8K
    CONTEXT_CAPACITY=8192
    PROMPT_TOKEN_COUNT=8155
    CONTEXT_TAG_SUFFIX=_8k
    SHORT_CONTEXT_ORACLE_TAG=greenfield_short_context_oracle_8k_20260807T172307269147351Z
    SHORT_CONTEXT_ORACLE_MANIFEST_SHA=e4fbcbdbf0fc8b1969e2f82ee457ab1563db4a8b37d2dea2bc4d1e828a13acf2
    SHORT_CONTEXT_DSA_ORACLE_TAG=greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z
    SHORT_CONTEXT_DSA_ORACLE_MANIFEST_SHA=f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da
    ;;
  *)
    echo "short decoder profile must be 2k or 8k" >&2
    exit 2
    ;;
esac
readonly CONTEXT_LABEL CONTEXT_NAME CONTEXT_CAPACITY PROMPT_TOKEN_COUNT
readonly CONTEXT_TAG_SUFFIX SHORT_CONTEXT_ORACLE_TAG
readonly SHORT_CONTEXT_ORACLE_MANIFEST_SHA SHORT_CONTEXT_DSA_ORACLE_TAG
readonly SHORT_CONTEXT_DSA_ORACLE_MANIFEST_SHA
readonly SHORT_CONTEXT_ORACLE_ROOT=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/$CONTEXT_LABEL/$SHORT_CONTEXT_ORACLE_TAG
readonly SHORT_CONTEXT_ORACLE_DIR=$SHORT_CONTEXT_ORACLE_ROOT/oracle
readonly SHORT_CONTEXT_DSA_ORACLE_ROOT=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/$CONTEXT_LABEL/$SHORT_CONTEXT_DSA_ORACLE_TAG
readonly SHORT_CONTEXT_DSA_ORACLE_DIR=$SHORT_CONTEXT_DSA_ORACLE_ROOT/oracle
readonly LAYER_RESIDUAL_OBSERVER=${GLM_GREENFIELD_LAYER_RESIDUAL_OBSERVER:-0}
readonly LAYER_RESIDUAL_POSITION=${GLM_GREENFIELD_LAYER_RESIDUAL_POSITION:-$((PROMPT_TOKEN_COUNT + 10))}
if [[ -n ${GLM_GREENFIELD_FEATURE_OUTPUT_TILE+x} ]]; then
  FEATURE_OUTPUT_TILE=$GLM_GREENFIELD_FEATURE_OUTPUT_TILE
elif [[ $RUNTIME_KIND == reference ]]; then
  FEATURE_OUTPUT_TILE=128
else
  FEATURE_OUTPUT_TILE=256
fi
readonly FEATURE_OUTPUT_TILE
readonly SOURCE_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/packed/PP8_LP4/greenfield_full_pack_pp8_20260805T182222755355852Z
readonly SOURCE_MANIFEST_SHA=0869493164a3a63797ea61d88c575f35bea8aa50790c46aa21ce6f0f7c4c78f1
readonly SOURCE_RUNTIME_TAG=greenfield_runtime_pack_pp8_20260806T002756318310857Z
readonly SOURCE_RUNTIME_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime/PP8_LP4/$SOURCE_RUNTIME_TAG
readonly SOURCE_RUNTIME_MANIFEST_SHA=fdedaae31fb3c094266272ed48dfe62bb098257a78272b93c14eafbd57e31dec
[[ $FEATURE_OUTPUT_TILE == 128 || $FEATURE_OUTPUT_TILE == 256 ]] || {
  echo "feature output tile must be 128 or 256" >&2
  exit 2
}
[[ $VERIFY_DEVICE_ROUNDTRIP == 0 || $VERIFY_DEVICE_ROUNDTRIP == 1 ]] || {
  echo "runtime device-roundtrip flag must be 0 or 1" >&2
  exit 2
}
[[ $FEATURE_FUSE_ROUTE_WEIGHTING == 0 || $FEATURE_FUSE_ROUTE_WEIGHTING == 1 ]] || {
  echo "feature route-weight fusion must be 0 or 1" >&2
  exit 2
}
[[ $FEATURE_RECONSTRUCT_DOWN_FP32 == 0 || $FEATURE_RECONSTRUCT_DOWN_FP32 == 1 ]] || {
  echo "feature FP32 reconstruction must be 0 or 1" >&2
  exit 2
}
[[ $COMPLETE_TOKEN_PATH == 0 || $COMPLETE_TOKEN_PATH == 1 ]] || {
  echo "complete token path must be 0 or 1" >&2
  exit 2
}
[[ $SPLIT_RESIDUAL_STATE == 0 || $SPLIT_RESIDUAL_STATE == 1 ]] || {
  echo "split residual state must be 0 or 1" >&2
  exit 2
}
[[ $PREFILL_INDEX_REPAIR == 0 || $PREFILL_INDEX_REPAIR == 1 ]] || {
  echo "prefill index repair must be 0 or 1" >&2
  exit 2
}
[[ $DSA_QUERY_EXACT_ASSOCIATION == 0 || $DSA_QUERY_EXACT_ASSOCIATION == 1 ]] || {
  echo "exact DSA query association must be 0 or 1" >&2
  exit 2
}
[[ $DSA_HEAD_KEY_EXACT_ASSOCIATION == 0 || $DSA_HEAD_KEY_EXACT_ASSOCIATION == 1 ]] || {
  echo "exact DSA head/key association must be 0 or 1" >&2
  exit 2
}
[[ $DSA_SCORE_DEFAULT_PRECISION == 0 || $DSA_SCORE_DEFAULT_PRECISION == 1 ]] || {
  echo "DSA score default-precision flag must be 0 or 1" >&2
  exit 2
}
[[ $MAIN_ROPE_TABLE == 0 || $MAIN_ROPE_TABLE == 1 ]] || {
  echo "main-RoPE table flag must be 0 or 1" >&2
  exit 2
}
[[ $PREGATHERED_B512_ATTENTION == 0 || $PREGATHERED_B512_ATTENTION == 1 ]] || {
  echo "pregathered-B512 attention flag must be 0 or 1" >&2
  exit 2
}
[[ $STRATEGY_ND_ATTENTION_PROJECTION == 0 || $STRATEGY_ND_ATTENTION_PROJECTION == 1 ]] || {
  echo "StrategyND attention-projection flag must be 0 or 1" >&2
  exit 2
}
[[ $SHORT_CONTEXT_ORACLE == 0 || $SHORT_CONTEXT_ORACLE == 1 ]] || {
  echo "short-context oracle flag must be 0 or 1" >&2
  exit 2
}
[[ $SHORT_CONTEXT_DSA_ORACLE == 0 || $SHORT_CONTEXT_DSA_ORACLE == 1 ]] || {
  echo "short-context DSA oracle flag must be 0 or 1" >&2
  exit 2
}
[[ $LAYER_RESIDUAL_OBSERVER == 0 || $LAYER_RESIDUAL_OBSERVER == 1 ]] || {
  echo "layer residual observer flag must be 0 or 1" >&2
  exit 2
}
[[ $DSA_INTERNAL_OBSERVER == 0 || $DSA_INTERNAL_OBSERVER == 1 ]] || {
  echo "DSA internal observer flag must be 0 or 1" >&2
  exit 2
}
[[ $LAYER0_RESIDUAL_VARIANTS == 0 || $LAYER0_RESIDUAL_VARIANTS == 1 ]] || {
  echo "layer-0 residual variant flag must be 0 or 1" >&2
  exit 2
}
[[ $LAYER0_SUBSHARD_VARIANTS == 0 || $LAYER0_SUBSHARD_VARIANTS == 1 ]] || {
  echo "layer-0 subshard variant flag must be 0 or 1" >&2
  exit 2
}
[[ $LAYER0_ATTENTION_VARIANTS == 0 || $LAYER0_ATTENTION_VARIANTS == 1 ]] || {
  echo "layer-0 attention variant flag must be 0 or 1" >&2
  exit 2
}
[[ $LAYER0_ATTENTION_OUTPUT_VARIANTS == 0 || $LAYER0_ATTENTION_OUTPUT_VARIANTS == 1 ]] || {
  echo "layer-0 attention-output variant flag must be 0 or 1" >&2
  exit 2
}
[[ $LAYER0_STRATEGY_ND_ROW0 == 0 || $LAYER0_STRATEGY_ND_ROW0 == 1 ]] || {
  echo "layer-0 StrategyND row-zero flag must be 0 or 1" >&2
  exit 2
}
[[ $LAYER0_INGREDIENTS == 0 || $LAYER0_INGREDIENTS == 1 ]] || {
  echo "layer-0 ingredient flag must be 0 or 1" >&2
  exit 2
}
((LAYER0_RESIDUAL_VARIANTS + LAYER0_SUBSHARD_VARIANTS + LAYER0_ATTENTION_VARIANTS + LAYER0_ATTENTION_OUTPUT_VARIANTS + LAYER0_STRATEGY_ND_ROW0 + LAYER0_INGREDIENTS <= 1)) || {
  echo "layer-0 diagnostic replays must remain mutually isolated" >&2
  exit 2
}
LAYER0_DISCRIMINATOR=0
if [[ $LAYER0_RESIDUAL_VARIANTS == 1 || $LAYER0_SUBSHARD_VARIANTS == 1 || $LAYER0_ATTENTION_VARIANTS == 1 || $LAYER0_ATTENTION_OUTPUT_VARIANTS == 1 || $LAYER0_STRATEGY_ND_ROW0 == 1 ]]; then
  LAYER0_DISCRIMINATOR=1
fi
readonly LAYER0_DISCRIMINATOR
[[ $LAYER_RESIDUAL_POSITION =~ ^[0-9]+$ ]] || {
  echo "layer residual position must be an integer" >&2
  exit 2
}
if [[ $SHORT_CONTEXT_ORACLE == 1 ]]; then
  [[ $COMPLETE_TOKEN_PATH == 1 ]] || {
    echo "short-context oracle requires complete token path" >&2
    exit 2
  }
  ((1 + WARMUP + ITERATIONS <= 20)) || {
    echo "correctness window exceeds the 20-token oracle" >&2
    exit 2
  }
  ((PROMPT_TOKEN_COUNT + WARMUP + ITERATIONS + TRACE_STEPS <= CONTEXT_CAPACITY)) || {
    echo "oracle prompt plus recurrent/trace steps exceeds $CONTEXT_NAME capacity" >&2
    exit 2
  }
fi
if [[ $SHORT_CONTEXT_DSA_ORACLE == 1 ]]; then
  [[ $SHORT_CONTEXT_ORACLE == 1 && $COMPLETE_TOKEN_PATH == 1 ]] || {
    echo "short-context DSA oracle requires token oracle and complete token path" >&2
    exit 2
  }
  [[ $WARMUP == 2 && $ITERATIONS == 10 && $TRACE_STEPS == 2 ]] || {
    echo "$CONTEXT_NAME Gate D requires warmup=2 iterations=10 trace_steps=2" >&2
    exit 2
  }
fi
if [[ $PREFILL_INDEX_REPAIR == 1 ]]; then
  [[ $PROFILE == 8k && $SHORT_CONTEXT_ORACLE == 1 && $SHORT_CONTEXT_DSA_ORACLE == 1 && $COMPLETE_TOKEN_PATH == 1 ]] || {
    echo "prefill index repair requires the protected 8K token/DSA Gate-D profile" >&2
    exit 2
  }
  [[ $SPLIT_RESIDUAL_STATE == 1 ]] || {
    echo "prefill index repair requires the accepted split residual state" >&2
    exit 2
  }
  # DSA internals use the separately compiled non-donating observer and are
  # guarded by an exact pinned production observation.  Returned residual
  # boundaries remain incompatible with the repaired prefill state.
  [[ $LAYER_RESIDUAL_OBSERVER == 0 ]] || {
    echo "prefill index repair must remain isolated from residual observation" >&2
    exit 2
  }
fi
if [[ $DSA_HEAD_KEY_EXACT_ASSOCIATION == 1 ]]; then
  [[ $DSA_QUERY_EXACT_ASSOCIATION == 1 && $PREFILL_INDEX_REPAIR == 1 ]] || {
    echo "exact DSA head/key association requires exact query and prefill repair" >&2
    exit 2
  }
fi
if [[ $DSA_SCORE_DEFAULT_PRECISION == 1 ]]; then
  [[ $PROFILE == 8k && $SHORT_CONTEXT_ORACLE == 1 && $SHORT_CONTEXT_DSA_ORACLE == 1 && $COMPLETE_TOKEN_PATH == 1 ]] || {
    echo "default DSA score precision requires the protected 8K token/DSA Gate-D profile" >&2
    exit 2
  }
  [[ $SPLIT_RESIDUAL_STATE == 1 && $PREFILL_INDEX_REPAIR == 1 && $DSA_QUERY_EXACT_ASSOCIATION == 1 && $DSA_HEAD_KEY_EXACT_ASSOCIATION == 1 ]] || {
    echo "default DSA score precision requires the exact repaired recurrent DSA chain" >&2
    exit 2
  }
fi
if [[ $MAIN_ROPE_TABLE == 1 ]]; then
  [[ $PROFILE == 8k && $SHORT_CONTEXT_ORACLE == 1 && $SHORT_CONTEXT_DSA_ORACLE == 1 && $COMPLETE_TOKEN_PATH == 1 ]] || {
    echo "main-RoPE table requires the protected 8K token/DSA Gate-D profile" >&2
    exit 2
  }
  [[ $SPLIT_RESIDUAL_STATE == 1 && $PREFILL_INDEX_REPAIR == 1 && $DSA_QUERY_EXACT_ASSOCIATION == 1 && $DSA_HEAD_KEY_EXACT_ASSOCIATION == 1 && $DSA_SCORE_DEFAULT_PRECISION == 1 ]] || {
    echo "main-RoPE table requires the exact repaired recurrent DSA chain" >&2
    exit 2
  }
  [[ $LAYER0_RESIDUAL_VARIANTS == 0 && $LAYER0_SUBSHARD_VARIANTS == 0 ]] || {
    echo "main-RoPE table admits only the isolated attention/StrategyND/ingredient diagnostics" >&2
    exit 2
  }
fi
if [[ $PREGATHERED_B512_ATTENTION == 1 ]]; then
  [[ $PROFILE == 8k && $SHORT_CONTEXT_ORACLE == 1 && $SHORT_CONTEXT_DSA_ORACLE == 1 ]] || {
    echo "pregathered-B512 attention requires the protected 8K token/DSA profile" >&2
    exit 2
  }
  [[ $RUNTIME_KIND == pallas_feature_linear && $COMPLETE_TOKEN_PATH == 1 && $SPLIT_RESIDUAL_STATE == 1 && $PREFILL_INDEX_REPAIR == 1 && $DSA_QUERY_EXACT_ASSOCIATION == 1 && $DSA_HEAD_KEY_EXACT_ASSOCIATION == 1 && $DSA_SCORE_DEFAULT_PRECISION == 1 && $MAIN_ROPE_TABLE == 1 ]] || {
    echo "pregathered-B512 attention requires the exact table-on production path" >&2
    exit 2
  }
  [[ $LAYER0_DISCRIMINATOR == 0 && $LAYER0_INGREDIENTS == 0 ]] || {
    echo "pregathered-B512 attention must remain isolated from layer-0 diagnostics" >&2
    exit 2
  }
fi
if [[ $STRATEGY_ND_ATTENTION_PROJECTION == 1 ]]; then
  [[ $PREGATHERED_B512_ATTENTION == 1 ]] || {
    echo "StrategyND attention projection requires pregathered-B512 attention" >&2
    exit 2
  }
  [[ $PROFILE == 8k && $SHORT_CONTEXT_ORACLE == 1 && $SHORT_CONTEXT_DSA_ORACLE == 1 && $LAYER0_DISCRIMINATOR == 0 && $LAYER0_INGREDIENTS == 0 ]] || {
    echo "StrategyND attention projection requires the protected 8K Gate-D path" >&2
    exit 2
  }
fi
if [[ $LAYER_RESIDUAL_OBSERVER == 1 ]]; then
  [[ $SHORT_CONTEXT_DSA_ORACLE == 1 ]] || {
    echo "layer residual observer requires the sealed DSA/token oracle" >&2
    exit 2
  }
  ((LAYER_RESIDUAL_POSITION >= PROMPT_TOKEN_COUNT && LAYER_RESIDUAL_POSITION < PROMPT_TOKEN_COUNT + 14)) || {
    echo "layer residual position must be in the sealed $PROMPT_TOKEN_COUNT..$((PROMPT_TOKEN_COUNT + 13)) window" >&2
    exit 2
  }
fi
if [[ $DSA_INTERNAL_OBSERVER == 1 ]]; then
  [[ $PROFILE == 8k && $SHORT_CONTEXT_DSA_ORACLE == 1 ]] || {
    echo "DSA internal observer requires the protected 8K DSA/token oracle" >&2
    exit 2
  }
  [[ $LAYER_RESIDUAL_OBSERVER == 0 ]] || {
    echo "DSA internal and residual observers must be isolated" >&2
    exit 2
  }
  [[ -r $DSA_INTERNAL_BASELINE_NPZ && -r $DSA_INTERNAL_LAYER0_REFERENCE_NPZ ]] || {
    echo "DSA internal observer inputs are unavailable" >&2
    exit 2
  }
  [[ $(sha256sum "$DSA_INTERNAL_BASELINE_NPZ" | awk '{print $1}') == "$DSA_INTERNAL_BASELINE_SHA" ]] || {
    echo "DSA internal baseline observation hash drifted" >&2
    exit 2
  }
  [[ $(sha256sum "$DSA_INTERNAL_LAYER0_REFERENCE_NPZ" | awk '{print $1}') == "$DSA_INTERNAL_LAYER0_REFERENCE_SHA" ]] || {
    echo "DSA internal layer-0 reference hash drifted" >&2
    exit 2
  }
fi
if [[ $LAYER0_DISCRIMINATOR == 1 ]]; then
  [[ $PROFILE == 8k && $SHORT_CONTEXT_DSA_ORACLE == 1 ]] || {
    echo "layer-0 discriminator requires the protected 8K DSA/token oracle" >&2
    exit 2
  }
  [[ $PREFILL_INDEX_REPAIR == 1 && $DSA_QUERY_EXACT_ASSOCIATION == 1 && $DSA_HEAD_KEY_EXACT_ASSOCIATION == 1 && $DSA_SCORE_DEFAULT_PRECISION == 1 ]] || {
    echo "layer-0 discriminator requires the complete exact recurrent DSA chain" >&2
    exit 2
  }
  [[ $LAYER_RESIDUAL_OBSERVER == 0 ]] || {
    echo "layer-0 discriminator must remain isolated from residual observation" >&2
    exit 2
  }
  [[ $DSA_INTERNAL_OBSERVER == 0 ]] || {
    echo "layer-0 discriminator must remain isolated from DSA internals" >&2
    exit 2
  }
  [[ -r $LAYER1_INTERNAL_REFERENCE_NPZ ]] || {
    echo "layer-1 internal reference is unavailable" >&2
    exit 2
  }
  [[ $(sha256sum "$LAYER1_INTERNAL_REFERENCE_NPZ" | awk '{print $1}') == "$LAYER1_INTERNAL_REFERENCE_SHA" ]] || {
    echo "layer-1 internal reference hash drifted" >&2
    exit 2
  }
fi
if [[ $PREGATHERED_B512_ATTENTION == 1 ]]; then
  readonly PREGATHERED_ATTENTION_PREREQUISITE_TAG=greenfield_layer0_attention_arithmetic_20260812T114701365714147Z
  readonly PREGATHERED_ATTENTION_PREREQUISITE_DIR=/home/gianl/glm-run/$PREGATHERED_ATTENTION_PREREQUISITE_TAG
  readonly PREGATHERED_ATTENTION_PREREQUISITE_REMOTE=$APPROVED_BUCKET/results/$PREGATHERED_ATTENTION_PREREQUISITE_TAG
  /home/gianl/vllm-env/bin/python - \
    "$PREGATHERED_ATTENTION_PREREQUISITE_DIR" "$RESULTS_DB" <<'PY'
import json
from hashlib import sha256
from pathlib import Path
import re
import sqlite3
import sys

run_dir = Path(sys.argv[1])
db_path = Path(sys.argv[2])
expected_hashes = {
    "SUCCESS": "ecc2b873c29ffd9bc6551136e90352b33523f50873827b00ae91279a4db3153c",
    "runner.json": "7961622c297567a021edf2ab335f0d046db6665e8130e210bbd78919a2d7a4ec",
    "attention_arithmetic.npz": "7d5ebe15dd006a70d77f17d41eb47f25f58c6d6784ee332fbc638f2916581f61",
    "summary.json": "9793f89aa0f0bbe2532106707e0a38600058d9ad5f30f36813f36c5fd7298540",
    "evidence.sha256": "2c877e8c90e9ad73f78ae92661ac877879ae5ff4fa91540b96ad382e8399f690",
    "results_ckpt.db": "cd6cd598ac9692b4cfdb27de8d0ef89e5901a244595312570be45e5031a63498",
    "census_pre.txt": "77e426bd6323913c3f13911b15def9a404730044c89ffe4bf6939c5eec67d7ca",
    "census_post.txt": "43f603392448470b7c82a45d6d6c0d88cc03546a0e5d346ccdfe3b0c98a99ebd",
    "remote_objects.json": "c3e3f5b9d5452ce6c2a7a699b1f781efc4b3d741d072022f2a98991faf9b621c",
}
for relative, expected in expected_hashes.items():
    path = run_dir / relative
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"DB537 prerequisite hash drifted: {relative}")

runner = json.loads((run_dir / "runner.json").read_text())
summary = json.loads((run_dir / "summary.json").read_text())
exact_arms = [
    "pregathered_h16_b512",
    "pregathered_attention_h2_b512",
    "pregathered_full_h2_b512",
]
accepted_output_sha = (
    "923e9bfeb65864868cef359cf98ebad67f2756794ad142ac87e66b71877a2d2a"
)
if (
    runner.get("status") != "SUCCESS"
    or runner.get("code_hash")
    != "83577222b4d48cb4b2fec544c06e3a71fbc821f5"
    or runner.get("classification") != "exact_arithmetic_arm_identified"
    or runner.get("exact_arms") != exact_arms
    or runner.get("greenfield_cache_exact_arms") != exact_arms
    or set(runner.get("arms", {})) != {
        "pregathered_h16_b128",
        "pregathered_h16_b512",
        "pregathered_attention_h2_b128",
        "pregathered_attention_h2_b512",
        "pregathered_full_h2_b512",
    }
    or not all(
        arm["hlo"]["contract"]["passed"]
        for arm in runner.get("arms", {}).values()
    )
    or summary.get("status") != "SUCCESS"
    or summary.get("results_db_run_id") != 537
    or summary.get("classification") != "exact_arithmetic_arm_identified"
    or summary.get("exact_arms") != exact_arms
    or summary.get("greenfield_cache_exact_arms") != exact_arms
):
    raise SystemExit("DB537 attention-arithmetic prerequisite drifted")
for name in exact_arms:
    arm = runner["arms"][name]
    for comparison_name in (
        "accepted_comparison",
        "greenfield_cache_comparison",
    ):
        comparison = arm[comparison_name]
        if (
            not comparison["elementwise_exact"]
            or comparison["mismatch_count"] != 0
            or comparison["expected_sha256"] != accepted_output_sha
            or comparison["observed_sha256"] != accepted_output_sha
        ):
            raise SystemExit(
                f"DB537 exact arm comparison drifted: {name}/{comparison_name}"
            )
for name in ("pregathered_h16_b128", "pregathered_attention_h2_b128"):
    comparison = runner["arms"][name]["accepted_comparison"]
    if (
        comparison["elementwise_exact"]
        or comparison["mismatch_count"] != 216
        or comparison["max_abs_error"] != 3.0517578125e-05
    ):
        raise SystemExit(f"DB537 rejected B128 control drifted: {name}")
for name in ("census_pre.txt", "census_post.txt"):
    workers = re.findall(
        r"^CENSUS_OK .*?-w-([0-7])$", (run_dir / name).read_text(), re.M
    )
    if sorted(workers) != list("01234567"):
        raise SystemExit(f"DB537 prerequisite {name} is not 8/8 clean")
with sqlite3.connect(db_path) as conn:
    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise SystemExit("live results DB integrity failed")
    row = conn.execute(
        "SELECT r.model, r.model_revision, r.harness_git, r.pod, "
        "i.id, i.benchmark, i.item_id, i.correct, i.score "
        "FROM runs r JOIN items i ON i.run_id = r.run_id "
        "WHERE r.run_id = 537"
    ).fetchone()
if row != (
    "zai-org/GLM-5.2-FP8:greenfield-layer0-attention-arithmetic",
    "native-jax-pregathered-lp4-v1",
    "8357722",
    "db-v4-64-od",
    1821,
    "greenfield_layer0_attention_arithmetic",
    "position8155",
    1,
    1.0,
):
    raise SystemExit("DB537 prerequisite DB linkage drifted")
PY
  for remote_file_sha in \
    "SUCCESS:ecc2b873c29ffd9bc6551136e90352b33523f50873827b00ae91279a4db3153c" \
    "runner.json:7961622c297567a021edf2ab335f0d046db6665e8130e210bbd78919a2d7a4ec" \
    "attention_arithmetic.npz:7d5ebe15dd006a70d77f17d41eb47f25f58c6d6784ee332fbc638f2916581f61" \
    "summary.json:9793f89aa0f0bbe2532106707e0a38600058d9ad5f30f36813f36c5fd7298540" \
    "remote_objects.json:c3e3f5b9d5452ce6c2a7a699b1f781efc4b3d741d072022f2a98991faf9b621c"
  do
    remote_file=${remote_file_sha%%:*}
    expected_remote_sha=${remote_file_sha#*:}
    observed_remote_sha=$(gcloud storage cat \
      "$PREGATHERED_ATTENTION_PREREQUISITE_REMOTE/$remote_file" | \
      sha256sum | awk '{print $1}')
    [[ $observed_remote_sha == "$expected_remote_sha" ]] || {
      echo "DB537 direct remote $remote_file hash drifted" >&2
      exit 2
    }
  done
fi
if [[ $STRATEGY_ND_ATTENTION_PROJECTION == 1 ]]; then
  readonly STRATEGY_ND_ATTENTION_PREREQUISITE_TAG=greenfield_legacy_layer0_attention_update_p8155_20260812T172809039093068Z
  readonly STRATEGY_ND_ATTENTION_PREREQUISITE_DIR=/home/gianl/glm-run/$STRATEGY_ND_ATTENTION_PREREQUISITE_TAG
  readonly STRATEGY_ND_ATTENTION_PREREQUISITE_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/attention_update/8k/$STRATEGY_ND_ATTENTION_PREREQUISITE_TAG
  /home/gianl/vllm-env/bin/python - \
    "$STRATEGY_ND_ATTENTION_PREREQUISITE_DIR" "$RESULTS_DB" <<'PY'
import json
from hashlib import sha256
from pathlib import Path
import re
import sqlite3
import sys

run_dir = Path(sys.argv[1])
db_path = Path(sys.argv[2])
expected_hashes = {
    "SUCCESS": "f52d3e88b060c19059392dad69ba959ce715ba489c44f3565b93c1215969f508",
    "attention_update_capture/capture.json": "f5b502cfed1e393d03c7d7e5d20c0b930f2fc1eb0e3fd56524073cb77665a91e",
    "attention_update_capture/attention_update.npz": "02c78d132f01c7564e416b001d378d1029b9ba6b1deaf044d1e6164e916acec6",
    "attention_update_comparison/comparison.json": "780cf3c2f1470119c73b2399082bd1e9c61b6f9908e28b8f1bc2a7dda73d435c",
    "results_ckpt.db": "c33f23bf69bbaa68f7d620fa74ee1e278805d26df5600921fdbab9587b2b4ea2",
    "census_pre.txt": "2b638eb85a20f637c90c3b2afaf0b927e679d2eefd678763cf1209c779f2e59d",
    "census_post.txt": "ac1aa438d6a189de51ca6e0d12389446b704427af8786c55df1411acc27f849a",
    "remote_objects.json": "3f3910fc1c52e2122b0c199c62ca83f6a3efc64e7d4b3a7e41142914561ee045",
}
for relative, expected in expected_hashes.items():
    path = run_dir / relative
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"DB539 prerequisite hash drifted: {relative}")

capture = json.loads(
    (run_dir / "attention_update_capture/capture.json").read_text()
)
comparison = json.loads(
    (run_dir / "attention_update_comparison/comparison.json").read_text()
)
accepted_sha = (
    "68afed86921584fb673abb11a563e359a1533210ec2483e71ee79b88c2b0bde7"
)
strategy = comparison.get("candidate_comparisons", {}).get("strategy_nd", {})
local = comparison.get("candidate_comparisons", {}).get("local", {})
if (
    capture.get("legacy_code_hash")
    != "23ab8780f3066ae1be12657d4f45daa7ea353761"
    or capture.get("oracle_pin")
    != "b3c25df47ac98783912dc658878181ec0a8ae16d"
    or capture.get("position") != 8155
    or capture.get("tensor", {}).get("tensor_sha256") != accepted_sha
    or comparison.get("status") != "SUCCESS"
    or comparison.get("classification")
    != "strategy_nd_attention_projection_exact"
    or comparison.get("exact_candidates") != ["strategy_nd"]
    or comparison.get("first_open_boundary") != "after_attention_projection"
    or comparison.get("probe", {}).get("results_db_run_id") != 538
    or comparison.get("probe", {}).get("code_hash")
    != "e2a3a74a3b2ef1fa8f3b9cb1c5d7ec65f833eafc"
    or not strategy.get("elementwise_exact")
    or strategy.get("mismatch_count") != 0
    or strategy.get("expected_sha256") != accepted_sha
    or strategy.get("observed_sha256") != accepted_sha
    or local.get("elementwise_exact")
    or local.get("mismatch_count") != 3652
):
    raise SystemExit("DB539 StrategyND attention prerequisite drifted")
for name in ("census_pre.txt", "census_post.txt"):
    workers = re.findall(
        r"^CENSUS_OK .*?-w-([0-7])$", (run_dir / name).read_text(), re.M
    )
    if sorted(workers) != list("01234567"):
        raise SystemExit(f"DB539 prerequisite {name} is not 8/8 clean")
with sqlite3.connect(db_path) as conn:
    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise SystemExit("live results DB integrity failed")
    row = conn.execute(
        "SELECT r.model, r.harness_git, r.pod, i.id, i.benchmark, "
        "i.item_id, i.correct, i.score FROM runs r JOIN items i "
        "ON i.run_id = r.run_id WHERE r.run_id = 539"
    ).fetchone()
if row != (
    "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8",
    "a4a17ac",
    "db-v4-64-od",
    1823,
    "passkey_L8192_d0.5",
    "t0",
    1,
    1.0,
):
    raise SystemExit("DB539 prerequisite DB linkage drifted")
PY
  for remote_file_sha in \
    "SUCCESS:f52d3e88b060c19059392dad69ba959ce715ba489c44f3565b93c1215969f508" \
    "attention_update_capture/capture.json:f5b502cfed1e393d03c7d7e5d20c0b930f2fc1eb0e3fd56524073cb77665a91e" \
    "attention_update_capture/attention_update.npz:02c78d132f01c7564e416b001d378d1029b9ba6b1deaf044d1e6164e916acec6" \
    "attention_update_comparison/comparison.json:780cf3c2f1470119c73b2399082bd1e9c61b6f9908e28b8f1bc2a7dda73d435c" \
    "remote_objects.json:3f3910fc1c52e2122b0c199c62ca83f6a3efc64e7d4b3a7e41142914561ee045"
  do
    remote_file=${remote_file_sha%%:*}
    expected_remote_sha=${remote_file_sha#*:}
    observed_remote_sha=$(gcloud storage cat \
      "$STRATEGY_ND_ATTENTION_PREREQUISITE_REMOTE/$remote_file" | \
      sha256sum | awk '{print $1}')
    [[ $observed_remote_sha == "$expected_remote_sha" ]] || {
      echo "DB539 direct remote $remote_file hash drifted" >&2
      exit 2
    }
  done
fi
if [[ $LAYER0_ATTENTION_VARIANTS == 1 ]]; then
  [[ $MAIN_ROPE_TABLE == 1 && $RUNTIME_KIND == pallas_feature_linear && $COMPLETE_TOKEN_PATH == 1 && $SPLIT_RESIDUAL_STATE == 1 ]] || {
    echo "layer-0 attention variants require the table-on complete split Pallas-linear path" >&2
    exit 2
  }
fi
if [[ $LAYER0_ATTENTION_OUTPUT_VARIANTS == 1 ]]; then
  [[ $MAIN_ROPE_TABLE == 1 && $RUNTIME_KIND == pallas_feature_linear && $COMPLETE_TOKEN_PATH == 1 && $SPLIT_RESIDUAL_STATE == 1 ]] || {
    echo "layer-0 attention-output variants require the table-on complete split Pallas-linear path" >&2
    exit 2
  }
fi
if [[ $LAYER0_STRATEGY_ND_ROW0 == 1 ]]; then
  [[ $MAIN_ROPE_TABLE == 1 && $RUNTIME_KIND == pallas_feature_linear && $COMPLETE_TOKEN_PATH == 1 && $SPLIT_RESIDUAL_STATE == 1 ]] || {
    echo "layer-0 StrategyND row-zero discriminator requires the table-on complete split Pallas-linear path" >&2
    exit 2
  }
fi
if [[ $LAYER0_INGREDIENTS == 1 ]]; then
  [[ $PROFILE == 8k && $SHORT_CONTEXT_DSA_ORACLE == 1 ]] || {
    echo "layer-0 ingredients require the protected 8K DSA/token oracle" >&2
    exit 2
  }
  [[ $RUNTIME_KIND == pallas_feature_linear && $COMPLETE_TOKEN_PATH == 1 && $SPLIT_RESIDUAL_STATE == 1 ]] || {
    echo "layer-0 ingredients require the complete split Pallas-linear path" >&2
    exit 2
  }
  [[ $MAIN_ROPE_TABLE == 1 ]] || {
    echo "layer-0 ingredients require the proven main-RoPE table" >&2
    exit 2
  }
  [[ $PREFILL_INDEX_REPAIR == 1 && $DSA_QUERY_EXACT_ASSOCIATION == 1 && $DSA_HEAD_KEY_EXACT_ASSOCIATION == 1 && $DSA_SCORE_DEFAULT_PRECISION == 1 ]] || {
    echo "layer-0 ingredients require the complete exact recurrent DSA chain" >&2
    exit 2
  }
  [[ $LAYER_RESIDUAL_OBSERVER == 0 ]] || {
    echo "layer-0 ingredients must remain isolated from residual observation" >&2
    exit 2
  }
  [[ $DSA_INTERNAL_OBSERVER == 0 ]] || {
    echo "layer-0 ingredients must remain isolated from DSA internals" >&2
    exit 2
  }
fi
if [[ $LAYER0_SUBSHARD_VARIANTS == 1 ]]; then
  readonly ISOLATED_RESIDUAL_PREREQUISITE_TAG=greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_oracle_dsa_layer0_residual_variants_trace2_20260810T221121969164909Z
  readonly ISOLATED_RESIDUAL_PREREQUISITE_DIR=/home/gianl/glm-run/$ISOLATED_RESIDUAL_PREREQUISITE_TAG
  readonly ISOLATED_RESIDUAL_PREREQUISITE_REMOTE=$APPROVED_BUCKET/results/$ISOLATED_RESIDUAL_PREREQUISITE_TAG
  /home/gianl/vllm-env/bin/python - "$ISOLATED_RESIDUAL_PREREQUISITE_DIR" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

run_dir = Path(sys.argv[1])
expected_hashes = {
    "census_pre.txt": "615dc21d544f97ea022a518b2d99f1161cc2a8679100fd9241de1c59ee059db9",
    "census_failure_exit.txt": "73d1e1dcfda00276c8523f1e497905f0d395ddffe2e8eff14f6cd1e1d3f0369d",
    "hlo/layer0_residual_discriminator.suite_contract.json": "31ed00930ab2876235ac202b626a5ba52ed152c4c906d298bde0f8c5fb49c75e",
    "layer0_residual_discriminator/contract.json": "1d3d270f2e71a7ace0fcf3056e92119222034713a10bf23fb3492dc557e70b0e",
    "layer0_residual_discriminator/position_8155_variants.npz": "f7323ea207a42ef3777bf3d8fdd898df6007b0e3fb6a090a245ec6e064eae45a",
}
for relative, expected in expected_hashes.items():
    path = run_dir / relative
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"isolated residual prerequisite hash drifted: {relative}")
contract = json.loads(
    (run_dir / "layer0_residual_discriminator/contract.json").read_text()
)
expected_names = [
    "baseline_bf16",
    "attention_output_fp32",
    "dense_down_fp32",
    "attention_output_and_dense_down_fp32",
]
if (
    not contract.get("passed")
    or not contract.get("reproduces_current_baseline")
    or contract.get("variant_names") != expected_names
    or contract["variant_comparisons"]["baseline_bf16"]["actual_sha256"]
    != "787c9ba7b39d6fd43b59876f713052b64a3ac2d3ecbc6074795ba5364abe153b"
    or any(
        item["elementwise_exact"]
        for item in contract["variant_comparisons"].values()
    )
):
    raise SystemExit("isolated residual prerequisite contract drifted")
for name in ("census_pre.txt", "census_failure_exit.txt"):
    workers = re.findall(
        r"^CENSUS_OK .*?-w-([0-7])$", (run_dir / name).read_text(), re.M
    )
    if sorted(workers) != list("01234567"):
        raise SystemExit(f"isolated residual prerequisite {name} is not 8/8 clean")
PY
  isolated_residual_remote_contract_sha=$(
    gcloud storage cat \
      "$ISOLATED_RESIDUAL_PREREQUISITE_REMOTE/layer0_residual_discriminator/contract.json" \
      | sha256sum | awk '{print $1}'
  )
  [[ $isolated_residual_remote_contract_sha == 1d3d270f2e71a7ace0fcf3056e92119222034713a10bf23fb3492dc557e70b0e ]] || {
    echo "isolated residual direct remote contract hash drifted" >&2
    exit 2
  }
fi
case "$RUNTIME_KIND" in
  reference)
    readonly RUNTIME_TAG=$SOURCE_RUNTIME_TAG
    readonly RUNTIME_ROOT=$SOURCE_RUNTIME_ROOT
    readonly RUNTIME_MANIFEST_SHA=$SOURCE_RUNTIME_MANIFEST_SHA
    readonly RUNTIME_LAYOUT_HASH=841a18f6dbbf329243482f97f01d28ca22211d63d323fca859477349cc0abcac
    readonly SPARSE_MOE_BACKEND=reference
    readonly HLO_BACKEND_CONTRACT=tpu_v4_pp8_reference
    ;;
  pallas_feature)
    readonly RUNTIME_TAG=${GLM_GREENFIELD_FEATURE_RUNTIME_TAG:-greenfield_runtime_feature_pack_pp8_20260806T064010287072141Z}
    readonly RUNTIME_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP8_LP4/$RUNTIME_TAG
    readonly RUNTIME_MANIFEST_SHA=${GLM_GREENFIELD_FEATURE_RUNTIME_MANIFEST_SHA:-54e2f89b1832b994acbf9ef36f5f6ce68c942d9146efc4d7c15360d68b6d9917}
    readonly RUNTIME_LAYOUT_HASH=${GLM_GREENFIELD_FEATURE_RUNTIME_LAYOUT_HASH:-ba21c4ec1500837f17a53047da98a7ed7c3a06782ddffe49d8bd796d4d0d1c9e}
    readonly SPARSE_MOE_BACKEND=pallas_feature
    readonly HLO_BACKEND_CONTRACT=tpu_v4_pp8_pallas_feature
    ;;
  pallas_feature_linear)
    readonly RUNTIME_TAG=${GLM_GREENFIELD_FEATURE_RUNTIME_TAG:-greenfield_runtime_feature_qkv_pack_pp8_20260808T141032190315066Z}
    readonly RUNTIME_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP8_LP4/$RUNTIME_TAG
    readonly RUNTIME_MANIFEST_SHA=${GLM_GREENFIELD_FEATURE_RUNTIME_MANIFEST_SHA:-123394906a153238e464fc096b626c77996b7cf22b95077b9377b8dcafbe699a}
    readonly RUNTIME_LAYOUT_HASH=${GLM_GREENFIELD_FEATURE_RUNTIME_LAYOUT_HASH:-523afb1dc1ff2b954a9795c4deabdc4fd599c244c0bf1a9e3b8f971700548cb4}
    readonly SPARSE_MOE_BACKEND=pallas_feature
    readonly HLO_BACKEND_CONTRACT=tpu_v4_pp8_pallas_feature_linear
    ;;
  *)
    echo "unsupported decoder runtime kind: $RUNTIME_KIND" >&2
    exit 2
    ;;
esac
read -r ATTENTION_PROJECTION_BACKEND EXPECTED_LOADED_PAYLOAD_BYTES < <(
  /home/gianl/vllm-env/bin/python - "$RUNTIME_ROOT/runtime_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
layout = manifest.get("attention_projection_layout", "separate_q_a_kv_a_v1")
backends = {
    "separate_q_a_kv_a_v1": "separate",
    "fused_qkv_a_virtual_tp32_n82_v1": "fused_n82_convolution",
}
if layout not in backends:
    raise SystemExit("runtime attention projection layout is unknown")
payload = manifest["runtime_payload_bytes"]
if payload % 8:
    raise SystemExit("runtime payload bytes do not divide over eight hosts")
print(backends[layout], payload // 8)
PY
)
readonly ATTENTION_PROJECTION_BACKEND EXPECTED_LOADED_PAYLOAD_BYTES
if [[ $PREFILL_INDEX_REPAIR == 1 && $ATTENTION_PROJECTION_BACKEND != fused_n82_convolution ]]; then
  echo "protected prefill repair requires the Gate-B-approved fused qkv-a runtime" >&2
  exit 2
fi
if [[ $DSA_QUERY_EXACT_ASSOCIATION == 1 && $ATTENTION_PROJECTION_BACKEND != fused_n82_convolution ]]; then
  echo "exact DSA query association requires the proven fused qkv-a runtime" >&2
  exit 2
fi
if [[ $RUNTIME_KIND == reference && $FEATURE_OUTPUT_TILE != 128 ]]; then
  echo "a non-default feature output tile requires a feature runtime" >&2
  exit 2
fi
if [[ $RUNTIME_KIND == reference && $FEATURE_FUSE_ROUTE_WEIGHTING != 0 ]]; then
  echo "feature route-weight fusion requires a feature runtime" >&2
  exit 2
fi
if [[ $RUNTIME_KIND == reference && $FEATURE_RECONSTRUCT_DOWN_FP32 != 0 ]]; then
  echo "feature FP32 reconstruction requires a feature runtime" >&2
  exit 2
fi
if [[ $FEATURE_RECONSTRUCT_DOWN_FP32 == 1 && $FEATURE_FUSE_ROUTE_WEIGHTING == 1 ]]; then
  echo "feature FP32 reconstruction is incompatible with route-weight fusion" >&2
  exit 2
fi

if [[ $PREFILL_INDEX_REPAIR == 1 ]]; then
  readonly PREFILL_REPAIR_PREREQUISITE_TAG=greenfield_layer0_prompt_key_norm_m64_20260809T122010714691723Z
  readonly PREFILL_REPAIR_PREREQUISITE_DIR=/home/gianl/glm-run/$PREFILL_REPAIR_PREREQUISITE_TAG
  readonly PREFILL_REPAIR_PREREQUISITE_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_key_norm_m64/8k/$PREFILL_REPAIR_PREREQUISITE_TAG
  readonly PREFILL_REPAIR_PREREQUISITE_SUCCESS_SHA=a8d370166257622875feafd4d1da3f8d666204a8609baaffef2573b659f6bfee
  readonly PREFILL_SPLIT_PREREQUISITE_TAG=greenfield_layer0_prompt_key_norm_m64_20260809T200559393031635Z
  readonly PREFILL_SPLIT_PREREQUISITE_DIR=/home/gianl/glm-run/$PREFILL_SPLIT_PREREQUISITE_TAG
  readonly PREFILL_SPLIT_PREREQUISITE_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_key_norm_m64/8k/$PREFILL_SPLIT_PREREQUISITE_TAG
  readonly PREFILL_SPLIT_PREREQUISITE_SUCCESS_SHA=643f80eb8699e18714213bb828a72df8a9f89ba2db1e798ad598dd914a2083ca
  /home/gianl/vllm-env/bin/python - "$PREFILL_REPAIR_PREREQUISITE_DIR" "$RESULTS_DB" "$PREFILL_SPLIT_PREREQUISITE_DIR" <<'PY'
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import sys

run_dir = Path(sys.argv[1])
db_path = Path(sys.argv[2])
split_run_dir = Path(sys.argv[3])
hashes = {
    "SUCCESS": "a8d370166257622875feafd4d1da3f8d666204a8609baaffef2573b659f6bfee",
    "summary.json": "8a8823de3dff03bc8827daef4cd38f5b09a80ad691ddea9aed0372808627a210",
    "comparison/comparison.json": "c9539e2f59016ee800bb452602d01e9680f5960c608f67af1dec6a98686c8ec7",
    "results_ckpt.db": "3752990d31011e89d01117c78804b7532dc5cc4153e2fcfdbe39c1e6b99cbd2b",
    "remote_objects.json": "80dc195b1384b18c2f99be290442887319c741f0e7737a044034052868cf9fa7",
    "census_pre.txt": "bc2c690ff56d25c62efc5be7e49dd81e01f3e116e96912470a172efbcad86283",
    "census_post.txt": "76ef7baf344d31ebf2cee778bcd6eb6cb2eca7c548d405c66866466e85c790d7",
    "evidence.sha256": "29c9b8b4df8637522cc856e4ee842a5ce5824371224429db2569d21506ba9a2a",
}
for relative, expected in hashes.items():
    path = run_dir / relative
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"DB518 prerequisite hash drifted: {relative}")
summary = json.loads((run_dir / "summary.json").read_text())
comparison = json.loads((run_dir / "comparison/comparison.json").read_text())
if (
    summary["status"] != "SUCCESS"
    or summary["results_db_run_id"] != 518
    or summary["results_db_item_row_id"] != 1803
    or summary["code_hash"] != "86243115452920fe4244bb77a9bbf4c44110aeab"
    or summary["comparison_manifest_sha256"] != "1d80d088561181a63e734a91cd0124c4011cfc4151198488c052740050d66fe5"
    or not summary["key_norm_association_restored"]
    or not summary["projection_input_elementwise_exact"]
    or not summary["state_elementwise_exact"]
    or not summary["cache_elementwise_exact"]
    or summary["candidate_cache_sha256"] != "3808d502f3ea1829bf12ab7585d66f15dd83bf640657a17c35daabf5ab1859d1"
):
    raise SystemExit("DB518 prerequisite summary drifted")
cache = comparison["cache_comparison"]
state = comparison["state_comparison"]
for name in ("cache", "states"):
    contract = comparison["hlo"][name]["contract"]
    if (
        not contract["passed"]
        or contract["accepted_convolution_count"] != 1
        or not contract["convolution_weight_f32"]
        or contract["physical_key_norm"]["grouped_sqrt_count"] != 0
        or not contract["required_shapes"]["physical_m64_projection"]
        or not contract["required_shapes"]["physical_m64_key_norm_sqrt"]
        or not contract["required_shapes"]["physical_m64_key_norm_affine"]
        or contract["forbidden_operations"]
        or contract["forbidden_shapes"]
    ):
        raise SystemExit(f"DB518 prerequisite {name} HLO drifted")
if (
    not cache["elementwise_exact"]
    or cache["mismatch_count"] != 0
    or cache["shape"] != [8155, 128]
    or not state["all_fields_elementwise_exact"]
    or state["classification"] != "producer_states_elementwise_exact"
    or state["first_divergent_field"] is not None
):
    raise SystemExit("DB518 prerequisite tensor evidence drifted")
for name in ("census_pre.txt", "census_post.txt"):
    workers = re.findall(r"^CENSUS_OK .*?-w-([0-7])$", (run_dir / name).read_text(), re.M)
    if sorted(workers) != list("01234567"):
        raise SystemExit(f"DB518 prerequisite {name} is not 8/8 clean")
split_hashes = {
    "SUCCESS": "643f80eb8699e18714213bb828a72df8a9f89ba2db1e798ad598dd914a2083ca",
    "summary.json": "260b7e2a2adabe4456604ecd6d832d377604e3149ad12417ad3db4a6280e911e",
    "comparison/comparison.json": "e0c637db7fc9add403ad16231b4c19e23ac06f5c8178034500e1c06bb3766ee9",
    "results_ckpt.db": "d466adc91aaca62e09c669f877b8577d9ee65b3f9959e01588730dc32d279581",
    "remote_objects.json": "599ba9f1270f6ca93c8f3d823b0209bfaa51c61d350ac2efde7569aed22daffd",
    "census_pre.txt": "c03e0acb293823a042d9c01742a6b75e9f2bbf856e89fd6539f6eeb5bafcaed9",
    "census_post.txt": "9c352ab39032bdf774a8cfbe9027e3800d893aae6aa5d2ce89ed37298fb21e90",
    "evidence.sha256": "b60f6627c3cf8265c155168afedbf6499336132929444b779c06abee0dd93add",
}
for relative, expected in split_hashes.items():
    path = split_run_dir / relative
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"DB520 prerequisite hash drifted: {relative}")
split_summary = json.loads((split_run_dir / "summary.json").read_text())
lp4 = split_summary.get("lp4_materialized_repair", {})
phase_hlo = lp4.get("hlo", {})
if (
    split_summary.get("status") != "SUCCESS"
    or split_summary.get("results_db_run_id") != 520
    or split_summary.get("results_db_item_row_id") != 1805
    or split_summary.get("code_hash") != "097702266d025f2887418e06898eae781615972c"
    or split_summary.get("comparison_manifest_sha256") != "1e94255557c475dac4335a0120f3448e8754135c44dde747a888cb3c2de08a59"
    or not split_summary.get("cache_elementwise_exact")
    or not split_summary.get("state_elementwise_exact")
    or not lp4.get("assembled_cache_elementwise_exact")
    or lp4.get("assembled_cache_sha256") != "3808d502f3ea1829bf12ab7585d66f15dd83bf640657a17c35daabf5ab1859d1"
    or not lp4.get("owner_isolation_exact")
    or lp4.get("per_lane_written_rows") != [2048, 2048, 2048, 2011]
    or len(lp4.get("materialized_shards", ())) != 4
    or not all(
        shard.get("comparison", {}).get("elementwise_exact")
        and shard.get("sha256") == "d680f7b1c2fed426174c41ee9153d9db47f5db8ec0e803e20d8853ec1f483469"
        for shard in lp4.get("materialized_shards", ())
    )
):
    raise SystemExit("DB520 prerequisite summary drifted")
for name, backend, hlo_sha in (
    ("lp4_wk_decode_bf16", "external_stage_local_raw_fp8_to_bf16", "08b6c59f96be27dc723337c24ba047d5501e20abfc150f07c3568bdf2b2ab2f9"),
    ("lp4_wk_promote_fp32", "external_stage_local_bf16_to_fp32", "1c107d68ee21356711f6b0ead7b721104e2266754c9d668fcd93516b81e5b1f8"),
):
    record = phase_hlo.get(name, {})
    contract = record.get("contract", {})
    if (
        not contract.get("passed")
        or contract.get("backend") != backend
        or contract.get("collective_count") != 0
        or contract.get("host_markers")
        or record.get("optimized_hlo_sha256") != hlo_sha
    ):
        raise SystemExit(f"DB520 prerequisite {name} HLO drifted")
for name in ("census_pre.txt", "census_post.txt"):
    workers = re.findall(
        r"^CENSUS_OK .*?-w-([0-7])$", (split_run_dir / name).read_text(), re.M
    )
    if sorted(workers) != list("01234567"):
        raise SystemExit(f"DB520 prerequisite {name} is not 8/8 clean")
with sqlite3.connect(db_path) as conn:
    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise SystemExit("live results DB integrity failed")
    run = conn.execute(
        "SELECT model,harness_git,pod FROM runs WHERE run_id=518"
    ).fetchone()
    item = conn.execute(
        "SELECT run_id,benchmark,item_id,correct,score,n_prompt_tokens FROM items WHERE id=1803"
    ).fetchone()
    split_run = conn.execute(
        "SELECT model,harness_git,pod FROM runs WHERE run_id=520"
    ).fetchone()
    split_item = conn.execute(
        "SELECT run_id,benchmark,item_id,correct,score,n_prompt_tokens FROM items WHERE id=1805"
    ).fetchone()
if run != (
    "zai-org/GLM-5.2-FP8:greenfield-layer0-prompt-key-norm-m64",
    "8624311",
    "db-v4-64-od",
) or item != (
    518,
    "greenfield_layer0_prompt_key_norm_association",
    "adapted_fp32_m64_projection_keynorm_lax_map",
    1,
    1.0,
    8155,
):
    raise SystemExit("live DB518/item1803 prerequisite linkage drifted")
if split_run != (
    "zai-org/GLM-5.2-FP8:greenfield-layer0-prompt-key-norm-m64",
    "0977022",
    "db-v4-64-od",
) or split_item != (
    520,
    "greenfield_layer0_prompt_key_norm_association",
    "adapted_fp32_m64_projection_keynorm_materialized_lp4_stage_local",
    1,
    1.0,
    8155,
):
    raise SystemExit("live DB520/item1805 prerequisite linkage drifted")
PY
  remote_success_sha=$(gcloud storage cat "$PREFILL_REPAIR_PREREQUISITE_REMOTE/SUCCESS" | sha256sum | awk '{print $1}')
  [[ $remote_success_sha == "$PREFILL_REPAIR_PREREQUISITE_SUCCESS_SHA" ]] || {
    echo "DB518 direct remote SUCCESS hash drifted" >&2
    exit 2
  }
  split_remote_success_sha=$(gcloud storage cat "$PREFILL_SPLIT_PREREQUISITE_REMOTE/SUCCESS" | sha256sum | awk '{print $1}')
  [[ $split_remote_success_sha == "$PREFILL_SPLIT_PREREQUISITE_SUCCESS_SHA" ]] || {
    echo "DB520 direct remote SUCCESS hash drifted" >&2
    exit 2
  }
fi

if [[ $DSA_QUERY_EXACT_ASSOCIATION == 1 ]]; then
  readonly DSA_QUERY_PREREQUISITE_TAG=greenfield_layer0_physical_lp4_dsa_head_geometry_20260810T060457076587721Z
  readonly DSA_QUERY_PREREQUISITE_DIR=/home/gianl/glm-run/$DSA_QUERY_PREREQUISITE_TAG
  readonly DSA_QUERY_PREREQUISITE_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/physical_lp4_dsa_head_geometry_association/8k/$DSA_QUERY_PREREQUISITE_TAG
  /home/gianl/vllm-env/bin/python - "$DSA_QUERY_PREREQUISITE_DIR" "$RESULTS_DB" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

run_dir = Path(sys.argv[1])
db_path = Path(sys.argv[2])
expected_hashes = {
    "SUCCESS": "5b547f24a360b89411f7773800a5ac5bef56d8cc953e4d141dbf2d498aa6d582",
    "evidence.sha256": "35f888c74705d56a404b10ab3b06403b29b50c01ea1aa063035b414f56217624",
    "runner.json": "da7acf8baddd14e276fefac3c466f2a6e0a41acb2ecb60c25cf7a28096d91dfc",
    "summary.json": "5bba9e4e5cc064e34100dcf294aab4dbaf3c87431322b99c20fd4084774a1932",
}
for name, expected in expected_hashes.items():
    if sha256((run_dir / name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"DB525 prerequisite {name} drifted")
summary = json.loads((run_dir / "summary.json").read_text())
runner = json.loads((run_dir / "runner.json").read_text())
candidate = runner["candidates"]["physical_owner_tuple4_barrier_m1_n1024"]
if (
    summary.get("code_hash")
    != "a749ff02e49d9a27d5a9ad36fec6597ffe0777a4"
    or summary.get("results_db_run_id") != 525
    or summary.get("exact_candidates")
    != ["physical_owner_tuple4_barrier_m1_n1024"]
    or not candidate["accepted_comparison"]["elementwise_exact"]
    or candidate["accepted_comparison"]["mismatch_count"] != 0
    or candidate["hlo"]["hlo_sha256"]
    != "40d9ef25681edc50e6611229c8a3ace2cc4bac6e4cd2459d14b12d44b5c74b7d"
    or not candidate["hlo"]["stablehlo"]["passed"]
    or candidate["input_weight_aliases"] != 4
):
    raise SystemExit("DB525 exact-association prerequisite drifted")
with sqlite3.connect(db_path) as connection:
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise SystemExit("live results DB integrity failed")
    run = connection.execute(
        "SELECT model,harness_git,pod FROM runs WHERE run_id=525"
    ).fetchone()
    item = connection.execute(
        "SELECT run_id,benchmark,item_id,correct,score FROM items WHERE id=1810"
    ).fetchone()
if run != (
    "zai-org/GLM-5.2-FP8:greenfield-layer0-query_lp4_head_geometry-association",
    "a749ff0",
    "db-v4-64-od",
) or item != (
    525,
    "greenfield_layer0_query_lp4_head_geometry_association",
    "layer0_position8155_query_lp4_head_geometry",
    1,
    1.0,
):
    raise SystemExit("DB525 prerequisite DB linkage drifted")
PY
  remote_success_sha=$(gcloud storage cat "$DSA_QUERY_PREREQUISITE_REMOTE/SUCCESS" | sha256sum | awk '{print $1}')
  [[ $remote_success_sha == 5b547f24a360b89411f7773800a5ac5bef56d8cc953e4d141dbf2d498aa6d582 ]] || {
    echo "DB525 direct remote SUCCESS hash drifted" >&2
    exit 2
  }
  readonly DSA_QUERY_PRODUCTION_PREREQUISITE_TAG=greenfield_layer0_physical_lp4_dsa_query_production_exact_20260810T080508327295662Z
  readonly DSA_QUERY_PRODUCTION_PREREQUISITE_DIR=/home/gianl/glm-run/$DSA_QUERY_PRODUCTION_PREREQUISITE_TAG
  readonly DSA_QUERY_PRODUCTION_PREREQUISITE_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/physical_lp4_dsa_query_production_exact/8k/$DSA_QUERY_PRODUCTION_PREREQUISITE_TAG
  /home/gianl/vllm-env/bin/python - "$DSA_QUERY_PRODUCTION_PREREQUISITE_DIR" "$RESULTS_DB" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

run_dir = Path(sys.argv[1])
db_path = Path(sys.argv[2])
expected_hashes = {
    "SUCCESS": "b3cff36beaff7eaaf71c30349007c8c5acb8c7423ae46fb82c40339061a23b11",
    "evidence.sha256": "8221919110485a3d9f6f70fcd7086012aa73666a96df1146f890aefacfb52b7d",
    "runner.json": "497dd6066c8480c643018f251c48566d84b00317372ccfd2543ed302cd2c96f2",
    "summary.json": "890a9d8e9ac4415fffad93f3b778ec1e79f9ca463e5d375239f1c720aff2212e",
    "physical_lp4_production_exact.npz": "b371ad77313c085268470c950f231a0cf23c4d17c7c7104b9c791cfc9f49d247",
}
for name, expected in expected_hashes.items():
    if sha256((run_dir / name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"DB526 prerequisite {name} drifted")
summary = json.loads((run_dir / "summary.json").read_text())
runner = json.loads((run_dir / "runner.json").read_text())
candidate = runner["candidates"][
    "physical_production_fused_q_a_tuple4_exact_m1_n1024"
]
materializer = runner["materializer"]
if (
    summary.get("code_hash")
    != "3aa9f9ca5fedc9e5b6b67aca00b9be10ff5f0dc1"
    or summary.get("results_db_run_id") != 526
    or summary.get("exact_candidates")
    != ["physical_production_fused_q_a_tuple4_exact_m1_n1024"]
    or not runner.get("production_helper")
    or not candidate["accepted_comparison"]["elementwise_exact"]
    or candidate["accepted_comparison"]["mismatch_count"] != 0
    or not candidate["q_a_comparison"]["elementwise_exact"]
    or candidate["q_a_comparison"]["mismatch_count"] != 0
    or candidate["input_weight_aliases"] != 4
    or candidate["hlo"]["tuple4_reduction_fusion_count"] != 1
    or candidate["hlo"]["hlo_sha256"]
    != "78c296746c1ee5bef3c2183256671fa5b2cd5dffd1d884217ee50b2fcf53069c"
    or candidate["hlo"]["stablehlo"]["stablehlo_sha256"]
    != "4e7f3dc3300c0aaf19ec7a11e0ed057bc521ad30cd8ddf4544932f9475c4190f"
    or not materializer["completed"]
    or not materializer["hlo"]["passed"]
    or materializer["local_fp32_bytes"] != 8 * 1024 * 1024
    or materializer["hlo"]["num_partitions"] != 4
    or materializer["hlo"]["custom_call_targets"]
    != ["AssumeGatherIndicesInBound", "GatherScatterIndicesBitpacked"]
    or materializer["hlo"]["forbidden_custom_call_targets"]
    or materializer["hlo"]["forbidden_operations"]
    or materializer["hlo"]["forbidden_global_shapes"]
    or materializer["hlo"]["host_markers"]
):
    raise SystemExit("DB526 production composition prerequisite drifted")
with sqlite3.connect(db_path) as connection:
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise SystemExit("live results DB integrity failed")
    run = connection.execute(
        "SELECT model,harness_git,pod FROM runs WHERE run_id=526"
    ).fetchone()
    item = connection.execute(
        "SELECT run_id,benchmark,item_id,correct,score FROM items WHERE id=1811"
    ).fetchone()
if run != (
    "zai-org/GLM-5.2-FP8:greenfield-layer0-query_lp4_production_exact-association",
    "3aa9f9c",
    "db-v4-64-od",
) or item != (
    526,
    "greenfield_layer0_query_lp4_production_exact_association",
    "layer0_position8155_query_lp4_production_exact",
    1,
    1.0,
):
    raise SystemExit("DB526 prerequisite DB linkage drifted")
PY
  production_remote_success_sha=$(gcloud storage cat "$DSA_QUERY_PRODUCTION_PREREQUISITE_REMOTE/SUCCESS" | sha256sum | awk '{print $1}')
  [[ $production_remote_success_sha == b3cff36beaff7eaaf71c30349007c8c5acb8c7423ae46fb82c40339061a23b11 ]] || {
    echo "DB526 direct remote SUCCESS hash drifted" >&2
    exit 2
  }
fi

if [[ $DSA_HEAD_KEY_EXACT_ASSOCIATION == 1 ]]; then
  readonly DSA_HEAD_KEY_PREREQUISITE_TAG=greenfield_layer0_physical_lp4_dsa_head_key_boundary_20260810T104647319991568Z
  readonly DSA_HEAD_KEY_PREREQUISITE_DIR=/home/gianl/glm-run/$DSA_HEAD_KEY_PREREQUISITE_TAG
  readonly DSA_HEAD_KEY_PREREQUISITE_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/physical_lp4_dsa_head_key_boundary_association/8k/$DSA_HEAD_KEY_PREREQUISITE_TAG
  /home/gianl/vllm-env/bin/python - "$DSA_HEAD_KEY_PREREQUISITE_DIR" "$RESULTS_DB" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import sys

run_dir = Path(sys.argv[1])
db_path = Path(sys.argv[2])
expected_hashes = {
    "SUCCESS": "0c961dd13d00d4db8f56fe2efac3fbb4a203b4f0fc6265e2bb7cc0842c015fb0",
    "evidence.sha256": "7c7563e714a8f57fbf46111a82133349d69726e07ed270eb77397ef03e397678",
    "runner.json": "f4d2518c725656baff63a5e8938bb985d8fcdc9bdec5d7c666f4199ce7541310",
    "summary.json": "e76e799b56126d76f5f2708c0dc8ebfef9cb4e89dc54b7ea6d91a89d2230c1ea",
    "physical_lp4_head_key_boundary.npz": "3ea5813fe01201be8df851b32befaa22e8a8d12311ab6040e0c1b874d40b5b52",
}
for name, expected in expected_hashes.items():
    path = run_dir / name
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"DB527 prerequisite {name} drifted")
summary = json.loads((run_dir / "summary.json").read_text())
runner = json.loads((run_dir / "runner.json").read_text())
candidate = runner["candidates"][
    "physical_normalized_barrier_materialized_divide_sqrt"
]
hlo = candidate["hlo"]
if (
    summary.get("code_hash")
    != "cd15aafb2fddcc1c6c2fcd32981c8bffd7d4b569"
    or summary.get("results_db_run_id") != 527
    or summary.get("exact_candidates")
    != ["physical_normalized_barrier_materialized_divide_sqrt"]
    or not runner.get("association_restored")
    or runner.get("current_reproducing_candidates")
    != ["physical_unbarriered_raw_pallas_rsqrt"]
    or not candidate["accepted_head_weights_comparison"]["elementwise_exact"]
    or candidate["accepted_head_weights_comparison"]["mismatch_count"] != 0
    or not candidate["accepted_current_key_comparison"]["elementwise_exact"]
    or candidate["accepted_current_key_comparison"]["mismatch_count"] != 0
    or candidate["key_norm_mode"] != "divide_sqrt"
    or not hlo["passed"]
    or not hlo["normalized_barrier"]
    or hlo["projection_source"] != "materialized"
    or hlo["tuple4_anchor"]
    or hlo["stablehlo_dot_count"] != 2
    or hlo["stablehlo_optimization_barrier_count"] != 1
    or hlo["hlo_sha256"]
    != "4dc288236a558fcdaccb33141c2d761aebd0eb90d1d1591b65e0b77c969303a1"
    or hlo["stablehlo_sha256"]
    != "d9fd33b4b6775f2bec296a3e8601ca3ba8dc5660c4120604d02892c1d811dbcb"
):
    raise SystemExit("DB527 exact head/key prerequisite drifted")
for name in ("census_pre.txt", "census_post.txt"):
    workers = re.findall(
        r"^CENSUS_OK .*?-w-([0-7])$", (run_dir / name).read_text(), re.M
    )
    if sorted(workers) != list("01234567"):
        raise SystemExit(f"DB527 prerequisite {name} is not 8/8 clean")
with sqlite3.connect(db_path) as connection:
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise SystemExit("live results DB integrity failed")
    run = connection.execute(
        "SELECT model,harness_git,pod FROM runs WHERE run_id=527"
    ).fetchone()
    item = connection.execute(
        "SELECT run_id,benchmark,item_id,correct,score FROM items WHERE id=1812"
    ).fetchone()
if run != (
    "zai-org/GLM-5.2-FP8:greenfield-layer0-query_lp4_head_key_boundary-association",
    "cd15aaf",
    "db-v4-64-od",
) or item != (
    527,
    "greenfield_layer0_query_lp4_head_key_boundary_association",
    "layer0_position8155_query_lp4_head_key_boundary",
    1,
    1.0,
):
    raise SystemExit("DB527 prerequisite DB linkage drifted")
PY
  head_key_remote_success_sha=$(gcloud storage cat "$DSA_HEAD_KEY_PREREQUISITE_REMOTE/SUCCESS" | sha256sum | awk '{print $1}')
  [[ $head_key_remote_success_sha == 0c961dd13d00d4db8f56fe2efac3fbb4a203b4f0fc6265e2bb7cc0842c015fb0 ]] || {
    echo "DB527 direct remote SUCCESS hash drifted" >&2
    exit 2
  }
fi
if [[ $DSA_SCORE_DEFAULT_PRECISION == 1 ]]; then
  readonly DSA_SCORE_PRECISION_PREREQUISITE_TAG=greenfield_layer0_dsa_scorer_association_20260810T164030202890642Z
  readonly DSA_SCORE_PRECISION_PREREQUISITE_DIR=/home/gianl/glm-run/$DSA_SCORE_PRECISION_PREREQUISITE_TAG
  readonly DSA_SCORE_PRECISION_PREREQUISITE_REMOTE=$APPROVED_BUCKET/results/$DSA_SCORE_PRECISION_PREREQUISITE_TAG
  /home/gianl/vllm-env/bin/python - "$DSA_SCORE_PRECISION_PREREQUISITE_DIR" "$RESULTS_DB" <<'PY'
import json
from hashlib import sha256
from pathlib import Path
import re
import sqlite3
import sys

run_dir = Path(sys.argv[1])
db_path = Path(sys.argv[2])
expected_hashes = {
    "SUCCESS": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "summary.json": "01de9dff6595180c968a7b9a1fac7e8fe1b4773ea79a2b71fb7e0d516348d7c0",
    "runner.json": "d2e35d1c54989a015da7fd683661cb4d182f26baf91fb5c8f6a0d5d0ff5d3ab3",
    "hlo/current_wide_score.optimized_hlo.txt": "211da7bf7e700466d18aeaff8d8ed92648fc19cade79219a056698e30eb39dbd",
    "hlo/default_wide_score.optimized_hlo.txt": "5e166b87e3714e576e3c5a9c56098b370722b447ad38004bb4b022a7f03375ef",
}
for relative, expected in expected_hashes.items():
    path = run_dir / relative
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"DB529 prerequisite hash drifted: {relative}")
summary = json.loads((run_dir / "summary.json").read_text())
runner = summary["runner"]
current = runner["current_wide_control"]
candidate = runner["default_precision_candidate"]
if (
    summary.get("status") != "SUCCESS"
    or summary.get("results_db_run_id") != 529
    or summary.get("code_hash")
    != "0cd3db063852630c6309f1966020ba3085aa9877"
    or not summary.get("candidate_restored")
    or not current.get("bitwise_exact")
    or current.get("logical_score_sha256")
    != "a567a1618e35fc595c509d337f0f224995211112c17be30a3d29186339bafc1f"
    or candidate.get("logical_score_sha256")
    != "529fadbaeb6646ef391fde902971c3a3576c56ae8da10f2e26adc2eba7baa023"
    or not candidate["comparison"]["passed"]
    or not candidate["comparison"]["selected_set_exact"]
    or not candidate["comparison"]["selected_order_exact"]
    or not candidate["score_delta"]["elementwise_exact"]
    or candidate["score_delta"]["mismatch_count"] != 0
    or candidate["score_delta"]["actual_sha256"]
    != "0e715f8b46464702902a7cc83425cdd6abd2f8df15210a7bd70e82545d7b1318"
    or runner["hlo"]["current_wide_score"]["contract"]["score_precision"]
    != "highest"
    or runner["hlo"]["default_wide_score"]["contract"]["score_precision"]
    != "default"
):
    raise SystemExit("DB529 exact score-precision prerequisite drifted")
for name in ("census_pre.txt", "census_post.txt"):
    workers = re.findall(
        r"^CENSUS_OK .*?-w-([0-7])$", (run_dir / name).read_text(), re.M
    )
    if sorted(workers) != list("01234567"):
        raise SystemExit(f"DB529 prerequisite {name} is not 8/8 clean")
with sqlite3.connect(db_path) as conn:
    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise SystemExit("live results DB integrity failed")
    row = conn.execute(
        "SELECT r.model, r.harness_git, r.pod, i.id, i.benchmark, "
        "i.item_id, i.correct, i.score FROM runs r JOIN items i "
        "ON i.run_id = r.run_id WHERE r.run_id = 529"
    ).fetchone()
if row != (
    "zai-org/GLM-5.2-FP8:greenfield-layer0-dsa-scorer-association",
    "0cd3db0",
    "db-v4-64-od",
    1814,
    "greenfield_layer0_dsa_scorer_association",
    "layer0_position8155_wide_highest_vs_default",
    1,
    1.0,
):
    raise SystemExit("DB529 prerequisite DB linkage drifted")
PY
  score_precision_remote_success_sha=$(gcloud storage cat "$DSA_SCORE_PRECISION_PREREQUISITE_REMOTE/SUCCESS" | sha256sum | awk '{print $1}')
  [[ $score_precision_remote_success_sha == e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 ]] || {
    echo "DB529 direct remote SUCCESS hash drifted" >&2
    exit 2
  }
fi
if [[ $MAIN_ROPE_TABLE == 1 ]]; then
  readonly MAIN_ROPE_PREREQUISITE_TAG=greenfield_layer0_main_rope_20260811T072231959104598Z
  readonly MAIN_ROPE_PREREQUISITE_DIR=/home/gianl/glm-run/$MAIN_ROPE_PREREQUISITE_TAG
  readonly MAIN_ROPE_PREREQUISITE_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/main_rope_association/8k/$MAIN_ROPE_PREREQUISITE_TAG
  /home/gianl/vllm-env/bin/python - "$MAIN_ROPE_PREREQUISITE_DIR" "$RESULTS_DB" <<'PY'
import json
from hashlib import sha256
from pathlib import Path
import re
import sqlite3
import sys

run_dir = Path(sys.argv[1])
db_path = Path(sys.argv[2])
expected_hashes = {
    "SUCCESS": "8f9763e4e8106db0fe30b9fa3820873f736ede2215edfc8b950fcd311d08b6d7",
    "summary.json": "0998cc0fc95094b2f29afdacee77a505062a962fa7c885a1a6c8784c14f44068",
    "runner.json": "d32d03576faa99c882913cf289a32d9aaa7f258a6b83588464058f36a8ad7370",
    "main_rope_candidate.npz": "52d2a36e88cc223f1aac78177d4dd955a5eaf54097117158bd33877668e84145",
    "hlo/accepted_table_fp32_final_round.optimized_hlo.txt": "c611c74d2316a3ffca2f2233799245cdd508d8130dbd032c3f1e6040fedcf623",
}
for relative, expected in expected_hashes.items():
    path = run_dir / relative
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"DB531 prerequisite hash drifted: {relative}")
summary = json.loads((run_dir / "summary.json").read_text())
runner = json.loads((run_dir / "runner.json").read_text())
candidate = runner["candidates"]["accepted_table_fp32_final_round"]
if (
    summary.get("status") != "SUCCESS"
    or summary.get("results_db_run_id") != 531
    or summary.get("code_hash")
    != "ed7c74f423e811f1acfbeb3c075fb8fad37d4f51"
    or not summary.get("association_restored")
    or runner.get("position") != 8155
    or runner.get("table", {}).get("position_row_sha256")
    != "67b01e3cab682d5ffd04ac9c8043e7e6825ee1275023428c41f9a1ae412dea1d"
    or not candidate["comparison"]["elementwise_exact"]
    or candidate["comparison"]["mismatch_count"] != 0
    or candidate["comparison"]["candidate_sha256"]
    != "e7c217ec92be8ef7335295bfff9cb1876703ae962242c941edb5c870fc283281"
    or not candidate["hlo"]["passed"]
    or candidate["hlo"]["fp32_multiply_count"] != 4
    or candidate["hlo"]["fp32_combine_count"] != 2
    or candidate["hlo"]["final_round_count"] != 2
    or candidate["hlo"]["bf16_arithmetic"]
    or candidate["hlo"]["forbidden_instructions"]
):
    raise SystemExit("DB531 main-RoPE prerequisite drifted")
for name in ("census_pre.txt", "census_post.txt"):
    workers = re.findall(
        r"^CENSUS_OK .*?-w-([0-7])$", (run_dir / name).read_text(), re.M
    )
    if sorted(workers) != list("01234567"):
        raise SystemExit(f"DB531 prerequisite {name} is not 8/8 clean")
with sqlite3.connect(db_path) as conn:
    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise SystemExit("live results DB integrity failed")
    row = conn.execute(
        "SELECT r.model, r.harness_git, r.pod, i.id, i.benchmark, "
        "i.item_id, i.correct, i.score FROM runs r JOIN items i "
        "ON i.run_id = r.run_id WHERE r.run_id = 531"
    ).fetchone()
if row != (
    "zai-org/GLM-5.2-FP8:greenfield-layer0-main_rope-association",
    "ed7c74f",
    "db-v4-64-od",
    1816,
    "greenfield_layer0_main_rope_association",
    "layer0_position8155_main_rope",
    1,
    1.0,
):
    raise SystemExit("DB531 prerequisite DB linkage drifted")
PY
  main_rope_remote_success_sha=$(gcloud storage cat "$MAIN_ROPE_PREREQUISITE_REMOTE/SUCCESS" | sha256sum | awk '{print $1}')
  [[ $main_rope_remote_success_sha == 8f9763e4e8106db0fe30b9fa3820873f736ede2215edfc8b950fcd311d08b6d7 ]] || {
    echo "DB531 direct remote SUCCESS hash drifted" >&2
    exit 2
  }
fi
if [[ $LAYER0_ATTENTION_VARIANTS == 1 ]]; then
  readonly ATTENTION_SCHEDULE_PREREQUISITE_TAG=greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_oracle_dsa_dsa_internal_trace2_20260811T113139003786245Z
  readonly ATTENTION_SCHEDULE_PREREQUISITE_DIR=/home/gianl/glm-run/$ATTENTION_SCHEDULE_PREREQUISITE_TAG
  readonly ATTENTION_SCHEDULE_PREREQUISITE_REMOTE=$APPROVED_BUCKET/results/$ATTENTION_SCHEDULE_PREREQUISITE_TAG
  /home/gianl/vllm-env/bin/python - "$ATTENTION_SCHEDULE_PREREQUISITE_DIR" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import numpy as np

run_dir = Path(sys.argv[1])
expected_hashes = {
    "census_pre.txt": "e3d6a952e161bc95dce62bc33897bdd7bda11795533b791cee4277c9a104a244",
    "census_failure_exit.txt": "6b460d489b2e736fd9f122a10f604400d675c4a835ef143f5382c30dacefcdfe",
    "dsa_internal_observer/contract.json": "7600e22f3682b8263a5a1771968331f04e1f6875f4cf01ad6b7ba74823063829",
    "dsa_internal_observer/position_8155_internals.npz": "96fe8d9bf0e8fa43a3f2ab92735854b8416f49a0082201515f6c720fd077af05",
}
for relative, expected in expected_hashes.items():
    path = run_dir / relative
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(
            f"table-on production-boundary prerequisite hash drifted: {relative}"
        )
contract = json.loads(
    (run_dir / "dsa_internal_observer/contract.json").read_text()
)
# This sealed observer intentionally stopped at the first layer-1 divergence;
# ``passed=False`` is required evidence, not a failed prerequisite check.
if (
    contract.get("decode_position") != 8155
    or contract.get("producer_layer_ids", [])[:3] != [0, 1, 2]
    or not contract.get("layer0_query_exact")
    or contract.get("passed") is not False
    or contract["field_records"]["normalized_hidden"]["sha256"]
    != "45357a341fe8b876e82c347a2bd532f78da4c4b29f6a6672ce76e34231953c47"
):
    raise SystemExit("table-on production-boundary contract drifted")
with np.load(
    run_dir / "dsa_internal_observer/position_8155_internals.npz",
    allow_pickle=False,
) as tensors:
    producer_ids = np.asarray(tensors["producer_layer_ids"], dtype=np.int32)
    matches = np.flatnonzero(producer_ids == 1)
    if matches.tolist() != [1]:
        raise SystemExit("table-on production-boundary layer-1 row drifted")
    layer1_bits = np.ascontiguousarray(
        tensors["normalized_hidden_bfloat16_bits"][matches[0]]
    )
if sha256(layer1_bits.tobytes(order="C")).hexdigest() != (
    "6c54c09a773e622fef35e753dc99929a83b73d9d89157fd732a5ace903149bca"
):
    raise SystemExit("table-on production-boundary layer-1 hash drifted")
for name in ("census_pre.txt", "census_failure_exit.txt"):
    workers = re.findall(
        r"^CENSUS_OK .*?-w-([0-7])$", (run_dir / name).read_text(), re.M
    )
    if sorted(workers) != list("01234567"):
        raise SystemExit(
            f"table-on production-boundary prerequisite {name} is not 8/8 clean"
        )
PY
  attention_schedule_remote_contract_sha=$(
    gcloud storage cat \
      "$ATTENTION_SCHEDULE_PREREQUISITE_REMOTE/dsa_internal_observer/contract.json" \
      | sha256sum | awk '{print $1}'
  )
  [[ $attention_schedule_remote_contract_sha == 7600e22f3682b8263a5a1771968331f04e1f6875f4cf01ad6b7ba74823063829 ]] || {
    echo "table-on production-boundary direct remote contract hash drifted" >&2
    exit 2
  }
  attention_schedule_remote_npz_sha=$(
    gcloud storage cat \
      "$ATTENTION_SCHEDULE_PREREQUISITE_REMOTE/dsa_internal_observer/position_8155_internals.npz" \
      | sha256sum | awk '{print $1}'
  )
  [[ $attention_schedule_remote_npz_sha == 96fe8d9bf0e8fa43a3f2ab92735854b8416f49a0082201515f6c720fd077af05 ]] || {
    echo "table-on production-boundary direct remote NPZ hash drifted" >&2
    exit 2
  }
fi
if [[ $LAYER0_ATTENTION_OUTPUT_VARIANTS == 1 ]]; then
  readonly ATTENTION_OUTPUT_PREREQUISITE_TAG=greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_oracle_dsa_layer0_attention_schedule_variants_trace2_20260811T142025290162247Z
  readonly ATTENTION_OUTPUT_PREREQUISITE_DIR=/home/gianl/glm-run/$ATTENTION_OUTPUT_PREREQUISITE_TAG
  readonly ATTENTION_OUTPUT_PREREQUISITE_REMOTE=$APPROVED_BUCKET/results/$ATTENTION_OUTPUT_PREREQUISITE_TAG
  /home/gianl/vllm-env/bin/python - "$ATTENTION_OUTPUT_PREREQUISITE_DIR" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

run_dir = Path(sys.argv[1])
expected_hashes = {
    "census_pre.txt": "cdea3d12e301e6bc22c7ecbdd475adb33b6bd0f51e82dfaa795640fa602dea82",
    "census_failure_exit.txt": "0d7108e95e8419360081ffc1da26bd3c21372e4a0d3b6f45d28a99a9eaa62ff0",
    "layer0_residual_discriminator/contract.json": "bacc8a785e3a9cfe26162b2452b8b47b9446e0dbae353576b702d091c61b7fbf",
    "layer0_residual_discriminator/position_8155_variants.npz": "ad64fff27fced824d76d82815800f9f9b75f18bb49d5042162cf166dca2f4f64",
}
for relative, expected in expected_hashes.items():
    path = run_dir / relative
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(
            f"attention-output prerequisite hash drifted: {relative}"
        )
contract = json.loads(
    (run_dir / "layer0_residual_discriminator/contract.json").read_text()
)
names = ["attention_schedule_control", "replicated_monolithic_attention"]
control = contract.get("variant_comparisons", {}).get(names[0], {})
challenger = contract.get("variant_comparisons", {}).get(names[1], {})
arms = contract.get("arm_contracts", {})
challenger_hlo = arms.get(names[1], {}).get("hlo_contract", {})
if (
    not contract.get("passed")
    or contract.get("discriminator_kind") != "attention_schedule"
    or contract.get("variant_names") != names
    or not contract.get("reproduces_current_baseline")
    or control.get("actual_sha256")
    != "6c54c09a773e622fef35e753dc99929a83b73d9d89157fd732a5ace903149bca"
    or control.get("mismatch_count") != 3984
    or challenger.get("actual_sha256")
    != "619f1b0bf5415e15139fb4fc42c64b1e8f728fc132966398453f6711269cd090"
    or challenger.get("mismatch_count") != 3954
    or not contract.get("hlo_suite_contract", {}).get("passed")
    or any(
        not arms.get(name, {}).get("passed")
        or not arms[name].get("selection_exact")
        for name in names
    )
    or len(challenger_hlo.get("cache_shaped_gathers", [])) != 1
    or len(challenger_hlo.get("monolithic_cache_gathers", [])) != 1
):
    raise SystemExit("attention-output prerequisite contract drifted")
for name in ("census_pre.txt", "census_failure_exit.txt"):
    workers = re.findall(
        r"^CENSUS_OK .*?-w-([0-7])$", (run_dir / name).read_text(), re.M
    )
    if sorted(workers) != list("01234567"):
        raise SystemExit(
            f"attention-output prerequisite {name} is not 8/8 clean"
        )
PY
  attention_output_remote_contract_sha=$(
    gcloud storage cat \
      "$ATTENTION_OUTPUT_PREREQUISITE_REMOTE/layer0_residual_discriminator/contract.json" \
      | sha256sum | awk '{print $1}'
  )
  [[ $attention_output_remote_contract_sha == bacc8a785e3a9cfe26162b2452b8b47b9446e0dbae353576b702d091c61b7fbf ]] || {
    echo "attention-output direct remote contract hash drifted" >&2
    exit 2
  }
  attention_output_remote_npz_sha=$(
    gcloud storage cat \
      "$ATTENTION_OUTPUT_PREREQUISITE_REMOTE/layer0_residual_discriminator/position_8155_variants.npz" \
      | sha256sum | awk '{print $1}'
  )
  [[ $attention_output_remote_npz_sha == ad64fff27fced824d76d82815800f9f9b75f18bb49d5042162cf166dca2f4f64 ]] || {
    echo "attention-output direct remote NPZ hash drifted" >&2
    exit 2
  }
fi
if [[ $LAYER0_STRATEGY_ND_ROW0 == 1 ]]; then
  readonly STRATEGY_ND_PREREQUISITE_TAG=greenfield_collective_association_20260811T213152133863450Z
  readonly STRATEGY_ND_PREREQUISITE_DIR=/home/gianl/glm-run/$STRATEGY_ND_PREREQUISITE_TAG
  readonly STRATEGY_ND_PREREQUISITE_REMOTE=$APPROVED_BUCKET/results/$STRATEGY_ND_PREREQUISITE_TAG
  /home/gianl/vllm-env/bin/python - \
    "$STRATEGY_ND_PREREQUISITE_DIR" "$RESULTS_DB" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import sys

run_dir = Path(sys.argv[1])
db_path = Path(sys.argv[2])
expected_hashes = {
    "SUCCESS": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "association/analysis.json": "e7e34828365ca3d6cae0052f8d0e2e802143c6ca83810153db3116423f994108",
    "association/input_bits.npy": "1a6c2fb5c8c04b904daad45f751e0b5308919b85cb1f3cf63879c5febf3ea0e9",
    "association/manifest.json": "9304077e523d705c12f4939501cf4338cc18ffd0cfadfa6c3c1bcf2c903d2060",
    "association/output_bits.npy": "fda0228123392c29db87e1d6dd01cc6830041196409f49e0144628054cc64bd9",
    "summary.json": "3ca82073f69fbe56526e1765594c7e8e9a738c73eb2a62b2df6d9a0c4d3136b7",
    "census_pre.txt": "bc7c68030a30897060b6fe7cd0a42024e4db41a9f6044b65948e614b35c2df13",
    "census_post.txt": "5f2d54b52b442717ddc448ad2c8c0804ea8c52303fa6d3243deb320d3b02705e",
    "hlo/strategy_nd_association_bfloat16_32x6144.hlo_contract.json": "966a5dd8be19c409ac616dc194fe8e7dc78ae6c253132c7a73dcdeb8e0c040bc",
}
for relative, expected in expected_hashes.items():
    path = run_dir / relative
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"DB533 StrategyND prerequisite hash drifted: {relative}")

analysis = json.loads((run_dir / "association/analysis.json").read_text())
summary = json.loads((run_dir / "summary.json").read_text())
model_to_device = [
    0, 8, 16, 24, 2, 10, 18, 26, 4, 12, 20, 28, 6, 14, 22, 30,
    1, 9, 17, 25, 3, 11, 19, 27, 5, 13, 21, 29, 7, 15, 23, 31,
]
expected_groups = [
    (list(range(0, 8)), "7239b23e8ccba21dd74cb11342b3a3cb8e79c9ebff35d03203da2c61898f51dc"),
    (list(range(8, 16)), "36450bf5c08e95b7c6b7505c3fa1fb23844a6b087aed7a282362c74cb7dcf9a8"),
    (list(range(16, 24)), "7e3fb5a1921697f49a57ab0431a9799501381f9f47c8b302d9ff68fe64f3f5ca"),
    (list(range(24, 32)), "f66edca7b9b2b47517f58638af696141cf6dc2f787021b430a4a785291e18c16"),
]
observed_groups = {
    (tuple(item["rows"]), item["row_output_bits_sha256"])
    for item in analysis["row_output_equivalence_groups"]
}
if (
    analysis.get("code_hash") != "a9e6307bdad70b883ba82456fbf8f4bdf8db5ac6"
    or analysis.get("compile_bucket_rows") != 32
    or analysis.get("candidate_count_per_row") != 54
    or analysis.get("minimum_row_union_exact_column_count") != 6144
    or analysis.get("maximum_row_union_exact_column_count") != 6144
    or analysis.get("accepted_model_axis_device_ids") != model_to_device
    or observed_groups
    != {(tuple(rows), digest) for rows, digest in expected_groups}
    or len(analysis.get("rows", [])) != 32
    or any(
        row.get("column_candidate_count_histogram") != {"1": 6144}
        or row.get("union_exact_column_count") != 6144
        for row in analysis["rows"]
    )
):
    raise SystemExit("DB533 StrategyND association summary drifted")

row0 = analysis["rows"][0]
if (
    row0.get("physical_row") != 0
    or row0.get("row_output_bits_sha256") != expected_groups[0][1]
    or len(row0.get("blocks", [])) != 48
):
    raise SystemExit("DB533 StrategyND row-zero identity drifted")
for block in row0["blocks"]:
    start = block["start"]
    d0 = "((0+3)+(1+2))" if 2048 <= start < 4096 else "((0+1)+(2+3))"
    d2 = "((0+3)+(1+2))" if (start // 256) % 2 else "((0+1)+(2+3))"
    expected_id = f"physical_y_x_z:phase=012:d0={d0}:d2={d2}"
    if (
        block.get("end") != start + 128
        or block.get("union_exact_columns") != 128
        or block.get("top_candidates") != [
            {"candidate_id": expected_id, "exact_columns": 128}
        ]
    ):
        raise SystemExit(f"DB533 row-zero block schedule drifted at {start}")

if (
    summary.get("mode") != "strategy_nd_fingerprint"
    or not summary.get("mechanism_only")
    or summary.get("results_db_run_id") != 533
    or summary.get("code_hash") != "a9e6307bdad70b883ba82456fbf8f4bdf8db5ac6"
    or len(summary.get("cases", [])) != 1
    or not summary["cases"][0].get("diagnostic_only")
    or summary["cases"][0].get("collective_groups") != [list(range(32))]
):
    raise SystemExit("DB533 StrategyND protected summary drifted")
for name in ("census_pre.txt", "census_post.txt"):
    workers = re.findall(
        r"^CENSUS_OK .*?-w-([0-7])$", (run_dir / name).read_text(), re.M
    )
    if sorted(workers) != list("01234567"):
        raise SystemExit(f"DB533 prerequisite {name} is not 8/8 clean")
with sqlite3.connect(db_path) as conn:
    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise SystemExit("live results DB integrity failed")
    row = conn.execute(
        "SELECT r.model,r.harness_git,r.pod,i.id,i.benchmark,i.item_id,i.correct,i.score "
        "FROM runs r JOIN items i ON i.run_id=r.run_id WHERE r.run_id=533"
    ).fetchone()
if row != (
    "zai-org/GLM-5.2-FP8:greenfield-strategy-nd-diagnostic-only",
    "a9e6307",
    "db-v4-64-od",
    1818,
    "greenfield_strategy_nd_fingerprint",
    "strategy_nd_association_fingerprint:g32:bfloat16:32x6144",
    1,
    None,
):
    raise SystemExit("DB533 prerequisite DB linkage drifted")
PY
  strategy_nd_remote_analysis_sha=$(
    gcloud storage cat \
      "$STRATEGY_ND_PREREQUISITE_REMOTE/association/analysis.json" \
      | sha256sum | awk '{print $1}'
  )
  [[ $strategy_nd_remote_analysis_sha == e7e34828365ca3d6cae0052f8d0e2e802143c6ca83810153db3116423f994108 ]] || {
    echo "DB533 direct remote analysis hash drifted" >&2
    exit 2
  }
  strategy_nd_remote_summary_sha=$(
    gcloud storage cat "$STRATEGY_ND_PREREQUISITE_REMOTE/summary.json" \
      | sha256sum | awk '{print $1}'
  )
  [[ $strategy_nd_remote_summary_sha == 3ca82073f69fbe56526e1765594c7e8e9a738c73eb2a62b2df6d9a0c4d3136b7 ]] || {
    echo "DB533 direct remote summary hash drifted" >&2
    exit 2
  }
  strategy_nd_remote_success_sha=$(
    gcloud storage cat "$STRATEGY_ND_PREREQUISITE_REMOTE/SUCCESS" \
      | sha256sum | awk '{print $1}'
  )
  [[ $strategy_nd_remote_success_sha == e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 ]] || {
    echo "DB533 direct remote SUCCESS hash drifted" >&2
    exit 2
  }
  strategy_nd_remote_input_sha=$(
    gcloud storage cat \
      "$STRATEGY_ND_PREREQUISITE_REMOTE/association/input_bits.npy" \
      | sha256sum | awk '{print $1}'
  )
  [[ $strategy_nd_remote_input_sha == "$STRATEGY_ND_CANARY_INPUT_SHA" ]] || {
    echo "DB533 direct remote canary input hash drifted" >&2
    exit 2
  }
  strategy_nd_remote_output_sha=$(
    gcloud storage cat \
      "$STRATEGY_ND_PREREQUISITE_REMOTE/association/output_bits.npy" \
      | sha256sum | awk '{print $1}'
  )
  [[ $strategy_nd_remote_output_sha == "$STRATEGY_ND_CANARY_OUTPUT_SHA" ]] || {
    echo "DB533 direct remote canary output hash drifted" >&2
    exit 2
  }
fi

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
ORACLE_PIN=$(git -C "$ORACLE_REPO" rev-parse HEAD)
TILE_SUFFIX=
if [[ $FEATURE_OUTPUT_TILE != 128 ]]; then
  TILE_SUFFIX=_ot${FEATURE_OUTPUT_TILE}
fi
readonly TILE_SUFFIX
FUSION_SUFFIX=
if [[ $FEATURE_FUSE_ROUTE_WEIGHTING == 1 ]]; then
  FUSION_SUFFIX=_wsum
fi
readonly FUSION_SUFFIX
RECONSTRUCTION_SUFFIX=
if [[ $FEATURE_RECONSTRUCT_DOWN_FP32 == 1 ]]; then
  RECONSTRUCTION_SUFFIX=_downf32
fi
readonly RECONSTRUCTION_SUFFIX
TOKEN_SUFFIX=
if [[ $COMPLETE_TOKEN_PATH == 1 ]]; then
  TOKEN_SUFFIX=_token
fi
readonly TOKEN_SUFFIX
SPLIT_RESIDUAL_SUFFIX=
if [[ $SPLIT_RESIDUAL_STATE == 1 ]]; then
  SPLIT_RESIDUAL_SUFFIX=_splitres
fi
readonly SPLIT_RESIDUAL_SUFFIX
PREFILL_REPAIR_SUFFIX=
if [[ $PREFILL_INDEX_REPAIR == 1 ]]; then
  PREFILL_REPAIR_SUFFIX=_prefill_keyfix
fi
readonly PREFILL_REPAIR_SUFFIX
DSA_QUERY_SUFFIX=
if [[ $DSA_QUERY_EXACT_ASSOCIATION == 1 ]]; then
  DSA_QUERY_SUFFIX=_queryexact
fi
readonly DSA_QUERY_SUFFIX
DSA_HEAD_KEY_SUFFIX=
if [[ $DSA_HEAD_KEY_EXACT_ASSOCIATION == 1 ]]; then
  DSA_HEAD_KEY_SUFFIX=_headkeyexact
fi
readonly DSA_HEAD_KEY_SUFFIX
DSA_SCORE_PRECISION_SUFFIX=
if [[ $DSA_SCORE_DEFAULT_PRECISION == 1 ]]; then
  DSA_SCORE_PRECISION_SUFFIX=_scoredefault
fi
readonly DSA_SCORE_PRECISION_SUFFIX
MAIN_ROPE_SUFFIX=
if [[ $MAIN_ROPE_TABLE == 1 ]]; then
  MAIN_ROPE_SUFFIX=_mainrope
fi
readonly MAIN_ROPE_SUFFIX
PREGATHERED_ATTENTION_SUFFIX=
if [[ $PREGATHERED_B512_ATTENTION == 1 ]]; then
  PREGATHERED_ATTENTION_SUFFIX=_pregatheredb512
fi
readonly PREGATHERED_ATTENTION_SUFFIX
STRATEGY_ND_ATTENTION_SUFFIX=
if [[ $STRATEGY_ND_ATTENTION_PROJECTION == 1 ]]; then
  STRATEGY_ND_ATTENTION_SUFFIX=_strategynd_o
fi
readonly STRATEGY_ND_ATTENTION_SUFFIX
ORACLE_SUFFIX=
if [[ $SHORT_CONTEXT_ORACLE == 1 ]]; then
  ORACLE_SUFFIX=_oracle
fi
if [[ $SHORT_CONTEXT_DSA_ORACLE == 1 ]]; then
  ORACLE_SUFFIX=_oracle_dsa
fi
readonly ORACLE_SUFFIX
RESIDUAL_SUFFIX=
if [[ $LAYER_RESIDUAL_OBSERVER == 1 ]]; then
  RESIDUAL_SUFFIX=_residual_p${LAYER_RESIDUAL_POSITION}
fi
readonly RESIDUAL_SUFFIX
DSA_INTERNAL_SUFFIX=
if [[ $DSA_INTERNAL_OBSERVER == 1 ]]; then
  DSA_INTERNAL_SUFFIX=_dsa_internal
fi
readonly DSA_INTERNAL_SUFFIX
LAYER0_VARIANT_SUFFIX=
if [[ $LAYER0_RESIDUAL_VARIANTS == 1 ]]; then
  LAYER0_VARIANT_SUFFIX=_layer0_residual_variants
elif [[ $LAYER0_SUBSHARD_VARIANTS == 1 ]]; then
  LAYER0_VARIANT_SUFFIX=_layer0_subshard_variants
elif [[ $LAYER0_ATTENTION_VARIANTS == 1 ]]; then
  LAYER0_VARIANT_SUFFIX=_layer0_attention_schedule_variants
elif [[ $LAYER0_ATTENTION_OUTPUT_VARIANTS == 1 ]]; then
  LAYER0_VARIANT_SUFFIX=_layer0_attention_output_variants
elif [[ $LAYER0_STRATEGY_ND_ROW0 == 1 ]]; then
  LAYER0_VARIANT_SUFFIX=_layer0_strategy_nd_row0
elif [[ $LAYER0_INGREDIENTS == 1 ]]; then
  LAYER0_VARIANT_SUFFIX=_layer0_ingredients
fi
readonly LAYER0_VARIANT_SUFFIX
ROUNDTRIP_SUFFIX=
if [[ $VERIFY_DEVICE_ROUNDTRIP == 1 ]]; then
  ROUNDTRIP_SUFFIX=_roundtrip
fi
readonly ROUNDTRIP_SUFFIX
TAG=${GLM_GREENFIELD_SHORT_DECODER_TAG:-greenfield_short_decoder_compile_pp8${CONTEXT_TAG_SUFFIX}_${RUNTIME_KIND}${TILE_SUFFIX}${RECONSTRUCTION_SUFFIX}${FUSION_SUFFIX}${TOKEN_SUFFIX}${SPLIT_RESIDUAL_SUFFIX}${PREFILL_REPAIR_SUFFIX}${DSA_QUERY_SUFFIX}${DSA_HEAD_KEY_SUFFIX}${DSA_SCORE_PRECISION_SUFFIX}${MAIN_ROPE_SUFFIX}${PREGATHERED_ATTENTION_SUFFIX}${STRATEGY_ND_ATTENTION_SUFFIX}${ORACLE_SUFFIX}${RESIDUAL_SUFFIX}${DSA_INTERNAL_SUFFIX}${LAYER0_VARIANT_SUFFIX}${ROUNDTRIP_SUFFIX}_trace${TRACE_STEPS}_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing decoder compile outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing decoder compile from a dirty worktree" >&2
  exit 2
}
[[ -f $SOURCE_ROOT/SUCCESS && -f $SOURCE_RUNTIME_ROOT/SUCCESS && -f $RUNTIME_ROOT/SUCCESS ]] || {
  echo "protected source/runtime checkpoint is unavailable" >&2
  exit 2
}
if [[ $SHORT_CONTEXT_ORACLE == 1 ]] && {
  [[ ! -f $SHORT_CONTEXT_ORACLE_ROOT/SUCCESS ]] ||
    [[ ! -f $SHORT_CONTEXT_ORACLE_DIR/manifest.json ]] ||
    [[ ! -f $SHORT_CONTEXT_ORACLE_DIR/tokens.safetensors ]]
}; then
  echo "protected short-context token oracle is unavailable" >&2
  exit 2
fi
if [[ $SHORT_CONTEXT_DSA_ORACLE == 1 ]] && {
  [[ ! -f $SHORT_CONTEXT_DSA_ORACLE_ROOT/SUCCESS ]] ||
    [[ ! -f $SHORT_CONTEXT_DSA_ORACLE_DIR/manifest.json ]] ||
    [[ ! -f $SHORT_CONTEXT_DSA_ORACLE_DIR/dsa_events.safetensors ]]
}; then
  echo "protected short-context DSA oracle is unavailable" >&2
  exit 2
fi
[[ -r $RESULTS_DB ]] || {
  echo "results database is unavailable" >&2
  exit 2
}
[[ ! -e $RUN_DIR ]] || {
  echo "append-only run directory exists: $RUN_DIR" >&2
  exit 2
}
mkdir -p "$RUN_DIR/host_records" "$RUN_DIR/host_logs" "$RUN_DIR/hlo" \
  "$RUN_DIR/traces" "$RUN_DIR/dsa_observer" \
  "$RUN_DIR/layer_residual_observer" "$RUN_DIR/dsa_internal_observer" \
  "$RUN_DIR/layer0_residual_discriminator"

say() {
  echo "[short-decoder-pp8 $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
}

set +e
remote_prefix_listing=$(gcloud storage ls "$REMOTE_PREFIX/**" 2>&1)
remote_prefix_rc=$?
set -e
printf '%s\n' "$remote_prefix_listing" >"$RUN_DIR/remote_prefix_preflight.txt"
if [[ $remote_prefix_rc -eq 0 ]]; then
  say "ABORT: append-only remote prefix already contains objects"
  exit 1
elif [[ $remote_prefix_rc -ne 1 ]] || ! grep -q "matched no objects" \
  "$RUN_DIR/remote_prefix_preflight.txt"; then
  say "ABORT: remote-prefix vacancy check failed"
  exit 1
fi

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  say "ABORT: another protected pod workflow holds the global lease"
  exit 1
}

has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l) -eq 8 ]] &&
    [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | sort -u | wc -l) -eq 8 ]]
}

strict_census() {
  local label=$1
  local out="$RUN_DIR/census_${label}.txt"
  local carrier="${TAG}_${label}"
  local ray_enum
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  local command
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[d]sa_throughput[.]py|[m]icrobench_collectives[.]py|[m]icrobench_pipeline_transport[.]py|[r]un_real_one_layer[.]py|[l]oad_full_checkpoint_stage[.]py|[p]ack_runtime_checkpoint[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

post_census_done=0
terminal_success_done=0

rollback_provisional_db() {
  /home/gianl/vllm-env/bin/python - "$RESULTS_DB" "$TAG" "$PIN" \
    "$RUNTIME_MANIFEST_SHA" "$CONTEXT_LABEL" "$SHORT_CONTEXT_ORACLE" \
    "$SHORT_CONTEXT_DSA_ORACLE" "$COMPLETE_TOKEN_PATH" \
    "$STRATEGY_ND_ATTENTION_PROJECTION" \
    >"$RUN_DIR/provisional_db_rollback.txt" <<'PY'
from __future__ import annotations

import json
import sqlite3
import sys

(
    db_path,
    run_tag,
    pin,
    runtime_manifest_sha256,
    context_label,
    short_context_oracle,
    short_context_dsa_oracle,
    complete_token_path,
    strategy_nd_attention_projection,
) = sys.argv[1:]
short_context_oracle = bool(int(short_context_oracle))
short_context_dsa_oracle = bool(int(short_context_dsa_oracle))
complete_token_path = bool(int(complete_token_path))
strategy_nd_attention_projection = bool(int(strategy_nd_attention_projection))
scope = (
    "gate-d"
    if short_context_dsa_oracle
    else (
        "token-oracle"
        if short_context_oracle
        else ("token-mechanism" if complete_token_path else "body")
    )
)

connection = sqlite3.connect(db_path)
connection.execute("BEGIN IMMEDIATE")
matches = []
for row in connection.execute(
    "SELECT run_id, model, model_revision, env_json, note FROM runs "
    "WHERE model_revision = ?",
    (runtime_manifest_sha256,),
):
    environment = json.loads(row[3])
    if environment.get("greenfield_run_tag") == run_tag:
        matches.append((row, environment))
if not matches:
    connection.rollback()
    print("NO_PROVISIONAL_DB_RUN")
    raise SystemExit(0)
if len(matches) != 1:
    connection.rollback()
    raise SystemExit("refusing ambiguous provisional DB rollback")
(run, environment), = matches
run_id = run[0]
expected_model = (
    f"zai-org/GLM-5.2-FP8:greenfield-78layer-{context_label}-{scope}"
)
expected_benchmark_prefix = (
    f"greenfield_78layer_{context_label}_{scope}_pp8"
)
expected_item_id = (
    "gate_d_exact_token_and_dsa"
    if short_context_dsa_oracle
    else (
        "raw_token_prefix"
        if short_context_oracle
        else ("token_step_mechanism" if complete_token_path else "body_step")
    )
)
items = connection.execute(
    "SELECT benchmark, item_id, correct, score FROM items WHERE run_id = ?",
    (run_id,),
).fetchall()
summaries = connection.execute(
    "SELECT benchmark, metric, value FROM summary WHERE run_id = ?",
    (run_id,),
).fetchall()
item_state_valid = not items or (
    len(items) == 1
    and items[0][0].startswith(expected_benchmark_prefix)
    and items[0][1:] == (expected_item_id, 1, 1.0)
    and (
        ("_strategynd_o" in items[0][0])
        == strategy_nd_attention_projection
    )
)
summary_state_valid = not summaries or (
    len(summaries) == 1
    and items
    and summaries[0]
    == (items[0][0], "contract_valid", 1.0)
)
if (
    run[1] != expected_model
    or run[2] != runtime_manifest_sha256
    or environment.get("GLM_ENGINE") != f"greenfield_pp8_decoder_{scope}"
    or environment.get("greenfield_code_hash") != pin
    or environment.get("runtime_manifest_sha256")
    != runtime_manifest_sha256
    or environment.get("greenfield_complete_token_path") != complete_token_path
    or environment.get("greenfield_short_context_oracle")
    != short_context_oracle
    or environment.get("greenfield_short_context_dsa_oracle")
    != short_context_dsa_oracle
    or environment.get("greenfield_strategy_nd_attention_projection")
    != strategy_nd_attention_projection
    or not run[4].startswith(
        f"Protected real 78-layer {context_label.upper()} transformer-body "
        "compile/run with "
    )
    or not item_state_valid
    or not summary_state_valid
):
    connection.rollback()
    raise SystemExit("refusing non-identical provisional DB rollback")
connection.execute("DELETE FROM summary WHERE run_id = ?", (run_id,))
connection.execute("DELETE FROM items WHERE run_id = ?", (run_id,))
connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
if connection.execute(
    "SELECT COUNT(*) FROM runs WHERE run_id = ?", (run_id,)
).fetchone()[0]:
    connection.rollback()
    raise SystemExit("provisional DB rollback did not remove run")
connection.commit()
print(f"ROLLED_BACK_PROVISIONAL_DB_RUN={run_id}")
PY
}

on_exit() {
  local status=$?
  if [[ $status -ne 0 && $terminal_success_done -eq 0 ]]; then
    rollback_provisional_db || true
  fi
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || true
  fi
  if [[ $status -ne 0 ]]; then
    say "FAILED status=$status; preserving diagnostics at $RUN_DIR"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic_local/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR PIN=$PIN PROFILE=$PROFILE CONTEXT_CAPACITY=$CONTEXT_CAPACITY PROMPT_TOKEN_COUNT=$PROMPT_TOKEN_COUNT RUNTIME_KIND=$RUNTIME_KIND VERIFY_DEVICE_ROUNDTRIP=$VERIFY_DEVICE_ROUNDTRIP FEATURE_OUTPUT_TILE=$FEATURE_OUTPUT_TILE FEATURE_FUSE_ROUTE_WEIGHTING=$FEATURE_FUSE_ROUTE_WEIGHTING FEATURE_RECONSTRUCT_DOWN_FP32=$FEATURE_RECONSTRUCT_DOWN_FP32 COMPLETE_TOKEN_PATH=$COMPLETE_TOKEN_PATH SPLIT_RESIDUAL_STATE=$SPLIT_RESIDUAL_STATE PREFILL_INDEX_REPAIR=$PREFILL_INDEX_REPAIR DSA_QUERY_EXACT_ASSOCIATION=$DSA_QUERY_EXACT_ASSOCIATION DSA_HEAD_KEY_EXACT_ASSOCIATION=$DSA_HEAD_KEY_EXACT_ASSOCIATION DSA_SCORE_DEFAULT_PRECISION=$DSA_SCORE_DEFAULT_PRECISION MAIN_ROPE_TABLE=$MAIN_ROPE_TABLE PREGATHERED_B512_ATTENTION=$PREGATHERED_B512_ATTENTION STRATEGY_ND_ATTENTION_PROJECTION=$STRATEGY_ND_ATTENTION_PROJECTION SHORT_CONTEXT_ORACLE=$SHORT_CONTEXT_ORACLE SHORT_CONTEXT_DSA_ORACLE=$SHORT_CONTEXT_DSA_ORACLE LAYER_RESIDUAL_OBSERVER=$LAYER_RESIDUAL_OBSERVER LAYER_RESIDUAL_POSITION=$LAYER_RESIDUAL_POSITION DSA_INTERNAL_OBSERVER=$DSA_INTERNAL_OBSERVER LAYER0_RESIDUAL_VARIANTS=$LAYER0_RESIDUAL_VARIANTS LAYER0_SUBSHARD_VARIANTS=$LAYER0_SUBSHARD_VARIANTS LAYER0_ATTENTION_VARIANTS=$LAYER0_ATTENTION_VARIANTS LAYER0_ATTENTION_OUTPUT_VARIANTS=$LAYER0_ATTENTION_OUTPUT_VARIANTS LAYER0_STRATEGY_ND_ROW0=$LAYER0_STRATEGY_ND_ROW0 LAYER0_INGREDIENTS=$LAYER0_INGREDIENTS WARMUP=$WARMUP ITERATIONS=$ITERATIONS TRACE_STEPS=$TRACE_STEPS"
say "RUNTIME=$RUNTIME_MANIFEST_SHA SOURCE_RUNTIME=$SOURCE_RUNTIME_MANIFEST_SHA SOURCE=$SOURCE_MANIFEST_SHA"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "syncing exact code and runtime/source artifacts"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$GREENFIELD_ORIGIN"'; wt='"$WORKTREE"'; source_root='"$SOURCE_ROOT"'; source_runtime_root='"$SOURCE_RUNTIME_ROOT"'; runtime_root='"$RUNTIME_ROOT"'; oracle_mode='"$SHORT_CONTEXT_ORACLE"'; oracle_root='"$SHORT_CONTEXT_ORACLE_ROOT"'; oracle_dir='"$SHORT_CONTEXT_ORACLE_DIR"'; dsa_oracle_mode='"$SHORT_CONTEXT_DSA_ORACLE"'; dsa_oracle_root='"$SHORT_CONTEXT_DSA_ORACLE_ROOT"'; dsa_oracle_dir='"$SHORT_CONTEXT_DSA_ORACLE_DIR"'; internal_mode='"$DSA_INTERNAL_OBSERVER"'; internal_baseline='"$DSA_INTERNAL_BASELINE_NPZ"'; internal_baseline_sha='"$DSA_INTERNAL_BASELINE_SHA"'; internal_ref='"$DSA_INTERNAL_LAYER0_REFERENCE_NPZ"'; internal_ref_sha='"$DSA_INTERNAL_LAYER0_REFERENCE_SHA"'; variant_mode='"$LAYER0_DISCRIMINATOR"'; layer1_ref='"$LAYER1_INTERNAL_REFERENCE_NPZ"'; layer1_ref_sha='"$LAYER1_INTERNAL_REFERENCE_SHA"'; if [[ ${HOSTNAME##*-w-} == 0 ]]; then [[ -e "$wt/.git" ]] && [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; else if [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin "$branch"; git -C "$wt" checkout -q --detach "$pin"; elif [[ -e "$wt" ]]; then echo "stale non-repository path $wt" >&2; exit 1; else git clone -q --filter=blob:none --no-checkout --single-branch --branch "$branch" "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; fi; oracle_ok=1; if [[ $oracle_mode == 1 ]]; then [[ -r "$oracle_root/SUCCESS" && -r "$oracle_dir/manifest.json" && -r "$oracle_dir/tokens.safetensors" ]] && findmnt -T "$oracle_root" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" || oracle_ok=0; fi; dsa_oracle_ok=1; if [[ $dsa_oracle_mode == 1 ]]; then [[ -r "$dsa_oracle_root/SUCCESS" && -r "$dsa_oracle_dir/manifest.json" && -r "$dsa_oracle_dir/dsa_events.safetensors" ]] && findmnt -T "$dsa_oracle_root" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" || dsa_oracle_ok=0; fi; internal_ok=1; if [[ $internal_mode == 1 ]]; then [[ -r "$internal_baseline" && -r "$internal_ref" && $(sha256sum "$internal_baseline" | awk "{print \$1}") == "$internal_baseline_sha" && $(sha256sum "$internal_ref" | awk "{print \$1}") == "$internal_ref_sha" ]] || internal_ok=0; fi; variant_ok=1; if [[ $variant_mode == 1 ]]; then [[ -r "$layer1_ref" && $(sha256sum "$layer1_ref" | awk "{print \$1}") == "$layer1_ref_sha" ]] && findmnt -T "$layer1_ref" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" || variant_ok=0; fi; [[ $oracle_ok == 1 && $dsa_oracle_ok == 1 && $internal_ok == 1 && $variant_ok == 1 ]] && [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]] && [[ -r "$source_root/SUCCESS" ]] && [[ -r "$source_runtime_root/SUCCESS" ]] && [[ -r "$runtime_root/SUCCESS" ]] && findmnt -T "$source_runtime_root" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && findmnt -T "$runtime_root" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: eight-host sync/artifact prerequisite failed"
  exit 1
}
if [[ $LAYER0_STRATEGY_ND_ROW0 == 1 ]]; then
  # shellcheck disable=SC2016
  canary_sync_command='set -euo pipefail; input='"$STRATEGY_ND_CANARY_INPUT_BITS"'; input_sha='"$STRATEGY_ND_CANARY_INPUT_SHA"'; output='"$STRATEGY_ND_CANARY_OUTPUT_BITS"'; output_sha='"$STRATEGY_ND_CANARY_OUTPUT_SHA"'; [[ -r "$input" && -r "$output" ]] && [[ $(sha256sum "$input" | awk "{print \$1}") == "$input_sha" ]] && [[ $(sha256sum "$output" | awk "{print \$1}") == "$output_sha" ]] && findmnt -T "$input" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && findmnt -T "$output" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && echo "STRATEGY_CANARY_OK $(hostname)"'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$canary_sync_command" >"$RUN_DIR/sync_strategy_canary.txt" 2>&1
  has_eight_unique_markers \
    "$RUN_DIR/sync_strategy_canary.txt" STRATEGY_CANARY_OK || {
    say "ABORT: eight-host DB533 canary prerequisite failed"
    exit 1
  }
fi

coordinator=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=0 \
  --command="hostname -I" 2>/dev/null | grep -Eo '192\.168\.[0-9]+\.[0-9]+' | head -1)
[[ -n $coordinator ]] || {
  say "ABORT: worker-0 coordinator address is unavailable"
  exit 1
}
coordinator="$coordinator:8476"
say "launching real 78-layer $CONTEXT_NAME load/compile coordinator=$coordinator"
# shellcheck disable=SC2016
execute_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; wt='"$WORKTREE"'; remote='"$REMOTE_PREFIX"'; verify_device_roundtrip='"$VERIFY_DEVICE_ROUNDTRIP"'; feature_output_tile='"$FEATURE_OUTPUT_TILE"'; feature_fuse_route_weighting='"$FEATURE_FUSE_ROUTE_WEIGHTING"'; feature_reconstruct_down_fp32='"$FEATURE_RECONSTRUCT_DOWN_FP32"'; complete_token_path='"$COMPLETE_TOKEN_PATH"'; split_residual_state='"$SPLIT_RESIDUAL_STATE"'; prefill_index_repair='"$PREFILL_INDEX_REPAIR"'; dsa_query_exact_association='"$DSA_QUERY_EXACT_ASSOCIATION"'; dsa_head_key_exact_association='"$DSA_HEAD_KEY_EXACT_ASSOCIATION"'; dsa_score_default_precision='"$DSA_SCORE_DEFAULT_PRECISION"'; main_rope_table='"$MAIN_ROPE_TABLE"'; pregathered_b512_attention='"$PREGATHERED_B512_ATTENTION"'; short_context_oracle='"$SHORT_CONTEXT_ORACLE"'; oracle_dir='"$SHORT_CONTEXT_ORACLE_DIR"'; oracle_sha='"$SHORT_CONTEXT_ORACLE_MANIFEST_SHA"'; short_context_dsa_oracle='"$SHORT_CONTEXT_DSA_ORACLE"'; dsa_oracle_dir='"$SHORT_CONTEXT_DSA_ORACLE_DIR"'; dsa_oracle_sha='"$SHORT_CONTEXT_DSA_ORACLE_MANIFEST_SHA"'; layer_residual_observer='"$LAYER_RESIDUAL_OBSERVER"'; layer_residual_position='"$LAYER_RESIDUAL_POSITION"'; dsa_internal_observer='"$DSA_INTERNAL_OBSERVER"'; internal_baseline='"$DSA_INTERNAL_BASELINE_NPZ"'; internal_baseline_sha='"$DSA_INTERNAL_BASELINE_SHA"'; internal_ref='"$DSA_INTERNAL_LAYER0_REFERENCE_NPZ"'; internal_ref_sha='"$DSA_INTERNAL_LAYER0_REFERENCE_SHA"'; layer0_variants='"$LAYER0_RESIDUAL_VARIANTS"'; layer0_subshard_variants='"$LAYER0_SUBSHARD_VARIANTS"'; layer0_attention_variants='"$LAYER0_ATTENTION_VARIANTS"'; layer0_attention_output_variants='"$LAYER0_ATTENTION_OUTPUT_VARIANTS"'; layer0_strategy_nd_row0='"$LAYER0_STRATEGY_ND_ROW0"'; strategy_nd_canary_input='"$STRATEGY_ND_CANARY_INPUT_BITS"'; strategy_nd_canary_input_sha='"$STRATEGY_ND_CANARY_INPUT_SHA"'; strategy_nd_canary_output='"$STRATEGY_ND_CANARY_OUTPUT_BITS"'; strategy_nd_canary_output_sha='"$STRATEGY_ND_CANARY_OUTPUT_SHA"'; layer0_ingredients='"$LAYER0_INGREDIENTS"'; layer1_ref='"$LAYER1_INTERNAL_REFERENCE_NPZ"'; layer1_ref_sha='"$LAYER1_INTERNAL_REFERENCE_SHA"'; run=/home/gianl/glm-run/$tag; mkdir -p "$run/hlo" "$run/layer_residual_observer" "$run/dsa_internal_observer" "$run/layer0_residual_discriminator" "$run/layer0_ingredients"; output="$run/decoder.rank${idx}.json"; log="$run/decoder.rank${idx}.log"; upload() { gcloud storage cp --no-clobber "$log" "$output" "$remote/host_records/" >/dev/null 2>&1 || true; if compgen -G "$run/hlo/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/hlo/* "$remote/hlo/" >/dev/null 2>&1 || true; fi; if compgen -G "$run/dsa_observer/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/dsa_observer/* "$remote/dsa_observer/" >/dev/null 2>&1 || true; fi; if compgen -G "$run/layer_residual_observer/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/layer_residual_observer/* "$remote/layer_residual_observer/" >/dev/null 2>&1 || true; fi; if compgen -G "$run/dsa_internal_observer/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/dsa_internal_observer/* "$remote/dsa_internal_observer/" >/dev/null 2>&1 || true; fi; if compgen -G "$run/layer0_residual_discriminator/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/layer0_residual_discriminator/* "$remote/layer0_residual_discriminator/" >/dev/null 2>&1 || true; fi; if compgen -G "$run/layer0_ingredients/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/layer0_ingredients/* "$remote/layer0_ingredients/" >/dev/null 2>&1 || true; fi; xplane=$(find "$run/trace" -type f -name "*.xplane.pb" 2>/dev/null | head -1 || true); if [[ -n $xplane ]]; then gcloud storage cp --no-clobber "$xplane" "$remote/traces/trace.rank${idx}.xplane.pb" >/dev/null 2>&1 || true; fi; }; trap upload EXIT; cd "$wt"; trace_args=(); if [[ '"$TRACE_STEPS"' -gt 0 ]]; then trace_args=(--trace-root "$run/trace" --trace-steps '"$TRACE_STEPS"'); fi; oracle_args=(); if [[ $short_context_oracle == 1 ]]; then oracle_args=(--short-context-oracle-dir "$oracle_dir" --short-context-oracle-manifest-sha256 "$oracle_sha"); fi; dsa_oracle_args=(); if [[ $short_context_dsa_oracle == 1 ]]; then dsa_oracle_args=(--short-context-dsa-oracle-dir "$dsa_oracle_dir" --short-context-dsa-oracle-manifest-sha256 "$dsa_oracle_sha"); fi; residual_args=(); if [[ $layer_residual_observer == 1 ]]; then residual_args=(--observe-layer-residuals 1 --layer-residual-position "$layer_residual_position"); fi; internal_args=(); if [[ $dsa_internal_observer == 1 ]]; then internal_args=(--observe-dsa-internals 1 --dsa-internal-baseline-observation-npz "$internal_baseline" --dsa-internal-baseline-observation-sha256 "$internal_baseline_sha" --dsa-internal-layer0-reference-npz "$internal_ref" --dsa-internal-layer0-reference-sha256 "$internal_ref_sha"); fi; variant_args=(); if [[ $layer0_variants == 1 ]]; then variant_args=(--observe-layer0-residual-variants 1 --layer1-internal-reference-npz "$layer1_ref" --layer1-internal-reference-sha256 "$layer1_ref_sha"); elif [[ $layer0_subshard_variants == 1 ]]; then variant_args=(--observe-layer0-subshard-variants 1 --layer1-internal-reference-npz "$layer1_ref" --layer1-internal-reference-sha256 "$layer1_ref_sha"); elif [[ $layer0_attention_variants == 1 ]]; then variant_args=(--observe-layer0-attention-schedule-variants 1 --layer1-internal-reference-npz "$layer1_ref" --layer1-internal-reference-sha256 "$layer1_ref_sha"); elif [[ $layer0_attention_output_variants == 1 ]]; then variant_args=(--observe-layer0-attention-output-association-variants 1 --layer1-internal-reference-npz "$layer1_ref" --layer1-internal-reference-sha256 "$layer1_ref_sha"); elif [[ $layer0_strategy_nd_row0 == 1 ]]; then variant_args=(--observe-layer0-strategy-nd-row0-association 1 --strategy-nd-canary-input-bits "$strategy_nd_canary_input" --strategy-nd-canary-input-sha256 "$strategy_nd_canary_input_sha" --strategy-nd-canary-output-bits "$strategy_nd_canary_output" --strategy-nd-canary-output-sha256 "$strategy_nd_canary_output_sha" --layer1-internal-reference-npz "$layer1_ref" --layer1-internal-reference-sha256 "$layer1_ref_sha"); elif [[ $layer0_ingredients == 1 ]]; then variant_args=(--observe-layer0-ingredients 1); fi; env JAX_PLATFORMS=tpu XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH="$wt" GLM_GREENFIELD_RUN_TAG="$tag" timeout --signal=TERM --kill-after=60 10800 /home/gianl/vllm-env/bin/python -u scripts/greenfield/compile_short_decoder.py --coordinator-address '"$coordinator"' --num-processes 8 --process-id "$idx" --expected-code-hash '"$PIN"' --runtime-kind '"$RUNTIME_KIND"' --verify-device-roundtrip "$verify_device_roundtrip" --feature-output-tile "$feature_output_tile" --feature-fuse-route-weighting "$feature_fuse_route_weighting" --feature-reconstruct-down-fp32 "$feature_reconstruct_down_fp32" --complete-token-path "$complete_token_path" --split-residual-state "$split_residual_state" --prefill-index-repair "$prefill_index_repair" --dsa-query-exact-association "$dsa_query_exact_association" --dsa-head-key-exact-association "$dsa_head_key_exact_association" --dsa-score-default-precision "$dsa_score_default_precision" --main-rope-table "$main_rope_table" --pregathered-b512-attention "$pregathered_b512_attention" --runtime-root '"$RUNTIME_ROOT"' --runtime-manifest-sha256 '"$RUNTIME_MANIFEST_SHA"' --source-runtime-root '"$SOURCE_RUNTIME_ROOT"' --source-runtime-manifest-sha256 '"$SOURCE_RUNTIME_MANIFEST_SHA"' --source-checkpoint-root '"$SOURCE_ROOT"' --source-packed-manifest-sha256 '"$SOURCE_MANIFEST_SHA"' --context-capacity '"$CONTEXT_CAPACITY"' --warmup '"$WARMUP"' --iterations '"$ITERATIONS"' "${trace_args[@]}" "${oracle_args[@]}" "${dsa_oracle_args[@]}" "${residual_args[@]}" "${internal_args[@]}" "${variant_args[@]}" --output "$output" >"$log" 2>&1; trap - EXIT; upload; echo "DECODER_HOST_OK $(hostname) rank=$idx"'
execute_command="export GLM_GREENFIELD_STRATEGY_ND_ATTENTION_PROJECTION=$STRATEGY_ND_ATTENTION_PROJECTION; $execute_command"
execute_status=0
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$execute_command" >"$RUN_DIR/execute.txt" 2>&1 || execute_status=$?
if [[ $execute_status -ne 0 ]] || \
  ! has_eight_unique_markers "$RUN_DIR/execute.txt" DECODER_HOST_OK; then
  say "retrieving fail-closed decoder diagnostic artifacts"
  gcloud storage cp "$REMOTE_PREFIX/host_records/*" \
    "$RUN_DIR/host_logs/" >"$RUN_DIR/diagnostic_downloads.txt" 2>&1 || true
  gcloud storage cp "$REMOTE_PREFIX/hlo/*" \
    "$RUN_DIR/hlo/" >>"$RUN_DIR/diagnostic_downloads.txt" 2>&1 || true
  gcloud storage cp "$REMOTE_PREFIX/dsa_observer/*" \
    "$RUN_DIR/dsa_observer/" >>"$RUN_DIR/diagnostic_downloads.txt" 2>&1 || true
  if [[ $LAYER_RESIDUAL_OBSERVER == 1 ]]; then
    gcloud storage cp "$REMOTE_PREFIX/layer_residual_observer/*" \
      "$RUN_DIR/layer_residual_observer/" \
      >>"$RUN_DIR/diagnostic_downloads.txt" 2>&1 || true
  fi
  if [[ $DSA_INTERNAL_OBSERVER == 1 ]]; then
    gcloud storage cp "$REMOTE_PREFIX/dsa_internal_observer/*" \
      "$RUN_DIR/dsa_internal_observer/" \
      >>"$RUN_DIR/diagnostic_downloads.txt" 2>&1 || true
  fi
  if [[ $LAYER0_DISCRIMINATOR == 1 ]]; then
    gcloud storage cp "$REMOTE_PREFIX/layer0_residual_discriminator/*" \
      "$RUN_DIR/layer0_residual_discriminator/" \
      >>"$RUN_DIR/diagnostic_downloads.txt" 2>&1 || true
  fi
  if [[ $LAYER0_INGREDIENTS == 1 ]]; then
    gcloud storage cp "$REMOTE_PREFIX/layer0_ingredients/*" \
      "$RUN_DIR/layer0_ingredients/" \
      >>"$RUN_DIR/diagnostic_downloads.txt" 2>&1 || true
    /home/gianl/vllm-env/bin/python - \
      "$RUN_DIR/layer0_ingredients/contract.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("layer-0 ingredient diagnostic contract is missing")
contract = json.loads(path.read_text())
main_rope = contract.get("hlo_contract", {}).get(
    "main_rope_table_contract", {}
)
if (
    not contract.get("passed")
    or contract.get("decode_position") != 8155
    or contract.get("main_rope_table_enabled") is not True
    or contract.get("main_rope_table_sha256")
    != "6a22140fc2aec475399738c6fc0f29be2a6c419feb0249aee35681c607c80701"
    or not main_rope.get("passed")
    or main_rope.get("table_parameter_count") != 1
    or main_rope.get("named_table_parameter_count") != 1
):
    raise SystemExit("layer-0 ingredient diagnostic contract failed")
print("LAYER0_INGREDIENTS_DIAGNOSTIC_CONTRACT_OK")
PY
    say "layer-0 ingredient diagnostic contract passed; preserving intentional diagnostic exit"
  fi
  if [[ $LAYER0_ATTENTION_VARIANTS == 1 ]]; then
    /home/gianl/vllm-env/bin/python - \
      "$RUN_DIR/layer0_residual_discriminator/contract.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("attention-schedule diagnostic contract is missing")
contract = json.loads(path.read_text())
expected_names = [
    "attention_schedule_control",
    "replicated_monolithic_attention",
]
expected_baseline = (
    "6c54c09a773e622fef35e753dc99929a83b73d9d89157fd732a5ace903149bca"
)
if (
    not contract.get("passed")
    or contract.get("discriminator_kind") != "attention_schedule"
    or contract.get("variant_names") != expected_names
    or not contract.get("reproduces_current_baseline")
    or contract.get("current_baseline_expected_sha256") != expected_baseline
    or contract["variant_comparisons"]["attention_schedule_control"][
        "actual_sha256"
    ]
    != expected_baseline
    or not contract.get("hlo_suite_contract", {}).get("passed")
    or any(
        not contract["arm_contracts"].get(name, {}).get("passed")
        for name in expected_names
    )
):
    raise SystemExit("attention-schedule diagnostic contract failed")
print("ATTENTION_SCHEDULE_DIAGNOSTIC_CONTRACT_OK")
PY
    say "attention-schedule diagnostic contract passed; preserving intentional diagnostic exit"
  fi
  if [[ $LAYER0_ATTENTION_OUTPUT_VARIANTS == 1 ]]; then
    /home/gianl/vllm-env/bin/python - \
      "$RUN_DIR/layer0_residual_discriminator/contract.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(
        "attention-output association diagnostic contract is missing"
    )
contract = json.loads(path.read_text())
expected_names = [
    "attention_output_association_control",
    "attention_output_dcp_then_model_sequential_bf16",
    "attention_output_dcp_then_model_pairwise_bf16",
    "attention_output_model_then_dcp_sequential_bf16",
    "attention_output_model_then_dcp_pairwise_bf16",
]
expected_baseline = (
    "6c54c09a773e622fef35e753dc99929a83b73d9d89157fd732a5ace903149bca"
)
arms = contract.get("arm_contracts", {})
suite = contract.get("hlo_suite_contract", {})
control = contract.get("variant_comparisons", {}).get(
    "attention_output_association_control", {}
)
if (
    not contract.get("passed")
    or contract.get("discriminator_kind")
    != "attention_output_association"
    or contract.get("variant_names") != expected_names
    or not contract.get("reproduces_current_baseline")
    or contract.get("current_baseline_expected_sha256")
    != expected_baseline
    or control.get("actual_sha256") != expected_baseline
    or control.get("mismatch_count") != 3984
    or not suite.get("passed")
    or suite.get("program_count") != len(expected_names)
    or not suite.get("all_distinct_hlo_modules")
    or any(
        not arms.get(name, {}).get("passed")
        or not arms[name].get("selection_exact")
        for name in expected_names
    )
):
    raise SystemExit(
        "attention-output association diagnostic contract failed"
    )
print("ATTENTION_OUTPUT_ASSOCIATION_DIAGNOSTIC_CONTRACT_OK")
PY
    say "attention-output association diagnostic contract passed; preserving intentional diagnostic exit"
  fi
  if [[ $LAYER0_STRATEGY_ND_ROW0 == 1 ]]; then
    /home/gianl/vllm-env/bin/python - \
      "$RUN_DIR/layer0_residual_discriminator/contract.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("StrategyND row-zero diagnostic contract is missing")
contract = json.loads(path.read_text())
expected_names = ["strategy_nd_row0_control", "strategy_nd_row0_both"]
expected_baseline = (
    "6c54c09a773e622fef35e753dc99929a83b73d9d89157fd732a5ace903149bca"
)
arms = contract.get("arm_contracts", {})
suite = contract.get("hlo_suite_contract", {})
comparisons = contract.get("variant_comparisons", {})
control = comparisons.get(expected_names[0], {})
candidate = comparisons.get(expected_names[1], {})
candidate_hlo = arms.get(expected_names[1], {}).get("hlo_contract", {})
kernel_counts = candidate_hlo.get("kernel_counts", {})
canary = contract.get("strategy_nd_canary_contract", {})
stablehlo = suite.get("strategy_nd_stablehlo_contract", {})
if (
    not contract.get("passed")
    or contract.get("discriminator_kind") != "strategy_nd_row0_association"
    or contract.get("variant_names") != expected_names
    or not contract.get("reproduces_current_baseline")
    or contract.get("current_baseline_expected_sha256") != expected_baseline
    or control.get("actual_sha256") != expected_baseline
    or control.get("mismatch_count") != 3984
    or not isinstance(candidate.get("actual_sha256"), str)
    or len(candidate["actual_sha256"]) != 64
    or not isinstance(candidate.get("mismatch_count"), int)
    or not 0 <= candidate["mismatch_count"] <= 6144
    or not suite.get("passed")
    or suite.get("program_count") != 2
    or not suite.get("all_distinct_hlo_modules")
    or any(
        not arms.get(name, {}).get("passed")
        or not arms[name].get("selection_exact")
        for name in expected_names
    )
    or len(candidate_hlo.get("strategy_nd_gathers", [])) != 2
    or candidate_hlo.get("strategy_nd_shaped_gather_count") != 2
    or candidate_hlo.get("strategy_nd_attention_gather_count") != 1
    or candidate_hlo.get("strategy_nd_dense_gather_count") != 1
    or candidate_hlo.get("cache_shaped_gathers")
    or not suite.get("strategy_nd_canary_passed")
    or not stablehlo.get("passed")
    or stablehlo.get("observed_delta") != 164
    or not canary.get("passed")
    or canary.get("trial_count") != 32
    or canary.get("lane_count") != 32
    or canary.get("mismatch_count") != 0
    or canary.get("first_mismatch") is not None
    or canary.get("stablehlo_optimization_barrier_count") != 82
    or not canary.get("hlo_contract", {}).get("passed")
    or canary.get("input_file_sha256")
    != "1a6c2fb5c8c04b904daad45f751e0b5308919b85cb1f3cf63879c5febf3ea0e9"
    or canary.get("output_file_sha256")
    != "fda0228123392c29db87e1d6dd01cc6830041196409f49e0144628054cc64bd9"
    or kernel_counts.get("greenfield_fp8_block_matmul_m8_k512_n6144") != 8
    or kernel_counts.get(
        "greenfield_fp8_fused_block_swiglu_m8_h6144_i384_o6144"
    ) != 8
):
    raise SystemExit("StrategyND row-zero diagnostic contract failed")
print("STRATEGY_ND_ROW0_DIAGNOSTIC_CONTRACT_OK")
PY
    say "StrategyND row-zero diagnostic contract passed; preserving intentional diagnostic exit"
  fi
  say "ABORT: real decoder load/compile did not pass 8/8"
  exit 1
fi

gcloud storage cp "$REMOTE_PREFIX/host_records/decoder.rank*.json" \
  "$RUN_DIR/host_records/" >/dev/null
gcloud storage cp "$REMOTE_PREFIX/host_records/decoder.rank*.log" \
  "$RUN_DIR/host_records/" >/dev/null
gcloud storage cp "$REMOTE_PREFIX/hlo/*" "$RUN_DIR/hlo/" >/dev/null
if [[ $TRACE_STEPS -gt 0 ]]; then
  gcloud storage cp "$REMOTE_PREFIX/traces/trace.rank*.xplane.pb" \
    "$RUN_DIR/traces/" >/dev/null
fi
if [[ $SHORT_CONTEXT_DSA_ORACLE == 1 ]]; then
  gcloud storage cp "$REMOTE_PREFIX/dsa_observer/*" \
    "$RUN_DIR/dsa_observer/" >/dev/null
fi
if [[ $LAYER_RESIDUAL_OBSERVER == 1 ]]; then
  gcloud storage cp "$REMOTE_PREFIX/layer_residual_observer/*" \
    "$RUN_DIR/layer_residual_observer/" >/dev/null
fi
if [[ $DSA_INTERNAL_OBSERVER == 1 ]]; then
  gcloud storage cp "$REMOTE_PREFIX/dsa_internal_observer/*" \
    "$RUN_DIR/dsa_internal_observer/" >/dev/null
fi
if [[ $LAYER0_INGREDIENTS == 1 ]]; then
  gcloud storage cp "$REMOTE_PREFIX/layer0_ingredients/*" \
    "$RUN_DIR/layer0_ingredients/" >/dev/null
fi

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

say "validating fleet agreement and recording diagnostic DB linkage"
/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$PIN" "$ORACLE_PIN" \
  "$RESULTS_DB" "$WORKTREE" "$ORACLE_REPO" "$RUNTIME_KIND" \
  "$FEATURE_OUTPUT_TILE" "$FEATURE_FUSE_ROUTE_WEIGHTING" "$FEATURE_RECONSTRUCT_DOWN_FP32" "$COMPLETE_TOKEN_PATH" "$SPLIT_RESIDUAL_STATE" "$PREFILL_INDEX_REPAIR" "$DSA_QUERY_EXACT_ASSOCIATION" "$DSA_HEAD_KEY_EXACT_ASSOCIATION" "$DSA_SCORE_DEFAULT_PRECISION" "$MAIN_ROPE_TABLE" "$PREGATHERED_B512_ATTENTION" "$STRATEGY_ND_ATTENTION_PROJECTION" "$SHORT_CONTEXT_ORACLE" "$SHORT_CONTEXT_ORACLE_MANIFEST_SHA" "$SHORT_CONTEXT_DSA_ORACLE" "$SHORT_CONTEXT_DSA_ORACLE_MANIFEST_SHA" "$SPARSE_MOE_BACKEND" "$HLO_BACKEND_CONTRACT" "$RUNTIME_MANIFEST_SHA" \
  "$RUNTIME_LAYOUT_HASH" "$WARMUP" "$ITERATIONS" "$TRACE_STEPS" \
  "$CONTEXT_LABEL" "$CONTEXT_CAPACITY" "$PROMPT_TOKEN_COUNT" \
  "$ATTENTION_PROJECTION_BACKEND" "$EXPECTED_LOADED_PAYLOAD_BYTES" \
  "$VERIFY_DEVICE_ROUNDTRIP" "$TAG" <<'PY'
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sqlite3
import sys

import numpy as np

(
    run_dir,
    pin,
    oracle_pin,
    db_path,
    repo,
    oracle_repo,
    runtime_kind,
    feature_output_tile,
    feature_fuse_route_weighting,
    feature_reconstruct_down_fp32,
    complete_token_path,
    split_residual_state,
    prefill_index_repair,
    dsa_query_exact_association,
    dsa_head_key_exact_association,
    dsa_score_default_precision,
    main_rope_table,
    pregathered_b512_attention,
    strategy_nd_attention_projection,
    short_context_oracle,
    short_context_oracle_manifest_sha256,
    short_context_dsa_oracle,
    short_context_dsa_oracle_manifest_sha256,
    sparse_moe_backend,
    hlo_backend_contract,
    runtime_manifest_sha256,
    runtime_layout_hash,
    warmup,
    iterations,
    trace_steps,
    context_label,
    context_capacity,
    prompt_token_count,
    attention_projection_backend,
    expected_loaded_payload_bytes,
    verify_device_roundtrip,
    greenfield_run_tag,
) = sys.argv[1:]
feature_output_tile = int(feature_output_tile)
feature_fuse_route_weighting = bool(int(feature_fuse_route_weighting))
feature_reconstruct_down_fp32 = bool(int(feature_reconstruct_down_fp32))
complete_token_path = bool(int(complete_token_path))
split_residual_state = bool(int(split_residual_state))
prefill_index_repair = bool(int(prefill_index_repair))
dsa_query_exact_association = bool(int(dsa_query_exact_association))
dsa_head_key_exact_association = bool(int(dsa_head_key_exact_association))
dsa_score_default_precision = bool(int(dsa_score_default_precision))
main_rope_table = bool(int(main_rope_table))
pregathered_b512_attention = bool(int(pregathered_b512_attention))
strategy_nd_attention_projection = bool(int(strategy_nd_attention_projection))
short_context_oracle = bool(int(short_context_oracle))
short_context_dsa_oracle = bool(int(short_context_dsa_oracle))
warmup = int(warmup)
iterations = int(iterations)
trace_steps = int(trace_steps)
context_capacity = int(context_capacity)
prompt_token_count = int(prompt_token_count)
expected_loaded_payload_bytes = int(expected_loaded_payload_bytes)
verify_device_roundtrip = bool(int(verify_device_roundtrip))
expected_prefill_fused_qkv_loops = (
    78 if attention_projection_backend == "fused_n82_convolution" else 0
)
expected_prefill_index_repair_loops = (
    21 * ((prompt_token_count + 2047) // 2048)
    if prefill_index_repair
    else 0
)
expected_prefill_loop_count = (
    1
    + expected_prefill_fused_qkv_loops
    + expected_prefill_index_repair_loops
)
context_name = context_label.upper()
decode_step_count = warmup + iterations + trace_steps
decode_end_exclusive = prompt_token_count + decode_step_count
run_dir = Path(run_dir)
records = [json.loads(path.read_text()) for path in sorted((run_dir / "host_records").glob("*.json"))]
if len(records) != 8:
    raise SystemExit(f"expected eight host records, got {len(records)}")
if {record["launch_process_id"] for record in records} != set(range(8)):
    raise SystemExit("launch process ids do not cover 0..7")
if {record["jax_process_index"] for record in records} != set(range(8)):
    raise SystemExit("JAX process indices do not cover 0..7")
if len({record["hostname"] for record in records}) != 8:
    raise SystemExit("host records are not fleet-distinct")
for field in (
    "code_hash",
    "dsa_observer_hlo_sha256",
    "dsa_observer_stablehlo_sha256",
    "dsa_query_materialization_hlo_sha256",
    "main_rope_table_sha256",
    "optimized_hlo_sha256",
    "plan_hash",
    "runtime_layout_hash",
    "runtime_manifest_sha256",
    "schedule_hash",
    "stablehlo_sha256",
    "state_layout_hash",
    "topology_hash",
):
    values = {record[field] for record in records}
    if len(values) != 1:
        raise SystemExit(f"fleet field {field} disagrees: {sorted(values)}")
if {record["code_hash"] for record in records} != {pin}:
    raise SystemExit("fleet used stale code")
if {record["runtime_kind"] for record in records} != {runtime_kind}:
    raise SystemExit("fleet runtime kind drifted")
if {record["feature_output_tile"] for record in records} != {
    feature_output_tile
}:
    raise SystemExit("fleet feature output tile drifted")
if {record["feature_fuse_route_weighting"] for record in records} != {
    feature_fuse_route_weighting
}:
    raise SystemExit("fleet feature route-weight fusion drifted")
if {record["feature_reconstruct_down_fp32"] for record in records} != {
    feature_reconstruct_down_fp32
}:
    raise SystemExit("fleet feature FP32 reconstruction drifted")
if {record["complete_token_path"] for record in records} != {
    complete_token_path
}:
    raise SystemExit("fleet complete token-path flag drifted")
if {record["split_residual_state"] for record in records} != {
    split_residual_state
}:
    raise SystemExit("fleet split residual-state flag drifted")
if {record["prefill_index_repair"] for record in records} != {
    prefill_index_repair
}:
    raise SystemExit("fleet prefill index-repair flag drifted")
if {record["dsa_query_exact_association"] for record in records} != {
    dsa_query_exact_association
}:
    raise SystemExit("fleet exact DSA query flag drifted")
if {record["dsa_head_key_exact_association"] for record in records} != {
    dsa_head_key_exact_association
}:
    raise SystemExit("fleet exact DSA head/key flag drifted")
if {record["dsa_score_default_precision"] for record in records} != {
    dsa_score_default_precision
}:
    raise SystemExit("fleet DSA score-precision flag drifted")
if {record["main_rope_table_enabled"] for record in records} != {
    main_rope_table
}:
    raise SystemExit("fleet main-RoPE table flag drifted")
if {record["pregathered_b512_attention"] for record in records} != {
    pregathered_b512_attention
}:
    raise SystemExit("fleet pregathered-B512 attention flag drifted")
if {record["strategy_nd_attention_projection"] for record in records} != {
    strategy_nd_attention_projection
}:
    raise SystemExit("fleet StrategyND attention-projection flag drifted")
expected_repair_backend = (
    "physical_m64_chunk" if prefill_index_repair else "none"
)
if {record["prefill_index_repair_backend"] for record in records} != {
    expected_repair_backend
}:
    raise SystemExit("fleet prefill index-repair backend drifted")
if {record["prefill_used"] for record in records} != {
    short_context_oracle
}:
    raise SystemExit("fleet short-context prefill flag drifted")
if {record["schema_version"] for record in records} != {18}:
    raise SystemExit("fleet decoder record schema drifted")

expected_main_rope_shape = [context_capacity, 64]
expected_main_rope_bytes = context_capacity * 64 * 2
expected_main_rope_sha256 = (
    "6a22140fc2aec475399738c6fc0f29be2a6c419feb0249aee35681c607c80701"
)
expected_main_rope_prerequisite = {
    "code_hash": "ed7c74f423e811f1acfbeb3c075fb8fad37d4f51",
    "position_8155_row_sha256": (
        "67b01e3cab682d5ffd04ac9c8043e7e6825ee1275023428c41f9a1ae412dea1d"
    ),
    "results_db_item_row_id": 1816,
    "results_db_run_id": 531,
    "runner_sha256": (
        "d32d03576faa99c882913cf289a32d9aaa7f258a6b83588464058f36a8ad7370"
    ),
    "success_sha256": (
        "8f9763e4e8106db0fe30b9fa3820873f736ede2215edfc8b950fcd311d08b6d7"
    ),
}

def validate_main_rope_hlo(executable, *, source):
    if executable["main_rope_table_enabled"] != main_rope_table:
        raise SystemExit(f"{source} main-RoPE flag drifted")
    contract = executable["main_rope_table_contract"]
    if (
        not contract["passed"]
        or contract["applicable"] != main_rope_table
        or contract["expected_table_shape"] != expected_main_rope_shape
        or contract["violations"]
        or contract["forbidden_instructions"]
        or contract["bf16_arithmetic"]
    ):
        raise SystemExit(f"{source} main-RoPE HLO contract failed")
    if main_rope_table:
        if (
            contract["table_parameter_count"] != 1
            or contract["named_table_parameter_count"] != 1
            or contract["scoped_instruction_count"] <= 0
            or contract["fp32_multiply_count"] < 78 * 8
            or contract["fp32_combine_count"] < 78 * 4
            or contract["final_round_count"] < 78 * 2
        ):
            raise SystemExit(f"{source} main-RoPE arithmetic drifted")
    elif (
        contract["table_parameter_count"] != 0
        or contract["named_table_parameter_count"] != 0
        or contract["scoped_instruction_count"] != 0
        or contract["fp32_multiply_count"] != 0
        or contract["fp32_combine_count"] != 0
        or contract["final_round_count"] != 0
    ):
        raise SystemExit(f"{source} default main-RoPE state is not absent")

def validate_pregathered_attention_hlo(executable, *, source):
    if executable["pregathered_b512_attention"] != pregathered_b512_attention:
        raise SystemExit(f"{source} pregathered-B512 flag drifted")
    contract = executable["pregathered_b512_attention_contract"]
    if (
        not contract["passed"]
        or contract["applicable"] != pregathered_b512_attention
        or contract["violations"]
    ):
        raise SystemExit(f"{source} pregathered-B512 HLO contract failed")
    if pregathered_b512_attention:
        if (
            contract["exchange_count"] != 78
            or not contract["exchange_bijection"]
            or contract["expected_exchange_count"] != 78
            or contract["expected_exchange_shape"] != "bf16[1,2048,640]"
            or contract["kernel_count"] != 78
            or len(contract["kernel_exchange_links"]) != 78
            or contract["kernel_name"]
            != "greenfield_pregathered_sparse_mla_h16_k2048_b512_w640"
            or contract["old_scope_instruction_count"] != 0
        ):
            raise SystemExit(f"{source} pregathered-B512 physical contract drifted")
    elif (
        contract["exchange_count"] != 0
        or contract["kernel_count"] != 0
        or contract["attention_instruction_count"] != 0
    ):
        raise SystemExit(f"{source} default pregathered-B512 state is not absent")

def validate_strategy_nd_attention_hlo(executable, *, source):
    if (
        executable["strategy_nd_attention_projection"]
        != strategy_nd_attention_projection
    ):
        raise SystemExit(f"{source} StrategyND attention flag drifted")
    contract = executable["strategy_nd_attention_projection_contract"]
    stable_contract = executable[
        "strategy_nd_attention_stablehlo_contract"
    ]
    if (
        not contract["passed"]
        or contract["applicable"] != strategy_nd_attention_projection
        or contract["violations"]
        or not stable_contract["passed"]
        or stable_contract["applicable"]
        != strategy_nd_attention_projection
        or stable_contract["violations"]
    ):
        raise SystemExit(f"{source} StrategyND attention HLO contract failed")
    if strategy_nd_attention_projection:
        if (
            contract["gather_count"] != 78
            or contract["expected_gather_count"] != 78
            or contract["kernel_count"] != 624
            or contract["expected_kernel_count"] != 624
            or contract["kernel_name"]
            != "greenfield_fp8_block_matmul_m8_k512_n6144"
            or not contract["partial_gather_bijection"]
            or not contract["exclusive_partial_dataflow"]
            or contract["exclusive_source_violations"]
            or not contract["all_gathers_live"]
            or set(contract["gather_source_counts"].values()) != {8}
            or stable_contract["expected_gather_count"] != 78
            or stable_contract["expected_kernel_count"] != 624
            or stable_contract["gather_count"] != 78
            or stable_contract["kernel_count"] != 624
            or stable_contract["matched_tree_count"] != 78
            or len(stable_contract["matched_trees"]) != 78
            or any(
                tree["kernel_count"] != 8
                for tree in stable_contract["matched_trees"]
            )
        ):
            raise SystemExit(
                f"{source} StrategyND attention physical contract drifted"
            )
    elif (
        contract["gather_count"] != 0
        or contract["kernel_count"] != 0
        or contract["scoped_instruction_count"] != 0
        or stable_contract["expected_gather_count"] != 0
        or stable_contract["expected_kernel_count"] != 0
        or stable_contract["gather_count"] != 0
        or stable_contract["kernel_count"] != 0
        or stable_contract["matched_tree_count"] != 0
        or stable_contract["matched_trees"]
    ):
        raise SystemExit(
            f"{source} default StrategyND attention state is not absent"
        )

for record in records:
    validate_main_rope_hlo(record["hlo_contract"], source="decoder")
    validate_pregathered_attention_hlo(
        record["hlo_contract"], source="decoder"
    )
    validate_strategy_nd_attention_hlo(
        record["hlo_contract"], source="decoder"
    )
    if (
        record["stablehlo_sha256"] is None
        or len(set(record["fleet_stablehlo_hashes"])) != 1
        or record["fleet_stablehlo_hashes"][0]
        != record["stablehlo_sha256"]
    ):
        raise SystemExit("fleet decoder StableHLO hash drifted")
    if main_rope_table:
        if (
            record["main_rope_table_sha256"]
            != expected_main_rope_sha256
            or record["main_rope_table_shape"] != expected_main_rope_shape
            or record["main_rope_table_bytes_per_device"]
            != expected_main_rope_bytes
            or record["main_rope_table_prerequisite"]
            != expected_main_rope_prerequisite
            or len(record["main_rope_table_local_device_sha256"]) != 4
            or set(record["main_rope_table_local_device_sha256"])
            != {expected_main_rope_sha256}
        ):
            raise SystemExit("fleet main-RoPE table asset drifted")
    elif (
        record["main_rope_table_sha256"] is not None
        or record["main_rope_table_shape"] is not None
        or record["main_rope_table_bytes_per_device"] != 0
        or record["main_rope_table_prerequisite"] is not None
        or record["main_rope_table_local_device_sha256"]
    ):
        raise SystemExit("default decoder materialized main-RoPE table state")
if short_context_oracle:
    for field in ("prefill_hlo_sha256", "prefill_stablehlo_sha256"):
        values = {record[field] for record in records}
        if len(values) != 1 or None in values:
            raise SystemExit(f"fleet field {field} disagrees: {values}")
if {record["sparse_moe_backend"] for record in records} != {sparse_moe_backend}:
    raise SystemExit("fleet sparse MoE backend drifted")
expected_linear_backend = (
    "pallas" if runtime_kind == "pallas_feature_linear" else "reference"
)
if {record["linear_backend"] for record in records} != {expected_linear_backend}:
    raise SystemExit("fleet FP8 linear backend drifted")
if {record["dsa_query_backend"] for record in records} != {"reference"}:
    raise SystemExit("fleet DSA query backend drifted")
for record in records:
    association = record["hlo_contract"][
        "dsa_query_association_contract"
    ]
    if (
        association["exact_association"]
        != dsa_query_exact_association
        or association["tuple4_reduction_fusion_count"]
        != (21 if dsa_query_exact_association else 0)
    ):
        raise SystemExit("fleet DSA query association HLO drifted")
    head_key_contracts = [record["hlo_contract"]]
    if short_context_dsa_oracle:
        head_key_contracts.append(record["dsa_observer_hlo_contract"])
        validate_main_rope_hlo(
            record["dsa_observer_hlo_contract"],
            source="DSA observer",
        )
        validate_pregathered_attention_hlo(
            record["dsa_observer_hlo_contract"],
            source="DSA observer",
        )
        validate_strategy_nd_attention_hlo(
            record["dsa_observer_hlo_contract"],
            source="DSA observer",
        )
    for executable_contract in head_key_contracts:
        live_score = executable_contract["live_tensor_contract"]
        expected_score_precision = (
            "default" if dsa_score_default_precision else "highest"
        )
        expected_highest_count = 0 if dsa_score_default_precision else 1
        if (
            executable_contract["dsa_score_default_precision"]
            != dsa_score_default_precision
            or live_score["score_precision"] != expected_score_precision
            or any(
                body["highest_precision_contraction_count"]
                != expected_highest_count
                for body in live_score["body_records"]
            )
        ):
            raise SystemExit("fleet DSA score-precision HLO drifted")
        head_key = executable_contract[
            "dsa_head_key_association_contract"
        ]
        if (
            not head_key["passed"]
            or head_key["exact_association"]
            != dsa_head_key_exact_association
            or head_key["forbidden_global_shapes"]
            or head_key["key_projection_count"]
            != (21 if dsa_head_key_exact_association else 0)
            or head_key["key_sqrt_count"]
            != (21 if dsa_head_key_exact_association else 0)
            or (
                dsa_head_key_exact_association
                and (
                    head_key["external_wk_parameter_count"] < 5
                    or head_key["normalized_f32_convert_count"] < 21
                )
            )
        ):
            raise SystemExit("fleet DSA head/key association HLO drifted")
    materializer = record["dsa_query_materialization_hlo_contract"]
    state = record["dsa_query_materialization_state"]
    if dsa_query_exact_association:
        if (
            materializer is None
            or not materializer["passed"]
            or materializer["forbidden_operations"]
            or materializer["forbidden_global_shapes"]
            or state is None
            or state["input_alias_count"] != 4
            or state["slot_count"] != 5
            or state["materialized_bytes_per_device"] != 41943040
            or state["source"]
            != "completed_stage_local_raw_fp8_to_fp32"
            or record["dsa_query_materialization_hlo_sha256"] is None
            or record[
                "fleet_dsa_query_materialization_hlo_hashes"
            ] is None
            or len(
                set(
                    record[
                        "fleet_dsa_query_materialization_hlo_hashes"
                    ]
                )
            )
            != 1
            or record[
                "fleet_dsa_query_materialization_hlo_hashes"
            ][0]
            != record["dsa_query_materialization_hlo_sha256"]
        ):
            raise SystemExit("fleet DSA query materialization drifted")
        shards = state["local_shards"]
        local_device_ids = {
            int(value)
            for value in record[
                "fleet_local_device_ids_in_runtime_order"
            ][record["jax_process_index"]]
        }
        if (
            len(shards) != 20
            or {shard["slot"] for shard in shards} != set(range(5))
            or any(shard["byte_count"] != 8 * 1024 * 1024 for shard in shards)
            or {shard["device_id"] for shard in shards} != local_device_ids
            or any(
                sum(value["slot"] == slot for value in shards) != 4
                for slot in range(5)
            )
        ):
            raise SystemExit("fleet DSA query materialized shards drifted")
    elif any(
        record[field] is not None
        for field in (
            "dsa_query_materialization_hlo_contract",
            "dsa_query_materialization_hlo_sha256",
            "dsa_query_materialization_state",
        )
    ):
        raise SystemExit("default decoder materialized DSA query weights")
if {record["attention_projection_backend"] for record in records} != {
    attention_projection_backend
}:
    raise SystemExit("fleet attention projection backend drifted")
if {record["runtime_manifest_sha256"] for record in records} != {runtime_manifest_sha256}:
    raise SystemExit("fleet runtime manifest drifted")
if {record["runtime_layout_hash"] for record in records} != {runtime_layout_hash}:
    raise SystemExit("fleet runtime layout drifted")
if {record["context_capacity"] for record in records} != {context_capacity}:
    raise SystemExit("fleet context capacity drifted")
for record in records:
    if (
        record["body_only"] == complete_token_path
        or record["transformer_body_timing_only"] == complete_token_path
        or record["raw_token_claim"] != short_context_oracle
        or not record["hlo_contract"]["passed"]
        or not record["metadata_passed"]
        or not record["token_passed"]
    ):
        raise SystemExit("step/HLO/metadata claim contract failed")
    if complete_token_path:
        token = record["token_contract"]
        if (
            token is None
            or token["synthetic_initial_state"] == short_context_oracle
            or token["prefill_used"] != short_context_oracle
            or not token["all_active_lanes_equal"]
            or not token["all_in_vocabulary"]
            or len(token["profiler_free_window_tokens"]) != iterations
        ):
            raise SystemExit("complete token-path output contract failed")
        if short_context_oracle:
            raw = token["raw_token_sequence"]
            oracle = token["short_context_oracle"]
            if (
                raw is None
                or not raw["exact_prefix_match"]
                or raw["compared_token_count"] != 1 + warmup + iterations
                or raw["observed_token_ids"] != raw["expected_token_ids"]
                or oracle is None
                or oracle["manifest_sha256"]
                != short_context_oracle_manifest_sha256
                or oracle["prompt_token_count"] != prompt_token_count
                or record["prefill_hlo_contract"] is None
                or not record["prefill_hlo_contract"]["passed"]
                or record["prefill_hlo_contract"]["outer_loop_count"] != 1
                or record["prefill_hlo_contract"]["loop_contract"][
                    "expected_fused_qkv_internal_loop_count"
                ]
                != expected_prefill_fused_qkv_loops
                or record["prefill_hlo_contract"]["loop_contract"][
                    "fused_qkv_internal_loop_count"
                ]
                != expected_prefill_fused_qkv_loops
                or record["prefill_hlo_contract"]["loop_contract"][
                    "expected_prefill_index_repair_loop_count"
                ]
                != expected_prefill_index_repair_loops
                or record["prefill_hlo_contract"]["loop_contract"][
                    "prefill_index_repair_loop_count"
                ]
                != expected_prefill_index_repair_loops
                or record["prefill_hlo_contract"]["loop_contract"][
                    "loop_count"
                ]
                != expected_prefill_loop_count
                or record["prefill_hlo_contract"]["loop_contract"][
                    "unclassified_loops"
                ]
                or record["prefill_compile_seconds"] is None
                or record["prefill_wall_ms"] is None
            ):
                raise SystemExit("real-prompt token/prefill contract failed")
        elif (
            token["raw_token_sequence"] is not None
            or token["short_context_oracle"] is not None
            or record["prefill_hlo_contract"] is not None
            or record["prefill_hlo_sha256"] is not None
            or record["prefill_stablehlo_sha256"] is not None
            or record["fleet_prefill_stablehlo_hashes"] is not None
        ):
            raise SystemExit("synthetic token path contains oracle evidence")
    elif record["token_contract"] is not None:
        raise SystemExit("body-only record unexpectedly contains a token")
if complete_token_path and len(
    {json.dumps(record["token_contract"], sort_keys=True) for record in records}
) != 1:
    raise SystemExit("fleet token contracts disagree")
if any(
    record["warmup"] != warmup
    or record["iterations"] != iterations
    or record["profiler_free_body_wall"]["count"] != iterations
    for record in records
):
    raise SystemExit("decoder timing configuration drifted")
if any(record["hlo_contract"]["violations"] for record in records):
    raise SystemExit("decoder HLO has violations")
if {record["hlo_contract"]["complete_token_path"] for record in records} != {
    complete_token_path
}:
    raise SystemExit("decoder HLO token-path contract drifted")
if {record["hlo_contract"]["split_residual_state"] for record in records} != {
    split_residual_state
}:
    raise SystemExit("decoder HLO split residual-state contract drifted")
expected_residual_components = 2 if split_residual_state else 1
expected_residual_bytes = expected_residual_components * 6144 * 2
expected_extra_bytes = 6144 * 2 if split_residual_state else 0
if any(
    record["residual_transport_components"] != expected_residual_components
    or record["residual_transport_bytes_per_stage"] != expected_residual_bytes
    or record["split_residual_extra_bytes_per_device"] != expected_extra_bytes
    or record["hlo_contract"]["residual_transport_count"] != 8
    or record["hlo_contract"]["residual_transport_dtype"] != "bf16"
    or record["hlo_contract"]["residual_transport_dimensions"]
    != ([2, 1, 6144] if split_residual_state else [1, 6144])
    for record in records
):
    raise SystemExit("decoder residual transport contract drifted")
if any(
    record["hlo_contract"]["backend_contract"] != hlo_backend_contract
    for record in records
):
    raise SystemExit("decoder HLO backend contract drifted")
if short_context_oracle:
    for record in records:
        prefill = record["prefill_hlo_contract"]
        validate_main_rope_hlo(
            prefill["decoder_contract"],
            source="teacher-forced prefill",
        )
        validate_pregathered_attention_hlo(
            prefill["decoder_contract"],
            source="teacher-forced prefill",
        )
        validate_strategy_nd_attention_hlo(
            prefill["decoder_contract"],
            source="teacher-forced prefill",
        )
        repair = prefill["index_repair_contract"]
        prefill_head_key = prefill["decoder_contract"][
            "dsa_head_key_association_contract"
        ]
        if (
            prefill["backend_contract"] != hlo_backend_contract
            or prefill["prompt_length"] != prompt_token_count
            or prefill["index_repair_backend"] != expected_repair_backend
            or prefill["decoder_contract"]["split_residual_state"]
            != split_residual_state
            or prefill["decoder_contract"]["prefill_index_repair"]
            != prefill_index_repair
            or prefill["decoder_contract"][
                "dsa_head_key_exact_association"
            ]
            != dsa_head_key_exact_association
            or prefill["decoder_contract"][
                "dsa_score_default_precision"
            ]
            != dsa_score_default_precision
            or prefill["decoder_contract"]["live_tensor_contract"][
                "score_precision"
            ]
            != (
                "default" if dsa_score_default_precision else "highest"
            )
            or not prefill_head_key["passed"]
            or prefill_head_key["key_projection_count"]
            != (21 if dsa_head_key_exact_association else 0)
            or prefill_head_key["key_sqrt_count"]
            != (21 if dsa_head_key_exact_association else 0)
            or prefill_head_key["forbidden_global_shapes"]
            or prefill["violations"]
            or len(set(record["fleet_prefill_hlo_hashes"])) != 1
            or record["fleet_prefill_hlo_hashes"][0]
            != record["prefill_hlo_sha256"]
            or record["prefill_stablehlo_sha256"] is None
            or len(set(record["fleet_prefill_stablehlo_hashes"])) != 1
            or record["fleet_prefill_stablehlo_hashes"][0]
            != record["prefill_stablehlo_sha256"]
        ):
            raise SystemExit("prefill HLO/fleet contract drifted")
        if prefill_index_repair:
            materializer = record[
                "prefill_wk_materialization_hlo_contract"
            ]
            if materializer is None:
                raise SystemExit(
                    "external prefill wk materialization contract is absent"
                )
            bf16_decoder = materializer["bf16_decode"]
            fp32_promoter = materializer["fp32_promote"]
            materialized_state = record[
                "prefill_wk_materialization_state"
            ]
            expected_materialized_slots = 5
            expected_materialized_shard_bytes = 128 * 6144 * 4
            expected_materialized_bytes = (
                expected_materialized_slots
                * expected_materialized_shard_bytes
            )
            if (
                not repair["passed"]
                or repair["backend"] != "physical_m64_chunk"
                or repair["chunk_count"] != 4
                or repair["full_indexer_layer_count"] != 21
                or repair["expected_call_count"] != 84
                or repair["projection_count"] != 84
                or repair["exact_projection_operand_count"] != 84
                or repair["physical_sqrt_count"] != 168
                or repair["physical_affine_count"] < 84
                or repair["cache_write_count"] < 84
                or repair["grouped_sqrt_count"] != 0
                or repair["history_estimated_bytes_per_device"] != 501043200
                or not repair["history_shapes"]
                or repair["full_pod_history_shapes"]
                or repair["repair_collectives"]
                or repair["forbidden_markers"]
                or repair["materialized_wk_parameter_count"]
                < expected_materialized_slots
                or repair["repair_weight_round_count"] != 0
                or repair["violations"]
            ):
                raise SystemExit("physical-M64 prefill repair HLO drifted")
            if (
                not materializer["passed"]
                or not bf16_decoder["passed"]
                or bf16_decoder["phase"] != "decode_bf16"
                or bf16_decoder["backend"]
                != "external_stage_local_raw_fp8_to_bf16"
                or bf16_decoder["expected_slot_count"]
                != expected_materialized_slots
                or bf16_decoder["raw_parameter_count"]
                != expected_materialized_slots
                or bf16_decoder["scale_parameter_count"]
                != expected_materialized_slots
                or bf16_decoder["bf16_parameter_count"] != 0
                or bf16_decoder["bf16_round_count"]
                < expected_materialized_slots
                or bf16_decoder["fp32_promotion_count"] != 0
                or bf16_decoder["collective_count"] != 0
                or bf16_decoder["host_markers"]
                or bf16_decoder["violations"]
                or not fp32_promoter["passed"]
                or fp32_promoter["phase"] != "promote_fp32"
                or fp32_promoter["backend"]
                != "external_stage_local_bf16_to_fp32"
                or fp32_promoter["expected_slot_count"]
                != expected_materialized_slots
                or fp32_promoter["raw_parameter_count"] != 0
                or fp32_promoter["scale_parameter_count"] != 0
                or fp32_promoter["bf16_parameter_count"]
                != expected_materialized_slots
                or fp32_promoter["bf16_round_count"] != 0
                or fp32_promoter["fp32_promotion_count"]
                < expected_materialized_slots
                or fp32_promoter["collective_count"] != 0
                or fp32_promoter["host_markers"]
                or fp32_promoter["violations"]
                or record[
                    "prefill_wk_materialization_compile_seconds"
                ] is None
                or record[
                    "prefill_wk_materialization_compile_seconds"
                ] <= 0
                or record[
                    "prefill_wk_materialization_execute_seconds"
                ] is None
                or record[
                    "prefill_wk_materialization_execute_seconds"
                ] <= 0
                or not record["prefill_wk_materialization_hlo_sha256"]
                or len(
                    set(
                        record[
                            "fleet_prefill_wk_materialization_hlo_hashes"
                        ]
                    )
                )
                != 1
                or record[
                    "fleet_prefill_wk_materialization_hlo_hashes"
                ][0]
                != record["prefill_wk_materialization_hlo_sha256"]
                or materialized_state is None
                or materialized_state["source"]
                != "completed_stage_local_raw_fp8_to_bf16_to_fp32"
                or materialized_state["materialized_bytes_per_device"]
                != expected_materialized_bytes
            ):
                raise SystemExit(
                    "external prefill wk materialization contract drifted"
                )
            shards = materialized_state["local_shards"]
            if len(shards) != 4 * expected_materialized_slots:
                raise SystemExit(
                    "external prefill wk materialization shard count drifted"
                )
            local_device_ids = {
                int(value)
                for value in record[
                    "fleet_local_device_ids_in_runtime_order"
                ][record["jax_process_index"]]
            }
            if {
                int(shard["device_id"]) for shard in shards
            } != local_device_ids:
                raise SystemExit(
                    "external prefill wk materialization ownership drifted"
                )
            for device_id in local_device_ids:
                device_shards = [
                    shard
                    for shard in shards
                    if int(shard["device_id"]) == device_id
                ]
                if (
                    {int(shard["slot"]) for shard in device_shards}
                    != set(range(expected_materialized_slots))
                    or any(
                        int(shard["byte_count"])
                        != expected_materialized_shard_bytes
                        or len(str(shard["sha256"])) != 64
                        or int(str(shard["sha256"]), 16) < 0
                        for shard in device_shards
                    )
                ):
                    raise SystemExit(
                        "external prefill wk materialization shard identity "
                        "drifted"
                    )
        elif repair != {"backend": "none", "passed": True, "violations": []}:
            raise SystemExit("unrequested prefill repair HLO evidence is present")
        elif any(
            record[field] is not None
            for field in (
                "fleet_prefill_wk_materialization_hlo_hashes",
                "prefill_wk_materialization_compile_seconds",
                "prefill_wk_materialization_execute_seconds",
                "prefill_wk_materialization_hlo_contract",
                "prefill_wk_materialization_hlo_sha256",
                "prefill_wk_materialization_state",
            )
        ):
            raise SystemExit(
                "unrequested prefill wk materialization evidence is present"
            )
if short_context_dsa_oracle:
    if len(
        {
            json.dumps(record["dsa_observer_contract"], sort_keys=True)
            for record in records
        }
    ) != 1:
        raise SystemExit("fleet DSA observer contracts disagree")
    for record in records:
        dsa = record["dsa_observer_contract"]
        observer_hlo = record["dsa_observer_hlo_contract"]
        isolation = record["dsa_observer_isolation_contract"]
        steps = dsa["step_records"] if dsa is not None else []
        if (
            dsa is None
            or not dsa["passed"]
            or not dsa["all_steps_passed"]
            or dsa["decode_step_count"] != decode_step_count
            or dsa["event_count"] != 21
            or dsa["manifest_sha256"]
            != short_context_dsa_oracle_manifest_sha256
            or not dsa["observer_executed_before_production"]
            or not dsa["prefill_state_preserved_without_donation"]
            or dsa["production_executable_observer_enabled"]
            or not dsa["score_comparison"]["compared"]
            or dsa["score_comparison"]["cross_backend_total_order_is_gate"]
            or not dsa["score_comparison"][
                "executing_score_order_and_ties_are_gate"
            ]
            or dsa["score_comparison"][
                "legacy_scores_use_position_aligned_bounded_gate"
            ]
            or not dsa["score_comparison"][
                "legacy_scores_use_position_aligned_diagnostic"
            ]
            or dsa["score_comparison"]["legacy_score_tolerance"]
            != {"max_abs": 0.125, "mean_abs": 0.01, "p99_abs": 0.03125}
            or not dsa["score_comparison"][
                "selected_set_against_legacy_is_exact_gate"
            ]
            or len(dsa["observation_artifacts"]) != decode_step_count
            or dsa["token_observation_candidates"] != 16
            or dsa["prefill_token_sequence"] is None
            or not dsa["prefill_token_sequence"]["exact_prefix_match"]
            or dsa["prefill_token_sequence"]["compared_token_count"] != 1
            or dsa["prefill_token_sequence"]["observed_token_ids"]
            != dsa["prefill_token_sequence"]["expected_token_ids"]
            or dsa["token_oracle_offset"] != 1
            or not dsa["token_sequence"]["exact_prefix_match"]
            or dsa["token_sequence"]["compared_token_count"]
            != decode_step_count
            or dsa["token_sequence"]["observed_token_ids"]
            != dsa["token_sequence"]["expected_token_ids"]
            or [step["decode_position"] for step in steps]
            != list(range(prompt_token_count, decode_end_exclusive))
            or not all(
                step["passed"]
                and step["position_passed"]
                and step["exact_selected_set_and_tail"]
                and step["actual_device_score_order_and_ties"]
                and step["event_count"] == 21
                and not step["lane_mismatch_stages"]
                and not step["padded_slot_mismatches"]
                and not step["producer_mismatches"]
                and not step["count_mismatches"]
                and not step["selected_set_mismatches"]
                and not step["tail_mismatches"]
                and not step["score_contract_mismatches"]
                and step["legacy_score_bounded_comparison"][
                    "coverage_complete"
                ]
                and len(step["observation_sha256"]) == 64
                and len(step["token_observation_sha256"]) == 64
                and step["token_observation"]["passed"]
                and step["token_observation"]["candidate_width"] == 16
                and len(step["token_observation"]["candidate_ids"]) == 16
                and len(step["token_observation"]["candidate_scores"]) == 16
                and step["token_observation"]["lane_replication"]
                and step["token_observation"][
                    "inactive_rows_are_sentinel"
                ]
                and step["token_observation"]["ids_valid"]
                and step["token_observation"]["scores_finite"]
                and step["token_observation"]["order_and_ties_valid"]
                and step["token_observation"]["winner_matches_output"]
                for step in steps
            )
            or observer_hlo is None
            or not observer_hlo["passed"]
            or observer_hlo["backend_contract"] != hlo_backend_contract
            or observer_hlo["token_observation_candidates"] != 16
            or observer_hlo["split_residual_state"]
            != split_residual_state
            or observer_hlo["collective_counts"]
            != record["hlo_contract"]["collective_counts"]
            or isolation is None
            or not isolation["passed"]
            or isolation["donate_argnums"]
            or isolation["input_output_alias_present"]
            or isolation["callback_markers"]
            or not isolation["collective_contract_matches_production"]
            or not isolation["non_token_result_shapes_match"]
            or not isolation["token_exchange_shape_difference_allowed"]
            or record["dsa_observer_compile_seconds"] is None
            or record["dsa_observer_hlo_sha256"] is None
            or len(set(record["fleet_dsa_observer_hlo_hashes"])) != 1
            or record["fleet_dsa_observer_hlo_hashes"][0]
            != record["dsa_observer_hlo_sha256"]
            or record["dsa_observer_stablehlo_sha256"] is None
            or len(
                set(record["fleet_dsa_observer_stablehlo_hashes"])
            )
            != 1
            or record["fleet_dsa_observer_stablehlo_hashes"][0]
            != record["dsa_observer_stablehlo_sha256"]
            or not record[
                "trace_and_timing_use_observer_free_production_executable"
            ]
        ):
            raise SystemExit("exact DSA observer/Gate D contract failed")
    artifact_records = records[0]["dsa_observer_contract"][
        "observation_artifacts"
    ]
    artifact_names = [record["filename"] for record in artifact_records]
    artifact_paths = sorted((run_dir / "dsa_observer").glob("*.npz"))
    if [path.name for path in artifact_paths] != artifact_names:
        raise SystemExit("DSA observer artifact inventory drifted")
    for record, path in zip(artifact_records, artifact_paths, strict=True):
        with np.load(path) as bundle:
            if set(bundle.files) != {
                "decode_position",
                "observation",
                "token_observation",
            }:
                raise SystemExit("DSA observer artifact fields drifted")
            decode_position = np.asarray(bundle["decode_position"])
            observation = np.asarray(bundle["observation"])
            token_observation = np.asarray(bundle["token_observation"])
        digest = hashlib.sha256(
            np.ascontiguousarray(observation).tobytes()
        ).hexdigest()
        token_digest = hashlib.sha256(
            np.ascontiguousarray(token_observation).tobytes()
        ).hexdigest()
        if (
            decode_position.shape != (1,)
            or int(decode_position[0])
            not in range(prompt_token_count, decode_end_exclusive)
            or observation.shape != (32, 5, 4098)
            or observation.dtype != np.dtype(np.int32)
            or digest != record["observation_sha256"]
            or token_observation.shape != (32, 32)
            or token_observation.dtype != np.dtype(np.int32)
            or token_digest != record["token_observation_sha256"]
        ):
            raise SystemExit("DSA observer artifact tensor/hash drifted")
else:
    for record in records:
        if any(
            record[name] is not None
            for name in (
                "dsa_observer_compile_seconds",
                "dsa_observer_contract",
                "dsa_observer_hlo_contract",
                "dsa_observer_hlo_sha256",
                "dsa_observer_stablehlo_sha256",
                "dsa_observer_isolation_contract",
                "fleet_dsa_observer_hlo_hashes",
                "fleet_dsa_observer_stablehlo_hashes",
            )
        ):
            raise SystemExit("unrequested DSA observer evidence is present")
if runtime_kind in ("pallas_feature", "pallas_feature_linear"):
    selected_kernel = "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512"
    if feature_output_tile != 128:
        selected_kernel += f"_ot{feature_output_tile}"
    if feature_reconstruct_down_fp32:
        selected_kernel += "_downf32"
    if feature_fuse_route_weighting:
        selected_kernel += "_wsum"
    expected_kernel_counts = {
        selected_kernel: 75,
        "greenfield_fp8_block_up_gate_m8_k6144_n512": 75,
        "greenfield_fp8_block_matmul_m8_k512_n6144": 75,
    }
    if feature_reconstruct_down_fp32:
        expected_kernel_counts["greenfield_fp32_to_bf16_r8_h6144"] = 75
    for record in records:
        feature = record["hlo_contract"]["pallas_feature_contract"]
        if (
            not feature["passed"]
            or feature["feature_output_tile"] != feature_output_tile
            or feature["fuse_route_weighting"]
            != feature_fuse_route_weighting
            or feature["reconstruct_down_fp32"]
            != feature_reconstruct_down_fp32
            or feature["kernel_counts"] != expected_kernel_counts
            or feature["expected_kernel_counts"] != expected_kernel_counts
            or feature["forbidden_decoded_expert_overlays"]
        ):
            raise SystemExit("feature-Pallas HLO kernel/overlay contract drifted")
for record in records:
    association = record["hlo_contract"]["dsa_query_association_contract"]
    if (
        not association["passed"]
        or association["backend"] != "reference"
        or association["local_owner_shape"] != "f32[1024,2048]"
        or association["local_owner_shape_occurrences"] < 21
        or association["forbidden_global_shapes"]
    ):
        raise SystemExit("local FP32 DSA query-owner contract drifted")
    fused_qkv = record["hlo_contract"]["fused_qkv_a_contract"]
    if attention_projection_backend == "fused_n82_convolution":
        if (
            fused_qkv["violations"]
            or fused_qkv["convolution_count"] != 78
            or fused_qkv["expected_convolution_count"] != 78
            or not all(fused_qkv["required_shapes"].values())
            or fused_qkv["forbidden_shapes"]
        ):
            raise SystemExit("fused qkv-a HLO contract drifted")
    elif fused_qkv:
        raise SystemExit("separate attention unexpectedly has a fused HLO contract")
if runtime_kind == "pallas_feature_linear":
    separate_qkv_a_calls = (
        78 if attention_projection_backend == "separate" else 0
    )
    expected_linear_kernel_counts = {
        "greenfield_fp8_block_matmul_m8_k6144_n2048": separate_qkv_a_calls,
        "greenfield_fp8_block_matmul_m8_k2048_n4096": 78,
        "greenfield_fp8_block_matmul_m8_k6144_n640": separate_qkv_a_calls,
        "greenfield_fp8_block_matmul_m8_k4096_n6144": (
            0 if strategy_nd_attention_projection else 78
        ),
        "greenfield_fp8_block_matmul_m8_k512_n6144": (
            624 if strategy_nd_attention_projection else 0
        ),
        "greenfield_fp8_structured_kv_b_q_absorb_h16_p192_l512": 78,
        "greenfield_fp8_structured_kv_b_value_h16_l512_v256": 78,
        "greenfield_fp8_block_matmul_f32_m8_k2048_n1024": 0,
        "greenfield_fp8_block_matmul_f32_m8_k6144_n128": 21,
        "greenfield_fp8_fused_block_swiglu_m8_h6144_i3072_o6144": 3,
    }
    for record in records:
        linear = record["hlo_contract"]["pallas_stage_linear_contract"]
        if (
            not linear["passed"]
            or linear["dsa_query_backend"] != "reference"
            or linear["attention_projection_backend"]
            != attention_projection_backend
            or linear["kernel_counts"] != expected_linear_kernel_counts
            or linear["expected_kernel_counts"]
            != expected_linear_kernel_counts
            or linear["forbidden_decoded_weight_overlays"]
        ):
            raise SystemExit("stage-linear Pallas HLO contract drifted")
if any(record["load_record"]["runtime_checkpoint_reshards"] != 0 for record in records):
    raise SystemExit("runtime loader performed a checkpoint reshard")
if any(record["load_record"]["host_global_concatenations"] != 0 for record in records):
    raise SystemExit("runtime loader performed a host global concat")
if any(record["load_record"]["fp8_host_dequantizations"] != 0 for record in records):
    raise SystemExit("runtime loader performed a host FP8 dequantization")
if any(record["load_record"]["fp8_device_dequantizations"] != 0 for record in records):
    raise SystemExit("runtime loader performed a device FP8 dequantization")
if any(
    record["load_record"]["loaded_payload_bytes"]
    != expected_loaded_payload_bytes
    for record in records
):
    raise SystemExit("runtime loader payload bytes per host drifted")
if any(
    record["load_record"]["device_roundtrip_verified"]
    != verify_device_roundtrip
    for record in records
):
    raise SystemExit("runtime loader device-roundtrip flag drifted")
expected_roundtrip_bytes = (
    expected_loaded_payload_bytes if verify_device_roundtrip else 0
)
if any(
    record["load_record"]["device_roundtrip_bytes"]
    != expected_roundtrip_bytes
    for record in records
):
    raise SystemExit("runtime loader device-roundtrip bytes drifted")
xplane = None
if trace_steps:
    traces = sorted((run_dir / "traces").glob("trace.rank*.xplane.pb"))
    if len(traces) != 8:
        raise SystemExit(f"expected eight decoder XPlanes, got {len(traces)}")
    for record, path in zip(records, traces, strict=True):
        trace = record["trace"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            trace is None
            or trace["steps"] != trace_steps
            or not trace["profiler_started_after_profiler_free_timing"]
            or len(trace["files"]) != 1
            or trace["files"][0]["sha256"] != digest
            or trace["files"][0]["size_bytes"] != path.stat().st_size
        ):
            raise SystemExit("decoder trace record drifted")
    sys.path.insert(0, str(Path(repo) / "scripts" / "analysis"))
    import parse_xplane

    xplane = parse_xplane.aggregate_fleet(
        run_dir / "traces",
        step_module_re=r"jit_mapped",
    )
    if (
        xplane["n_files"] != 8
        or xplane["n_cores"] != 64
        or xplane["steps_per_core"] != trace_steps
    ):
        raise SystemExit("decoder fleet XPlane inventory drifted")
else:
    if any(record["trace"] is not None for record in records):
        raise SystemExit("unrequested decoder trace was captured")

def peak(record):
    values = []
    for stats in record["device_memory_after_execute"]:
        if stats is not None:
            for key in ("peak_bytes_in_use", "peak_bytes", "bytes_in_use"):
                if key in stats:
                    values.append(stats[key])
                    break
    return max(values) if values else None

def hbm_headrooms(record):
    values = []
    for stats in record["device_memory_after_execute"]:
        if stats is None or "bytes_limit" not in stats:
            continue
        peak_value = next(
            (
                stats[key]
                for key in ("peak_bytes_in_use", "peak_bytes", "bytes_in_use")
                if key in stats
            ),
            None,
        )
        if peak_value is not None:
            values.append(stats["bytes_limit"] - peak_value)
    return values

fleet_p50 = max(record["profiler_free_body_wall"]["p50_ms"] for record in records)
fleet_p99 = max(record["profiler_free_body_wall"]["p99_ms"] for record in records)
peaks = [value for value in map(peak, records) if value is not None]
headrooms = [
    value
    for record in records
    for value in hbm_headrooms(record)
]
if (prefill_index_repair or dsa_query_exact_association or main_rope_table) and (
    len(headrooms) != 32 or min(headrooms) <= 0
):
    raise SystemExit("materialized exactness state lacks positive HBM headroom")
summary = {
    "artifact_kind": (
        f"greenfield_real_78layer_{context_label}_decoder_"
        + (
            "gate_d_fleet"
            if short_context_dsa_oracle
            else (
                "token_oracle_fleet"
                if short_context_oracle
                else (
                    "token_mechanism_fleet"
                    if complete_token_path
                    else "body_fleet"
                )
            )
        )
    ),
    "answer_tokens_per_second": (
        1000.0 / fleet_p50 if short_context_dsa_oracle else None
    ),
    "body_only": not complete_token_path,
    "code_hash": pin,
    "compile_seconds_max": max(record["compile_seconds"] for record in records),
    "context_capacity": context_capacity,
    "fleet_p50_body_ms": fleet_p50,
    "fleet_p99_body_ms": fleet_p99,
    "fleet_p50_complete_step_ms": (
        fleet_p50 if complete_token_path else None
    ),
    "fleet_p99_complete_step_ms": (
        fleet_p99 if complete_token_path else None
    ),
    "feature_output_tile": feature_output_tile,
    "feature_fuse_route_weighting": feature_fuse_route_weighting,
    "feature_reconstruct_down_fp32": feature_reconstruct_down_fp32,
    "split_residual_state": split_residual_state,
    "prefill_index_repair": prefill_index_repair,
    "dsa_query_exact_association": dsa_query_exact_association,
    "dsa_head_key_exact_association": dsa_head_key_exact_association,
    "dsa_score_default_precision": dsa_score_default_precision,
    "main_rope_table_enabled": main_rope_table,
    "pregathered_b512_attention": pregathered_b512_attention,
    "strategy_nd_attention_projection": strategy_nd_attention_projection,
    "main_rope_table_sha256": records[0]["main_rope_table_sha256"],
    "main_rope_table_shape": records[0]["main_rope_table_shape"],
    "main_rope_table_bytes_per_device": records[0][
        "main_rope_table_bytes_per_device"
    ],
    "main_rope_table_prerequisite": records[0][
        "main_rope_table_prerequisite"
    ],
    "pregathered_b512_attention_prerequisite": (
        {
            "accepted_output_sha256": (
                "923e9bfeb65864868cef359cf98ebad67f2756794ad142ac87e66b71877a2d2a"
            ),
            "code_hash": "83577222b4d48cb4b2fec544c06e3a71fbc821f5",
            "exact_arm": "pregathered_h16_b512",
            "results_db_item_row_id": 1821,
            "results_db_run_id": 537,
            "runner_sha256": (
                "7961622c297567a021edf2ab335f0d046db6665e8130e210bbd78919a2d7a4ec"
            ),
            "success_sha256": (
                "ecc2b873c29ffd9bc6551136e90352b33523f50873827b00ae91279a4db3153c"
            ),
        }
        if pregathered_b512_attention
        else None
    ),
    "strategy_nd_attention_projection_prerequisite": (
        {
            "accepted_output_sha256": (
                "68afed86921584fb673abb11a563e359a1533210ec2483e71ee79b88c2b0bde7"
            ),
            "classification": "strategy_nd_attention_projection_exact",
            "code_hash": "a4a17ac",
            "exact_candidate": "strategy_nd",
            "results_db_item_row_id": 1823,
            "results_db_run_id": 539,
            "success_sha256": (
                "f52d3e88b060c19059392dad69ba959ce715ba489c44f3565b93c1215969f508"
            ),
        }
        if strategy_nd_attention_projection
        else None
    ),
    "dsa_score_default_precision_prerequisite": (
        {
            "code_hash": "0cd3db063852630c6309f1966020ba3085aa9877",
            "default_logical_score_sha256": "529fadbaeb6646ef391fde902971c3a3576c56ae8da10f2e26adc2eba7baa023",
            "default_optimized_hlo_sha256": "5e166b87e3714e576e3c5a9c56098b370722b447ad38004bb4b022a7f03375ef",
            "default_selected_score_sha256": "0e715f8b46464702902a7cc83425cdd6abd2f8df15210a7bd70e82545d7b1318",
            "results_db_item_row_id": 1814,
            "results_db_run_id": 529,
            "runner_sha256": "d2e35d1c54989a015da7fd683661cb4d182f26baf91fb5c8f6a0d5d0ff5d3ab3",
            "summary_sha256": "01de9dff6595180c968a7b9a1fac7e8fe1b4773ea79a2b71fb7e0d516348d7c0",
        }
        if dsa_score_default_precision
        else None
    ),
    "dsa_head_key_exact_association_prerequisite": (
        {
            "code_hash": "cd15aafb2fddcc1c6c2fcd32981c8bffd7d4b569",
            "current_key_sha256": "9f1fb991de6ba3adbb4308ff8bd6e6c2afc03ecd9f6f54e6130be5f51f2dbbc5",
            "exact_candidate": "physical_normalized_barrier_materialized_divide_sqrt",
            "head_weights_sha256": "ec66b475563836340f570df247da3414272321984fd50522ca4b4b001ae1725e",
            "optimized_hlo_sha256": "4dc288236a558fcdaccb33141c2d761aebd0eb90d1d1591b65e0b77c969303a1",
            "results_db_item_row_id": 1812,
            "results_db_run_id": 527,
            "stablehlo_sha256": "d9fd33b4b6775f2bec296a3e8601ca3ba8dc5660c4120604d02892c1d811dbcb",
            "success_sha256": "0c961dd13d00d4db8f56fe2efac3fbb4a203b4f0fc6265e2bb7cc0842c015fb0",
        }
        if dsa_head_key_exact_association
        else None
    ),
    "dsa_query_exact_association_prerequisite": (
        {
            "code_hash": "a749ff02e49d9a27d5a9ad36fec6597ffe0777a4",
            "exact_candidate": "physical_owner_tuple4_barrier_m1_n1024",
            "optimized_hlo_sha256": "40d9ef25681edc50e6611229c8a3ace2cc4bac6e4cd2459d14b12d44b5c74b7d",
            "results_db_item_row_id": 1810,
            "results_db_run_id": 525,
            "success_sha256": "5b547f24a360b89411f7773800a5ac5bef56d8cc953e4d141dbf2d498aa6d582",
        }
        if dsa_query_exact_association
        else None
    ),
    "dsa_query_exact_association_production_prerequisite": (
        {
            "code_hash": "3aa9f9ca5fedc9e5b6b67aca00b9be10ff5f0dc1",
            "exact_candidate": "physical_production_fused_q_a_tuple4_exact_m1_n1024",
            "optimized_hlo_sha256": "78c296746c1ee5bef3c2183256671fa5b2cd5dffd1d884217ee50b2fcf53069c",
            "results_db_item_row_id": 1811,
            "results_db_run_id": 526,
            "success_sha256": "b3cff36beaff7eaaf71c30349007c8c5acb8c7423ae46fb82c40339061a23b11",
        }
        if dsa_query_exact_association
        else None
    ),
    "prefill_index_repair_backend": expected_repair_backend,
    "prefill_index_repair_prerequisite": (
        {
            "code_hash": "86243115452920fe4244bb77a9bbf4c44110aeab",
            "comparison_manifest_sha256": "1d80d088561181a63e734a91cd0124c4011cfc4151198488c052740050d66fe5",
            "results_db_item_row_id": 1803,
            "results_db_run_id": 518,
            "success_sha256": "a8d370166257622875feafd4d1da3f8d666204a8609baaffef2573b659f6bfee",
        }
        if prefill_index_repair
        else None
    ),
    "prefill_index_weight_materialization_prerequisite": (
        {
            "candidate_cache_sha256": "8fd4a8c27602c2f855645c52f2687c62c7b47a47dc9715452f16ca2a1dd5df08",
            "code_hash": "5e1cbb5e7814eb49a76c0d6446871b6e51b9ea19",
            "comparison_manifest_sha256": "b96697994106a2a0058a800d2c4de9ec4b45e382c181541041fcead548848d42",
            "results_db_item_row_id": 1804,
            "results_db_run_id": 519,
        }
        if prefill_index_repair
        else None
    ),
    "prefill_index_weight_split_prerequisite": (
        {
            "candidate_cache_sha256": "3808d502f3ea1829bf12ab7585d66f15dd83bf640657a17c35daabf5ab1859d1",
            "code_hash": "097702266d025f2887418e06898eae781615972c",
            "comparison_manifest_sha256": "1e94255557c475dac4335a0120f3448e8754135c44dde747a888cb3c2de08a59",
            "results_db_item_row_id": 1805,
            "results_db_run_id": 520,
            "success_sha256": "643f80eb8699e18714213bb828a72df8a9f89ba2db1e798ad598dd914a2083ca",
        }
        if prefill_index_repair
        else None
    ),
    "residual_transport_components": records[0][
        "residual_transport_components"
    ],
    "residual_transport_bytes_per_stage": records[0][
        "residual_transport_bytes_per_stage"
    ],
    "split_residual_extra_bytes_per_device": records[0][
        "split_residual_extra_bytes_per_device"
    ],
    "complete_token_path": complete_token_path,
    "dsa_observer_compile_seconds_max": (
        max(record["dsa_observer_compile_seconds"] for record in records)
        if short_context_dsa_oracle
        else None
    ),
    "dsa_observer_contract": records[0]["dsa_observer_contract"],
    "dsa_observer_hlo_contract": records[0]["dsa_observer_hlo_contract"],
    "dsa_observer_hlo_sha256": records[0]["dsa_observer_hlo_sha256"],
    "dsa_observer_isolation_contract": records[0][
        "dsa_observer_isolation_contract"
    ],
    "dsa_query_materialization_compile_seconds_max": (
        max(
            record["dsa_query_materialization_compile_seconds"]
            for record in records
        )
        if dsa_query_exact_association
        else None
    ),
    "dsa_query_materialization_execute_seconds_max": (
        max(
            record["dsa_query_materialization_execute_seconds"]
            for record in records
        )
        if dsa_query_exact_association
        else None
    ),
    "dsa_query_materialization_hlo_contract": records[0][
        "dsa_query_materialization_hlo_contract"
    ],
    "dsa_query_materialization_hlo_sha256": records[0][
        "dsa_query_materialization_hlo_sha256"
    ],
    "gate_d_passed": short_context_dsa_oracle,
    "hlo_contract": records[0]["hlo_contract"],
    "host_count": 8,
    "iterations": iterations,
    "maximum_peak_hbm_bytes": max(peaks) if peaks else None,
    "minimum_measured_hbm_headroom_bytes": (
        min(headrooms) if headrooms else None
    ),
    "optimized_hlo_sha256": records[0]["optimized_hlo_sha256"],
    "plan_hash": records[0]["plan_hash"],
    "prefill_compile_seconds_max": (
        max(record["prefill_compile_seconds"] for record in records)
        if short_context_oracle
        else None
    ),
    "prefill_hlo_contract": records[0]["prefill_hlo_contract"],
    "prefill_hlo_sha256": records[0]["prefill_hlo_sha256"],
    "prefill_wk_materialization_compile_seconds_max": (
        max(
            record["prefill_wk_materialization_compile_seconds"]
            for record in records
        )
        if prefill_index_repair
        else None
    ),
    "prefill_wk_materialization_execute_seconds_max": (
        max(
            record["prefill_wk_materialization_execute_seconds"]
            for record in records
        )
        if prefill_index_repair
        else None
    ),
    "prefill_wk_materialization_hlo_contract": records[0][
        "prefill_wk_materialization_hlo_contract"
    ],
    "prefill_wk_materialization_hlo_sha256": records[0][
        "prefill_wk_materialization_hlo_sha256"
    ],
    "prefill_wk_materialization_states": (
        [record["prefill_wk_materialization_state"] for record in records]
        if prefill_index_repair
        else None
    ),
    "prefill_used": short_context_oracle,
    "prefill_wall_ms_max": (
        max(record["prefill_wall_ms"] for record in records)
        if short_context_oracle
        else None
    ),
    "raw_token_claim": short_context_oracle,
    "runtime_kind": runtime_kind,
    "runtime_device_roundtrip_verified": verify_device_roundtrip,
    "runtime_layout_hash": records[0]["runtime_layout_hash"],
    "runtime_manifest_sha256": records[0]["runtime_manifest_sha256"],
    "schedule_hash": records[0]["schedule_hash"],
    "state_layout_hash": records[0]["state_layout_hash"],
    "sparse_moe_backend": sparse_moe_backend,
    "topology_hash": records[0]["topology_hash"],
    "trace_steps": trace_steps,
    "short_context_oracle_manifest_sha256": (
        short_context_oracle_manifest_sha256
        if short_context_oracle
        else None
    ),
    "short_context_dsa_oracle_manifest_sha256": (
        short_context_dsa_oracle_manifest_sha256
        if short_context_dsa_oracle
        else None
    ),
    "synthetic_initial_state": (
        complete_token_path and not short_context_oracle
    ),
    "token_contract": records[0]["token_contract"],
    "transformer_body_timing_only": not complete_token_path,
    "warmup": warmup,
    "xplane": xplane,
}
(run_dir / "xplane_summary.json").write_text(
    json.dumps(xplane, indent=2, sort_keys=True) + "\n"
)
sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

conn = pv.connect(db_path)
scope = (
    "gate-d"
    if short_context_dsa_oracle
    else (
        "token-oracle"
        if short_context_oracle
        else ("token-mechanism" if complete_token_path else "body")
    )
)
run_id = pv.start_run(
    conn,
    model=(
        f"zai-org/GLM-5.2-FP8:greenfield-78layer-{context_label}-{scope}"
    ),
    revision=records[0]["runtime_manifest_sha256"],
    env={
        "GLM_ENGINE": f"greenfield_pp8_decoder_{scope}",
        "greenfield_complete_token_path": complete_token_path,
        "greenfield_split_residual_state": split_residual_state,
        "greenfield_prefill_index_repair": prefill_index_repair,
        "greenfield_dsa_query_exact_association": (
            dsa_query_exact_association
        ),
        "greenfield_dsa_query_exact_association_prerequisite_db_run": (
            525 if dsa_query_exact_association else None
        ),
        "greenfield_dsa_query_exact_association_production_prerequisite_db_run": (
            526 if dsa_query_exact_association else None
        ),
        "greenfield_dsa_head_key_exact_association": (
            dsa_head_key_exact_association
        ),
        "greenfield_dsa_head_key_exact_association_prerequisite_db_run": (
            527 if dsa_head_key_exact_association else None
        ),
        "greenfield_dsa_score_default_precision": (
            dsa_score_default_precision
        ),
        "greenfield_dsa_score_default_precision_prerequisite_db_run": (
            529 if dsa_score_default_precision else None
        ),
        "greenfield_main_rope_table": main_rope_table,
        "greenfield_main_rope_table_sha256": (
            records[0]["main_rope_table_sha256"]
        ),
        "greenfield_main_rope_table_prerequisite_db_run": (
            531 if main_rope_table else None
        ),
        "greenfield_pregathered_b512_attention": (
            pregathered_b512_attention
        ),
        "greenfield_pregathered_b512_attention_prerequisite_db_run": (
            537 if pregathered_b512_attention else None
        ),
        "greenfield_strategy_nd_attention_projection": (
            strategy_nd_attention_projection
        ),
        "greenfield_strategy_nd_attention_projection_prerequisite_db_run": (
            539 if strategy_nd_attention_projection else None
        ),
        "greenfield_prefill_index_repair_backend": expected_repair_backend,
        "greenfield_prefill_index_repair_prerequisite_db_run": (
            518 if prefill_index_repair else None
        ),
        "greenfield_prefill_index_weight_materialization_prerequisite_db_run": (
            519 if prefill_index_repair else None
        ),
        "greenfield_short_context_oracle": short_context_oracle,
        "greenfield_short_context_oracle_manifest_sha256": (
            short_context_oracle_manifest_sha256
            if short_context_oracle
            else None
        ),
        "greenfield_short_context_dsa_oracle": short_context_dsa_oracle,
        "greenfield_short_context_dsa_oracle_manifest_sha256": (
            short_context_dsa_oracle_manifest_sha256
            if short_context_dsa_oracle
            else None
        ),
        "greenfield_runtime_kind": runtime_kind,
        "greenfield_runtime_device_roundtrip_verified": (
            verify_device_roundtrip
        ),
        "greenfield_feature_output_tile": feature_output_tile,
        "greenfield_feature_fuse_route_weighting": (
            feature_fuse_route_weighting
        ),
        "greenfield_feature_reconstruct_down_fp32": (
            feature_reconstruct_down_fp32
        ),
        "greenfield_sparse_moe_backend": sparse_moe_backend,
        "greenfield_code_hash": pin,
        "greenfield_run_tag": greenfield_run_tag,
        "legacy_oracle_code_hash": oracle_pin,
        "hlo_sha256": records[0]["optimized_hlo_sha256"],
        "dsa_observer_hlo_sha256": records[0][
            "dsa_observer_hlo_sha256"
        ],
        "plan_hash": records[0]["plan_hash"],
        "runtime_layout_hash": records[0]["runtime_layout_hash"],
        "runtime_manifest_sha256": records[0]["runtime_manifest_sha256"],
        "state_layout_hash": records[0]["state_layout_hash"],
    },
    note=(
        f"Protected real 78-layer {context_name} transformer-body compile/run with "
        f"feature output tile {feature_output_tile} and fused route weighting "
        f"{feature_fuse_route_weighting}, FP32 routed-down reconstruction "
        f"{feature_reconstruct_down_fp32}; "
        f"split residual state {split_residual_state}; "
        f"prefill physical-M64 index repair {prefill_index_repair}; "
        f"exact tuple4 DSA query association {dsa_query_exact_association}; "
        f"exact recurrent DSA head/key association "
        f"{dsa_head_key_exact_association}; "
        f"exact default-precision DSA scorer "
        f"{dsa_score_default_precision}; "
        f"device-resident main-RoPE table {main_rope_table}; "
        f"selected-cache LP4 exchange plus block-512 attention "
        f"{pregathered_b512_attention}; "
        f"exact StrategyND attention projection "
        f"{strategy_nd_attention_projection}; "
        f"runtime device round-trip verification {verify_device_roundtrip}; "
        + (
            f"real {prompt_token_count:,}-token prompt with exact raw tokens and "
            f"exact all-event DSA observer evidence; protected {context_name} "
            "Gate D."
            if short_context_dsa_oracle
            else (
                f"real {prompt_token_count:,}-token prompt with an exact sealed "
                "raw-token prefix; "
                "not Gate D until full DSA event-order evidence passes."
                if short_context_oracle
                else (
                    "complete token mechanism from synthetic state, no correctness "
                    "or tok/s claim."
                    if complete_token_path
                    else "no token or tok/s claim."
                )
            )
        )
    ),
    harness_repo=repo,
    fork_repo=oracle_repo,
)
(run_dir / "provisional_db_run_id.txt").write_text(f"{run_id}\n")
pv.record_item(
    conn,
    run_id,
    benchmark=(
        f"greenfield_78layer_{context_label}_{scope}_pp8"
        + (f"_ot{feature_output_tile}" if feature_output_tile != 128 else "")
        + ("_downf32" if feature_reconstruct_down_fp32 else "")
        + ("_wsum" if feature_fuse_route_weighting else "")
        + ("_splitres" if split_residual_state else "")
        + ("_prefill_keyfix" if prefill_index_repair else "")
        + ("_queryexact" if dsa_query_exact_association else "")
        + ("_headkeyexact" if dsa_head_key_exact_association else "")
        + ("_scoredefault" if dsa_score_default_precision else "")
        + ("_mainrope" if main_rope_table else "")
        + ("_pregatheredb512" if pregathered_b512_attention else "")
        + ("_strategynd_o" if strategy_nd_attention_projection else "")
        + ("_roundtrip" if verify_device_roundtrip else "")
    ),
    item_id=(
        "gate_d_exact_token_and_dsa"
        if short_context_dsa_oracle
        else (
            "raw_token_prefix"
            if short_context_oracle
            else ("token_step_mechanism" if complete_token_path else "body_step")
        )
    ),
    prompt=(
        f"Execute the sealed {prompt_token_count:,}-token short-context prompt "
        "through device-resident prefill and recurrent PP8 decode."
        if short_context_oracle
        else (
            "Execute the real 78-layer PP8 decoder step with "
            f"feature output tile {feature_output_tile} and fused route weighting "
            f"{feature_fuse_route_weighting}, FP32 routed-down reconstruction "
            f"{feature_reconstruct_down_fp32}."
        )
    ),
    gold=(
        "Exact tokens and executing-device DSA sets/ties; legacy scores retained diagnostically."
        if short_context_dsa_oracle
        else (
            "Exact raw token IDs against the sealed accepted legacy prefix."
            if short_context_oracle
            else (
                "Local-only HLO, exact pipeline state, direct runtime load, and "
                "bounded token mechanism without a raw-correctness claim."
            )
        )
    ),
    raw_output=json.dumps(summary, sort_keys=True),
    extracted=str(summary["hlo_contract"]["collective_counts"]),
    correct=True,
    score=1.0,
    latency_ms=fleet_p50,
)
pv.finalize(
    conn,
    run_id,
    benchmark=(
        f"greenfield_78layer_{context_label}_{scope}_pp8"
        + (f"_ot{feature_output_tile}" if feature_output_tile != 128 else "")
        + ("_downf32" if feature_reconstruct_down_fp32 else "")
        + ("_wsum" if feature_fuse_route_weighting else "")
        + ("_splitres" if split_residual_state else "")
        + ("_prefill_keyfix" if prefill_index_repair else "")
        + ("_queryexact" if dsa_query_exact_association else "")
        + ("_headkeyexact" if dsa_head_key_exact_association else "")
        + ("_scoredefault" if dsa_score_default_precision else "")
        + ("_mainrope" if main_rope_table else "")
        + ("_pregatheredb512" if pregathered_b512_attention else "")
        + ("_strategynd_o" if strategy_nd_attention_projection else "")
        + ("_roundtrip" if verify_device_roundtrip else "")
    ),
    metric="contract_valid",
    value=1.0,
    note=(
        f"Protected real-prompt {context_name} Gate D passes exact raw tokens, "
        "all-event DSA, "
        "state/cache, observer isolation, wall/HBM, fresh XPlane, and cleanup."
        if short_context_dsa_oracle
        else (
            "Exact raw-token prefix and real-prompt recurrent wall pass; full Gate D "
            "still requires exact all-event DSA evidence."
            if short_context_oracle
            else (
                "Synthetic-state complete token timing is not raw-token correctness "
                "or answer tok/s."
                if complete_token_path
                else "Profiler-free body timing is not complete decode latency or tok/s."
            )
        )
    ),
)
conn.close()
summary["results_db_run_id"] = run_id
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
source = sqlite3.connect(db_path)
snapshot = sqlite3.connect(run_dir / "results_ckpt.db")
source.backup(snapshot)
snapshot.close()
source.close()
if sqlite3.connect(run_dir / "results_ckpt.db").execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("results DB snapshot integrity failed")
print(f"SHORT_DECODER_STEP_VALID db_run={run_id} p50_ms={fleet_p50:.6f}")
PY

say "sealing and archiving protected diagnostic evidence"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find host_records host_logs hlo traces dsa_observer \
    layer_residual_observer dsa_internal_observer \
    layer0_residual_discriminator layer0_ingredients -type f -print0 | \
    sort -z | xargs -0 sha256sum
  sha256sum summary.json xplane_summary.json results_ckpt.db census_pre.txt census_post.txt \
    sync.txt execute.txt orchestrator.sealed.log remote_prefix_preflight.txt \
    provisional_db_run_id.txt
) >"$RUN_DIR/evidence.sha256"
DB_RUN=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["results_db_run_id"])' \
  "$RUN_DIR/summary.json")
gcloud storage cp --recursive --no-clobber "$RUN_DIR"/* \
  "$REMOTE_PREFIX/" >/dev/null

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" \
  >"$RUN_DIR/remote_objects.json" <<'PY'
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import google_crc32c

root = Path(sys.argv[1])
prefix = sys.argv[2]
paths = [
    (path, path.relative_to(root).as_posix())
    for path in sorted(root.rglob("*"))
    if path.is_file() and path.name not in {"SUCCESS", "remote_objects.json"}
]


def local_crc32c(path):
    checksum = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode()


def describe(item):
    path, relative = item
    value = json.loads(
        subprocess.run(
            [
                "gcloud",
                "storage",
                "objects",
                "describe",
                f"{prefix}/{relative}",
                "--format=json",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    crc32c = value.get("crc32c_hash") or value.get("crc32c")
    if int(value["size"]) != path.stat().st_size or crc32c != local_crc32c(path):
        raise SystemExit(f"remote object verification failed: {relative}")
    return {
        "crc32c": crc32c,
        "generation": value["generation"],
        "path": relative,
        "size": int(value["size"]),
    }


with ThreadPoolExecutor(max_workers=16) as executor:
    records = list(executor.map(describe, paths))
print(json.dumps({"objects": records}, indent=2, sort_keys=True))
PY
gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" \
  "$REMOTE_PREFIX/remote_objects.json" >/dev/null
local_remote_objects_sha=$(sha256sum "$RUN_DIR/remote_objects.json" | awk '{print $1}')
remote_remote_objects_sha=$(gcloud storage cat "$REMOTE_PREFIX/remote_objects.json" |
  sha256sum | awk '{print $1}')
[[ $local_remote_objects_sha == "$remote_remote_objects_sha" ]] || {
  say "ABORT: remote object ledger checksum mismatch"
  exit 1
}

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$REMOTE_PREFIX" <<'PY'
from pathlib import Path
import subprocess
import sys

from glm_tpu.greenfield.validation.attention_update import (
    validate_exact_remote_object_set,
)

root = Path(sys.argv[1])
prefix = sys.argv[2]
listing = subprocess.run(
    ["gcloud", "storage", "ls", f"{prefix}/**"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
validate_exact_remote_object_set(root, prefix, listing)
PY

/home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$REMOTE_PREFIX" "$PIN" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
values = {
    "artifact_kind": summary["artifact_kind"],
    "code_hash": sys.argv[3],
    "results_db_run_id": summary["results_db_run_id"],
    "gate_d_passed": str(summary["gate_d_passed"]).lower(),
    "plan_hash": summary["plan_hash"],
    "strategy_nd_attention_projection": str(
        summary["strategy_nd_attention_projection"]
    ).lower(),
    "evidence_sha256": sha256((root / "evidence.sha256").read_bytes()).hexdigest(),
    "remote_objects_sha256": sha256(
        (root / "remote_objects.json").read_bytes()
    ).hexdigest(),
    "remote_prefix": sys.argv[2],
}
(root / "SUCCESS").write_text(
    "".join(f"{key}={value}\n" for key, value in values.items())
)
PY
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" \
  "$REMOTE_PREFIX/SUCCESS" >/dev/null
local_success_sha=$(sha256sum "$RUN_DIR/SUCCESS" | awk '{print $1}')
remote_success_sha=$(gcloud storage cat "$REMOTE_PREFIX/SUCCESS" |
  sha256sum | awk '{print $1}')
[[ $local_success_sha == "$remote_success_sha" ]] || {
  say "ABORT: remote SUCCESS checksum mismatch"
  exit 1
}
terminal_success_done=1

trap - EXIT
echo "SHORT_DECODER_STEP_OK $TAG DB=$DB_RUN"
