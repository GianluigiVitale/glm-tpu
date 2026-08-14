#!/usr/bin/env bash
# Protected four-chip discriminator for layer-0 projection/reduction association.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db

readonly ATTENTION_TAG=greenfield_layer0_attention_arithmetic_20260812T114701365714147Z
readonly ATTENTION_DIR=/home/gianl/gcs-models/results/$ATTENTION_TAG
readonly ATTENTION_TENSOR=$ATTENTION_DIR/attention_arithmetic.npz
readonly ATTENTION_RUNNER=$ATTENTION_DIR/runner.json
readonly ATTENTION_SUCCESS=$ATTENTION_DIR/SUCCESS
readonly ATTENTION_TENSOR_SHA=7d5ebe15dd006a70d77f17d41eb47f25f58c6d6784ee332fbc638f2916581f61
readonly ATTENTION_RUNNER_SHA=7961622c297567a021edf2ab335f0d046db6665e8130e210bbd78919a2d7a4ec
readonly ATTENTION_SUCCESS_SHA=ecc2b873c29ffd9bc6551136e90352b33523f50873827b00ae91279a4db3153c
readonly ATTENTION_REMOTE=$APPROVED_BUCKET/results/$ATTENTION_TAG

readonly INGREDIENT_TAG=greenfield_table_on_layer0_ingredients_p8155_20260812T021718885910564Z
readonly INGREDIENT_DIR=/home/gianl/gcs-models/results/$INGREDIENT_TAG/layer0_ingredients
readonly INGREDIENT_NPZ=$INGREDIENT_DIR/position_8155_ingredients.npz
readonly INGREDIENT_JSON=$INGREDIENT_DIR/contract.json
readonly INGREDIENT_NPZ_SHA=c06fe575f0518e981c3099a8033c1978b01fd0cff843ecbbd7a25cebed8d0e95
readonly INGREDIENT_JSON_SHA=02f0f1c3e22b08c6646167ac5a7ffaf5056932003ee722752f7d40e26e714864
readonly INGREDIENT_REMOTE=$APPROVED_BUCKET/results/$INGREDIENT_TAG/layer0_ingredients

readonly ACCEPTED_TAG=greenfield_legacy_layer0_attention_projection_p8155_20260812T090000000000000Z
readonly ACCEPTED_ROOT=/home/gianl/gcs-models/oracles/greenfield/glm52/attention_projection/8k/$ACCEPTED_TAG
readonly ACCEPTED_NPZ=$ACCEPTED_ROOT/attention_projection_capture/attention_projection.npz
readonly ACCEPTED_JSON=$ACCEPTED_ROOT/attention_projection_capture/capture.json
readonly ACCEPTED_SUCCESS=$ACCEPTED_ROOT/SUCCESS
readonly ACCEPTED_NPZ_SHA=3a619a0985fbb9ba6e1be9347bbc745c120fa74553ce1de5de830190a29c0a30
readonly ACCEPTED_JSON_SHA=f517b408f4ab6d2fb4d53e1aede696cc74e9d2d08ff70b8aa135291c4dcdaa32
readonly ACCEPTED_SUCCESS_SHA=88691576033fea623414034171441c213472ac26a3d5e7a04388cff434a0a47c
readonly ACCEPTED_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/attention_projection/8k/$ACCEPTED_TAG

readonly LAYER1_TAG=greenfield_layer1_dsa_internal_comparison_20260808T115135394251231Z
readonly LAYER1_ROOT=/home/gianl/gcs-models/oracles/greenfield/glm52/dsa_internals/8k/layer1/$LAYER1_TAG
readonly LAYER1_REFERENCE=$LAYER1_ROOT/internals.npz
readonly LAYER1_COMPARISON=$LAYER1_ROOT/comparison.json
readonly LAYER1_SEAL=$LAYER1_ROOT/diagnostic_seal.json
readonly LAYER1_REFERENCE_SHA=79b813daa8e194b6c9a9ad883a0199f4a938ca4d4ab7277d20a291b480349054
readonly LAYER1_COMPARISON_SHA=1bc43a8e231323c85b6782926182f2999f3014b8adbafa6f4db18b74e3419ad5
readonly LAYER1_SEAL_SHA=283e5e88b7aa836bd237e1b49e6d35ff2039db7639d2e34feccb61ec1fcb10d5
readonly LAYER1_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/dsa_internals/8k/layer1/$LAYER1_TAG

readonly ASSOCIATION_TAG=greenfield_collective_association_20260811T213152133863450Z
readonly ASSOCIATION_ROOT=/home/gianl/gcs-models/results/$ASSOCIATION_TAG
readonly ASSOCIATION_ANALYSIS=$ASSOCIATION_ROOT/association/analysis.json
readonly ASSOCIATION_SUCCESS=$ASSOCIATION_ROOT/SUCCESS
readonly ASSOCIATION_ANALYSIS_SHA=e7e34828365ca3d6cae0052f8d0e2e802143c6ca83810153db3116423f994108
readonly ASSOCIATION_SUCCESS_SHA=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
readonly ASSOCIATION_REMOTE=$APPROVED_BUCKET/results/$ASSOCIATION_TAG

readonly CHECKPOINT_TAG=greenfield_runtime_feature_qkv_pack_pp8_20260808T141032190315066Z
readonly CHECKPOINT_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP8_LP4/$CHECKPOINT_TAG
readonly CHECKPOINT_SUCCESS_SHA=368ef308c7937258c181cdc42fecddf8a7f7ca7bab04470d39cb622aaa3c5b24
readonly CHECKPOINT_MANIFEST_SHA=de46d38e404c637209f95505291105e89a6e7f95270fe91375a55ea79b5f7134
readonly CHECKPOINT_REMOTE=$APPROVED_BUCKET/checkpoints/greenfield/glm52/runtime_feature/PP8_LP4/$CHECKPOINT_TAG

readonly DB538_TAG=greenfield_layer0_projection_reduction_20260812T164052560787241Z
readonly DB538_DIR=/home/gianl/glm-run/$DB538_TAG
readonly DB538_RUNNER=$DB538_DIR/runner.json
readonly DB538_TENSOR=$DB538_DIR/projection_reduction.npz
readonly DB538_SUMMARY=$DB538_DIR/summary.json
readonly DB538_SUCCESS=$DB538_DIR/SUCCESS
readonly DB538_RUNNER_SHA=303dd91eed1d75e0cd443645c5f1ef745f259c51596d0651646bb141db8f16f8
readonly DB538_TENSOR_SHA=e801d5471697fefd1477c46603698289de93818d08d214bdf56e576f52819e0e
readonly DB538_SUMMARY_SHA=90090ba9999812082727ed56b734163f07e3c27b4eb88fa049b9c2504516e782
readonly DB538_SUCCESS_SHA=7744356f63b67cc813901499d0828c029ea5a9c985a5dac65525700457f79985
readonly DB538_CODE_HASH=e2a3a74a3b2ef1fa8f3b9cb1c5d7ec65f833eafc
readonly DB538_RUN_ID=538
readonly DB538_REMOTE=$APPROVED_BUCKET/results/$DB538_TAG
readonly POST_ATTENTION_RESIDUAL_SHA=a105fdbd429adb1d06a70bf71598a72a91d7b6faa83360005487ce11ce099f8e
readonly ATTENTION_UPDATE_SHA=68afed86921584fb673abb11a563e359a1533210ec2483e71ee79b88c2b0bde7
readonly COMBINED_RESIDUAL_SHA=02d045b9a0ec5ab22a711bd6a964564f707be0848683381104e83331020e31a3
readonly NORMALIZED_MLP_SHA=082125fead43b25f10686705c1b6473153f4092dd5bc476f8e01a86629f0758f

readonly ACCEPTED_M32_TAG=greenfield_accepted_decode_projection_lowering_20260811T184908676873350Z
readonly ACCEPTED_M32_ROOT=/home/gianl/gcs-models/oracles/greenfield/glm52/decode_projection_lowering/8k/$ACCEPTED_M32_TAG
readonly ACCEPTED_M32_HLO=$ACCEPTED_M32_ROOT/accepted_decode_projection_lowering/jit_step_fun_impl.m32.after_codegen_hlo.txt.gz
readonly ACCEPTED_M32_SUMMARY=$ACCEPTED_M32_ROOT/accepted_decode_projection_lowering/summary.json
readonly ACCEPTED_M32_SUCCESS=$ACCEPTED_M32_ROOT/SUCCESS
readonly ACCEPTED_M32_HLO_SHA=25041bfbcf319b6c6fc4c5888cb22548b246cccba784791796fe9e8f57199e4c
readonly ACCEPTED_M32_SUMMARY_SHA=409c845c2c9d67a1d6de36f0cccd25d2982850ee86c35645b0839fc78a1507a3
readonly ACCEPTED_M32_SUCCESS_SHA=6ef516dc42e046a996aa1fe542a4b11af5c2a14450e1aba7bfac98ddbf257278
readonly ACCEPTED_M32_RAW_HLO_SHA=3cd750810982608f9a3a7d557497c58f61159cc3dcdeb521f1377ba8c93fb775
readonly ACCEPTED_M32_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/decode_projection_lowering/8k/$ACCEPTED_M32_TAG

readonly CAPTURED_RMS_SOURCE_TAG=greenfield_layer0_dense_partial_capture_20260813T200736889447458Z
readonly CAPTURED_RMS_SOURCE_DIR=/home/gianl/glm-run/$CAPTURED_RMS_SOURCE_TAG
readonly CAPTURED_RMS_SOURCE_TENSOR=$CAPTURED_RMS_SOURCE_DIR/dense_partial_capture.npz
readonly CAPTURED_RMS_SOURCE_RUNNER=$CAPTURED_RMS_SOURCE_DIR/runner.json
readonly CAPTURED_RMS_SOURCE_SUMMARY=$CAPTURED_RMS_SOURCE_DIR/summary.json
readonly CAPTURED_RMS_SOURCE_SUCCESS=$CAPTURED_RMS_SOURCE_DIR/SUCCESS
readonly CAPTURED_RMS_SOURCE_TENSOR_SHA=f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298
readonly CAPTURED_RMS_SOURCE_RUNNER_SHA=d22b35f664409485b697fdb579665dc4b1ecf42683a2aaff9e9a4d7481ee8343
readonly CAPTURED_RMS_SOURCE_SUMMARY_SHA=c349b5fd458f34986d8cc59c0f026af6b0a4c4e998f8691d6f6aa83a9b55e416
readonly CAPTURED_RMS_SOURCE_SUCCESS_SHA=6cac897695fc1e78d0a10c0e36c993cffd281c6a88721d8955fa470bd44b0b85
readonly CAPTURED_RMS_SOURCE_REMOTE=$APPROVED_BUCKET/results/$CAPTURED_RMS_SOURCE_TAG

readonly CAPTURED_RMS_DB548_TAG=greenfield_layer0_dense_envelope_cross_layer_20260813T120703034434907Z
readonly CAPTURED_RMS_DB548_DIR=/home/gianl/glm-run/$CAPTURED_RMS_DB548_TAG
readonly CAPTURED_RMS_DB548_TENSOR=$CAPTURED_RMS_DB548_DIR/dense_envelope_cross_layer.npz
readonly CAPTURED_RMS_DB548_RUNNER=$CAPTURED_RMS_DB548_DIR/runner.json
readonly CAPTURED_RMS_DB548_SUMMARY=$CAPTURED_RMS_DB548_DIR/summary.json
readonly CAPTURED_RMS_DB548_SUCCESS=$CAPTURED_RMS_DB548_DIR/SUCCESS
readonly CAPTURED_RMS_DB548_HLO=$CAPTURED_RMS_DB548_DIR/hlo/dense_convolution.optimized_hlo.txt
readonly CAPTURED_RMS_DB548_TENSOR_SHA=6cb76623bd79e1712b6c323fa786e516abd05f6f0ca51a367bc871c4a920480f
readonly CAPTURED_RMS_DB548_RUNNER_SHA=0c9035409382db3e96bfee080af73f46c226e559908fc6869387da4a21f49112
readonly CAPTURED_RMS_DB548_SUMMARY_SHA=916f6ea202575c663e57ad768413a0015a04cf72450207a753e125965c9ae080
readonly CAPTURED_RMS_DB548_SUCCESS_SHA=a81433c33c14de0083c05d2d8dc1ebf34df0908356f18ca45c410efecf1d39f1
readonly CAPTURED_RMS_DB548_HLO_SHA=5f4dd83793da67be6a8c580949920e93f8c64fe8205816738e7d04890640a877
readonly CAPTURED_RMS_DB548_REMOTE=$APPROVED_BUCKET/results/$CAPTURED_RMS_DB548_TAG

readonly CAPTURED_RMS_DB549_TAG=greenfield_layer0_dense_envelope_split_rms_20260813T134012434338842Z
readonly CAPTURED_RMS_DB549_DIR=/home/gianl/glm-run/$CAPTURED_RMS_DB549_TAG
readonly CAPTURED_RMS_DB549_TENSOR=$CAPTURED_RMS_DB549_DIR/dense_envelope_split_rms.npz
readonly CAPTURED_RMS_DB549_RUNNER=$CAPTURED_RMS_DB549_DIR/runner.json
readonly CAPTURED_RMS_DB549_SUMMARY=$CAPTURED_RMS_DB549_DIR/summary.json
readonly CAPTURED_RMS_DB549_SUCCESS=$CAPTURED_RMS_DB549_DIR/SUCCESS
readonly CAPTURED_RMS_DB549_HLO=$CAPTURED_RMS_DB549_DIR/hlo/dense_convolution.optimized_hlo.txt
readonly CAPTURED_RMS_DB549_TENSOR_SHA=89ef1e3893bc39a349827debdbe37fcc8f3e009f9a5d3b8e48f87667863a1153
readonly CAPTURED_RMS_DB549_RUNNER_SHA=7f53f4e6dc4f400b8a47b893507ba5a293a46a844d2e22f06d815648df334e10
readonly CAPTURED_RMS_DB549_SUMMARY_SHA=8d3c6b7350928003f63d902bc4c6d54a59ed95687ba09c33502ed513cef12822
readonly CAPTURED_RMS_DB549_SUCCESS_SHA=894f4164b0924a7df100e8989d52c0c010c3588f862a9961147364e99fec4a0b
readonly CAPTURED_RMS_DB549_HLO_SHA=faf38fa987d174421d2631bc4921deee88bc30b50d557fbb02046c50fd9741c9
readonly CAPTURED_RMS_DB549_REMOTE=$APPROVED_BUCKET/results/$CAPTURED_RMS_DB549_TAG

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
HARNESS_GIT=$(git -C "$WORKTREE" rev-parse --short HEAD)
FORK_GIT=$(git -C /home/gianl/tpu-inference rev-parse --short HEAD)
DENSE_CONVOLUTION=${GLM_GREENFIELD_DENSE_CONVOLUTION_PROBE:-0}
DENSE_COMPILE_ROWS=${GLM_GREENFIELD_DENSE_CONVOLUTION_COMPILE_ROWS:-1}
DENSE_LAYER1_ONLY=${GLM_GREENFIELD_DENSE_LAYER1_ONLY:-0}
DENSE_FINAL_LAYOUT=${GLM_GREENFIELD_DENSE_FINAL_LAYOUT:-0}
DENSE_ENVELOPE=${GLM_GREENFIELD_DENSE_ENVELOPE:-0}
DENSE_SPLIT_LAYER1_RMS=${GLM_GREENFIELD_DENSE_SPLIT_LAYER1_RMS:-0}
DENSE_CAPTURE_PARTIALS=${GLM_GREENFIELD_DENSE_CAPTURE_PARTIALS:-0}
CAPTURED_RMS_REPLAY=${GLM_GREENFIELD_CAPTURED_RMS_REPLAY:-0}
ISOLATED_DENSE_REPLAY=${GLM_GREENFIELD_ISOLATED_DENSE_REPLAY:-0}
[[ $DENSE_CONVOLUTION == 0 || $DENSE_CONVOLUTION == 1 ]] || {
  echo "GLM_GREENFIELD_DENSE_CONVOLUTION_PROBE must be 0 or 1" >&2
  exit 2
}
[[ $DENSE_COMPILE_ROWS == 1 || $DENSE_COMPILE_ROWS == 32 ]] || {
  echo "GLM_GREENFIELD_DENSE_CONVOLUTION_COMPILE_ROWS must be 1 or 32" >&2
  exit 2
}
[[ $DENSE_LAYER1_ONLY == 0 || $DENSE_LAYER1_ONLY == 1 ]] || {
  echo "GLM_GREENFIELD_DENSE_LAYER1_ONLY must be 0 or 1" >&2
  exit 2
}
[[ $DENSE_FINAL_LAYOUT == 0 || $DENSE_FINAL_LAYOUT == 1 ]] || {
  echo "GLM_GREENFIELD_DENSE_FINAL_LAYOUT must be 0 or 1" >&2
  exit 2
}
[[ $DENSE_ENVELOPE == 0 || $DENSE_ENVELOPE == 1 ]] || {
  echo "GLM_GREENFIELD_DENSE_ENVELOPE must be 0 or 1" >&2
  exit 2
}
[[ $DENSE_SPLIT_LAYER1_RMS == 0 || $DENSE_SPLIT_LAYER1_RMS == 1 ]] || {
  echo "GLM_GREENFIELD_DENSE_SPLIT_LAYER1_RMS must be 0 or 1" >&2
  exit 2
}
[[ $DENSE_CAPTURE_PARTIALS == 0 || $DENSE_CAPTURE_PARTIALS == 1 ]] || {
  echo "GLM_GREENFIELD_DENSE_CAPTURE_PARTIALS must be 0 or 1" >&2
  exit 2
}
[[ $CAPTURED_RMS_REPLAY == 0 || $CAPTURED_RMS_REPLAY == 1 ]] || {
  echo "GLM_GREENFIELD_CAPTURED_RMS_REPLAY must be 0 or 1" >&2
  exit 2
}
[[ $ISOLATED_DENSE_REPLAY == 0 || $ISOLATED_DENSE_REPLAY == 1 ]] || {
  echo "GLM_GREENFIELD_ISOLATED_DENSE_REPLAY must be 0 or 1" >&2
  exit 2
}
if [[ $CAPTURED_RMS_REPLAY == 1 && $ISOLATED_DENSE_REPLAY == 1 ]]; then
  echo "captured RMS and isolated dense replays are mutually exclusive" >&2
  exit 2
fi
if [[ ( $CAPTURED_RMS_REPLAY == 1 || $ISOLATED_DENSE_REPLAY == 1 ) && \
      ! ( $DENSE_CONVOLUTION == 0 && $DENSE_COMPILE_ROWS == 1 && \
          $DENSE_LAYER1_ONLY == 0 && $DENSE_FINAL_LAYOUT == 0 && \
          $DENSE_ENVELOPE == 0 && $DENSE_SPLIT_LAYER1_RMS == 0 && \
          $DENSE_CAPTURE_PARTIALS == 0 ) ]]; then
  echo "bounded replays are exclusive of dense/checkpoint modes" >&2
  exit 2
fi
if [[ $DENSE_CONVOLUTION == 0 && $DENSE_COMPILE_ROWS != 1 ]]; then
  echo "projection/reduction mode requires compile rows 1" >&2
  exit 2
fi
if [[ $DENSE_LAYER1_ONLY == 1 && \
      ! ( $DENSE_CONVOLUTION == 1 && $DENSE_COMPILE_ROWS == 32 ) ]]; then
  echo "layer-1-only mode requires the dense M32 discriminator" >&2
  exit 2
fi
if [[ $DENSE_FINAL_LAYOUT == 1 && \
      ! ( $DENSE_CONVOLUTION == 1 && $DENSE_COMPILE_ROWS == 32 && \
          $DENSE_LAYER1_ONLY == 1 ) ]]; then
  echo "final dense layout requires the M32 layer-1-only discriminator" >&2
  exit 2
fi
if [[ $DENSE_ENVELOPE == 1 && $DENSE_FINAL_LAYOUT != 1 ]]; then
  echo "dense envelope requires the final-layout discriminator" >&2
  exit 2
fi
if [[ $DENSE_SPLIT_LAYER1_RMS == 1 && $DENSE_ENVELOPE != 1 ]]; then
  echo "split layer-1 RMS requires the dense-envelope discriminator" >&2
  exit 2
fi
if [[ $DENSE_CAPTURE_PARTIALS == 1 && \
      ! ( $DENSE_CONVOLUTION == 1 && $DENSE_COMPILE_ROWS == 32 && \
          $DENSE_LAYER1_ONLY == 1 && $DENSE_FINAL_LAYOUT == 1 && \
          $DENSE_ENVELOPE == 1 && $DENSE_SPLIT_LAYER1_RMS == 0 ) ]]; then
  echo "dense partial capture requires the unsplit M32 final-layout envelope" >&2
  exit 2
fi
if [[ $ISOLATED_DENSE_REPLAY == 1 ]]; then
  TAG=${GLM_GREENFIELD_ISOLATED_DENSE_TAG:-greenfield_layer0_isolated_dense_replay_$(date -u +%Y%m%dT%H%M%S%NZ)}
  TENSOR_BASENAME=isolated_dense_replay.npz
elif [[ $CAPTURED_RMS_REPLAY == 1 ]]; then
  TAG=${GLM_GREENFIELD_CAPTURED_RMS_TAG:-greenfield_layer0_captured_rms_replay_$(date -u +%Y%m%dT%H%M%S%NZ)}
  TENSOR_BASENAME=captured_rms_replay.npz
elif [[ $DENSE_CONVOLUTION == 1 ]]; then
  if [[ $DENSE_CAPTURE_PARTIALS == 1 ]]; then
    TAG=${GLM_GREENFIELD_DENSE_CONVOLUTION_TAG:-greenfield_layer0_dense_partial_capture_$(date -u +%Y%m%dT%H%M%S%NZ)}
    TENSOR_BASENAME=dense_partial_capture.npz
  elif [[ $DENSE_SPLIT_LAYER1_RMS == 1 ]]; then
    TAG=${GLM_GREENFIELD_DENSE_CONVOLUTION_TAG:-greenfield_layer0_dense_envelope_split_rms_$(date -u +%Y%m%dT%H%M%S%NZ)}
    TENSOR_BASENAME=dense_envelope_split_rms.npz
  elif [[ $DENSE_ENVELOPE == 1 ]]; then
    TAG=${GLM_GREENFIELD_DENSE_CONVOLUTION_TAG:-greenfield_layer0_dense_envelope_cross_layer_$(date -u +%Y%m%dT%H%M%S%NZ)}
    TENSOR_BASENAME=dense_envelope_cross_layer.npz
  elif [[ $DENSE_FINAL_LAYOUT == 1 ]]; then
    TAG=${GLM_GREENFIELD_DENSE_CONVOLUTION_TAG:-greenfield_layer0_dense_final_layout_cross_layer_$(date -u +%Y%m%dT%H%M%S%NZ)}
    TENSOR_BASENAME=dense_final_layout_cross_layer.npz
  elif [[ $DENSE_LAYER1_ONLY == 1 ]]; then
    TAG=${GLM_GREENFIELD_DENSE_CONVOLUTION_TAG:-greenfield_layer0_dense_m32_cross_layer_$(date -u +%Y%m%dT%H%M%S%NZ)}
    TENSOR_BASENAME=dense_m32_cross_layer.npz
  elif [[ $DENSE_COMPILE_ROWS == 32 ]]; then
    TAG=${GLM_GREENFIELD_DENSE_CONVOLUTION_TAG:-greenfield_layer0_dense_m32_convolution_$(date -u +%Y%m%dT%H%M%S%NZ)}
    TENSOR_BASENAME=dense_m32_convolution.npz
  else
    TAG=${GLM_GREENFIELD_DENSE_CONVOLUTION_TAG:-greenfield_layer0_dense_convolution_$(date -u +%Y%m%dT%H%M%S%NZ)}
    TENSOR_BASENAME=dense_convolution.npz
  fi
else
  TAG=${GLM_GREENFIELD_PROJECTION_REDUCTION_TAG:-greenfield_layer0_projection_reduction_$(date -u +%Y%m%dT%H%M%S%NZ)}
  TENSOR_BASENAME=projection_reduction.npz
fi
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
m32_probe_args=()
if [[ $DENSE_COMPILE_ROWS == 32 ]]; then
  m32_probe_args=(
    --accepted-m32-hlo "$ACCEPTED_M32_HLO"
    --accepted-m32-hlo-sha256 "$ACCEPTED_M32_HLO_SHA"
    --accepted-m32-summary "$ACCEPTED_M32_SUMMARY"
    --accepted-m32-summary-sha256 "$ACCEPTED_M32_SUMMARY_SHA"
    --accepted-m32-success "$ACCEPTED_M32_SUCCESS"
    --accepted-m32-success-sha256 "$ACCEPTED_M32_SUCCESS_SHA"
  )
fi
layer1_probe_args=()
if [[ $DENSE_LAYER1_ONLY == 1 ]]; then
  layer1_probe_args=(--layer1-only)
fi
final_layout_probe_args=()
if [[ $DENSE_FINAL_LAYOUT == 1 ]]; then
  final_layout_probe_args=(--final-dense-layout)
fi
dense_envelope_probe_args=()
if [[ $DENSE_ENVELOPE == 1 ]]; then
  dense_envelope_probe_args=(--dense-envelope)
fi
split_layer1_rms_probe_args=()
if [[ $DENSE_SPLIT_LAYER1_RMS == 1 ]]; then
  split_layer1_rms_probe_args=(--split-layer1-rms)
fi
capture_partials_probe_args=()
validate_captured_rms_replay() {
  /home/gianl/vllm-env/bin/python - "$RUN_DIR" "$PIN" "$elapsed" "$TAG" \
    "$CAPTURED_RMS_SOURCE_TENSOR_SHA" "$CAPTURED_RMS_SOURCE_RUNNER_SHA" \
    "$CAPTURED_RMS_SOURCE_SUMMARY_SHA" "$CAPTURED_RMS_SOURCE_SUCCESS_SHA" \
    "$CAPTURED_RMS_DB548_TENSOR_SHA" "$CAPTURED_RMS_DB548_RUNNER_SHA" \
    "$CAPTURED_RMS_DB548_SUMMARY_SHA" "$CAPTURED_RMS_DB548_SUCCESS_SHA" \
    "$CAPTURED_RMS_DB548_HLO_SHA" "$CAPTURED_RMS_DB549_TENSOR_SHA" \
    "$CAPTURED_RMS_DB549_RUNNER_SHA" "$CAPTURED_RMS_DB549_SUMMARY_SHA" \
    "$CAPTURED_RMS_DB549_SUCCESS_SHA" "$CAPTURED_RMS_DB549_HLO_SHA" <<'PY'
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np

(
    run_dir_text,
    pin,
    elapsed_text,
    run_tag,
    capture_tensor_sha,
    capture_runner_sha,
    capture_summary_sha,
    capture_success_sha,
    db548_tensor_sha,
    db548_runner_sha,
    db548_summary_sha,
    db548_success_sha,
    db548_hlo_sha,
    db549_tensor_sha,
    db549_runner_sha,
    db549_summary_sha,
    db549_success_sha,
    db549_hlo_sha,
) = sys.argv[1:]
run_dir = Path(run_dir_text)
runner_path = run_dir / "runner.json"
tensor_path = run_dir / "captured_rms_replay.npz"
runner = json.loads(runner_path.read_text())

def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()

def array_sha(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()

def exact_int(value: object) -> bool:
    return type(value) is int

def finite_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value)

expected_source = {
    "capture_runner_sha256": capture_runner_sha,
    "capture_summary_sha256": capture_summary_sha,
    "capture_success_sha256": capture_success_sha,
    "capture_tensor_sha256": capture_tensor_sha,
    "db548_hlo_sha256": db548_hlo_sha,
    "db548_runner_sha256": db548_runner_sha,
    "db548_summary_sha256": db548_summary_sha,
    "db548_success_sha256": db548_success_sha,
    "db548_tensor_sha256": db548_tensor_sha,
    "db549_hlo_sha256": db549_hlo_sha,
    "db549_runner_sha256": db549_runner_sha,
    "db549_summary_sha256": db549_summary_sha,
    "db549_success_sha256": db549_success_sha,
    "db549_tensor_sha256": db549_tensor_sha,
}
arms = runner.get("arms")
if not (
    runner.get("artifact_kind") == "glm52_layer0_captured_rms_replay"
    and isinstance(arms, dict)
    and set(arms) == {"control", "accepted_split"}
    and runner.get("code_hash") == pin
    and runner.get("control_admissible") is True
    and runner.get("dense_update_sha256")
    == "efde853254c03dd18a5f5f22733630ce0e785dfbb4eba09c41eea9085e47b4fc"
    and runner.get("live_rows") == 1
    and runner.get("performance_claim") is False
    and runner.get("position") == 8155
    and runner.get("source") == expected_source
    and runner.get("status") == "SUCCESS"
):
    raise SystemExit("captured RMS runner identity drifted")

expected_collectives = {
    "all_gather": 1,
    "all_reduce": 0,
    "all_to_all": 0,
    "collective_broadcast": 0,
    "collective_permute": 0,
    "reduce_scatter": 0,
}
for name, split in (("control", False), ("accepted_split", True)):
    arm = arms[name]
    stable = arm.get("hlo", {}).get("stablehlo_contract", {})
    optimized = arm.get("hlo", {}).get("optimized_contract", {})
    stable_path = run_dir / "hlo" / f"{name}.stablehlo.mlir"
    optimized_path = run_dir / "hlo" / f"{name}.optimized_hlo.txt"
    if not (
        arm.get("split_layer1_rms") is split
        and stable.get("passed") is True
        and stable.get("violations") == []
        and stable.get("collective_counts") == expected_collectives
        and stable.get("exact_result_binding") is True
        and stable.get("live_rows") == 1
        and stable.get("split_layer1_rms") is split
        and optimized.get("passed") is True
        and optimized.get("violations") == []
        and optimized.get("async_collectives") == []
        and optimized.get("collective_count") == 1
        and optimized.get("exact_output_fusion") is True
        and optimized.get("exact_result_binding") is True
        and optimized.get("exact_scheduled_reduction_binding") is True
        and optimized.get("live_rows") == 1
        and optimized.get("num_partitions") == 4
        and optimized.get("num_replicas") in (None, 1)
        and optimized.get("performance_claim") is False
        and optimized.get("split_layer1_rms") is split
        and (
            len(optimized.get("accepted_scheduled_reduction_values", [])) == 1
            if split
            else optimized.get("accepted_scheduled_reduction_values") == []
        )
        and arm.get("hlo", {}).get("stablehlo_sha256")
        == file_sha(stable_path)
        and arm.get("hlo", {}).get("optimized_sha256")
        == file_sha(optimized_path)
    ):
        raise SystemExit(f"captured RMS {name} HLO contract drifted")

control = arms["control"]
control_comparison = control.get("comparison", {})
if not (
    control.get("output_sha256")
    == "9b52a04e2852719237f4465b28665cbc213b635763303b554bb12345e99a4005"
    and control_comparison.get("elementwise_exact") is True
    and control_comparison.get("expected_sha256") == control["output_sha256"]
    and control_comparison.get("observed_sha256") == control["output_sha256"]
    and exact_int(control_comparison.get("mismatch_count"))
    and control_comparison.get("mismatch_count") == 0
    and control_comparison.get("first_mismatch_index") is None
    and finite_float(control_comparison.get("max_abs_error"))
    and control_comparison.get("max_abs_error") == 0.0
    and finite_float(control_comparison.get("mean_abs_error"))
    and control_comparison.get("mean_abs_error") == 0.0
    and control_comparison.get("shape") == [6144]
):
    raise SystemExit("captured RMS control comparison drifted")

split = arms["accepted_split"]
comparison = split.get("comparison", {})
exact = runner.get("exact")
expected_sha = "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039"
common = bool(
    comparison.get("expected_sha256") == expected_sha
    and comparison.get("observed_sha256") == split.get("output_sha256")
    and comparison.get("shape") == [6144]
    and exact_int(comparison.get("mismatch_count"))
    and finite_float(comparison.get("max_abs_error"))
    and finite_float(comparison.get("mean_abs_error"))
)
exact_schema = bool(
    exact is True
    and runner.get("classification") == "captured_partials_split_rms_exact"
    and runner.get("exact_arms") == ["accepted_split"]
    and split.get("output_sha256") == expected_sha
    and comparison.get("elementwise_exact") is True
    and exact_int(comparison.get("mismatch_count"))
    and comparison.get("mismatch_count") == 0
    and comparison.get("first_mismatch_index") is None
    and finite_float(comparison.get("max_abs_error"))
    and comparison.get("max_abs_error") == 0.0
    and finite_float(comparison.get("mean_abs_error"))
    and comparison.get("mean_abs_error") == 0.0
)
nonexact_schema = bool(
    exact is False
    and runner.get("classification") == "captured_partials_split_rms_nonexact"
    and runner.get("exact_arms") == []
    and split.get("output_sha256") != expected_sha
    and comparison.get("elementwise_exact") is False
    and 1 <= comparison.get("mismatch_count", 0) <= 6144
    and exact_int(comparison.get("first_mismatch_index"))
    and 0 <= comparison["first_mismatch_index"] < 6144
    and 0.0 < comparison.get("mean_abs_error", 0.0)
    <= comparison.get("max_abs_error", 0.0)
)
if not common or not (exact_schema or nonexact_schema):
    raise SystemExit("captured RMS split comparison drifted")

with np.load(tensor_path, allow_pickle=False) as payload:
    if set(payload.files) != {
        "accepted_layer1_normalized_bfloat16_bits",
        "control_layer1_normalized_bfloat16_bits",
        "db548_layer1_normalized_bfloat16_bits",
        "split_layer1_normalized_bfloat16_bits",
    }:
        raise SystemExit("captured RMS tensor key set drifted")
    expected_arrays = {
        "accepted_layer1_normalized_bfloat16_bits": expected_sha,
        "control_layer1_normalized_bfloat16_bits": control["output_sha256"],
        "db548_layer1_normalized_bfloat16_bits": control["output_sha256"],
        "split_layer1_normalized_bfloat16_bits": split["output_sha256"],
    }
    for name, expected_array_sha in expected_arrays.items():
        value = np.ascontiguousarray(payload[name])
        if (
            value.dtype != np.uint16
            or value.shape != (6144,)
            or array_sha(value) != expected_array_sha
        ):
            raise SystemExit(f"captured RMS tensor drifted: {name}")

summary = {
    "artifact_kind": runner["artifact_kind"],
    "arms": {
        name: {
            "comparison": arms[name]["comparison"],
            "hlo": {
                "optimized_sha256": arms[name]["hlo"]["optimized_sha256"],
                "stablehlo_sha256": arms[name]["hlo"]["stablehlo_sha256"],
            },
            "output_sha256": arms[name]["output_sha256"],
            "split_layer1_rms": arms[name]["split_layer1_rms"],
        }
        for name in ("control", "accepted_split")
    },
    "classification": runner["classification"],
    "code_hash": pin,
    "control_admissible": True,
    "elapsed_seconds": int(elapsed_text),
    "exact": exact,
    "exact_arms": runner["exact_arms"],
    "live_rows": 1,
    "performance_claim": False,
    "position": 8155,
    "results_db_run_id": None,
    "run_tag": run_tag,
    "runner_sha256": file_sha(runner_path),
    "source": expected_source,
    "status": "SUCCESS",
    "tensor_sha256": file_sha(tensor_path),
}
(run_dir / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("CAPTURED_RMS_REPLAY_VALID")
PY
}
validate_isolated_dense_replay() {
  /home/gianl/vllm-env/bin/python - "$RUN_DIR" "$PIN" "$elapsed" "$TAG" \
    "$CAPTURED_RMS_SOURCE_TENSOR_SHA" "$CAPTURED_RMS_SOURCE_RUNNER_SHA" \
    "$CAPTURED_RMS_SOURCE_SUMMARY_SHA" "$CAPTURED_RMS_SOURCE_SUCCESS_SHA" \
    "$CAPTURED_RMS_DB548_TENSOR_SHA" "$CAPTURED_RMS_DB548_RUNNER_SHA" \
    "$CAPTURED_RMS_DB548_SUMMARY_SHA" "$CAPTURED_RMS_DB548_SUCCESS_SHA" \
    "$CAPTURED_RMS_DB548_HLO_SHA" "$CAPTURED_RMS_DB549_TENSOR_SHA" \
    "$CAPTURED_RMS_DB549_RUNNER_SHA" "$CAPTURED_RMS_DB549_SUMMARY_SHA" \
    "$CAPTURED_RMS_DB549_SUCCESS_SHA" "$CAPTURED_RMS_DB549_HLO_SHA" \
    "$ACCEPTED_M32_HLO_SHA" "$ACCEPTED_M32_SUMMARY_SHA" \
    "$ACCEPTED_M32_SUCCESS_SHA" \
    "$CHECKPOINT_MANIFEST_SHA" <<'PY'
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import ml_dtypes
import numpy as np

(
    run_dir_text,
    pin,
    elapsed_text,
    run_tag,
    capture_tensor_sha,
    capture_runner_sha,
    capture_summary_sha,
    capture_success_sha,
    db548_tensor_sha,
    db548_runner_sha,
    db548_summary_sha,
    db548_success_sha,
    db548_hlo_sha,
    db549_tensor_sha,
    db549_runner_sha,
    db549_summary_sha,
    db549_success_sha,
    db549_hlo_sha,
    accepted_m32_hlo_sha,
    accepted_m32_summary_sha,
    accepted_m32_success_sha,
    checkpoint_manifest_sha,
) = sys.argv[1:]
root = Path(run_dir_text)
runner_path = root / "runner.json"
tensor_path = root / "isolated_dense_replay.npz"
runner = json.loads(runner_path.read_text())

def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()

def array_sha(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()

def compare(reference: np.ndarray, observed: np.ndarray) -> dict[str, object]:
    mismatch = reference != observed
    indices = np.flatnonzero(mismatch)
    expected_float = reference.view(ml_dtypes.bfloat16).astype(np.float32)
    observed_float = observed.view(ml_dtypes.bfloat16).astype(np.float32)
    errors = np.abs(expected_float - observed_float)
    return {
        "elementwise_exact": not bool(indices.size),
        "expected_sha256": array_sha(reference),
        "first_mismatch_index": None if not indices.size else int(indices[0]),
        "max_abs_error": float(errors.max(initial=0.0)),
        "mean_abs_error": float(errors.mean()),
        "mismatch_count": int(indices.size),
        "observed_sha256": array_sha(observed),
        "shape": list(observed.shape),
    }

expected_source = {
    "accepted_m32_hlo_raw_sha256": "3cd750810982608f9a3a7d557497c58f61159cc3dcdeb521f1377ba8c93fb775",
    "accepted_m32_hlo_sha256": accepted_m32_hlo_sha,
    "accepted_m32_success_sha256": accepted_m32_success_sha,
    "accepted_m32_summary_sha256": accepted_m32_summary_sha,
    "capture_runner_sha256": capture_runner_sha,
    "capture_summary_sha256": capture_summary_sha,
    "capture_success_sha256": capture_success_sha,
    "capture_tensor_sha256": capture_tensor_sha,
    "checkpoint_manifest_sha256": checkpoint_manifest_sha,
    "db548_hlo_sha256": db548_hlo_sha,
    "db548_runner_sha256": db548_runner_sha,
    "db548_summary_sha256": db548_summary_sha,
    "db548_success_sha256": db548_success_sha,
    "db548_tensor_sha256": db548_tensor_sha,
    "db549_hlo_sha256": db549_hlo_sha,
    "db549_runner_sha256": db549_runner_sha,
    "db549_summary_sha256": db549_summary_sha,
    "db549_success_sha256": db549_success_sha,
    "db549_tensor_sha256": db549_tensor_sha,
}
expected_records = {
    "dense.slot_00.merged_gate_up.weight_bits_in_out": {
        "dtype": "float8_e4m3fn",
        "sha256": "82c93c0fafda7afa3853e3a689aace78be61bec88c5ef779bb0e9efeb834facf",
        "shape": [4, 8, 6144, 768],
    },
    "dense.slot_00.merged_gate_up.scale_inv_in_out": {
        "dtype": "float32",
        "sha256": "9b4bfee8b15d04545a277ba4e1ea0d3b73427a0f2cebf0a5f5fa4d7ab32bf8b3",
        "shape": [4, 8, 48, 768],
    },
    "dense.slot_00.down.weight_bits_in_out": {
        "dtype": "float8_e4m3fn",
        "sha256": "8654c1ebb6f0ef29b1d3919699c08ca2b81e058bf3f9cd0827a9889994d72f7e",
        "shape": [4, 8, 384, 6144],
    },
    "dense.slot_00.down.scale_inv_in_out": {
        "dtype": "float32",
        "sha256": "f37e87987745d263d152ff17415d21d572fa3c73ad5888207f1aa44916f74bdd",
        "shape": [4, 8, 3, 6144],
    },
}
expected_gate_singleton_discriminator = {
    "accepted_internal_gate_dequant_count": 3,
    "accepted_rank2_gate_count": 0,
    "accepted_rank3_gate_count": 3,
    "challenger_selected": True,
    "db548_materialized_bf16_gate_count": 8,
    "db548_rank2_gate_count": 8,
    "db548_rank3_gate_count": 0,
}
if not (
    runner.get("accepted_gate_dequant_fusion") is True
    and runner.get("accepted_gate_singleton") is True
    and runner.get("artifact_kind") == "glm52_layer0_isolated_dense_replay"
    and runner.get("code_hash") == pin
    and runner.get("control_admissible") is True
    and type(runner.get("exact")) is bool
    and runner.get("exact_arms")
    == (["isolated_virtual_contractions"] if runner["exact"] else [])
    and runner.get("final_layout_records") == expected_records
    and runner.get("gate_singleton_discriminator")
    == expected_gate_singleton_discriminator
    and runner.get("performance_claim") is False
    and runner.get("position") == 8155
    and runner.get("source") == expected_source
    and runner.get("status") == "SUCCESS"
    and runner.get("virtual_contractions_per_chip") == 1
    and runner.get("virtual_rank_batches") == 8
):
    raise SystemExit("isolated dense runner identity drifted")

with np.load(tensor_path, allow_pickle=False) as payload:
    expected_keys = {
        "accepted_layer1_normalized_bfloat16_bits",
        "captured_dense_virtual_partials_bfloat16_bits",
        "control_layer1_normalized_bfloat16_bits",
        "isolated_dense_virtual_partials_bfloat16_bits",
        "isolated_layer1_normalized_bfloat16_bits",
        "sensitivity_candidate_dense_update_bfloat16_bits",
        "sensitivity_candidate_model_rank",
        "sensitivity_candidate_partial_bit_delta",
        "sensitivity_candidate_partial_bits",
        "sensitivity_layer1_normalized_bfloat16_bits",
    }
    if set(payload.files) != expected_keys:
        raise SystemExit("isolated dense tensor key set drifted")
    accepted = np.ascontiguousarray(
        payload["accepted_layer1_normalized_bfloat16_bits"]
    )
    captured = np.ascontiguousarray(
        payload["captured_dense_virtual_partials_bfloat16_bits"]
    )
    control = np.ascontiguousarray(
        payload["control_layer1_normalized_bfloat16_bits"]
    )
    isolated_partials = np.ascontiguousarray(
        payload["isolated_dense_virtual_partials_bfloat16_bits"]
    )
    isolated = np.ascontiguousarray(
        payload["isolated_layer1_normalized_bfloat16_bits"]
    )
    sensitivity_dense = np.ascontiguousarray(
        payload["sensitivity_candidate_dense_update_bfloat16_bits"]
    )
    sensitivity_rank = np.ascontiguousarray(
        payload["sensitivity_candidate_model_rank"]
    )
    sensitivity_delta = np.ascontiguousarray(
        payload["sensitivity_candidate_partial_bit_delta"]
    )
    sensitivity_partial_bits = np.ascontiguousarray(
        payload["sensitivity_candidate_partial_bits"]
    )
    sensitivity_outputs = np.ascontiguousarray(
        payload["sensitivity_layer1_normalized_bfloat16_bits"]
    )
if not (
    all(value.dtype == np.uint16 for value in (accepted, captured, control, isolated_partials, isolated))
    and accepted.shape == control.shape == isolated.shape == (6144,)
    and captured.shape == isolated_partials.shape == (4, 8, 1, 6144)
    and array_sha(accepted)
    == "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039"
    and array_sha(captured)
    == "9d9f65dddc7b622875872a33a6522c330c8fb5490c8cba14526553c211516e35"
    and array_sha(control)
    == "9b52a04e2852719237f4465b28665cbc213b635763303b554bb12345e99a4005"
    and sensitivity_dense.dtype == np.uint16
    and sensitivity_rank.dtype == np.int16
    and sensitivity_delta.dtype == np.int16
    and sensitivity_partial_bits.dtype == np.int32
    and sensitivity_outputs.dtype == np.uint16
    and sensitivity_dense.ndim == sensitivity_rank.ndim
    == sensitivity_delta.ndim == sensitivity_partial_bits.ndim == 1
    and sensitivity_outputs.shape == (sensitivity_dense.size, 6144)
    and sensitivity_rank.shape == sensitivity_delta.shape
    == sensitivity_partial_bits.shape == sensitivity_dense.shape
):
    raise SystemExit("isolated dense tensor identity drifted")
comparison = compare(accepted, isolated)
if runner.get("isolated_layer1_comparison") != comparison:
    raise SystemExit("isolated dense layer1 comparison drifted")
if runner.get("isolated_layer1_sha256") != array_sha(isolated):
    raise SystemExit("isolated dense layer1 SHA drifted")
partial_mismatch = captured != isolated_partials
first_partial = np.argwhere(partial_mismatch)
expected_partial = {
    "elementwise_exact": not bool(first_partial.size),
    "first_mismatch_index": (
        None
        if not first_partial.size
        else [int(value) for value in first_partial[0]]
    ),
    "mismatch_count": int(np.count_nonzero(partial_mismatch)),
    "per_virtual_rank": [
        {
            "mismatch_count": int(np.count_nonzero(partial_mismatch[:, rank])),
            "virtual_rank": rank,
        }
        for rank in range(8)
    ],
}
if runner.get("partial_comparison") != expected_partial:
    raise SystemExit("isolated dense partial comparison drifted")

physical_model_positions = (
    0, 16, 4, 20, 8, 24, 12, 28,
    1, 17, 5, 21, 9, 25, 13, 29,
    2, 18, 6, 22, 10, 26, 14, 30,
    3, 19, 7, 23, 11, 27, 15, 31,
)

def reduce_feature_bits(value: np.ndarray, hidden_index: int) -> int:
    model = value.reshape(32, 1, 6144)[:, 0, hidden_index].view(
        ml_dtypes.bfloat16
    )
    physical = np.stack(tuple(model[index] for index in physical_model_positions))
    physical_y_x_z = np.transpose(physical.reshape(4, 4, 2), (1, 2, 0))

    def add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.asarray(left + right, dtype=ml_dtypes.bfloat16)

    def reduce_four(values: np.ndarray, cross: bool) -> np.ndarray:
        if cross:
            return add(add(values[0], values[3]), add(values[1], values[2]))
        return add(add(values[0], values[1]), add(values[2], values[3]))

    y_reduced = reduce_four(
        physical_y_x_z, 2048 <= hidden_index < 4096
    )
    x_reduced = add(y_reduced[0], y_reduced[1])
    reduced = reduce_four(x_reduced, bool((hidden_index // 256) % 2))
    return int(np.asarray(reduced).view(np.uint16))

def expected_sensitivity_records() -> list[dict[str, object]]:
    hidden_index = 2795
    baseline = reduce_feature_bits(captured, hidden_index)
    flat = captured.reshape(32, 1, 6144)
    producers: dict[int, list[tuple[int, int, int, int]]] = {
        baseline: [(-1, 0, -1, -1)]
    }
    for model_rank in range(32):
        original = int(flat[model_rank, 0, hidden_index])
        for delta in range(-32, 33):
            if delta == 0 or not 0 <= original + delta <= 0xFFFF:
                continue
            mutated = captured.copy()
            mutated.reshape(32, 1, 6144)[model_rank, 0, hidden_index] = (
                np.uint16(original + delta)
            )
            dense_bits = reduce_feature_bits(mutated, hidden_index)
            producers.setdefault(dense_bits, []).append(
                (model_rank, delta, original, original + delta)
            )
    records: list[dict[str, object]] = []
    for candidate_id, dense_bits in enumerate(sorted(producers)):
        available = producers[dense_bits]
        representative = min(
            available,
            key=lambda item: (
                abs(item[1]),
                item[0] if item[0] >= 0 else -1,
                item[1],
            ),
        )
        model_rank, delta, original, mutated = representative
        minimum_abs = min(abs(item[1]) for item in available)
        records.append({
            "candidate_id": candidate_id,
            "dense_update_bits": dense_bits,
            "minimum_abs_partial_bit_delta": minimum_abs,
            "minimum_delta_producers": [
                {"model_rank": item[0], "partial_bit_delta": item[1]}
                for item in available
                if abs(item[1]) == minimum_abs
            ],
            "representative_model_rank": model_rank,
            "representative_owner": model_rank // 8 if model_rank >= 0 else -1,
            "representative_virtual_rank": model_rank % 8 if model_rank >= 0 else -1,
            "representative_original_partial_bits": original,
            "representative_mutated_partial_bits": mutated,
            "representative_partial_bit_delta": delta,
        })
    return records

expected_sensitivity = expected_sensitivity_records()
actual_sensitivity = runner.get("sensitivity", {})
actual_candidates = actual_sensitivity.get("candidates")
if not isinstance(actual_candidates, list) or len(actual_candidates) != len(
    expected_sensitivity
):
    raise SystemExit("isolated dense sensitivity candidate count drifted")
exact_candidate_ids = []
for index, (expected_candidate, actual_candidate) in enumerate(
    zip(expected_sensitivity, actual_candidates, strict=True)
):
    candidate_comparison = compare(accepted, sensitivity_outputs[index])
    expected_with_result = {
        **expected_candidate,
        "comparison": candidate_comparison,
        "output_sha256": array_sha(sensitivity_outputs[index]),
    }
    if actual_candidate != expected_with_result:
        raise SystemExit("isolated dense sensitivity candidate drifted")
    if candidate_comparison["elementwise_exact"]:
        exact_candidate_ids.append(index)
if not (
    actual_sensitivity.get("baseline_dense_update_bits") == 47808
    and actual_sensitivity.get("candidate_count") == len(expected_sensitivity)
    and actual_sensitivity.get("exact_candidate_ids") == exact_candidate_ids
    and actual_sensitivity.get("hidden_index") == 2795
    and actual_sensitivity.get("partial_bit_delta_limit") == 32
    and np.array_equal(
        sensitivity_dense,
        np.asarray(
            [item["dense_update_bits"] for item in expected_sensitivity],
            dtype=np.uint16,
        ),
    )
    and np.array_equal(
        sensitivity_rank,
        np.asarray(
            [item["representative_model_rank"] for item in expected_sensitivity],
            dtype=np.int16,
        ),
    )
    and np.array_equal(
        sensitivity_delta,
        np.asarray(
            [
                item["representative_partial_bit_delta"]
                for item in expected_sensitivity
            ],
            dtype=np.int16,
        ),
    )
    and np.array_equal(
        sensitivity_partial_bits,
        np.asarray(
            [
                item["representative_mutated_partial_bits"]
                for item in expected_sensitivity
            ],
            dtype=np.int32,
        ),
    )
):
    raise SystemExit("isolated dense sensitivity manifest drifted")
exact = runner["exact"]
if not (
    runner.get("classification")
    == (
        "isolated_virtual_contractions_exact"
        if exact
        else "isolated_virtual_contractions_nonexact"
    )
    and comparison["elementwise_exact"] is exact
):
    raise SystemExit("isolated dense classification drifted")

isolated_hlo = runner.get("hlo", {}).get("isolated_dense", {})
rms_hlo = runner.get("hlo", {}).get("layer1_replay", {})
isolated_stable = isolated_hlo.get("stablehlo_contract", {})
isolated_optimized = isolated_hlo.get("optimized_contract", {})
rms_stable = rms_hlo.get("stablehlo_contract", {})
rms_optimized = rms_hlo.get("optimized_contract", {})
expected_hlo_files = {
    "isolated_dense": (
        root / "hlo/isolated_dense.stablehlo.mlir",
        root / "hlo/isolated_dense.optimized_hlo.txt",
    ),
    "layer1_replay": (
        root / "hlo/layer1_replay.stablehlo.mlir",
        root / "hlo/layer1_replay.optimized_hlo.txt",
    ),
}
if not (
    isolated_stable.get("accepted_gate_dequant_fusion") is True
    and isolated_stable.get("accepted_gate_singleton") is True
    and isolated_stable.get("passed") is True
    and isolated_stable.get("violations") == []
    and isolated_stable.get("convolution_count") == 2
    and isolated_stable.get("exact_result_binding") is True
    and isolated_stable.get("gate_up_layout_constraint_count") == 0
    and isolated_stable.get("matched_virtual_shards") == [0]
    and isolated_stable.get("virtual_contractions_per_chip") == 1
    and not any(isolated_stable.get("collective_counts", {}).values())
    and isolated_optimized.get("accepted_gate_dequant_fusion") is True
    and isolated_optimized.get("accepted_gate_singleton") is True
    and isolated_optimized.get("passed") is True
    and isolated_optimized.get("violations") == []
    and isolated_optimized.get("accepted_gate_up_schedule") is True
    and isolated_optimized.get("accepted_down_schedule") is True
    and isolated_optimized.get("async_collectives") == []
    and isolated_optimized.get("collective_count") == 0
    and isolated_optimized.get("convolution_count") == 2
    and isolated_optimized.get("gate_up_convolution_count") == 1
    and isolated_optimized.get("down_convolution_count") == 1
    and isolated_optimized.get("exact_accepted_kernel_geometry") is True
    and isolated_optimized.get("exact_accepted_weight_layout") is True
    and isolated_optimized.get("exact_activation_graph") is True
    and isolated_optimized.get("exact_carried_residual_binding") is True
    and isolated_optimized.get(
        "exact_gate_singleton_external_boundary"
    ) is True
    and isolated_optimized.get("exact_gate_dequant_fusion_boundary") is True
    and isolated_optimized.get("exact_packed_weight_lineage") is True
    and isolated_optimized.get("exact_result_binding") is True
    and isolated_optimized.get("final_dense_layout") is True
    and isolated_optimized.get("dense_envelope") is True
    and isolated_optimized.get("isolated_dense") is True
    and isolated_optimized.get("live_rows") == 1
    and isolated_optimized.get("num_partitions") == 4
    and isolated_optimized.get("num_replicas") in (None, 1)
    and isolated_optimized.get("performance_claim") is False
    and rms_stable.get("passed") is True
    and rms_stable.get("violations") == []
    and rms_stable.get("exact_result_binding") is True
    and rms_stable.get("split_layer1_rms") is False
    and rms_optimized.get("passed") is True
    and rms_optimized.get("violations") == []
    and rms_optimized.get("collective_count") == 1
    and rms_optimized.get("exact_result_binding") is True
    and rms_optimized.get("split_layer1_rms") is False
):
    raise SystemExit("isolated dense HLO contract drifted")
for name, (stable_path, optimized_path) in expected_hlo_files.items():
    if not (
        runner["hlo"][name]["stablehlo_sha256"] == file_sha(stable_path)
        and runner["hlo"][name]["optimized_sha256"] == file_sha(optimized_path)
    ):
        raise SystemExit("isolated dense HLO file hash drifted")

summary = {
    "accepted_gate_dequant_fusion": True,
    "accepted_gate_singleton": True,
    "artifact_kind": runner["artifact_kind"],
    "classification": runner["classification"],
    "code_hash": pin,
    "control_admissible": True,
    "elapsed_seconds": int(elapsed_text),
    "exact": exact,
    "exact_arms": runner["exact_arms"],
    "final_layout_records": expected_records,
    "gate_singleton_discriminator": expected_gate_singleton_discriminator,
    "hlo": {
        name: {
            "optimized_sha256": runner["hlo"][name]["optimized_sha256"],
            "stablehlo_sha256": runner["hlo"][name]["stablehlo_sha256"],
        }
        for name in ("isolated_dense", "layer1_replay")
    },
    "isolated_layer1_comparison": comparison,
    "isolated_layer1_sha256": array_sha(isolated),
    "partial_comparison": expected_partial,
    "performance_claim": False,
    "position": 8155,
    "sensitivity": actual_sensitivity,
    "results_db_run_id": None,
    "run_tag": run_tag,
    "runner_sha256": file_sha(runner_path),
    "source": expected_source,
    "status": "SUCCESS",
    "tensor_sha256": file_sha(tensor_path),
    "virtual_contractions_per_chip": 1,
    "virtual_rank_batches": 8,
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("ISOLATED_DENSE_REPLAY_VALID")
PY
}
if [[ $DENSE_CAPTURE_PARTIALS == 1 ]]; then
  capture_partials_probe_args=(--capture-partials)
fi

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing layer-0 arithmetic probe outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing projection/reduction probe from a dirty worktree" >&2
  exit 2
}
[[ -r $RESULTS_DB && ! -e $RUN_DIR ]] || {
  echo "results DB missing or append-only run path already exists" >&2
  exit 2
}
mkdir -p "$RUN_DIR/hlo"

say() {
  echo "[projection-reduction $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
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

require_sha() {
  local path=$1 expected=$2 label=$3
  [[ -r $path && $(sha256sum "$path" | awk '{print $1}') == "$expected" ]] || {
    say "ABORT: $label prerequisite hash drifted"
    exit 1
  }
}

require_remote_sha() {
  local uri=$1 expected=$2 label=$3
  [[ $(gcloud storage cat "$uri" | sha256sum | awk '{print $1}') == "$expected" ]] || {
    say "ABORT: $label remote hash drifted"
    exit 1
  }
}

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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[p]robe_layer0_projection_reduction[.]py|[p]robe_layer0_dense_convolution[.]py|[p]robe_layer0_captured_rms[.]py|[p]robe_layer0_isolated_dense[.]py|[p]robe_layer0_attention_arithmetic[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

post_census_done=0
terminal_success_done=0

rollback_provisional_db() {
  if [[ $CAPTURED_RMS_REPLAY == 1 || $ISOLATED_DENSE_REPLAY == 1 ]]; then
    echo "NO_PROVISIONAL_DB_RUN" >"$RUN_DIR/provisional_db_rollback.txt"
    return 0
  fi
  /home/gianl/vllm-env/bin/python - "$RESULTS_DB" "$TAG" "$PIN" \
    "$DENSE_CONVOLUTION" "$RUN_DIR" "$DB538_TENSOR_SHA" \
    "$CHECKPOINT_MANIFEST_SHA" "$HARNESS_GIT" "$FORK_GIT" \
    "$DENSE_COMPILE_ROWS" "$DENSE_LAYER1_ONLY" "$DENSE_FINAL_LAYOUT" \
    "$DENSE_ENVELOPE" "$DENSE_SPLIT_LAYER1_RMS" "$DENSE_CAPTURE_PARTIALS" \
    "$ACCEPTED_M32_HLO_SHA" \
    "$ACCEPTED_M32_SUMMARY_SHA" "$ACCEPTED_M32_SUCCESS_SHA" \
    "$ACCEPTED_M32_RAW_HLO_SHA" \
    >"$RUN_DIR/provisional_db_rollback.txt" <<'PY'
from __future__ import annotations

import json
import sqlite3
import sys

(
    db_path,
    run_tag,
    pin,
    dense_text,
    run_dir,
    db538_tensor_sha,
    checkpoint_manifest_sha,
    harness_git,
    fork_git,
    dense_compile_rows_text,
    dense_layer1_only_text,
    dense_final_layout_text,
    dense_envelope_text,
    split_layer1_rms_text,
    capture_partials_text,
    accepted_m32_hlo_sha,
    accepted_m32_summary_sha,
    accepted_m32_success_sha,
    accepted_m32_raw_hlo_sha,
) = sys.argv[1:]
dense = dense_text == "1"
compile_rows = int(dense_compile_rows_text)
layer1_only = dense_layer1_only_text == "1"
final_layout = dense_final_layout_text == "1"
dense_envelope = dense_envelope_text == "1"
split_layer1_rms = split_layer1_rms_text == "1"
capture_partials = capture_partials_text == "1"
if capture_partials:
    print("NO_PROVISIONAL_DB_RUN")
    raise SystemExit(0)
expected_final_layout_records = {
    "dense.slot_00.merged_gate_up.weight_bits_in_out": {
        "shape": [4, 8, 6144, 768],
        "dtype": "float8_e4m3fn",
        "sha256": "82c93c0fafda7afa3853e3a689aace78be61bec88c5ef779bb0e9efeb834facf",
    },
    "dense.slot_00.merged_gate_up.scale_inv_in_out": {
        "shape": [4, 8, 48, 768],
        "dtype": "float32",
        "sha256": "9b4bfee8b15d04545a277ba4e1ea0d3b73427a0f2cebf0a5f5fa4d7ab32bf8b3",
    },
    "dense.slot_00.down.weight_bits_in_out": {
        "shape": [4, 8, 384, 6144],
        "dtype": "float8_e4m3fn",
        "sha256": "8654c1ebb6f0ef29b1d3919699c08ca2b81e058bf3f9cd0827a9889994d72f7e",
    },
    "dense.slot_00.down.scale_inv_in_out": {
        "shape": [4, 8, 3, 6144],
        "dtype": "float32",
        "sha256": "f37e87987745d263d152ff17415d21d572fa3c73ad5888207f1aa44916f74bdd",
    },
}
final_layout_records_sha256 = (
    "45bfd64e45956627516c36603ccee53d85d713ce6476761e68f1170c51901ba4"
)
runner_path = __import__("pathlib").Path(run_dir) / "runner.json"
if not runner_path.is_file():
    raise SystemExit("refusing rollback without the producing runner")
runner = json.loads(runner_path.read_text())
dense_arm = (
    "accepted_m32_dense_envelope_split_rms"
    if split_layer1_rms
    else "accepted_m32_dense_envelope_cross_layer"
    if dense_envelope
    else "accepted_m32_dense_final_layout_cross_layer"
    if final_layout
    else "accepted_m32_dense_cross_layer"
    if layer1_only
    else (
        "accepted_m32_dense_convolution"
        if compile_rows == 32
        else "accepted_dense_convolution"
    )
)
model = (
    (
        "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-envelope-split-rms"
        if split_layer1_rms
        else "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-envelope-cross-layer"
        if dense_envelope
        else "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-final-layout-cross-layer"
        if final_layout
        else "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-m32-cross-layer"
        if layer1_only
        else (
            "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-m32-convolution"
            if compile_rows == 32
            else "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-convolution"
        )
    )
    if dense
    else "zai-org/GLM-5.2-FP8:greenfield-layer0-projection-reduction"
)
revision = (
    (
        "native-jax-accepted-dense-envelope-split-rms-v1"
        if split_layer1_rms
        else "native-jax-accepted-dense-envelope-v1"
        if dense_envelope
        else "native-jax-accepted-dense-final-layout-v1"
        if final_layout
        else "native-jax-db542-dense-cross-layer-v1"
        if layer1_only
        else (
            "native-jax-db532-dense-m32-discriminator-v1"
            if compile_rows == 32
            else "native-jax-db538-dense-convolution-v1"
        )
    )
    if dense
    else "native-jax-db537-strategy-nd-v1"
)
benchmark = (
    (
        "greenfield_layer0_dense_envelope_split_rms"
        if split_layer1_rms
        else "greenfield_layer0_dense_envelope_cross_layer"
        if dense_envelope
        else "greenfield_layer0_dense_final_layout_cross_layer"
        if final_layout
        else "greenfield_layer0_dense_m32_cross_layer"
        if layer1_only
        else (
            "greenfield_layer0_dense_m32_convolution"
            if compile_rows == 32
            else "greenfield_layer0_dense_convolution"
        )
    )
    if dense
    else "greenfield_layer0_projection_reduction"
)
note = (
    (
        "Protected layer-1 accepted split-RMS schedule discriminator; no performance claim."
        if split_layer1_rms
        else "Protected layer-0 accepted dense fusion-envelope discriminator; no performance claim."
        if dense_envelope
        else "Protected layer-0 accepted final-layout dense cross-layer discriminator; no performance claim."
        if final_layout
        else "Protected layer-0 accepted-M32 dense cross-layer fusion discriminator; no performance claim."
        if layer1_only
        else (
            "Protected layer-0 accepted-M32 dense arithmetic discriminator; no performance claim."
            if compile_rows == 32
            else "Protected layer-0 dense convolution discriminator; no performance claim."
        )
    )
    if dense
    else "Protected layer-0 projection/reduction discriminator; no performance claim."
)
engine = (
    (
        "greenfield_dense_envelope_split_rms_probe"
        if split_layer1_rms
        else "greenfield_dense_envelope_cross_layer_probe"
        if dense_envelope
        else "greenfield_dense_final_layout_cross_layer_probe"
        if final_layout
        else "greenfield_dense_m32_cross_layer_probe"
        if layer1_only
        else (
            "greenfield_dense_m32_convolution_probe"
            if compile_rows == 32
            else "greenfield_dense_convolution_probe"
        )
    )
    if dense
    else "greenfield_projection_reduction_probe"
)
expected_environment = {
    "GLM_ENGINE": engine,
    "greenfield_code_hash": pin,
    "greenfield_run_tag": run_tag,
    "classification": runner.get("classification"),
    **(
        {
            "checkpoint_manifest_sha256": checkpoint_manifest_sha,
            "db538_tensor_sha256": db538_tensor_sha,
            "compile_rows": compile_rows,
            "dense_envelope": dense_envelope,
            "final_dense_layout": final_layout,
            "split_layer1_rms": split_layer1_rms,
            "result_mode": (
                "layer1_only" if layer1_only else "dense_and_layer1"
            ),
            **(
                {
                    "attention_update_sha256": "68afed86921584fb673abb11a563e359a1533210ec2483e71ee79b88c2b0bde7",
                    "combined_residual_sha256": "02d045b9a0ec5ab22a711bd6a964564f707be0848683381104e83331020e31a3",
                    "normalized_mlp_sha256": "082125fead43b25f10686705c1b6473153f4092dd5bc476f8e01a86629f0758f",
                }
                if dense_envelope
                else {}
            ),
            **(
                {
                    "final_layout_records_sha256": (
                        final_layout_records_sha256
                    ),
                    "exact_accepted_kernel_geometry": True,
                    "scheduled_kernel_geometry_required": True,
                }
                if final_layout
                else {}
            ),
            **(
                {
                    "exact_accepted_scheduled_reduction": True,
                    "split_recompute_exact": True,
                    "split_output_fusion_exact": True,
                }
                if split_layer1_rms
                else {}
            ),
            **(
                {
                    "accepted_m32_hlo_raw_sha256": accepted_m32_raw_hlo_sha,
                    "accepted_m32_hlo_sha256": accepted_m32_hlo_sha,
                    "accepted_m32_summary_sha256": accepted_m32_summary_sha,
                    "accepted_m32_success_sha256": accepted_m32_success_sha,
                }
                if compile_rows == 32
                else {}
            ),
        }
        if dense
        else {
            "attention_tensor_sha256": runner.get("source", {}).get(
                "attention_arithmetic_sha256"
            ),
            "association_analysis_sha256": runner.get("source", {}).get(
                "association_analysis_sha256"
            ),
        }
    ),
}
if (
    runner.get("status") != "SUCCESS"
    or runner.get("code_hash") != pin
    or (
        dense
        and (
            runner.get("artifact_kind")
            != "glm52_layer0_dense_convolution_probe"
            or runner.get("source", {}).get("db538_tensor_sha256")
            != db538_tensor_sha
            or runner.get("compile_rows") != compile_rows
            or runner.get("dense_envelope") is not dense_envelope
            or runner.get("final_dense_layout") is not final_layout
            or runner.get("split_layer1_rms") is not split_layer1_rms
            or runner.get("hlo", {})
            .get("stablehlo_contract", {})
            .get("split_layer1_rms")
            is not split_layer1_rms
            or runner.get("hlo", {})
            .get("optimized_contract", {})
            .get("split_layer1_rms")
            is not split_layer1_rms
            or runner.get("final_layout_records")
            != (expected_final_layout_records if final_layout else {})
            or (
                final_layout
                and (
                    runner.get("hlo", {})
                    .get("optimized_contract", {})
                    .get("scheduled_kernel_geometry_required")
                    is not True
                    or runner.get("hlo", {})
                    .get("optimized_contract", {})
                    .get("exact_accepted_kernel_geometry")
                    is not True
                    or (
                        split_layer1_rms
                        and (
                            runner.get("hlo", {})
                            .get("optimized_contract", {})
                            .get("lineage", {})
                            .get("rmsnorm_contract", {})
                            .get("split_recompute_exact")
                            is not True
                            or runner.get("hlo", {})
                            .get("optimized_contract", {})
                            .get("lineage", {})
                            .get("rmsnorm_contract", {})
                            .get("exact_accepted_scheduled_reduction")
                            is not True
                            or runner.get("hlo", {})
                            .get("optimized_contract", {})
                            .get("lineage", {})
                            .get("rmsnorm_contract", {})
                            .get("split_output_fusion_exact")
                            is not True
                        )
                    )
                    or set(
                        runner.get("hlo", {})
                        .get("optimized_contract", {})
                        .get("accepted_weight_layouts", {})
                    )
                    != {"gate_up", "down"}
                    or any(
                        len(records) != 8
                        for records in runner.get("hlo", {})
                        .get("optimized_contract", {})
                        .get("accepted_weight_layouts", {})
                        .values()
                    )
                    or any(
                        record.get("exact_convolution_tiling") is not True
                        for records in runner.get("hlo", {})
                        .get("optimized_contract", {})
                        .get("accepted_weight_layouts", {})
                        .values()
                        for record in records
                    )
                )
            )
            or runner.get("live_rows") != 1
            or runner.get("diagnostic_dead_rows") != compile_rows - 1
            or runner.get("result_mode")
            != ("layer1_only" if layer1_only else "dense_and_layer1")
            or runner.get("classification")
            not in {
                f"{dense_arm}_exact",
                f"{dense_arm}_nonexact",
            }
            or (
                compile_rows == 32
                and runner.get("source", {})
                != {
                    **runner.get("source", {}),
                    "accepted_m32_hlo_raw_sha256": accepted_m32_raw_hlo_sha,
                    "accepted_m32_hlo_sha256": accepted_m32_hlo_sha,
                    "accepted_m32_summary_sha256": accepted_m32_summary_sha,
                    "accepted_m32_success_sha256": accepted_m32_success_sha,
                }
            )
            or (
                dense_envelope
                and (
                    runner.get("source", {}).get("attention_update_sha256")
                    != "68afed86921584fb673abb11a563e359a1533210ec2483e71ee79b88c2b0bde7"
                    or runner.get("source", {}).get("combined_residual_sha256")
                    != "02d045b9a0ec5ab22a711bd6a964564f707be0848683381104e83331020e31a3"
                    or runner.get("source", {}).get("normalized_mlp_sha256")
                    != "082125fead43b25f10686705c1b6473153f4092dd5bc476f8e01a86629f0758f"
                )
            )
        )
    )
    or (
        not dense
        and (
            runner.get("artifact_kind")
            != "glm52_layer0_projection_reduction_probe"
            or runner.get("classification")
            != (
                "exact_projection_reduction_arm_identified"
                if runner.get("exact_arms")
                else "projection_reduction_unresolved"
            )
        )
    )
):
    raise SystemExit("refusing rollback from an unauthenticated runner")
connection = sqlite3.connect(db_path)
connection.execute("BEGIN IMMEDIATE")
matches = []
for row in connection.execute(
    "SELECT run_id, model, model_revision, harness_git, fork_git, env_json, "
    "pod, note FROM runs "
    "WHERE model = ? AND model_revision = ?",
    (
        model,
        revision,
    ),
):
    environment = json.loads(row[5])
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
items = connection.execute(
    "SELECT benchmark, item_id, prompt, gold, raw_output, extracted, correct, "
    "score, n_prompt_tokens, n_gen_tokens, latency_ms, seed, finish_reason, "
    "truncated FROM items WHERE run_id = ?",
    (run_id,),
).fetchall()
summaries = connection.execute(
    "SELECT benchmark, n, metric, value, card_value, delta, note FROM summary "
    "WHERE run_id = ?",
    (run_id,),
).fetchall()
expected_prompt = (
    (
        "Sealed accepted scalar-only layer-1 RMS schedule with exact recompute."
        if split_layer1_rms
        else "Sealed exact pre-dense RMSNorm boundary with accepted dense fusion envelope."
        if dense_envelope
        else "Sealed exact StrategyND attention boundary with accepted final-layout dense fusion."
        if final_layout
        else "Sealed exact StrategyND attention boundary with layer-1-only accepted-M32 dense fusion."
        if layer1_only
        else (
            "Sealed exact StrategyND attention boundary with diagnostic accepted-M32 dense geometry."
            if compile_rows == 32
            else "Sealed exact StrategyND attention boundary at first 8K decode row."
        )
    )
    if dense
    else "Sealed exact B512 latent and layer-0 residual at first 8K decode row."
)
expected_raw = json.dumps(
    {
        "classification": runner["classification"],
        "exact_arms": runner["exact_arms"],
        "mismatch_counts": (
            {
                dense_arm: runner["layer1_comparison"]["mismatch_count"]
            }
            if dense
            else {
                name: arm["layer1_comparison"]["mismatch_count"]
                for name, arm in runner["arms"].items()
            }
        ),
    },
    sort_keys=True,
)
expected_item = (
    benchmark,
    "position8155",
    expected_prompt,
    "Exact accepted BF16 layer-1 normalized hidden [6144].",
    expected_raw,
    ",".join(runner["exact_arms"]) or "none",
    int(bool(runner["exact_arms"])),
    float(bool(runner["exact_arms"])),
    None,
    None,
    None,
    None,
    None,
    None,
)
item_state_valid = not items or items == [expected_item]
summary_state_valid = not summaries or summaries == [(
    benchmark,
    1,
    "probe_contract_valid",
    1.0,
    None,
    None,
    "Diagnostic layer-0 arithmetic classification only; no decoder claim.",
)]
if (
    run[1] != model
    or run[2] != revision
    or run[3] != harness_git
    or run[4] != fork_git
    or environment != expected_environment
    or run[6] != "db-v4-64-od"
    or run[7] != note
    or not item_state_valid
    or not summary_state_valid
    or (summaries and not items)
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
    say "FAILED status=$status; partial evidence preserved at $RUN_DIR"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR PIN=$PIN"
if [[ $CAPTURED_RMS_REPLAY == 1 || $ISOLATED_DENSE_REPLAY == 1 ]]; then
  require_sha "$CAPTURED_RMS_SOURCE_TENSOR" "$CAPTURED_RMS_SOURCE_TENSOR_SHA" "captured RMS source tensor"
  require_sha "$CAPTURED_RMS_SOURCE_RUNNER" "$CAPTURED_RMS_SOURCE_RUNNER_SHA" "captured RMS source runner"
  require_sha "$CAPTURED_RMS_SOURCE_SUMMARY" "$CAPTURED_RMS_SOURCE_SUMMARY_SHA" "captured RMS source summary"
  require_sha "$CAPTURED_RMS_SOURCE_SUCCESS" "$CAPTURED_RMS_SOURCE_SUCCESS_SHA" "captured RMS source SUCCESS"
  require_sha "$CAPTURED_RMS_DB548_TENSOR" "$CAPTURED_RMS_DB548_TENSOR_SHA" "captured RMS DB548 tensor"
  require_sha "$CAPTURED_RMS_DB548_RUNNER" "$CAPTURED_RMS_DB548_RUNNER_SHA" "captured RMS DB548 runner"
  require_sha "$CAPTURED_RMS_DB548_SUMMARY" "$CAPTURED_RMS_DB548_SUMMARY_SHA" "captured RMS DB548 summary"
  require_sha "$CAPTURED_RMS_DB548_SUCCESS" "$CAPTURED_RMS_DB548_SUCCESS_SHA" "captured RMS DB548 SUCCESS"
  require_sha "$CAPTURED_RMS_DB548_HLO" "$CAPTURED_RMS_DB548_HLO_SHA" "captured RMS DB548 HLO"
  require_sha "$CAPTURED_RMS_DB549_TENSOR" "$CAPTURED_RMS_DB549_TENSOR_SHA" "captured RMS DB549 tensor"
  require_sha "$CAPTURED_RMS_DB549_RUNNER" "$CAPTURED_RMS_DB549_RUNNER_SHA" "captured RMS DB549 runner"
  require_sha "$CAPTURED_RMS_DB549_SUMMARY" "$CAPTURED_RMS_DB549_SUMMARY_SHA" "captured RMS DB549 summary"
  require_sha "$CAPTURED_RMS_DB549_SUCCESS" "$CAPTURED_RMS_DB549_SUCCESS_SHA" "captured RMS DB549 SUCCESS"
  require_sha "$CAPTURED_RMS_DB549_HLO" "$CAPTURED_RMS_DB549_HLO_SHA" "captured RMS DB549 HLO"

  require_remote_sha "$CAPTURED_RMS_SOURCE_REMOTE/dense_partial_capture.npz" "$CAPTURED_RMS_SOURCE_TENSOR_SHA" "captured RMS source tensor"
  require_remote_sha "$CAPTURED_RMS_SOURCE_REMOTE/runner.json" "$CAPTURED_RMS_SOURCE_RUNNER_SHA" "captured RMS source runner"
  require_remote_sha "$CAPTURED_RMS_SOURCE_REMOTE/summary.json" "$CAPTURED_RMS_SOURCE_SUMMARY_SHA" "captured RMS source summary"
  require_remote_sha "$CAPTURED_RMS_SOURCE_REMOTE/SUCCESS" "$CAPTURED_RMS_SOURCE_SUCCESS_SHA" "captured RMS source SUCCESS"
  require_remote_sha "$CAPTURED_RMS_DB548_REMOTE/dense_envelope_cross_layer.npz" "$CAPTURED_RMS_DB548_TENSOR_SHA" "captured RMS DB548 tensor"
  require_remote_sha "$CAPTURED_RMS_DB548_REMOTE/runner.json" "$CAPTURED_RMS_DB548_RUNNER_SHA" "captured RMS DB548 runner"
  require_remote_sha "$CAPTURED_RMS_DB548_REMOTE/summary.json" "$CAPTURED_RMS_DB548_SUMMARY_SHA" "captured RMS DB548 summary"
  require_remote_sha "$CAPTURED_RMS_DB548_REMOTE/SUCCESS" "$CAPTURED_RMS_DB548_SUCCESS_SHA" "captured RMS DB548 SUCCESS"
  require_remote_sha "$CAPTURED_RMS_DB548_REMOTE/hlo/dense_convolution.optimized_hlo.txt" "$CAPTURED_RMS_DB548_HLO_SHA" "captured RMS DB548 HLO"
  require_remote_sha "$CAPTURED_RMS_DB549_REMOTE/dense_envelope_split_rms.npz" "$CAPTURED_RMS_DB549_TENSOR_SHA" "captured RMS DB549 tensor"
  require_remote_sha "$CAPTURED_RMS_DB549_REMOTE/runner.json" "$CAPTURED_RMS_DB549_RUNNER_SHA" "captured RMS DB549 runner"
  require_remote_sha "$CAPTURED_RMS_DB549_REMOTE/summary.json" "$CAPTURED_RMS_DB549_SUMMARY_SHA" "captured RMS DB549 summary"
  require_remote_sha "$CAPTURED_RMS_DB549_REMOTE/SUCCESS" "$CAPTURED_RMS_DB549_SUCCESS_SHA" "captured RMS DB549 SUCCESS"
  require_remote_sha "$CAPTURED_RMS_DB549_REMOTE/hlo/dense_convolution.optimized_hlo.txt" "$CAPTURED_RMS_DB549_HLO_SHA" "captured RMS DB549 HLO"
  if [[ $ISOLATED_DENSE_REPLAY == 1 ]]; then
    require_sha "$ACCEPTED_M32_HLO" "$ACCEPTED_M32_HLO_SHA" "accepted M32 HLO"
    require_sha "$ACCEPTED_M32_SUMMARY" "$ACCEPTED_M32_SUMMARY_SHA" "accepted M32 summary"
    require_sha "$ACCEPTED_M32_SUCCESS" "$ACCEPTED_M32_SUCCESS_SHA" "accepted M32 SUCCESS"
    require_sha "$CHECKPOINT_ROOT/SUCCESS" "$CHECKPOINT_SUCCESS_SHA" "checkpoint SUCCESS"
    require_sha "$CHECKPOINT_ROOT/runtime_manifest.json" "$CHECKPOINT_MANIFEST_SHA" "checkpoint manifest"
    require_remote_sha "$ACCEPTED_M32_REMOTE/accepted_decode_projection_lowering/jit_step_fun_impl.m32.after_codegen_hlo.txt.gz" "$ACCEPTED_M32_HLO_SHA" "accepted M32 HLO"
    require_remote_sha "$ACCEPTED_M32_REMOTE/accepted_decode_projection_lowering/summary.json" "$ACCEPTED_M32_SUMMARY_SHA" "accepted M32 summary"
    require_remote_sha "$ACCEPTED_M32_REMOTE/SUCCESS" "$ACCEPTED_M32_SUCCESS_SHA" "accepted M32 SUCCESS"
    require_remote_sha "$CHECKPOINT_REMOTE/SUCCESS" "$CHECKPOINT_SUCCESS_SHA" "checkpoint SUCCESS"
    require_remote_sha "$CHECKPOINT_REMOTE/runtime_manifest.json" "$CHECKPOINT_MANIFEST_SHA" "checkpoint manifest"
  fi
else
require_sha "$ATTENTION_TENSOR" "$ATTENTION_TENSOR_SHA" "DB537 tensor"
require_sha "$ATTENTION_RUNNER" "$ATTENTION_RUNNER_SHA" "DB537 runner"
require_sha "$ATTENTION_SUCCESS" "$ATTENTION_SUCCESS_SHA" "DB537 SUCCESS"
require_sha "$INGREDIENT_NPZ" "$INGREDIENT_NPZ_SHA" "ingredient tensor"
require_sha "$INGREDIENT_JSON" "$INGREDIENT_JSON_SHA" "ingredient contract"
require_sha "$ACCEPTED_NPZ" "$ACCEPTED_NPZ_SHA" "accepted projection"
require_sha "$ACCEPTED_JSON" "$ACCEPTED_JSON_SHA" "accepted capture"
require_sha "$ACCEPTED_SUCCESS" "$ACCEPTED_SUCCESS_SHA" "accepted SUCCESS"
require_sha "$LAYER1_REFERENCE" "$LAYER1_REFERENCE_SHA" "layer-1 reference"
require_sha "$LAYER1_COMPARISON" "$LAYER1_COMPARISON_SHA" "layer-1 comparison"
require_sha "$LAYER1_SEAL" "$LAYER1_SEAL_SHA" "layer-1 diagnostic seal"
require_sha "$ASSOCIATION_ANALYSIS" "$ASSOCIATION_ANALYSIS_SHA" "association analysis"
require_sha "$ASSOCIATION_SUCCESS" "$ASSOCIATION_SUCCESS_SHA" "association SUCCESS"
require_sha "$CHECKPOINT_ROOT/SUCCESS" "$CHECKPOINT_SUCCESS_SHA" "checkpoint SUCCESS"
require_sha "$CHECKPOINT_ROOT/runtime_manifest.json" "$CHECKPOINT_MANIFEST_SHA" "checkpoint manifest"
if [[ $DENSE_CONVOLUTION == 1 ]]; then
  require_sha "$DB538_RUNNER" "$DB538_RUNNER_SHA" "DB538 runner"
  require_sha "$DB538_TENSOR" "$DB538_TENSOR_SHA" "DB538 tensor"
  require_sha "$DB538_SUMMARY" "$DB538_SUMMARY_SHA" "DB538 summary"
  require_sha "$DB538_SUCCESS" "$DB538_SUCCESS_SHA" "DB538 SUCCESS"
  if [[ $DENSE_COMPILE_ROWS == 32 ]]; then
    require_sha "$ACCEPTED_M32_HLO" "$ACCEPTED_M32_HLO_SHA" "accepted M32 HLO"
    require_sha "$ACCEPTED_M32_SUMMARY" "$ACCEPTED_M32_SUMMARY_SHA" "accepted M32 summary"
    require_sha "$ACCEPTED_M32_SUCCESS" "$ACCEPTED_M32_SUCCESS_SHA" "accepted M32 SUCCESS"
  fi
  /home/gianl/vllm-env/bin/python - "$RESULTS_DB" "$DB538_RUN_ID" \
    "$DB538_TAG" "$DB538_CODE_HASH" <<'PY'
import json
import sqlite3
import sys

db_path, run_id_text, run_tag, code_hash = sys.argv[1:]
run_id = int(run_id_text)
connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
run = connection.execute(
    "SELECT model, model_revision, env_json, note FROM runs WHERE run_id = ?",
    (run_id,),
).fetchone()
items = connection.execute(
    "SELECT benchmark, item_id, gold, correct, score FROM items WHERE run_id = ?",
    (run_id,),
).fetchall()
summaries = connection.execute(
    "SELECT benchmark, metric, value FROM summary WHERE run_id = ?",
    (run_id,),
).fetchall()
connection.close()
if run is None:
    raise SystemExit("protected DB538 run is absent")
environment = json.loads(run[2])
if (
    run[0] != "zai-org/GLM-5.2-FP8:greenfield-layer0-projection-reduction"
    or run[1] != "native-jax-db537-strategy-nd-v1"
    or run[3]
    != "Protected layer-0 projection/reduction discriminator; no performance claim."
    or environment.get("greenfield_run_tag") != run_tag
    or environment.get("greenfield_code_hash") != code_hash
    or environment.get("classification") != "projection_reduction_unresolved"
    or items != [(
        "greenfield_layer0_projection_reduction",
        "position8155",
        "Exact accepted BF16 layer-1 normalized hidden [6144].",
        0,
        0.0,
    )]
    or summaries != [(
        "greenfield_layer0_projection_reduction",
        "probe_contract_valid",
        1.0,
    )]
):
    raise SystemExit("protected DB538 live DB identity drifted")
PY
fi

require_remote_sha "$ATTENTION_REMOTE/attention_arithmetic.npz" "$ATTENTION_TENSOR_SHA" "DB537 tensor"
require_remote_sha "$ATTENTION_REMOTE/runner.json" "$ATTENTION_RUNNER_SHA" "DB537 runner"
require_remote_sha "$ATTENTION_REMOTE/SUCCESS" "$ATTENTION_SUCCESS_SHA" "DB537 SUCCESS"
require_remote_sha "$INGREDIENT_REMOTE/position_8155_ingredients.npz" "$INGREDIENT_NPZ_SHA" "ingredient tensor"
require_remote_sha "$INGREDIENT_REMOTE/contract.json" "$INGREDIENT_JSON_SHA" "ingredient contract"
require_remote_sha "$ACCEPTED_REMOTE/attention_projection_capture/attention_projection.npz" "$ACCEPTED_NPZ_SHA" "accepted projection"
require_remote_sha "$ACCEPTED_REMOTE/attention_projection_capture/capture.json" "$ACCEPTED_JSON_SHA" "accepted capture"
require_remote_sha "$ACCEPTED_REMOTE/SUCCESS" "$ACCEPTED_SUCCESS_SHA" "accepted SUCCESS"
require_remote_sha "$LAYER1_REMOTE/internals.npz" "$LAYER1_REFERENCE_SHA" "layer-1 reference"
require_remote_sha "$LAYER1_REMOTE/comparison.json" "$LAYER1_COMPARISON_SHA" "layer-1 comparison"
require_remote_sha "$LAYER1_REMOTE/diagnostic_seal.json" "$LAYER1_SEAL_SHA" "layer-1 diagnostic seal"
require_remote_sha "$ASSOCIATION_REMOTE/association/analysis.json" "$ASSOCIATION_ANALYSIS_SHA" "association analysis"
require_remote_sha "$ASSOCIATION_REMOTE/SUCCESS" "$ASSOCIATION_SUCCESS_SHA" "association SUCCESS"
require_remote_sha "$CHECKPOINT_REMOTE/SUCCESS" "$CHECKPOINT_SUCCESS_SHA" "checkpoint SUCCESS"
require_remote_sha "$CHECKPOINT_REMOTE/runtime_manifest.json" "$CHECKPOINT_MANIFEST_SHA" "checkpoint manifest"
if [[ $DENSE_CONVOLUTION == 1 ]]; then
  require_remote_sha "$DB538_REMOTE/runner.json" "$DB538_RUNNER_SHA" "DB538 runner"
  require_remote_sha "$DB538_REMOTE/projection_reduction.npz" "$DB538_TENSOR_SHA" "DB538 tensor"
  require_remote_sha "$DB538_REMOTE/summary.json" "$DB538_SUMMARY_SHA" "DB538 summary"
  require_remote_sha "$DB538_REMOTE/SUCCESS" "$DB538_SUCCESS_SHA" "DB538 SUCCESS"
  if [[ $DENSE_COMPILE_ROWS == 32 ]]; then
    require_remote_sha "$ACCEPTED_M32_REMOTE/accepted_decode_projection_lowering/jit_step_fun_impl.m32.after_codegen_hlo.txt.gz" "$ACCEPTED_M32_HLO_SHA" "accepted M32 HLO"
    require_remote_sha "$ACCEPTED_M32_REMOTE/accepted_decode_projection_lowering/summary.json" "$ACCEPTED_M32_SUMMARY_SHA" "accepted M32 summary"
    require_remote_sha "$ACCEPTED_M32_REMOTE/SUCCESS" "$ACCEPTED_M32_SUCCESS_SHA" "accepted M32 SUCCESS"
  fi
fi
fi

strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

started=$(date +%s)
if [[ $ISOLATED_DENSE_REPLAY == 1 ]]; then
  say "replaying one layer-0 virtual contraction per physical chip"
  (
    cd "$WORKTREE"
    JAX_PLATFORMS=tpu \
      TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
      TPU_PROCESS_BOUNDS=1,1,1 \
      TPU_VISIBLE_DEVICES=0,1,2,3 \
      PYTHONPATH="$WORKTREE" \
      /home/gianl/vllm-env/bin/python \
        scripts/greenfield/probe_layer0_isolated_dense.py \
        --expected-code-hash "$PIN" \
        --capture-tensor "$CAPTURED_RMS_SOURCE_TENSOR" \
        --capture-tensor-sha256 "$CAPTURED_RMS_SOURCE_TENSOR_SHA" \
        --capture-runner "$CAPTURED_RMS_SOURCE_RUNNER" \
        --capture-runner-sha256 "$CAPTURED_RMS_SOURCE_RUNNER_SHA" \
        --capture-summary "$CAPTURED_RMS_SOURCE_SUMMARY" \
        --capture-summary-sha256 "$CAPTURED_RMS_SOURCE_SUMMARY_SHA" \
        --capture-success "$CAPTURED_RMS_SOURCE_SUCCESS" \
        --capture-success-sha256 "$CAPTURED_RMS_SOURCE_SUCCESS_SHA" \
        --db548-tensor "$CAPTURED_RMS_DB548_TENSOR" \
        --db548-tensor-sha256 "$CAPTURED_RMS_DB548_TENSOR_SHA" \
        --db548-runner "$CAPTURED_RMS_DB548_RUNNER" \
        --db548-runner-sha256 "$CAPTURED_RMS_DB548_RUNNER_SHA" \
        --db548-summary "$CAPTURED_RMS_DB548_SUMMARY" \
        --db548-summary-sha256 "$CAPTURED_RMS_DB548_SUMMARY_SHA" \
        --db548-success "$CAPTURED_RMS_DB548_SUCCESS" \
        --db548-success-sha256 "$CAPTURED_RMS_DB548_SUCCESS_SHA" \
        --db548-hlo "$CAPTURED_RMS_DB548_HLO" \
        --db548-hlo-sha256 "$CAPTURED_RMS_DB548_HLO_SHA" \
        --db549-tensor "$CAPTURED_RMS_DB549_TENSOR" \
        --db549-tensor-sha256 "$CAPTURED_RMS_DB549_TENSOR_SHA" \
        --db549-runner "$CAPTURED_RMS_DB549_RUNNER" \
        --db549-runner-sha256 "$CAPTURED_RMS_DB549_RUNNER_SHA" \
        --db549-summary "$CAPTURED_RMS_DB549_SUMMARY" \
        --db549-summary-sha256 "$CAPTURED_RMS_DB549_SUMMARY_SHA" \
        --db549-success "$CAPTURED_RMS_DB549_SUCCESS" \
        --db549-success-sha256 "$CAPTURED_RMS_DB549_SUCCESS_SHA" \
        --db549-hlo "$CAPTURED_RMS_DB549_HLO" \
        --db549-hlo-sha256 "$CAPTURED_RMS_DB549_HLO_SHA" \
        --accepted-m32-hlo "$ACCEPTED_M32_HLO" \
        --accepted-m32-hlo-sha256 "$ACCEPTED_M32_HLO_SHA" \
        --accepted-m32-summary "$ACCEPTED_M32_SUMMARY" \
        --accepted-m32-summary-sha256 "$ACCEPTED_M32_SUMMARY_SHA" \
        --accepted-m32-success "$ACCEPTED_M32_SUCCESS" \
        --accepted-m32-success-sha256 "$ACCEPTED_M32_SUCCESS_SHA" \
        --checkpoint-root "$CHECKPOINT_ROOT" \
        --checkpoint-manifest-sha256 "$CHECKPOINT_MANIFEST_SHA" \
        --output "$RUN_DIR/runner.json" \
        --tensor-output "$RUN_DIR/$TENSOR_BASENAME" \
        --hlo-dir "$RUN_DIR/hlo"
  ) >"$RUN_DIR/runner.log" 2>&1
elif [[ $CAPTURED_RMS_REPLAY == 1 ]]; then
  say "replaying sealed dense partials through isolated layer-1 RMS arms"
  (
    cd "$WORKTREE"
    JAX_PLATFORMS=tpu \
      TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
      TPU_PROCESS_BOUNDS=1,1,1 \
      TPU_VISIBLE_DEVICES=0,1,2,3 \
      PYTHONPATH="$WORKTREE" \
      /home/gianl/vllm-env/bin/python \
        scripts/greenfield/probe_layer0_captured_rms.py \
        --expected-code-hash "$PIN" \
        --capture-tensor "$CAPTURED_RMS_SOURCE_TENSOR" \
        --capture-tensor-sha256 "$CAPTURED_RMS_SOURCE_TENSOR_SHA" \
        --capture-runner "$CAPTURED_RMS_SOURCE_RUNNER" \
        --capture-runner-sha256 "$CAPTURED_RMS_SOURCE_RUNNER_SHA" \
        --capture-summary "$CAPTURED_RMS_SOURCE_SUMMARY" \
        --capture-summary-sha256 "$CAPTURED_RMS_SOURCE_SUMMARY_SHA" \
        --capture-success "$CAPTURED_RMS_SOURCE_SUCCESS" \
        --capture-success-sha256 "$CAPTURED_RMS_SOURCE_SUCCESS_SHA" \
        --db548-tensor "$CAPTURED_RMS_DB548_TENSOR" \
        --db548-tensor-sha256 "$CAPTURED_RMS_DB548_TENSOR_SHA" \
        --db548-runner "$CAPTURED_RMS_DB548_RUNNER" \
        --db548-runner-sha256 "$CAPTURED_RMS_DB548_RUNNER_SHA" \
        --db548-summary "$CAPTURED_RMS_DB548_SUMMARY" \
        --db548-summary-sha256 "$CAPTURED_RMS_DB548_SUMMARY_SHA" \
        --db548-success "$CAPTURED_RMS_DB548_SUCCESS" \
        --db548-success-sha256 "$CAPTURED_RMS_DB548_SUCCESS_SHA" \
        --db548-hlo "$CAPTURED_RMS_DB548_HLO" \
        --db548-hlo-sha256 "$CAPTURED_RMS_DB548_HLO_SHA" \
        --db549-tensor "$CAPTURED_RMS_DB549_TENSOR" \
        --db549-tensor-sha256 "$CAPTURED_RMS_DB549_TENSOR_SHA" \
        --db549-runner "$CAPTURED_RMS_DB549_RUNNER" \
        --db549-runner-sha256 "$CAPTURED_RMS_DB549_RUNNER_SHA" \
        --db549-summary "$CAPTURED_RMS_DB549_SUMMARY" \
        --db549-summary-sha256 "$CAPTURED_RMS_DB549_SUMMARY_SHA" \
        --db549-success "$CAPTURED_RMS_DB549_SUCCESS" \
        --db549-success-sha256 "$CAPTURED_RMS_DB549_SUCCESS_SHA" \
        --db549-hlo "$CAPTURED_RMS_DB549_HLO" \
        --db549-hlo-sha256 "$CAPTURED_RMS_DB549_HLO_SHA" \
        --output "$RUN_DIR/runner.json" \
        --tensor-output "$RUN_DIR/$TENSOR_BASENAME" \
        --hlo-dir "$RUN_DIR/hlo"
  ) >"$RUN_DIR/runner.log" 2>&1
elif [[ $DENSE_CONVOLUTION == 1 ]]; then
  say "running accepted dense-convolution arithmetic arm at M=$DENSE_COMPILE_ROWS"
  (
    cd "$WORKTREE"
    JAX_PLATFORMS=tpu \
      TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
      TPU_PROCESS_BOUNDS=1,1,1 \
      TPU_VISIBLE_DEVICES=0,1,2,3 \
      PYTHONPATH="$WORKTREE" \
      /home/gianl/vllm-env/bin/python \
        scripts/greenfield/probe_layer0_dense_convolution.py \
        --expected-code-hash "$PIN" \
        --compile-rows "$DENSE_COMPILE_ROWS" \
        "${layer1_probe_args[@]}" \
        "${final_layout_probe_args[@]}" \
        "${dense_envelope_probe_args[@]}" \
        "${split_layer1_rms_probe_args[@]}" \
        "${capture_partials_probe_args[@]}" \
        --db538-runner "$DB538_RUNNER" \
        --db538-runner-sha256 "$DB538_RUNNER_SHA" \
        --db538-tensor "$DB538_TENSOR" \
        --db538-tensor-sha256 "$DB538_TENSOR_SHA" \
        --db538-summary "$DB538_SUMMARY" \
        --db538-summary-sha256 "$DB538_SUMMARY_SHA" \
        --db538-success "$DB538_SUCCESS" \
        --db538-success-sha256 "$DB538_SUCCESS_SHA" \
        --checkpoint-root "$CHECKPOINT_ROOT" \
        --checkpoint-manifest-sha256 "$CHECKPOINT_MANIFEST_SHA" \
        "${m32_probe_args[@]}" \
        --output "$RUN_DIR/runner.json" \
        --tensor-output "$RUN_DIR/$TENSOR_BASENAME" \
        --hlo-dir "$RUN_DIR/hlo"
  ) >"$RUN_DIR/runner.log" 2>&1
else
  say "running four isolated projection/reduction association arms"
  (
    cd "$WORKTREE"
    JAX_PLATFORMS=tpu \
      TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
      TPU_PROCESS_BOUNDS=1,1,1 \
      TPU_VISIBLE_DEVICES=0,1,2,3 \
      PYTHONPATH="$WORKTREE" \
      /home/gianl/vllm-env/bin/python \
        scripts/greenfield/probe_layer0_projection_reduction.py \
        --expected-code-hash "$PIN" \
        --attention-arithmetic "$ATTENTION_TENSOR" \
        --attention-arithmetic-sha256 "$ATTENTION_TENSOR_SHA" \
        --attention-runner "$ATTENTION_RUNNER" \
        --attention-runner-sha256 "$ATTENTION_RUNNER_SHA" \
        --accepted-projection "$ACCEPTED_NPZ" \
        --accepted-projection-sha256 "$ACCEPTED_NPZ_SHA" \
        --accepted-projection-capture "$ACCEPTED_JSON" \
        --accepted-projection-capture-sha256 "$ACCEPTED_JSON_SHA" \
        --ingredients "$INGREDIENT_NPZ" \
        --ingredients-sha256 "$INGREDIENT_NPZ_SHA" \
        --ingredients-contract "$INGREDIENT_JSON" \
        --ingredients-contract-sha256 "$INGREDIENT_JSON_SHA" \
        --layer1-reference "$LAYER1_REFERENCE" \
        --layer1-reference-sha256 "$LAYER1_REFERENCE_SHA" \
        --association-analysis "$ASSOCIATION_ANALYSIS" \
        --association-analysis-sha256 "$ASSOCIATION_ANALYSIS_SHA" \
        --checkpoint-root "$CHECKPOINT_ROOT" \
        --checkpoint-manifest-sha256 "$CHECKPOINT_MANIFEST_SHA" \
        --output "$RUN_DIR/runner.json" \
        --tensor-output "$RUN_DIR/$TENSOR_BASENAME" \
        --hlo-dir "$RUN_DIR/hlo"
  ) >"$RUN_DIR/runner.log" 2>&1
fi
elapsed=$(( $(date +%s) - started ))
say "probe completed in ${elapsed}s"

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

if [[ $ISOLATED_DENSE_REPLAY == 1 ]]; then
  validate_isolated_dense_replay
elif [[ $CAPTURED_RMS_REPLAY == 1 ]]; then
  validate_captured_rms_replay
elif [[ $DENSE_CAPTURE_PARTIALS == 1 ]]; then
  /home/gianl/vllm-env/bin/python - "$RUN_DIR" "$PIN" "$elapsed" "$TAG" \
    "$DB538_RUNNER_SHA" "$DB538_TENSOR_SHA" "$DB538_SUMMARY_SHA" \
    "$DB538_SUCCESS_SHA" "$CHECKPOINT_MANIFEST_SHA" \
    "$ACCEPTED_M32_HLO_SHA" "$ACCEPTED_M32_SUMMARY_SHA" \
    "$ACCEPTED_M32_SUCCESS_SHA" "$ACCEPTED_M32_RAW_HLO_SHA" <<'PY'
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

(
    run_dir_text,
    pin,
    elapsed_text,
    run_tag,
    db538_runner_sha,
    db538_tensor_sha,
    db538_summary_sha,
    db538_success_sha,
    checkpoint_manifest_sha,
    accepted_m32_hlo_sha,
    accepted_m32_summary_sha,
    accepted_m32_success_sha,
    accepted_m32_raw_hlo_sha,
) = sys.argv[1:]
run_dir = Path(run_dir_text)
runner_path = run_dir / "runner.json"
tensor_path = run_dir / "dense_partial_capture.npz"
runner = json.loads(runner_path.read_text())

def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()

def array_sha(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()

expected_sources = {
    "accepted_m32_hlo_raw_sha256": accepted_m32_raw_hlo_sha,
    "accepted_m32_hlo_sha256": accepted_m32_hlo_sha,
    "accepted_m32_summary_sha256": accepted_m32_summary_sha,
    "accepted_m32_success_sha256": accepted_m32_success_sha,
    "checkpoint_manifest_sha256": checkpoint_manifest_sha,
    "db538_runner_sha256": db538_runner_sha,
    "db538_tensor_sha256": db538_tensor_sha,
    "db538_summary_sha256": db538_summary_sha,
    "db538_success_sha256": db538_success_sha,
    "attention_update_sha256": "68afed86921584fb673abb11a563e359a1533210ec2483e71ee79b88c2b0bde7",
    "combined_residual_sha256": "02d045b9a0ec5ab22a711bd6a964564f707be0848683381104e83331020e31a3",
    "normalized_mlp_sha256": "082125fead43b25f10686705c1b6473153f4092dd5bc476f8e01a86629f0758f",
    "post_attention_residual_sha256": "a105fdbd429adb1d06a70bf71598a72a91d7b6faa83360005487ce11ce099f8e",
}
stable = runner.get("hlo", {}).get("stablehlo_contract", {})
optimized = runner.get("hlo", {}).get("optimized_contract", {})
records = runner.get("capture_records")
if not (
    runner.get("artifact_kind") == "glm52_layer0_dense_partial_capture"
    and runner.get("classification") == "accepted_m32_dense_partials_captured"
    and runner.get("code_hash") == pin
    and runner.get("compile_rows") == 32
    and runner.get("diagnostic_dead_rows") == 31
    and runner.get("capture_partials") is True
    and runner.get("dense_envelope") is True
    and runner.get("final_dense_layout") is True
    and runner.get("split_layer1_rms") is False
    and runner.get("exact") is False
    and runner.get("exact_arms") == []
    and runner.get("layer1_comparison") is None
    and runner.get("live_rows") == 1
    and runner.get("result_mode") == "partials_only"
    and runner.get("performance_claim") is False
    and runner.get("position") == 8155
    and runner.get("status") == "SUCCESS"
    and runner.get("source") == expected_sources
    and stable.get("passed") is True
    and stable.get("partials_only") is True
    and stable.get("result_mode") == "partials_only"
    and stable.get("matched_virtual_shards") == list(range(8))
    and stable.get("convolution_count") == 16
    and stable.get("collective_counts") == {
        "all_gather": 1,
        "all_reduce": 0,
        "all_to_all": 0,
        "collective_broadcast": 0,
        "collective_permute": 0,
        "reduce_scatter": 0,
    }
    and optimized.get("passed") is True
    and optimized.get("accepted_gate_up_ranks") == list(range(8))
    and optimized.get("accepted_down_ranks") == list(range(8))
    and optimized.get("async_collectives") == []
    and optimized.get("exact_capture_lineage") is True
    and optimized.get("exact_live_schedule_bijection") is True
    and optimized.get("exact_result_binding") is True
    and optimized.get("capture_gather_count") == 1
    and optimized.get("collective_count") == 1
    and optimized.get("performance_claim") is False
):
    raise SystemExit("dense partial-capture runner contract drifted")

expected_record_shapes = {
    "dense_virtual_partials_bfloat16_bits": (
        [4, 8, 1, 6144],
        ["pp8_owner", "virtual_rank", "row", "hidden"],
    ),
    "post_attention_m32_bfloat16_bits": (
        [32, 6144],
        ["row", "hidden"],
    ),
}
if not isinstance(records, dict) or set(records) != set(expected_record_shapes):
    raise SystemExit("dense partial-capture record set drifted")
for name, (shape, order) in expected_record_shapes.items():
    record = records[name]
    if (
        record.get("shape") != shape
        or record.get("axis_order") != order
        or record.get("dtype") != "uint16"
        or not isinstance(record.get("sha256"), str)
        or len(record["sha256"]) != 64
    ):
        raise SystemExit(f"dense partial-capture record drifted: {name}")

expected_keys = {
    "accepted_layer1_normalized_bfloat16_bits",
    "attention_update_bfloat16_bits",
    "combined_residual_bfloat16_bits",
    "compile_rows",
    "dense_virtual_partials_bfloat16_bits",
    "layer1_input_norm_bfloat16_bits",
    "normalized_mlp_bfloat16_bits",
    "post_attention_m32_bfloat16_bits",
    "post_attention_residual_bfloat16_bits",
}
with np.load(tensor_path, allow_pickle=False) as payload:
    if set(payload.files) != expected_keys:
        raise SystemExit("dense partial-capture NPZ key set drifted")
    if payload["compile_rows"].shape != () or int(payload["compile_rows"]) != 32:
        raise SystemExit("dense partial-capture compile rows drifted")
    for name, (shape, _order) in expected_record_shapes.items():
        value = np.ascontiguousarray(payload[name])
        if (
            list(value.shape) != shape
            or value.dtype != np.uint16
            or array_sha(value) != records[name]["sha256"]
        ):
            raise SystemExit(f"dense partial-capture tensor drifted: {name}")
    for name, (shape, expected_sha) in {
        "accepted_layer1_normalized_bfloat16_bits": (
            (6144,),
            "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039",
        ),
        "attention_update_bfloat16_bits": (
            (1, 6144),
            "68afed86921584fb673abb11a563e359a1533210ec2483e71ee79b88c2b0bde7",
        ),
        "combined_residual_bfloat16_bits": (
            (1, 6144),
            "02d045b9a0ec5ab22a711bd6a964564f707be0848683381104e83331020e31a3",
        ),
        "layer1_input_norm_bfloat16_bits": (
            (6144,),
            "10e34f4f99c638b29557526283205071c1ac8f81f168f4a6817e7e1def4b6c87",
        ),
        "normalized_mlp_bfloat16_bits": (
            (1, 6144),
            "082125fead43b25f10686705c1b6473153f4092dd5bc476f8e01a86629f0758f",
        ),
        "post_attention_residual_bfloat16_bits": (
            (1, 6144),
            "a105fdbd429adb1d06a70bf71598a72a91d7b6faa83360005487ce11ce099f8e",
        ),
    }.items():
        value = payload[name]
        if (
            value.shape != shape
            or value.dtype != np.uint16
            or array_sha(value) != expected_sha
        ):
            raise SystemExit(f"dense partial-capture source tensor drifted: {name}")

optimized_path = run_dir / "hlo/dense_convolution.optimized_hlo.txt"
stable_path = run_dir / "hlo/dense_convolution.stablehlo.mlir"
if (
    file_sha(optimized_path) != runner["hlo"]["optimized_sha256"]
    or file_sha(stable_path) != runner["hlo"]["stablehlo_sha256"]
):
    raise SystemExit("dense partial-capture HLO file identity drifted")
summary = {
    "artifact_kind": runner["artifact_kind"],
    "capture_records": records,
    "classification": runner["classification"],
    "code_hash": pin,
    "compile_rows": 32,
    "diagnostic_dead_rows": 31,
    "elapsed_seconds": int(elapsed_text),
    "exact_arms": [],
    "hlo": {
        "optimized_sha256": runner["hlo"]["optimized_sha256"],
        "stablehlo_sha256": runner["hlo"]["stablehlo_sha256"],
    },
    "live_rows": 1,
    "performance_claim": False,
    "result_mode": "partials_only",
    "results_db_run_id": None,
    "run_tag": run_tag,
    "runner_sha256": file_sha(runner_path),
    "source": expected_sources,
    "status": "SUCCESS",
    "tensor_sha256": file_sha(tensor_path),
}
(run_dir / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("DENSE_PARTIAL_CAPTURE_VALID")
PY
else
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$RESULTS_DB" "$WORKTREE" "$elapsed" "$TAG" \
  "$DENSE_CONVOLUTION" "$DB538_RUNNER_SHA" "$DB538_TENSOR_SHA" \
  "$DB538_SUMMARY_SHA" "$DB538_SUCCESS_SHA" \
  "$POST_ATTENTION_RESIDUAL_SHA" "$CHECKPOINT_MANIFEST_SHA" \
  "$DENSE_COMPILE_ROWS" "$DENSE_LAYER1_ONLY" "$DENSE_FINAL_LAYOUT" \
  "$DENSE_ENVELOPE" "$DENSE_SPLIT_LAYER1_RMS" \
  "$ATTENTION_UPDATE_SHA" "$COMBINED_RESIDUAL_SHA" \
  "$NORMALIZED_MLP_SHA" \
  "$ACCEPTED_M32_HLO_SHA" \
  "$ACCEPTED_M32_SUMMARY_SHA" "$ACCEPTED_M32_SUCCESS_SHA" \
  "$ACCEPTED_M32_RAW_HLO_SHA" <<'PY'
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
import sys

(
    run_dir,
    pin,
    db_path,
    repo,
    elapsed,
    run_tag,
    dense_text,
    db538_runner_sha,
    db538_tensor_sha,
    db538_summary_sha,
    db538_success_sha,
    post_attention_sha,
    checkpoint_manifest_sha,
    dense_compile_rows_text,
    dense_layer1_only_text,
    dense_final_layout_text,
    dense_envelope_text,
    split_layer1_rms_text,
    attention_update_sha,
    combined_residual_sha,
    normalized_mlp_sha,
    accepted_m32_hlo_sha,
    accepted_m32_summary_sha,
    accepted_m32_success_sha,
    accepted_m32_raw_hlo_sha,
) = sys.argv[1:]
dense = dense_text == "1"
compile_rows = int(dense_compile_rows_text)
layer1_only = dense_layer1_only_text == "1"
final_layout = dense_final_layout_text == "1"
dense_envelope = dense_envelope_text == "1"
split_layer1_rms = split_layer1_rms_text == "1"
run_dir = Path(run_dir)
runner = json.loads((run_dir / "runner.json").read_text())
if dense:
    exact = runner.get("exact")
    arm = (
        "accepted_m32_dense_envelope_split_rms"
        if split_layer1_rms
        else "accepted_m32_dense_envelope_cross_layer"
        if dense_envelope
        else "accepted_m32_dense_final_layout_cross_layer"
        if final_layout
        else "accepted_m32_dense_cross_layer"
        if layer1_only
        else (
            "accepted_m32_dense_convolution"
            if compile_rows == 32
            else "accepted_dense_convolution"
        )
    )
    classification = f"{arm}_{'exact' if exact is True else 'nonexact'}"
    comparison = runner.get("layer1_comparison", {})
    stable = runner.get("hlo", {}).get("stablehlo_contract", {})
    optimized = runner.get("hlo", {}).get("optimized_contract", {})
    source = runner.get("source", {})
    weight_records = runner.get("weight_records")
    final_layout_records = runner.get("final_layout_records")
    sha_pattern = __import__("re").compile(r"[0-9a-f]{64}")
    mismatch_count = comparison.get("mismatch_count")
    first_mismatch_index = comparison.get("first_mismatch_index")
    max_abs_error = comparison.get("max_abs_error")
    mean_abs_error = comparison.get("mean_abs_error")
    if exact is True:
        numerical_comparison_valid = (
            type(mismatch_count) is int
            and mismatch_count == 0
            and first_mismatch_index is None
            and type(max_abs_error) in (int, float)
            and math.isfinite(max_abs_error)
            and max_abs_error == 0.0
            and type(mean_abs_error) in (int, float)
            and math.isfinite(mean_abs_error)
            and mean_abs_error == 0.0
        )
    elif exact is False:
        numerical_comparison_valid = (
            type(mismatch_count) is int
            and 1 <= mismatch_count <= 6144
            and type(first_mismatch_index) is int
            and 0 <= first_mismatch_index < 6144
            and type(max_abs_error) in (int, float)
            and math.isfinite(max_abs_error)
            and type(mean_abs_error) in (int, float)
            and math.isfinite(mean_abs_error)
            and 0.0 < mean_abs_error <= max_abs_error
        )
    else:
        numerical_comparison_valid = False
    expected_source = {
        "checkpoint_manifest_sha256": checkpoint_manifest_sha,
        "db538_runner_sha256": db538_runner_sha,
        "db538_tensor_sha256": db538_tensor_sha,
        "db538_summary_sha256": db538_summary_sha,
        "db538_success_sha256": db538_success_sha,
        "post_attention_residual_sha256": post_attention_sha,
    }
    if dense_envelope:
        expected_source.update({
            "attention_update_sha256": attention_update_sha,
            "combined_residual_sha256": combined_residual_sha,
            "normalized_mlp_sha256": normalized_mlp_sha,
        })
    if compile_rows == 32:
        expected_source.update({
            "accepted_m32_hlo_raw_sha256": accepted_m32_raw_hlo_sha,
            "accepted_m32_hlo_sha256": accepted_m32_hlo_sha,
            "accepted_m32_summary_sha256": accepted_m32_summary_sha,
            "accepted_m32_success_sha256": accepted_m32_success_sha,
        })
    source_valid = source == expected_source
    comparison_valid = (
        comparison.get("shape") == [6144]
        and comparison.get("expected_sha256")
        == "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039"
        and comparison.get("elementwise_exact") is exact
        and numerical_comparison_valid
        and isinstance(comparison.get("observed_sha256"), str)
        and sha_pattern.fullmatch(comparison["observed_sha256"]) is not None
        and (
            comparison["observed_sha256"] == comparison["expected_sha256"]
        ) is exact
    )
    stable_valid = (
        stable.get("passed") is True
        and stable.get("violations") == []
        and stable.get("compile_rows") == compile_rows
        and stable.get("dense_envelope") is dense_envelope
        and stable.get("final_dense_layout") is final_layout
        and stable.get("split_layer1_rms") is split_layer1_rms
        and stable.get("live_rows") == 1
        and stable.get("result_mode")
        == ("layer1_only" if layer1_only else "dense_and_layer1")
        and stable.get("collective_counts") == {
            "all_gather": 1,
            "all_reduce": 0,
            "all_to_all": 0,
            "collective_broadcast": 0,
            "collective_permute": 0,
            "reduce_scatter": 0,
        }
        and stable.get("convolution_count") == 16
        and stable.get("gate_up_convolution_count") == 8
        and stable.get("down_convolution_count") == 8
        and stable.get("gate_up_layout_constraint_count")
        == (8 if final_layout else 0)
        and stable.get("matched_virtual_shards") == list(range(8))
    )
    optimized_lineage = optimized.get("lineage", {})
    association_graph = optimized_lineage.get("association_edge_graph", {})
    expected_pairings = {
        str(component): (
            [[0, 1], [2, 3]]
            if component % 2 == 0
            else [[0, 3], [1, 2]]
        )
        for component in range(24)
    }
    x_association_valid = association_graph.get("x") == {
        "component_count": 1,
        "exact": True,
        "leaf_rows": [0, 1],
    }
    y_association = association_graph.get("y", {})
    y_association_valid = y_association == {
        "component_count": 3,
        "component_ids": list(range(3)),
        "exact": True,
        "leaf_pairings": {
            key: value
            for key, value in expected_pairings.items()
            if int(key) < 3
        },
        "ordered_components": [[component] for component in range(3)],
    }
    z_association_valid = association_graph.get("z") == {
        "component_count": 24,
        "component_ids": list(range(24)),
        "exact": True,
        "leaf_pairings": expected_pairings,
        "ordered_components": [[component] for component in range(24)],
    }
    optimized_valid = (
        optimized.get("passed") is True
        and optimized.get("violations") == []
        and optimized.get("compile_rows") == compile_rows
        and optimized.get("dense_envelope") is dense_envelope
        and optimized.get("final_dense_layout") is final_layout
        and optimized.get("split_layer1_rms") is split_layer1_rms
        and optimized.get("live_rows") == 1
        and optimized.get("result_mode")
        == ("layer1_only" if layer1_only else "dense_and_layer1")
        and optimized.get("async_collectives") == []
        and optimized.get("collective_count") == 1
        and optimized.get("convolution_count") == 16
        and optimized.get("gate_up_convolution_count") == 8
        and optimized.get("down_convolution_count") == 8
        and optimized.get("unexpected_convolutions") == []
        and optimized.get("num_partitions") == 4
        and optimized.get("num_replicas") in (None, 1)
        and optimized_lineage.get("gate_up_virtual_ranks") == list(range(8))
        and optimized_lineage.get("down_virtual_ranks") == list(range(8))
        and optimized_lineage.get("ordered_stack_sources")
        == [[rank] for rank in range(8)]
        and optimized_lineage.get("association_add_shapes")
        == {"y": 9, "x": 1, "z": 72}
        and x_association_valid
        and y_association_valid
        and z_association_valid
        and set(optimized_lineage.get("activation_contract", {}))
        == {str(rank) for rank in range(8)}
        and all(
            {
                opcode: len(items)
                for opcode, items in optimized_lineage[
                    "activation_contract"
                ][str(rank)].items()
                if opcode != "exact_operand_graph"
            }
            == {
                "negate": 1,
                "exponential": 1,
                "add": 1,
                "divide": 1,
                "multiply": 2,
            }
            and optimized_lineage["activation_contract"][str(rank)].get(
                "exact_operand_graph"
            ) is True
            for rank in range(8)
        )
        and optimized_lineage.get("rmsnorm_contract", {}).get(
            "direct_exact_operand_graph"
        ) is True
        and optimized_lineage.get("rmsnorm_contract", {}).get(
            "exact_result_binding"
        ) is True
        and optimized_lineage.get("rmsnorm_contract", {}).get(
            "exact_weighted_operand_graph"
        ) is True
        and optimized_lineage.get("rmsnorm_contract", {}).get(
            "exact_reduction_operand_graph"
        ) is True
        and optimized_lineage.get("rmsnorm_contract", {}).get(
            "exact_cross_fusion_reduction_lineage"
        ) is True
        and optimized_lineage.get("rmsnorm_contract", {}).get(
            "exact_m32_reduction_geometry"
        ) is True
        and optimized_lineage.get("rmsnorm_contract", {}).get(
            "split_layer1_rms"
        ) is split_layer1_rms
        and (
            not split_layer1_rms
            or (
                optimized_lineage.get("rmsnorm_contract", {}).get(
                    "split_recompute_exact"
                ) is True
                and optimized_lineage.get("rmsnorm_contract", {}).get(
                    "exact_accepted_scheduled_reduction"
                ) is True
                and optimized_lineage.get("rmsnorm_contract", {}).get(
                    "split_output_fusion_exact"
                ) is True
                and len(
                    optimized_lineage.get("rmsnorm_contract", {}).get(
                        "accepted_scheduled_reduction_values", []
                    )
                ) == 1
            )
        )
        and {
            "add",
            "div",
            "mul",
            "reduce_sum",
            "rsqrt",
            "square",
        }.issubset(
            optimized_lineage.get("rmsnorm_contract", {}).get(
                "semantic_counts", {}
            )
        )
        and (
            (
                len(optimized_lineage.get("result_parameter_sources", [])) == 1
                and len(optimized_lineage["result_parameter_sources"][0])
                == (8 if dense_envelope else 7 if final_layout else 9)
            )
            if layer1_only
            else {
                "bf16[1,6144]",
                "bf16[6144]",
            }.issubset(
                set(optimized_lineage.get("layer1_only_parameter_shapes", []))
            )
        )
        and len(optimized_lineage.get("collective_convolution_sources", []))
        == 8
        and (
            compile_rows == 1
            or len(optimized_lineage.get("m32_live_row_slices", [])) == 1
        )
        and (
            not final_layout
            or (
                optimized.get("exact_accepted_weight_layout") is True
                and optimized.get("exact_packed_weight_lineage") is True
                and optimized.get("scheduled_kernel_geometry_required") is True
                and optimized.get("exact_accepted_kernel_geometry") is True
                and set(optimized.get("accepted_weight_layouts", {}))
                == {"gate_up", "down"}
                and all(
                    len(records) == 8
                    and {
                        record.get("virtual_rank") for record in records
                    } == set(range(8))
                    and all(
                        record.get("accepted") is True
                        and record.get("accepted_layout") is True
                        and record.get("exact_packed_dequant") is True
                        and record.get("exact_convolution_tiling") is True
                        and len(record.get("parameter_sources", [])) == 2
                        for record in records
                    )
                    for records in optimized["accepted_weight_layouts"].values()
                )
            )
        )
        and (
            not dense_envelope
            or (
                optimized_lineage.get("predense_rmsnorm_contract", {}).get(
                    "enabled"
                ) is True
                and optimized_lineage.get(
                    "predense_rmsnorm_contract", {}
                ).get("exact_operand_graph") is True
                and optimized_lineage.get(
                    "predense_rmsnorm_contract", {}
                ).get("exact_gate_input_binding") is True
                and optimized_lineage.get(
                    "predense_rmsnorm_contract", {}
                ).get("exact_fused_gate_ownership") is True
                and optimized_lineage.get(
                    "predense_rmsnorm_contract", {}
                ).get("fused_gate_binding_count") == 8
                and optimized_lineage.get(
                    "predense_rmsnorm_contract", {}
                ).get("exact_carried_residual_binding") is True
                and optimized_lineage.get(
                    "predense_rmsnorm_contract", {}
                ).get("weighted_value_count") in {1, 8}
            )
        )
    )
    expected_final_layout = {
        "dense.slot_00.merged_gate_up.weight_bits_in_out": {
            "shape": [4, 8, 6144, 768],
            "dtype": "float8_e4m3fn",
            "sha256": "82c93c0fafda7afa3853e3a689aace78be61bec88c5ef779bb0e9efeb834facf",
        },
        "dense.slot_00.merged_gate_up.scale_inv_in_out": {
            "shape": [4, 8, 48, 768],
            "dtype": "float32",
            "sha256": "9b4bfee8b15d04545a277ba4e1ea0d3b73427a0f2cebf0a5f5fa4d7ab32bf8b3",
        },
        "dense.slot_00.down.weight_bits_in_out": {
            "shape": [4, 8, 384, 6144],
            "dtype": "float8_e4m3fn",
            "sha256": "8654c1ebb6f0ef29b1d3919699c08ca2b81e058bf3f9cd0827a9889994d72f7e",
        },
        "dense.slot_00.down.scale_inv_in_out": {
            "shape": [4, 8, 3, 6144],
            "dtype": "float32",
            "sha256": "f37e87987745d263d152ff17415d21d572fa3c73ad5888207f1aa44916f74bdd",
        },
    }
    final_layout_records_valid = (
        isinstance(final_layout_records, dict)
        and (
            final_layout_records == expected_final_layout
            if final_layout
            else final_layout_records == {}
        )
    )
    weights_valid = (
        isinstance(weight_records, list)
        and len(weight_records) == 4
        and [item.get("device_slot") for item in weight_records]
        == list(range(4))
        and all(
            item.get("destination_filename")
            == (
                "base_decoder_runtime_feature/stage_00/"
                f"device_slot_{slot:02d}.safetensors"
            )
            and sha_pattern.fullmatch(item.get("evidence_file_sha256", ""))
            is not None
            and sha_pattern.fullmatch(item.get("header_sha256", ""))
            is not None
            and isinstance(item.get("tensors"), list)
            and {
                tensor.get("name") for tensor in item["tensors"]
            }
            == {
                "attention.slot_00.kv_b.weight_bits",
                "attention.slot_00.kv_b.scale_inv",
                "attention.slot_00.o.weight_bits",
                "attention.slot_00.o.scale_inv",
                "attention.slot_00.post_norm",
                "dense.slot_00.gate.weight_bits",
                "dense.slot_00.gate.scale_inv",
                "dense.slot_00.up.weight_bits",
                "dense.slot_00.up.scale_inv",
                "dense.slot_00.down.weight_bits",
                "dense.slot_00.down.scale_inv",
                "attention.slot_01.input_norm",
            }
            and all(
                isinstance(tensor.get("name"), str)
                and sha_pattern.fullmatch(tensor.get("sha256", "")) is not None
                for tensor in item["tensors"]
            )
            for slot, item in enumerate(weight_records)
        )
    )
    runner_valid = (
        isinstance(exact, bool)
        and
        runner["status"] == "SUCCESS"
        and runner["code_hash"] == pin
        and runner["artifact_kind"] == "glm52_layer0_dense_convolution_probe"
        and runner.get("position") == 8155
        and runner.get("performance_claim") is False
        and runner.get("compile_rows") == compile_rows
        and runner.get("dense_envelope") is dense_envelope
        and runner.get("final_dense_layout") is final_layout
        and runner.get("split_layer1_rms") is split_layer1_rms
        and runner.get("live_rows") == 1
        and runner.get("diagnostic_dead_rows") == compile_rows - 1
        and runner.get("result_mode")
        == ("layer1_only" if layer1_only else "dense_and_layer1")
        and runner.get("classification") == classification
        and runner["exact_arms"]
        == ([arm] if exact else [])
        and comparison_valid
        and source_valid
        and stable_valid
        and optimized_valid
        and final_layout_records_valid
        and weights_valid
        and sha_pattern.fullmatch(runner.get("hlo", {}).get("stablehlo_sha256", ""))
        is not None
        and sha_pattern.fullmatch(runner.get("hlo", {}).get("optimized_sha256", ""))
        is not None
    )
else:
    runner_valid = (
        runner["status"] == "SUCCESS"
        and runner["code_hash"] == pin
        and runner["artifact_kind"]
        == "glm52_layer0_projection_reduction_probe"
        and len(runner["arms"]) == 4
        and all(
            arm["hlo"]["contract"]["passed"]
            for arm in runner["arms"].values()
        )
        and all(
            arm["value_comparison"]["elementwise_exact"]
            for arm in runner["arms"].values()
        )
    )
if not runner_valid:
    raise SystemExit("layer-0 arithmetic runner contract failed")

model = (
    (
        "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-envelope-split-rms"
        if split_layer1_rms
        else "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-envelope-cross-layer"
        if dense_envelope
        else "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-final-layout-cross-layer"
        if final_layout
        else "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-m32-cross-layer"
        if layer1_only
        else (
            "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-m32-convolution"
            if compile_rows == 32
            else "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-convolution"
        )
    )
    if dense
    else "zai-org/GLM-5.2-FP8:greenfield-layer0-projection-reduction"
)
revision = (
    (
        "native-jax-accepted-dense-envelope-split-rms-v1"
        if split_layer1_rms
        else "native-jax-accepted-dense-envelope-v1"
        if dense_envelope
        else "native-jax-accepted-dense-final-layout-v1"
        if final_layout
        else "native-jax-db542-dense-cross-layer-v1"
        if layer1_only
        else (
            "native-jax-db532-dense-m32-discriminator-v1"
            if compile_rows == 32
            else "native-jax-db538-dense-convolution-v1"
        )
    )
    if dense
    else "native-jax-db537-strategy-nd-v1"
)
benchmark = (
    (
        "greenfield_layer0_dense_envelope_split_rms"
        if split_layer1_rms
        else "greenfield_layer0_dense_envelope_cross_layer"
        if dense_envelope
        else "greenfield_layer0_dense_final_layout_cross_layer"
        if final_layout
        else "greenfield_layer0_dense_m32_cross_layer"
        if layer1_only
        else (
            "greenfield_layer0_dense_m32_convolution"
            if compile_rows == 32
            else "greenfield_layer0_dense_convolution"
        )
    )
    if dense
    else "greenfield_layer0_projection_reduction"
)
engine = (
    (
        "greenfield_dense_envelope_split_rms_probe"
        if split_layer1_rms
        else "greenfield_dense_envelope_cross_layer_probe"
        if dense_envelope
        else "greenfield_dense_final_layout_cross_layer_probe"
        if final_layout
        else "greenfield_dense_m32_cross_layer_probe"
        if layer1_only
        else (
            "greenfield_dense_m32_convolution_probe"
            if compile_rows == 32
            else "greenfield_dense_convolution_probe"
        )
    )
    if dense
    else "greenfield_projection_reduction_probe"
)
note = (
    (
        "Protected layer-1 accepted split-RMS schedule discriminator; no performance claim."
        if split_layer1_rms
        else "Protected layer-0 accepted dense fusion-envelope discriminator; no performance claim."
        if dense_envelope
        else "Protected layer-0 accepted final-layout dense cross-layer discriminator; no performance claim."
        if final_layout
        else "Protected layer-0 accepted-M32 dense cross-layer fusion discriminator; no performance claim."
        if layer1_only
        else (
            "Protected layer-0 accepted-M32 dense arithmetic discriminator; no performance claim."
            if compile_rows == 32
            else "Protected layer-0 dense convolution discriminator; no performance claim."
        )
    )
    if dense
    else "Protected layer-0 projection/reduction discriminator; no performance claim."
)

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

connection = pv.connect(db_path)
run_id = pv.start_run(
    connection,
    model=model,
    revision=revision,
    env={
        "GLM_ENGINE": engine,
        "greenfield_code_hash": pin,
        "greenfield_run_tag": run_tag,
        "classification": runner["classification"],
        **(
            {
                "checkpoint_manifest_sha256": runner["source"][
                    "checkpoint_manifest_sha256"
                ],
                "db538_tensor_sha256": runner["source"]["db538_tensor_sha256"],
                "compile_rows": compile_rows,
                "dense_envelope": dense_envelope,
                "final_dense_layout": final_layout,
                "split_layer1_rms": split_layer1_rms,
                "result_mode": runner["result_mode"],
                **(
                    {
                        "attention_update_sha256": runner["source"][
                            "attention_update_sha256"
                        ],
                        "combined_residual_sha256": runner["source"][
                            "combined_residual_sha256"
                        ],
                        "normalized_mlp_sha256": runner["source"][
                            "normalized_mlp_sha256"
                        ],
                    }
                    if dense_envelope
                    else {}
                ),
                **(
                    {
                        "final_layout_records_sha256": (
                            "45bfd64e45956627516c36603ccee53d85d713ce6476761e68f1170c51901ba4"
                        ),
                        "exact_accepted_kernel_geometry": True,
                        "scheduled_kernel_geometry_required": True,
                    }
                    if final_layout
                    else {}
                ),
                **(
                    {
                        "exact_accepted_scheduled_reduction": True,
                        "split_recompute_exact": True,
                        "split_output_fusion_exact": True,
                    }
                    if split_layer1_rms
                    else {}
                ),
                **(
                    {
                        "accepted_m32_hlo_raw_sha256": runner["source"][
                            "accepted_m32_hlo_raw_sha256"
                        ],
                        "accepted_m32_hlo_sha256": runner["source"][
                            "accepted_m32_hlo_sha256"
                        ],
                        "accepted_m32_summary_sha256": runner["source"][
                            "accepted_m32_summary_sha256"
                        ],
                        "accepted_m32_success_sha256": runner["source"][
                            "accepted_m32_success_sha256"
                        ],
                    }
                    if compile_rows == 32
                    else {}
                ),
            }
            if dense
            else {
                "attention_tensor_sha256": runner["source"][
                    "attention_arithmetic_sha256"
                ],
                "association_analysis_sha256": runner["source"][
                    "association_analysis_sha256"
                ],
            }
        ),
    },
    note=note,
    harness_repo=repo,
    fork_repo=None,
)
(run_dir / "provisional_db_run_id.txt").write_text(f"{run_id}\n")
pv.record_item(
    connection,
    run_id,
    benchmark=benchmark,
    item_id="position8155",
    prompt=(
        (
            "Sealed accepted scalar-only layer-1 RMS schedule with exact recompute."
            if split_layer1_rms
            else "Sealed exact StrategyND attention boundary with accepted final-layout dense fusion."
            if final_layout and not dense_envelope
            else "Sealed exact pre-dense RMSNorm boundary with accepted dense fusion envelope."
            if dense_envelope
            else "Sealed exact StrategyND attention boundary with layer-1-only accepted-M32 dense fusion."
            if layer1_only
            else (
                "Sealed exact StrategyND attention boundary with diagnostic accepted-M32 dense geometry."
                if compile_rows == 32
                else "Sealed exact StrategyND attention boundary at first 8K decode row."
            )
        )
        if dense
        else "Sealed exact B512 latent and layer-0 residual at first 8K decode row."
    ),
    gold="Exact accepted BF16 layer-1 normalized hidden [6144].",
    raw_output=json.dumps(
        {
            "classification": runner["classification"],
            "exact_arms": runner["exact_arms"],
            "mismatch_counts": (
                {arm: runner["layer1_comparison"]["mismatch_count"]}
                if dense
                else {
                    name: arm["layer1_comparison"]["mismatch_count"]
                    for name, arm in runner["arms"].items()
                }
            ),
        },
        sort_keys=True,
    ),
    extracted=",".join(runner["exact_arms"]) or "none",
    correct=bool(runner["exact_arms"]),
    score=float(bool(runner["exact_arms"])),
)
pv.finalize(
    connection,
    run_id,
    benchmark=benchmark,
    metric="probe_contract_valid",
    value=1.0,
    note="Diagnostic layer-0 arithmetic classification only; no decoder claim.",
)
connection.close()
summary = {
    "artifact_kind": runner["artifact_kind"],
    "classification": runner["classification"],
    "code_hash": pin,
    "compile_rows": compile_rows if dense else 1,
    "diagnostic_dead_rows": compile_rows - 1 if dense else 0,
    "elapsed_seconds": int(elapsed),
    "exact_arms": runner["exact_arms"],
    "dense_envelope": dense_envelope if dense else False,
    "final_dense_layout": final_layout if dense else False,
    "split_layer1_rms": split_layer1_rms if dense else False,
    "final_layout_records": (
        runner.get("final_layout_records", {}) if dense else {}
    ),
    "final_layout_records_sha256": sha256(
        json.dumps(
            runner.get("final_layout_records", {}) if dense else {},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest(),
    "exact_accepted_kernel_geometry": (
        runner.get("hlo", {})
        .get("optimized_contract", {})
        .get("exact_accepted_kernel_geometry", False)
        if dense and final_layout
        else False
    ),
    "scheduled_kernel_geometry_required": (
        runner.get("hlo", {})
        .get("optimized_contract", {})
        .get("scheduled_kernel_geometry_required", False)
        if dense and final_layout
        else False
    ),
    "exact_accepted_scheduled_reduction": (
        runner.get("hlo", {})
        .get("optimized_contract", {})
        .get("lineage", {})
        .get("rmsnorm_contract", {})
        .get("exact_accepted_scheduled_reduction", False)
        if dense and split_layer1_rms
        else False
    ),
    "split_recompute_exact": (
        runner.get("hlo", {})
        .get("optimized_contract", {})
        .get("lineage", {})
        .get("rmsnorm_contract", {})
        .get("split_recompute_exact", False)
        if dense and split_layer1_rms
        else False
    ),
    "split_output_fusion_exact": (
        runner.get("hlo", {})
        .get("optimized_contract", {})
        .get("lineage", {})
        .get("rmsnorm_contract", {})
        .get("split_output_fusion_exact", False)
        if dense and split_layer1_rms
        else False
    ),
    "live_rows": 1,
    "result_mode": (
        runner.get("result_mode") if dense else "dense_and_layer1"
    ),
    "performance_claim": False,
    "results_db_run_id": run_id,
    "source": runner.get("source", {}),
    "status": "SUCCESS",
}
(run_dir / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
source = sqlite3.connect(db_path)
snapshot = sqlite3.connect(run_dir / "results_ckpt.db")
source.backup(snapshot)
snapshot.close()
source.close()
if sqlite3.connect(run_dir / "results_ckpt.db").execute(
    "PRAGMA integrity_check"
).fetchone()[0] != "ok":
    raise SystemExit("projection/reduction DB snapshot failed integrity check")
print(f"PROJECTION_REDUCTION_VALID db_run={run_id}")
PY
fi

say "freezing and archiving projection/reduction evidence"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find hlo -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum "$TENSOR_BASENAME" runner.json runner.log summary.json \
    census_pre.txt census_post.txt orchestrator.sealed.log
  if [[ -f results_ckpt.db ]]; then
    sha256sum results_ckpt.db
  fi
) >"$RUN_DIR/evidence.sha256"
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
    value = json.loads(subprocess.run(
        ["gcloud", "storage", "objects", "describe", f"{prefix}/{relative}",
         "--format=json"],
        check=True, capture_output=True, text=True,
    ).stdout)
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

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" "$PIN" <<'PY'
from hashlib import sha256
import base64
import json
from pathlib import Path
import sys

import google_crc32c

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
if summary.get("artifact_kind") == "glm52_layer0_isolated_dense_replay":
    runner = json.loads((root / "runner.json").read_text())
    exact = summary.get("exact")
    exact_schema = bool(
        exact is True
        and summary.get("classification")
        == "isolated_virtual_contractions_exact"
        and summary.get("exact_arms") == ["isolated_virtual_contractions"]
    )
    nonexact_schema = bool(
        exact is False
        and summary.get("classification")
        == "isolated_virtual_contractions_nonexact"
        and summary.get("exact_arms") == []
    )
    actual_hlo = {
        name: {
            "optimized_sha256": sha256(
                (root / "hlo" / f"{name}.optimized_hlo.txt").read_bytes()
            ).hexdigest(),
            "stablehlo_sha256": sha256(
                (root / "hlo" / f"{name}.stablehlo.mlir").read_bytes()
            ).hexdigest(),
        }
        for name in ("isolated_dense", "layer1_replay")
    }
    evidence_records = {}
    for line in (root / "evidence.sha256").read_text().splitlines():
        digest, relative = line.split(None, 1)
        evidence_records[relative.strip()] = digest

    def local_crc32c(path: Path) -> str:
        checksum = google_crc32c.Checksum()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                checksum.update(chunk)
        return base64.b64encode(checksum.digest()).decode()

    ledger = json.loads((root / "remote_objects.json").read_text())
    ledger_by_path = {
        record["path"]: record for record in ledger.get("objects", ())
    }
    expected_ledger_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"SUCCESS", "remote_objects.json"}
    }
    exact_ledger = bool(
        set(ledger_by_path) == expected_ledger_paths
        and all(
            record.get("size") == (root / relative).stat().st_size
            and record.get("crc32c") == local_crc32c(root / relative)
            and isinstance(record.get("generation"), str)
            and record["generation"]
            for relative, record in ledger_by_path.items()
        )
    )
    evidence_bound = bool(
        evidence_records.get("summary.json")
        == sha256((root / "summary.json").read_bytes()).hexdigest()
        and evidence_records.get("runner.json")
        == sha256((root / "runner.json").read_bytes()).hexdigest()
        and evidence_records.get("isolated_dense_replay.npz")
        == sha256((root / "isolated_dense_replay.npz").read_bytes()).hexdigest()
        and all(
            evidence_records.get(f"hlo/{name}.{suffix}") == digest
            for name in ("isolated_dense", "layer1_replay")
            for suffix, digest in (
                ("optimized_hlo.txt", actual_hlo[name]["optimized_sha256"]),
                ("stablehlo.mlir", actual_hlo[name]["stablehlo_sha256"]),
            )
        )
    )
    if not (
        summary.get("accepted_gate_dequant_fusion") is True
        and runner.get("accepted_gate_dequant_fusion") is True
        and summary.get("accepted_gate_singleton") is True
        and runner.get("accepted_gate_singleton") is True
        and summary.get("code_hash") == sys.argv[3]
        and runner.get("artifact_kind") == summary["artifact_kind"]
        and runner.get("code_hash") == summary["code_hash"]
        and runner.get("classification") == summary["classification"]
        and runner.get("control_admissible") is True
        and runner.get("exact") is exact
        and runner.get("exact_arms") == summary.get("exact_arms")
        and runner.get("source") == summary.get("source")
        and runner.get("final_layout_records")
        == summary.get("final_layout_records")
        and runner.get("gate_singleton_discriminator")
        == summary.get("gate_singleton_discriminator")
        and runner.get("isolated_layer1_comparison")
        == summary.get("isolated_layer1_comparison")
        and runner.get("isolated_layer1_sha256")
        == summary.get("isolated_layer1_sha256")
        and runner.get("partial_comparison")
        == summary.get("partial_comparison")
        and runner.get("sensitivity") == summary.get("sensitivity")
        and all(
            actual_hlo[name] == summary["hlo"][name]
            and actual_hlo[name]["optimized_sha256"]
            == runner["hlo"][name]["optimized_sha256"]
            and actual_hlo[name]["stablehlo_sha256"]
            == runner["hlo"][name]["stablehlo_sha256"]
            for name in ("isolated_dense", "layer1_replay")
        )
        and evidence_bound
        and exact_ledger
        and summary.get("performance_claim") is False
        and summary.get("position") == 8155
        and summary.get("results_db_run_id") is None
        and summary.get("runner_sha256")
        == sha256((root / "runner.json").read_bytes()).hexdigest()
        and summary.get("tensor_sha256")
        == sha256((root / "isolated_dense_replay.npz").read_bytes()).hexdigest()
        and summary.get("status") == "SUCCESS"
        and summary.get("virtual_contractions_per_chip") == 1
        and summary.get("virtual_rank_batches") == 8
        and (exact_schema or nonexact_schema)
    ):
        raise SystemExit("isolated dense replay summary drifted before SUCCESS")
    values = {
        "accepted_gate_dequant_fusion": "true",
        "accepted_gate_singleton": "true",
        "artifact_kind": summary["artifact_kind"],
        "classification": summary["classification"],
        "code_hash": sys.argv[3],
        "control_admissible": "true",
        "evidence_sha256": sha256(
            (root / "evidence.sha256").read_bytes()
        ).hexdigest(),
        "exact": str(exact).lower(),
        "exact_arms": ",".join(summary["exact_arms"]) or "none",
        "isolated_layer1_sha256": summary["isolated_layer1_sha256"],
        "gate_singleton_accepted_rank3_count": str(
            summary["gate_singleton_discriminator"][
                "accepted_rank3_gate_count"
            ]
        ),
        "gate_dequant_accepted_internal_count": str(
            summary["gate_singleton_discriminator"][
                "accepted_internal_gate_dequant_count"
            ]
        ),
        "gate_dequant_db548_materialized_count": str(
            summary["gate_singleton_discriminator"][
                "db548_materialized_bf16_gate_count"
            ]
        ),
        "gate_singleton_db548_rank2_count": str(
            summary["gate_singleton_discriminator"][
                "db548_rank2_gate_count"
            ]
        ),
        "performance_claim": "false",
        "remote_objects_sha256": sha256(
            (root / "remote_objects.json").read_bytes()
        ).hexdigest(),
        "remote_prefix": sys.argv[2],
        "results_db_run_id": "none",
        "runner_sha256": summary["runner_sha256"],
        "sensitivity_baseline_dense_update_bits": str(
            summary["sensitivity"]["baseline_dense_update_bits"]
        ),
        "sensitivity_candidate_count": str(
            summary["sensitivity"]["candidate_count"]
        ),
        "sensitivity_exact_candidate_ids": (
            ",".join(
                str(value)
                for value in summary["sensitivity"]["exact_candidate_ids"]
            )
            or "none"
        ),
        "sensitivity_hidden_index": str(
            summary["sensitivity"]["hidden_index"]
        ),
        "tensor_sha256": summary["tensor_sha256"],
        "virtual_contractions_per_chip": "1",
        "virtual_rank_batches": "8",
    }
    for name, hashes in sorted(actual_hlo.items()):
        values[f"{name}_optimized_hlo_sha256"] = hashes[
            "optimized_sha256"
        ]
        values[f"{name}_stablehlo_sha256"] = hashes["stablehlo_sha256"]
    for key, value in sorted(summary.get("source", {}).items()):
        values[f"source_{key}"] = value
    (root / "SUCCESS").write_text(
        "".join(f"{key}={value}\n" for key, value in sorted(values.items()))
    )
    raise SystemExit(0)
if summary.get("artifact_kind") == "glm52_layer0_captured_rms_replay":
    runner = json.loads((root / "runner.json").read_text())
    arms = summary.get("arms")
    exact = summary.get("exact")
    exact_schema = bool(
        exact is True
        and summary.get("classification")
        == "captured_partials_split_rms_exact"
        and summary.get("exact_arms") == ["accepted_split"]
    )
    nonexact_schema = bool(
        exact is False
        and summary.get("classification")
        == "captured_partials_split_rms_nonexact"
        and summary.get("exact_arms") == []
    )
    expected_arms = {
        arm_name: {
            "comparison": runner["arms"][arm_name]["comparison"],
            "hlo": {
                "optimized_sha256": runner["arms"][arm_name]["hlo"][
                    "optimized_sha256"
                ],
                "stablehlo_sha256": runner["arms"][arm_name]["hlo"][
                    "stablehlo_sha256"
                ],
            },
            "output_sha256": runner["arms"][arm_name]["output_sha256"],
            "split_layer1_rms": runner["arms"][arm_name][
                "split_layer1_rms"
            ],
        }
        for arm_name in ("control", "accepted_split")
    }
    actual_hlo_hashes = {
        arm_name: {
            "optimized_sha256": sha256(
                (root / "hlo" / f"{arm_name}.optimized_hlo.txt").read_bytes()
            ).hexdigest(),
            "stablehlo_sha256": sha256(
                (root / "hlo" / f"{arm_name}.stablehlo.mlir").read_bytes()
            ).hexdigest(),
        }
        for arm_name in ("control", "accepted_split")
    }
    evidence_records = {}
    for line in (root / "evidence.sha256").read_text().splitlines():
        digest, relative = line.split(None, 1)
        evidence_records[relative.strip()] = digest

    def local_crc32c(path: Path) -> str:
        checksum = google_crc32c.Checksum()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                checksum.update(chunk)
        return base64.b64encode(checksum.digest()).decode()

    ledger = json.loads((root / "remote_objects.json").read_text())
    ledger_by_path = {
        record["path"]: record for record in ledger.get("objects", ())
    }
    expected_ledger_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"SUCCESS", "remote_objects.json"}
    }
    exact_ledger = bool(
        set(ledger_by_path) == expected_ledger_paths
        and all(
            record.get("size") == (root / relative).stat().st_size
            and record.get("crc32c") == local_crc32c(root / relative)
            and isinstance(record.get("generation"), str)
            and record["generation"]
            for relative, record in ledger_by_path.items()
        )
    )
    evidence_bound = bool(
        evidence_records.get("summary.json")
        == sha256((root / "summary.json").read_bytes()).hexdigest()
        and evidence_records.get("runner.json")
        == sha256((root / "runner.json").read_bytes()).hexdigest()
        and evidence_records.get("captured_rms_replay.npz")
        == sha256((root / "captured_rms_replay.npz").read_bytes()).hexdigest()
        and all(
            evidence_records.get(f"hlo/{arm_name}.{suffix}") == digest
            for arm_name in ("control", "accepted_split")
            for suffix, digest in (
                ("optimized_hlo.txt", actual_hlo_hashes[arm_name]["optimized_sha256"]),
                ("stablehlo.mlir", actual_hlo_hashes[arm_name]["stablehlo_sha256"]),
            )
        )
    )
    if not (
        summary.get("code_hash") == sys.argv[3]
        and runner.get("artifact_kind") == summary["artifact_kind"]
        and runner.get("code_hash") == summary["code_hash"]
        and runner.get("classification") == summary.get("classification")
        and runner.get("control_admissible")
        is summary.get("control_admissible")
        and runner.get("exact") is summary.get("exact")
        and runner.get("exact_arms") == summary.get("exact_arms")
        and runner.get("source") == summary.get("source")
        and arms == expected_arms
        and all(
            actual_hlo_hashes[arm_name] == arms[arm_name]["hlo"]
            for arm_name in ("control", "accepted_split")
        )
        and evidence_bound
        and exact_ledger
        and summary.get("control_admissible") is True
        and summary.get("live_rows") == 1
        and summary.get("performance_claim") is False
        and summary.get("position") == 8155
        and summary.get("results_db_run_id") is None
        and summary.get("status") == "SUCCESS"
        and (exact_schema or nonexact_schema)
        and isinstance(arms, dict)
        and set(arms) == {"control", "accepted_split"}
        and arms["control"].get("output_sha256")
        == "9b52a04e2852719237f4465b28665cbc213b635763303b554bb12345e99a4005"
        and arms["control"].get("comparison", {}).get("elementwise_exact")
        is True
        and arms["accepted_split"].get("comparison", {}).get(
            "elementwise_exact"
        )
        is exact
        and summary.get("runner_sha256")
        == sha256((root / "runner.json").read_bytes()).hexdigest()
        and summary.get("tensor_sha256")
        == sha256((root / "captured_rms_replay.npz").read_bytes()).hexdigest()
    ):
        raise SystemExit("captured RMS replay summary drifted before SUCCESS")
    values = {
        "artifact_kind": summary["artifact_kind"],
        "classification": summary["classification"],
        "code_hash": sys.argv[3],
        "control_admissible": "true",
        "evidence_sha256": sha256(
            (root / "evidence.sha256").read_bytes()
        ).hexdigest(),
        "exact": str(exact).lower(),
        "exact_arms": ",".join(summary["exact_arms"]) or "none",
        "live_rows": "1",
        "performance_claim": "false",
        "remote_objects_sha256": sha256(
            (root / "remote_objects.json").read_bytes()
        ).hexdigest(),
        "remote_prefix": sys.argv[2],
        "results_db_run_id": "none",
        "runner_sha256": summary["runner_sha256"],
        "tensor_sha256": summary["tensor_sha256"],
    }
    for arm_name, arm in sorted(arms.items()):
        values[f"arm_{arm_name}_optimized_hlo_sha256"] = arm["hlo"][
            "optimized_sha256"
        ]
        values[f"arm_{arm_name}_output_sha256"] = arm["output_sha256"]
        values[f"arm_{arm_name}_stablehlo_sha256"] = arm["hlo"][
            "stablehlo_sha256"
        ]
    for key, value in sorted(summary.get("source", {}).items()):
        values[f"source_{key}"] = value
    (root / "SUCCESS").write_text(
        "".join(f"{key}={value}\n" for key, value in sorted(values.items()))
    )
    raise SystemExit(0)
if summary.get("artifact_kind") == "glm52_layer0_dense_partial_capture":
    records = summary.get("capture_records")
    records_sha = sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if not (
        summary.get("classification")
        == "accepted_m32_dense_partials_captured"
        and summary.get("code_hash") == sys.argv[3]
        and summary.get("compile_rows") == 32
        and summary.get("diagnostic_dead_rows") == 31
        and summary.get("exact_arms") == []
        and summary.get("live_rows") == 1
        and summary.get("performance_claim") is False
        and summary.get("result_mode") == "partials_only"
        and summary.get("results_db_run_id") is None
        and summary.get("status") == "SUCCESS"
        and isinstance(summary.get("runner_sha256"), str)
        and summary["runner_sha256"]
        == sha256((root / "runner.json").read_bytes()).hexdigest()
        and isinstance(summary.get("tensor_sha256"), str)
        and summary["tensor_sha256"]
        == sha256((root / "dense_partial_capture.npz").read_bytes()).hexdigest()
        and isinstance(records, dict)
        and set(records)
        == {
            "dense_virtual_partials_bfloat16_bits",
            "post_attention_m32_bfloat16_bits",
        }
    ):
        raise SystemExit("dense partial-capture summary drifted before SUCCESS")
    values = {
        "artifact_kind": summary["artifact_kind"],
        "capture_records_sha256": records_sha,
        "classification": summary["classification"],
        "code_hash": sys.argv[3],
        "compile_rows": summary["compile_rows"],
        "diagnostic_dead_rows": summary["diagnostic_dead_rows"],
        "evidence_sha256": sha256(
            (root / "evidence.sha256").read_bytes()
        ).hexdigest(),
        "exact_arms": "none",
        "live_rows": summary["live_rows"],
        "performance_claim": "false",
        "remote_objects_sha256": sha256(
            (root / "remote_objects.json").read_bytes()
        ).hexdigest(),
        "remote_prefix": sys.argv[2],
        "result_mode": summary["result_mode"],
        "results_db_run_id": "none",
        "runner_sha256": summary["runner_sha256"],
        "tensor_sha256": summary["tensor_sha256"],
    }
    for key, value in sorted(summary.get("source", {}).items()):
        values[f"source_{key}"] = value
    (root / "SUCCESS").write_text(
        "".join(f"{key}={value}\n" for key, value in sorted(values.items()))
    )
    raise SystemExit(0)
records_sha = sha256(
    json.dumps(
        summary["final_layout_records"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()
if records_sha != summary["final_layout_records_sha256"]:
    raise SystemExit("final-layout record manifest hash drifted before SUCCESS")
if summary["final_dense_layout"] and records_sha != (
    "45bfd64e45956627516c36603ccee53d85d713ce6476761e68f1170c51901ba4"
):
    raise SystemExit("pinned final-layout record manifest drifted before SUCCESS")
if summary["final_dense_layout"] and (
    summary.get("scheduled_kernel_geometry_required") is not True
    or summary.get("exact_accepted_kernel_geometry") is not True
):
    raise SystemExit("accepted scheduled kernel geometry drifted before SUCCESS")
if summary["split_layer1_rms"] and (
    summary.get("split_recompute_exact") is not True
    or summary.get("split_output_fusion_exact") is not True
    or summary.get("exact_accepted_scheduled_reduction") is not True
):
    raise SystemExit("accepted split RMS schedule drifted before SUCCESS")
values = {
    "artifact_kind": summary["artifact_kind"],
    "code_hash": sys.argv[3],
    "compile_rows": summary["compile_rows"],
    "live_rows": summary["live_rows"],
    "diagnostic_dead_rows": summary["diagnostic_dead_rows"],
    "result_mode": summary["result_mode"],
    "results_db_run_id": summary["results_db_run_id"],
    "classification": summary["classification"],
    "exact_arms": ",".join(summary["exact_arms"]) or "none",
    "dense_envelope": str(summary["dense_envelope"]).lower(),
    "final_dense_layout": str(summary["final_dense_layout"]).lower(),
    "split_layer1_rms": str(summary["split_layer1_rms"]).lower(),
    "final_layout_records_sha256": records_sha,
    "exact_accepted_kernel_geometry": str(
        summary["exact_accepted_kernel_geometry"]
    ).lower(),
    "scheduled_kernel_geometry_required": str(
        summary["scheduled_kernel_geometry_required"]
    ).lower(),
    "exact_accepted_scheduled_reduction": str(
        summary["exact_accepted_scheduled_reduction"]
    ).lower(),
    "split_recompute_exact": str(summary["split_recompute_exact"]).lower(),
    "split_output_fusion_exact": str(
        summary["split_output_fusion_exact"]
    ).lower(),
    "performance_claim": "false",
    "evidence_sha256": sha256((root / "evidence.sha256").read_bytes()).hexdigest(),
    "remote_objects_sha256": sha256(
        (root / "remote_objects.json").read_bytes()
    ).hexdigest(),
    "remote_prefix": sys.argv[2],
}
for key, value in sorted(summary.get("source", {}).items()):
    values[f"source_{key}"] = value
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
db_run=$(sed -n 's/^results_db_run_id=//p' "$RUN_DIR/SUCCESS")
exact=$(sed -n 's/^exact_arms=//p' "$RUN_DIR/SUCCESS")
say "SUCCESS DB=$db_run exact=$exact"
say "ARCHIVE=$REMOTE_PREFIX"
