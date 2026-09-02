#!/usr/bin/env bash
# Capture and seal one fresh, flat-scatter short-context DSA event oracle.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly RAY_FIREWALL_RULE=allow-ray-pod-internal
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly HARNESS_REPO=/home/gianl/glm-tpu
readonly RAY_LAUNCHER_SHA=b587a8a39262a185c3386989bc9471953b407dc9adbfcc0ae30bb50db68bea23
readonly RAY_VALIDATOR_SHA=00bc87d99e087434eb429a66a3e10df9f2bd9140eac92c2c60ac62ac39ae2db1
readonly ORACLE_REPO=/home/gianl/tpu-inference
readonly ORACLE_PIN=b3c25df47ac98783912dc658878181ec0a8ae16d
readonly INTERNAL_CAPTURE=${GLM_GREENFIELD_DSA_INTERNALS_CAPTURE:-0}
readonly INTERNAL_MODE=${GLM_GREENFIELD_DSA_INTERNALS_MODE:-scorer}
readonly INTERNAL_POSITION_OVERRIDE=${GLM_GREENFIELD_DSA_INTERNALS_POSITION:-}
readonly PROMPT_CACHE_CAPTURE=${GLM_GREENFIELD_PROMPT_CACHE_CAPTURE:-0}
# Full-indexer layer whose prompt index cache the legacy dump captures; the
# legacy kv_caches slot is derived from the sealed registration order.
readonly PROMPT_CACHE_LAYER_ID=${GLM_GREENFIELD_PROMPT_CACHE_LAYER_ID:-0}
readonly PREFILL_PROJECTION_CAPTURE=${GLM_GREENFIELD_ACCEPTED_PREFILL_PROJECTION_CAPTURE:-0}
readonly DECODE_PROJECTION_CAPTURE=${GLM_GREENFIELD_ACCEPTED_DECODE_PROJECTION_CAPTURE:-0}
readonly MAIN_CACHE_CAPTURE=${GLM_GREENFIELD_MAIN_CACHE_CAPTURE:-0}
readonly INTERNAL_LAYER_ID=${GLM_GREENFIELD_DSA_INTERNALS_LAYER_ID:-0}
[[ $INTERNAL_CAPTURE == 0 || $INTERNAL_CAPTURE == 1 ]] || {
  echo "GLM_GREENFIELD_DSA_INTERNALS_CAPTURE must be 0 or 1" >&2
  exit 2
}
case "$INTERNAL_MODE" in
  scorer)
    readonly PROMPT_KEY_CAPTURE=0
    readonly ATTENTION_OUTPUT_CAPTURE=0
    readonly ATTENTION_PROJECTION_CAPTURE=0
    readonly ATTENTION_UPDATE_CAPTURE=0
    readonly DENSE_BOUNDARY_CAPTURE=0
    readonly DENSE_INPUT_CAPTURE=0
    readonly DENSE_PARTIAL_CAPTURE=0
    readonly LAYER1_RMS_INPUT_CAPTURE=0
    ;;
  prompt_key | prompt_key_input)
    readonly PROMPT_KEY_CAPTURE=1
    readonly ATTENTION_OUTPUT_CAPTURE=0
    readonly ATTENTION_PROJECTION_CAPTURE=0
    readonly ATTENTION_UPDATE_CAPTURE=0
    readonly DENSE_BOUNDARY_CAPTURE=0
    readonly DENSE_INPUT_CAPTURE=0
    readonly DENSE_PARTIAL_CAPTURE=0
    readonly LAYER1_RMS_INPUT_CAPTURE=0
    ;;
  attention_output)
    readonly PROMPT_KEY_CAPTURE=0
    readonly ATTENTION_OUTPUT_CAPTURE=1
    readonly ATTENTION_PROJECTION_CAPTURE=0
    readonly ATTENTION_UPDATE_CAPTURE=0
    readonly DENSE_BOUNDARY_CAPTURE=0
    readonly DENSE_INPUT_CAPTURE=0
    readonly DENSE_PARTIAL_CAPTURE=0
    readonly LAYER1_RMS_INPUT_CAPTURE=0
    ;;
  attention_projection)
    readonly PROMPT_KEY_CAPTURE=0
    readonly ATTENTION_OUTPUT_CAPTURE=0
    readonly ATTENTION_PROJECTION_CAPTURE=1
    readonly ATTENTION_UPDATE_CAPTURE=0
    readonly DENSE_BOUNDARY_CAPTURE=0
    readonly DENSE_INPUT_CAPTURE=0
    readonly DENSE_PARTIAL_CAPTURE=0
    readonly LAYER1_RMS_INPUT_CAPTURE=0
    ;;
  attention_update)
    readonly PROMPT_KEY_CAPTURE=0
    readonly ATTENTION_OUTPUT_CAPTURE=0
    readonly ATTENTION_PROJECTION_CAPTURE=0
    readonly ATTENTION_UPDATE_CAPTURE=1
    readonly DENSE_BOUNDARY_CAPTURE=0
    readonly DENSE_INPUT_CAPTURE=0
    readonly DENSE_PARTIAL_CAPTURE=0
    readonly LAYER1_RMS_INPUT_CAPTURE=0
    ;;
  dense_boundary)
    readonly PROMPT_KEY_CAPTURE=0
    readonly ATTENTION_OUTPUT_CAPTURE=0
    readonly ATTENTION_PROJECTION_CAPTURE=0
    readonly ATTENTION_UPDATE_CAPTURE=0
    readonly DENSE_BOUNDARY_CAPTURE=1
    readonly DENSE_INPUT_CAPTURE=0
    readonly DENSE_PARTIAL_CAPTURE=0
    readonly LAYER1_RMS_INPUT_CAPTURE=0
    ;;
  dense_input)
    readonly PROMPT_KEY_CAPTURE=0
    readonly ATTENTION_OUTPUT_CAPTURE=0
    readonly ATTENTION_PROJECTION_CAPTURE=0
    readonly ATTENTION_UPDATE_CAPTURE=0
    readonly DENSE_BOUNDARY_CAPTURE=0
    readonly DENSE_INPUT_CAPTURE=1
    readonly DENSE_PARTIAL_CAPTURE=0
    readonly LAYER1_RMS_INPUT_CAPTURE=0
    ;;
  dense_partial)
    readonly PROMPT_KEY_CAPTURE=0
    readonly ATTENTION_OUTPUT_CAPTURE=0
    readonly ATTENTION_PROJECTION_CAPTURE=0
    readonly ATTENTION_UPDATE_CAPTURE=0
    readonly DENSE_BOUNDARY_CAPTURE=0
    readonly DENSE_INPUT_CAPTURE=0
    readonly DENSE_PARTIAL_CAPTURE=1
    readonly LAYER1_RMS_INPUT_CAPTURE=0
    ;;
  layer1_rms_input)
    echo "REFUSED REJECTED_OBSERVER_PERTURBATION: layer1_rms_input callback mode reproduced DB551 in protected run greenfield_legacy_layer1_rms_input_p8155_20260829T192233297523063Z; no cloud, JAX, Ray, or model action is permitted" >&2
    exit 2
    ;;
  *)
    echo "unsupported GLM_GREENFIELD_DSA_INTERNALS_MODE=$INTERNAL_MODE" >&2
    exit 2
    ;;
esac
[[ $PROMPT_CACHE_LAYER_ID =~ ^[0-9]+$ ]] || {
  echo "GLM_GREENFIELD_PROMPT_CACHE_LAYER_ID must be a nonnegative integer" >&2
  exit 2
}
PROMPT_CACHE_SLOT=$(PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python -c '
import sys
from glm_tpu.greenfield.validation.prompt_index_cache import expected_prompt_cache_slot
print(expected_prompt_cache_slot(int(sys.argv[1])))
' "$PROMPT_CACHE_LAYER_ID") || {
  echo "GLM_GREENFIELD_PROMPT_CACHE_LAYER_ID=$PROMPT_CACHE_LAYER_ID owns no DSA prompt index cache" >&2
  exit 2
}
readonly PROMPT_CACHE_SLOT
[[ $PROMPT_CACHE_CAPTURE == 0 || $PROMPT_CACHE_CAPTURE == 1 ]] || {
  echo "GLM_GREENFIELD_PROMPT_CACHE_CAPTURE must be 0 or 1" >&2
  exit 2
}
[[ $PREFILL_PROJECTION_CAPTURE == 0 || $PREFILL_PROJECTION_CAPTURE == 1 ]] || {
  echo "GLM_GREENFIELD_ACCEPTED_PREFILL_PROJECTION_CAPTURE must be 0 or 1" >&2
  exit 2
}
[[ $DECODE_PROJECTION_CAPTURE == 0 || $DECODE_PROJECTION_CAPTURE == 1 ]] || {
  echo "GLM_GREENFIELD_ACCEPTED_DECODE_PROJECTION_CAPTURE must be 0 or 1" >&2
  exit 2
}
[[ $MAIN_CACHE_CAPTURE == 0 || $MAIN_CACHE_CAPTURE == 1 ]] || {
  echo "GLM_GREENFIELD_MAIN_CACHE_CAPTURE must be 0 or 1" >&2
  exit 2
}
if [[ ! $INTERNAL_LAYER_ID =~ ^[0-9]+$ ]] ||
  ! ((INTERNAL_LAYER_ID <= 2 || (INTERNAL_LAYER_ID >= 6 && INTERNAL_LAYER_ID < 78 && (INTERNAL_LAYER_ID - 6) % 4 == 0))); then
  echo "DSA internal layer must be a full-indexer producer" >&2
  exit 2
fi
if [[ $INTERNAL_LAYER_ID == 0 && $INTERNAL_MODE == scorer ]]; then
  readonly INTERNAL_COMPARE_LAYER0=1
else
  readonly INTERNAL_COMPARE_LAYER0=0
fi
if [[ $MAIN_CACHE_CAPTURE == 1 ]]; then
  readonly OBSERVER_DEV_REPO=/home/gianl/tpu-inference-greenfield-main-cache-observer
  readonly OBSERVER_BRANCH=greenfield/legacy-main-cache-observer
  readonly OBSERVER_RUNTIME_REPO=/home/gianl/tpu-inference-main-cache-3443515d9
  readonly OBSERVER_COMMIT_DISTANCE=1
  readonly LEGACY_PIN=3443515d9d3c42412558b778c608aaf07c6c89ff
  readonly LEGACY_REPO=$OBSERVER_RUNTIME_REPO
  readonly LEGACY_SOURCE_REPO=$OBSERVER_DEV_REPO
elif [[ $INTERNAL_CAPTURE == 1 ]]; then
  if [[ $LAYER1_RMS_INPUT_CAPTURE == 1 ]]; then
    readonly OBSERVER_DEV_REPO=/home/gianl/tpu-inference-greenfield-layer1-rms-input-observer
    readonly OBSERVER_BRANCH=greenfield/legacy-layer1-rms-input-observer
    readonly OBSERVER_RUNTIME_REPO=/home/gianl/tpu-inference-dsa-internal-8dc7d20fe
    readonly OBSERVER_COMMIT_DISTANCE=12
    readonly LEGACY_PIN=8dc7d20fedca5a98c27bfd1774827305973fa4c1
  else
    readonly OBSERVER_DEV_REPO=/home/gianl/tpu-inference-greenfield-dsa-internal-observer
    readonly OBSERVER_BRANCH=greenfield/legacy-dsa-internal-observer
  fi
  if [[ $LAYER1_RMS_INPUT_CAPTURE == 1 ]]; then
    :
  elif [[ $DENSE_PARTIAL_CAPTURE == 1 ]]; then
    readonly OBSERVER_RUNTIME_REPO=/home/gianl/tpu-inference-dsa-internal-4e3aa9666
    readonly OBSERVER_COMMIT_DISTANCE=12
    readonly LEGACY_PIN=4e3aa9666cefa38deba9c2824d5125c2e32ab2cf
  elif [[ $DENSE_INPUT_CAPTURE == 1 ]]; then
    readonly OBSERVER_RUNTIME_REPO=/home/gianl/tpu-inference-dsa-internal-0c2f7f28a
    readonly OBSERVER_COMMIT_DISTANCE=11
    readonly LEGACY_PIN=0c2f7f28a075a51f5eb51dc98bbb74e363d3290f
  elif [[ $DENSE_BOUNDARY_CAPTURE == 1 ]]; then
    # The residual-only contract must never materialize the dense output.
    readonly OBSERVER_RUNTIME_REPO=/home/gianl/tpu-inference-dsa-internal-2c4fbc155
    readonly OBSERVER_COMMIT_DISTANCE=13
    readonly LEGACY_PIN=2c4fbc155157ad52a4e61cf59f92984d1101e142
  elif [[ $ATTENTION_UPDATE_CAPTURE == 1 ]]; then
    readonly OBSERVER_RUNTIME_REPO=/home/gianl/tpu-inference-dsa-internal-23ab8780f
    readonly OBSERVER_COMMIT_DISTANCE=9
    readonly LEGACY_PIN=23ab8780f3066ae1be12657d4f45daa7ea353761
  elif [[ $ATTENTION_PROJECTION_CAPTURE == 1 ]]; then
    readonly OBSERVER_RUNTIME_REPO=/home/gianl/tpu-inference-dsa-internal-11c250648
    readonly OBSERVER_COMMIT_DISTANCE=8
    readonly LEGACY_PIN=11c2506480e98902d66a88309533f624c994d202
  elif [[ $ATTENTION_OUTPUT_CAPTURE == 1 ]]; then
    readonly OBSERVER_RUNTIME_REPO=/home/gianl/tpu-inference-dsa-internal-bf8a03e26
    readonly OBSERVER_COMMIT_DISTANCE=7
    readonly LEGACY_PIN=bf8a03e264971c8efba99a346d1e8189ef0ff518
  elif [[ $INTERNAL_MODE == prompt_key_input ]]; then
    readonly OBSERVER_RUNTIME_REPO=/home/gianl/tpu-inference-dsa-internal-89fc453b6
    readonly OBSERVER_COMMIT_DISTANCE=6
    readonly LEGACY_PIN=89fc453b6116ac3df71e666db6f4659775b313c3
  elif [[ $INTERNAL_MODE == prompt_key ]]; then
    readonly OBSERVER_RUNTIME_REPO=/home/gianl/tpu-inference-dsa-internal-9c1d6b3b9
    readonly OBSERVER_COMMIT_DISTANCE=3
    readonly LEGACY_PIN=9c1d6b3b950d5c5dd45bdf885058202517097eba
  else
    readonly OBSERVER_RUNTIME_REPO=/home/gianl/tpu-inference-dsa-internal-83ff4a357
    readonly OBSERVER_COMMIT_DISTANCE=2
    readonly LEGACY_PIN=83ff4a3576602ca844ea090550139a2ff00b0bb1
  fi
  readonly LEGACY_REPO=$OBSERVER_RUNTIME_REPO
  readonly LEGACY_SOURCE_REPO=$OBSERVER_DEV_REPO
else
  readonly LEGACY_REPO=$ORACLE_REPO
  readonly LEGACY_SOURCE_REPO=$ORACLE_REPO
  readonly LEGACY_PIN=$ORACLE_PIN
fi
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly VLLM_REFERENCE_REPO=/home/gianl/vllm-build-a30addc
readonly VLLM_ARCHIVE_REPO=$VLLM_REFERENCE_REPO
readonly VLLM_PIN=a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c
readonly VLLM_IR_LAYERNORM_SHA=d8e4380ca97d2c719836a7e15fb410a73d354b15d79fc54275b573bbb1c06910
readonly VLLM_EXECUTOR_LAYERNORM_SHA=53c6abdab25dc1675f26f4c8fc5ba2094f1fb4a106334e581f630436210d0c9b
readonly GOLDEN_SOURCE_DIR=/home/gianl/gcs-models/manifests/golden_v1
readonly GOLDEN_MANIFEST_SHA=916d421a10de9495086c0ad52645c9937746c88bae87a5ca1a69483bfe60d45d
readonly GOLDEN_MANIFEST_BYTES=321146
readonly LAYER1_OBSERVER_TRACKED_FILE_COUNT=947
readonly OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly DISK_MIN_FREE_GB=10
readonly DISK_WARN_FREE_GB=15
readonly MODEL_ID=zai-org/GLM-5.2-FP8
readonly REFERENCE_8K_DSA_ORACLE_ROOT=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z
readonly REFERENCE_8K_DSA_ORACLE=$REFERENCE_8K_DSA_ORACLE_ROOT/oracle
readonly REFERENCE_8K_DSA_ORACLE_MANIFEST_SHA=f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da
readonly REFERENCE_8K_DSA_ORACLE_SUCCESS_SHA=0b798974ae8a9f95c32d3aa2eff532213624f1e2ae7f1de809a161e18dbdf1b9
readonly LAYER0_INPUT_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z
readonly LAYER0_INPUT_MANIFEST_SHA=574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141
readonly DISTRIBUTED_Q_A_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_association_20260807T231449677046310Z/distributed_q_a_norm_artifact
readonly DISTRIBUTED_Q_A_MANIFEST_SHA=7518e7eff0487f0dc02cd4b0ff1c3d0fc3ef9ca7c43dcded7d809120e30d8c16
readonly DISTRIBUTED_Q_A_CODE_HASH=ea879a24d196f61e238a22ee5bb393d3b6fa938d
readonly ACCEPTED_PROMPT_CACHE_SHA=3808d502f3ea1829bf12ab7585d66f15dd83bf640657a17c35daabf5ab1859d1
readonly DB512_PROMPT_CACHE_SHA=52bf55ed5e9ea74a59551351a21ae830fb39e289e82eaff636fb9ab84d7dcd8a
readonly MAIN_CACHE_INGREDIENTS_DIR=/home/gianl/glm-run/greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_oracle_dsa_layer0_ingredients_trace2_20260811T033957390557074Z/layer0_ingredients
readonly MAIN_CACHE_INGREDIENTS_CODE_HASH=4d4e2c5da53554b3ab015d26cc6d1e058a7b48dd
readonly MAIN_CACHE_INGREDIENTS_CONTRACT_SHA=9c3ec9fae5be31f791d737804adf6bcbaaa7f481120ce706be7a43e5dcd37693
readonly MAIN_CACHE_INGREDIENTS_TENSOR_SHA=fd76cd4c6e61be773fe49bfffaceb1467c0865420d68016471df1ebe5140249c
readonly ATTENTION_PROJECTION_INGREDIENTS_TAG=greenfield_table_on_layer0_ingredients_p8155_20260812T021718885910564Z
readonly ATTENTION_PROJECTION_INGREDIENTS_DIR=/home/gianl/gcs-models/results/$ATTENTION_PROJECTION_INGREDIENTS_TAG/layer0_ingredients
readonly ATTENTION_PROJECTION_INGREDIENTS_CODE_HASH=d4862e5601bb81761a6c3b695df6a439fb87c909
readonly ATTENTION_PROJECTION_INGREDIENTS_CONTRACT_SHA=02f0f1c3e22b08c6646167ac5a7ffaf5056932003ee722752f7d40e26e714864
readonly ATTENTION_PROJECTION_INGREDIENTS_TENSOR_SHA=c06fe575f0518e981c3099a8033c1978b01fd0cff843ecbbd7a25cebed8d0e95
readonly MAIN_ROPE_TABLE_SHA=6a22140fc2aec475399738c6fc0f29be2a6c419feb0249aee35681c607c80701
readonly PROJECTION_REDUCTION_TAG=greenfield_layer0_projection_reduction_20260812T164052560787241Z
readonly PROJECTION_REDUCTION_DIR=/home/gianl/glm-run/$PROJECTION_REDUCTION_TAG
readonly PROJECTION_REDUCTION_CODE_HASH=e2a3a74a3b2ef1fa8f3b9cb1c5d7ec65f833eafc
readonly PROJECTION_REDUCTION_RUNNER_SHA=303dd91eed1d75e0cd443645c5f1ef745f259c51596d0651646bb141db8f16f8
readonly PROJECTION_REDUCTION_TENSOR_SHA=e801d5471697fefd1477c46603698289de93818d08d214bdf56e576f52819e0e
readonly PROJECTION_REDUCTION_SUMMARY_SHA=90090ba9999812082727ed56b734163f07e3c27b4eb88fa049b9c2504516e782
readonly PROJECTION_REDUCTION_SUCCESS_SHA=7744356f63b67cc813901499d0828c029ea5a9c985a5dac65525700457f79985
readonly PROJECTION_REDUCTION_RUN_ID=538
readonly PROJECTION_REDUCTION_REMOTE=$APPROVED_BUCKET/results/$PROJECTION_REDUCTION_TAG
readonly DENSE_CONVOLUTION_TAG=greenfield_layer0_dense_convolution_20260813T005213127235575Z
readonly DENSE_CONVOLUTION_DIR=/home/gianl/glm-run/$DENSE_CONVOLUTION_TAG
readonly DENSE_CONVOLUTION_CODE_HASH=2f63779309b25c71c1cc7d35ff97715ae4bf631e
readonly DENSE_CONVOLUTION_RUNNER_SHA=876353e2504d728343223f03be9092a08d2662924ceb4b88d6f09563101bad91
readonly DENSE_CONVOLUTION_TENSOR_SHA=2cdf128976eb7e04d5c84e012f066d1766361af57896cd22f83c01c7d704e1b9
readonly DENSE_CONVOLUTION_SUMMARY_SHA=9b277ca495d3b9e9ce497d2bf78a520a2672f133e68228d14a10937d4a4b1449
readonly DENSE_CONVOLUTION_SUCCESS_SHA=d4c01377daae55ea23b329b1b6dc819b9595dc80f7d35521d0cbdd7d28caa799
readonly DENSE_CONVOLUTION_RUN_ID=540
readonly DENSE_CONVOLUTION_REMOTE=$APPROVED_BUCKET/results/$DENSE_CONVOLUTION_TAG
readonly DENSE_PARTIAL_PROBE_TAG=greenfield_layer0_dense_partial_capture_20260813T200736889447458Z
readonly DENSE_PARTIAL_PROBE_DIR=/home/gianl/glm-run/$DENSE_PARTIAL_PROBE_TAG
readonly DENSE_PARTIAL_PROBE_RUNNER_SHA=d22b35f664409485b697fdb579665dc4b1ecf42683a2aaff9e9a4d7481ee8343
readonly DENSE_PARTIAL_PROBE_TENSOR_SHA=f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298
readonly DENSE_PARTIAL_PROBE_SUMMARY_SHA=c349b5fd458f34986d8cc59c0f026af6b0a4c4e998f8691d6f6aa83a9b55e416
readonly DENSE_PARTIAL_PROBE_SUCCESS_SHA=6cac897695fc1e78d0a10c0e36c993cffd281c6a88721d8955fa470bd44b0b85
readonly DENSE_PARTIAL_PROBE_REMOTE=$APPROVED_BUCKET/results/$DENSE_PARTIAL_PROBE_TAG
readonly DB550_BOUNDARY=/home/gianl/gcs-models/results/greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/dense_partial_capture.npz
readonly DB550_BOUNDARY_SHA=f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298
readonly STRADDLER_CLASSIFICATION=$WORKTREE/docs/artifacts/pp16-feature2-layer1-straddler-classification.json
readonly STRADDLER_CLASSIFICATION_SHA=eebe1c5d5ba475a5faf000243d881657754fc47ed2345fe2691d923c1d457b36
if [[ $LAYER1_RMS_INPUT_CAPTURE == 1 ]]; then
  readonly INTERNAL_LAYER=model.layers.1.input_layernorm
else
  readonly INTERNAL_LAYER=model.layers.${INTERNAL_LAYER_ID}.self_attn.attn
fi

PROFILE=${GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE:-2k}
case "$PROFILE" in
  2k)
    TOKEN_ORACLE_TAG=greenfield_short_context_oracle_20260806T202544155912103Z
    TOKEN_ORACLE_SHA=f580c14954bcbd0d973b6fe8158520992a18a1375ed88cff9cceb8e01c7efe19
    BENCHMARK_LENGTH=2040
    BENCHMARK_DEPTH=0.25
    BENCHMARK_MAX_LEN=2560
    BENCHMARK_MAX_BATCHED_TOKENS=2048
    BENCHMARK_NUM_BLOCKS=8
    EXPECTED_BENCHMARK=passkey_L2040_d0.25
    EXPECTED_PROMPT_TOKENS=2034
    EXPECTED_GENERATED_TOKENS=20
    EXPECTED_SEED=283835
    EXPECTED_GOLD=110391
    FIRST_SOURCE_STEP=2
    FIRST_DECODE_POSITION=2034
    TAG_PREFIX=greenfield_short_context_dsa_oracle
    ;;
  8k)
    TOKEN_ORACLE_TAG=greenfield_short_context_oracle_8k_20260807T172307269147351Z
    TOKEN_ORACLE_SHA=e4fbcbdbf0fc8b1969e2f82ee457ab1563db4a8b37d2dea2bc4d1e828a13acf2
    BENCHMARK_LENGTH=8192
    BENCHMARK_DEPTH=0.5
    BENCHMARK_MAX_LEN=8704
    BENCHMARK_MAX_BATCHED_TOKENS=2048
    BENCHMARK_NUM_BLOCKS=24
    EXPECTED_BENCHMARK=passkey_L8192_d0.5
    EXPECTED_PROMPT_TOKENS=8155
    EXPECTED_GENERATED_TOKENS=20
    EXPECTED_SEED=1093997
    EXPECTED_GOLD=881446
    # Four 2,048-token prefill chunks precede recurrent decode at 8K.
    FIRST_SOURCE_STEP=5
    FIRST_DECODE_POSITION=8155
    TAG_PREFIX=greenfield_short_context_dsa_oracle_8k
    ;;
  *)
    echo "unsupported short-context DSA oracle profile: $PROFILE" >&2
    exit 2
    ;;
esac
readonly PROFILE TOKEN_ORACLE_TAG TOKEN_ORACLE_SHA BENCHMARK_LENGTH
readonly BENCHMARK_DEPTH BENCHMARK_MAX_LEN BENCHMARK_MAX_BATCHED_TOKENS
readonly BENCHMARK_NUM_BLOCKS EXPECTED_BENCHMARK EXPECTED_PROMPT_TOKENS
readonly EXPECTED_GENERATED_TOKENS EXPECTED_SEED EXPECTED_GOLD
readonly FIRST_SOURCE_STEP FIRST_DECODE_POSITION TAG_PREFIX
INTERNAL_TARGET_POSITION=${INTERNAL_POSITION_OVERRIDE:-$FIRST_DECODE_POSITION}
[[ $INTERNAL_TARGET_POSITION =~ ^[0-9]+$ ]] || {
  echo "DSA internal position must be a nonnegative integer" >&2
  exit 2
}
if [[ $PROMPT_KEY_CAPTURE == 1 ]]; then
  [[ $INTERNAL_CAPTURE == 1 && $INTERNAL_LAYER_ID == 0 && \
     $PROFILE == 8k && $PROMPT_CACHE_CAPTURE == 1 && $PROMPT_CACHE_LAYER_ID == 0 && \
     $INTERNAL_TARGET_POSITION -lt $EXPECTED_PROMPT_TOKENS ]] || {
    echo "prompt-key capture requires layer 0, 8K, layer-0 prompt cache, and a prompt position" >&2
    exit 2
  }
fi
if [[ $ATTENTION_OUTPUT_CAPTURE == 1 || $ATTENTION_PROJECTION_CAPTURE == 1 || \
      $ATTENTION_UPDATE_CAPTURE == 1 || $DENSE_BOUNDARY_CAPTURE == 1 || \
      $DENSE_INPUT_CAPTURE == 1 || $DENSE_PARTIAL_CAPTURE == 1 ]]; then
  [[ $INTERNAL_CAPTURE == 1 && $INTERNAL_LAYER_ID == 0 && \
     $PROFILE == 8k && $INTERNAL_TARGET_POSITION == 8155 && \
     $PROMPT_CACHE_CAPTURE == 0 && $PREFILL_PROJECTION_CAPTURE == 0 && \
     $DECODE_PROJECTION_CAPTURE == 0 && $MAIN_CACHE_CAPTURE == 0 ]] || {
    echo "layer-0 boundary capture requires isolated 8K position 8155 mode" >&2
    exit 2
  }
fi
if [[ $LAYER1_RMS_INPUT_CAPTURE == 1 ]]; then
  [[ $INTERNAL_CAPTURE == 1 && $INTERNAL_LAYER_ID == 1 && \
     $PROFILE == 8k && $INTERNAL_TARGET_POSITION == 8155 && \
     $PROMPT_CACHE_CAPTURE == 0 && $PREFILL_PROJECTION_CAPTURE == 0 && \
     $DECODE_PROJECTION_CAPTURE == 0 && $MAIN_CACHE_CAPTURE == 0 ]] || {
    echo "layer-1 RMS-input capture requires isolated 8K position 8155 mode" >&2
    exit 2
  }
  [[ -r $DB550_BOUNDARY && -r $STRADDLER_CLASSIFICATION && \
     $(sha256sum "$DB550_BOUNDARY" | awk '{print $1}') == "$DB550_BOUNDARY_SHA" && \
     $(sha256sum "$STRADDLER_CLASSIFICATION" | awk '{print $1}') == "$STRADDLER_CLASSIFICATION_SHA" ]] || {
    echo "layer-1 RMS-input source contract drifted" >&2
    exit 2
  }
  vllm_source_ok=0
  if [[ -r $VLLM_REFERENCE_REPO/vllm/ir/ops/layernorm.py &&
        -r $VLLM_REFERENCE_REPO/vllm/model_executor/layers/layernorm.py &&
        $(git -C "$VLLM_REFERENCE_REPO" rev-parse HEAD) == "$VLLM_PIN" &&
        -z $(git -C "$VLLM_REFERENCE_REPO" status --porcelain --untracked-files=no) &&
        $(sha256sum "$VLLM_REFERENCE_REPO/vllm/ir/ops/layernorm.py" | awk '{print $1}') == "$VLLM_IR_LAYERNORM_SHA" &&
        $(sha256sum "$VLLM_REFERENCE_REPO/vllm/model_executor/layers/layernorm.py" | awk '{print $1}') == "$VLLM_EXECUTOR_LAYERNORM_SHA" ]]; then
    vllm_source_ok=1
  fi
  [[ $vllm_source_ok == 1 ]] || {
    echo "layer-1 RMS-input source contract drifted" >&2
    exit 2
  }
fi
if [[ $DENSE_PARTIAL_CAPTURE == 1 ]]; then
  [[ -r $DENSE_PARTIAL_PROBE_DIR/runner.json &&
     -r $DENSE_PARTIAL_PROBE_DIR/dense_partial_capture.npz &&
     -r $DENSE_PARTIAL_PROBE_DIR/summary.json &&
     -r $DENSE_PARTIAL_PROBE_DIR/SUCCESS ]] || {
    echo "protected DB548 dense-partial evidence is unavailable" >&2
    exit 2
  }
  [[ $(sha256sum "$DENSE_PARTIAL_PROBE_DIR/runner.json" | awk '{print $1}') == \
     "$DENSE_PARTIAL_PROBE_RUNNER_SHA" &&
     $(sha256sum "$DENSE_PARTIAL_PROBE_DIR/dense_partial_capture.npz" | awk '{print $1}') == \
     "$DENSE_PARTIAL_PROBE_TENSOR_SHA" &&
     $(sha256sum "$DENSE_PARTIAL_PROBE_DIR/summary.json" | awk '{print $1}') == \
     "$DENSE_PARTIAL_PROBE_SUMMARY_SHA" &&
     $(sha256sum "$DENSE_PARTIAL_PROBE_DIR/SUCCESS" | awk '{print $1}') == \
     "$DENSE_PARTIAL_PROBE_SUCCESS_SHA" ]] || {
    echo "protected DB548 dense-partial evidence identity drifted" >&2
    exit 2
  }
fi
if [[ $DENSE_BOUNDARY_CAPTURE == 1 || $DENSE_INPUT_CAPTURE == 1 ]]; then
  [[ -r $DENSE_CONVOLUTION_DIR/runner.json &&
     -r $DENSE_CONVOLUTION_DIR/dense_convolution.npz &&
     -r $DENSE_CONVOLUTION_DIR/summary.json &&
     -r $DENSE_CONVOLUTION_DIR/SUCCESS ]] || {
    echo "protected dense-convolution evidence is unavailable" >&2
    exit 2
  }
  [[ $(sha256sum "$DENSE_CONVOLUTION_DIR/runner.json" | awk '{print $1}') == \
     "$DENSE_CONVOLUTION_RUNNER_SHA" &&
     $(sha256sum "$DENSE_CONVOLUTION_DIR/dense_convolution.npz" | awk '{print $1}') == \
     "$DENSE_CONVOLUTION_TENSOR_SHA" &&
     $(sha256sum "$DENSE_CONVOLUTION_DIR/summary.json" | awk '{print $1}') == \
     "$DENSE_CONVOLUTION_SUMMARY_SHA" &&
     $(sha256sum "$DENSE_CONVOLUTION_DIR/SUCCESS" | awk '{print $1}') == \
     "$DENSE_CONVOLUTION_SUCCESS_SHA" ]] || {
    echo "protected dense-convolution evidence identity drifted" >&2
    exit 2
  }
  /home/gianl/vllm-env/bin/python - "$RESULTS_DB" \
    "$DENSE_CONVOLUTION_RUN_ID" "$DENSE_CONVOLUTION_TAG" \
    "$DENSE_CONVOLUTION_CODE_HASH" <<'PY'
import json
import sqlite3
import sys

db_path, run_id_text, run_tag, code_hash = sys.argv[1:]
run_id = int(run_id_text)
connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
run = connection.execute(
    "SELECT run_id, created_utc, model, model_revision, harness_git, fork_git, "
    "env_json, pod, note FROM runs WHERE run_id = ?",
    (run_id,),
).fetchone()
items = connection.execute(
    "SELECT id, run_id, benchmark, item_id, asked_utc, prompt, gold, "
    "raw_output, extracted, correct, score, n_prompt_tokens, n_gen_tokens, "
    "latency_ms, seed, finish_reason, truncated FROM items WHERE run_id = ?",
    (run_id,),
).fetchall()
summaries = connection.execute(
    "SELECT id, run_id, benchmark, created_utc, n, metric, value, card_value, "
    "delta, note FROM summary WHERE run_id = ?",
    (run_id,),
).fetchall()
connection.close()
if run is None:
    raise SystemExit("protected DB540 run is absent")
environment = json.loads(run[6])
expected_environment = {
    "GLM_ENGINE": "greenfield_dense_convolution_probe",
    "checkpoint_manifest_sha256": (
        "de46d38e404c637209f95505291105e89a6e7f95270fe91375a55ea79b5f7134"
    ),
    "classification": "accepted_dense_convolution_nonexact",
    "db538_tensor_sha256": (
        "e801d5471697fefd1477c46603698289de93818d08d214bdf56e576f52819e0e"
    ),
    "greenfield_code_hash": code_hash,
    "greenfield_run_tag": run_tag,
}
if (
    run != (
        540,
        "2026-08-13T00:53:15+00:00",
        "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-convolution",
        "native-jax-db538-dense-convolution-v1",
        "2f63779",
        "b3c25df47",
        run[6],
        "db-v4-64-od",
        "Protected layer-0 dense convolution discriminator; no performance claim.",
    )
    or environment != expected_environment
    or items != [(
        1824,
        540,
        "greenfield_layer0_dense_convolution",
        "position8155",
        "2026-08-13T00:53:15+00:00",
        "Sealed exact StrategyND attention boundary at first 8K decode row.",
        "Exact accepted BF16 layer-1 normalized hidden [6144].",
        '{"classification": "accepted_dense_convolution_nonexact", '
        '"exact_arms": [], "mismatch_counts": '
        '{"accepted_dense_convolution": 1073}}',
        "none",
        0,
        0.0,
        None,
        None,
        None,
        None,
        None,
        None,
    )]
    or summaries != [(
        771,
        540,
        "greenfield_layer0_dense_convolution",
        "2026-08-13T00:53:15+00:00",
        1,
        "probe_contract_valid",
        1.0,
        None,
        None,
        "Diagnostic layer-0 arithmetic classification only; no decoder claim.",
    )]
):
    raise SystemExit("protected DB540 live DB identity drifted")
PY
fi
if [[ $ATTENTION_UPDATE_CAPTURE == 1 ]]; then
  [[ -r $PROJECTION_REDUCTION_DIR/runner.json &&
     -r $PROJECTION_REDUCTION_DIR/projection_reduction.npz &&
     -r $PROJECTION_REDUCTION_DIR/summary.json &&
     -r $PROJECTION_REDUCTION_DIR/SUCCESS ]] || {
    echo "protected projection/reduction evidence is unavailable" >&2
    exit 2
  }
  [[ $(sha256sum "$PROJECTION_REDUCTION_DIR/runner.json" | awk '{print $1}') == \
     "$PROJECTION_REDUCTION_RUNNER_SHA" &&
     $(sha256sum "$PROJECTION_REDUCTION_DIR/projection_reduction.npz" | awk '{print $1}') == \
     "$PROJECTION_REDUCTION_TENSOR_SHA" &&
     $(sha256sum "$PROJECTION_REDUCTION_DIR/summary.json" | awk '{print $1}') == \
     "$PROJECTION_REDUCTION_SUMMARY_SHA" &&
     $(sha256sum "$PROJECTION_REDUCTION_DIR/SUCCESS" | awk '{print $1}') == \
     "$PROJECTION_REDUCTION_SUCCESS_SHA" ]] || {
    echo "protected projection/reduction evidence identity drifted" >&2
    exit 2
  }
  /home/gianl/vllm-env/bin/python - "$RESULTS_DB" \
    "$PROJECTION_REDUCTION_RUN_ID" "$PROJECTION_REDUCTION_TAG" \
    "$PROJECTION_REDUCTION_CODE_HASH" <<'PY'
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
if [[ $PREFILL_PROJECTION_CAPTURE == 1 ]]; then
  [[ $PROFILE == 8k && $INTERNAL_CAPTURE == 0 && $PROMPT_CACHE_CAPTURE == 0 && \
     $DECODE_PROJECTION_CAPTURE == 0 && $MAIN_CACHE_CAPTURE == 0 ]] || {
    echo "accepted prefill projection capture requires plain accepted 8K oracle mode" >&2
    exit 2
  }
fi
if [[ $DECODE_PROJECTION_CAPTURE == 1 ]]; then
  [[ $PROFILE == 8k && $INTERNAL_CAPTURE == 0 && \
     $PROMPT_CACHE_CAPTURE == 0 && $PREFILL_PROJECTION_CAPTURE == 0 && \
     $MAIN_CACHE_CAPTURE == 0 ]] || {
    echo "accepted decode projection capture requires isolated accepted 8K oracle mode" >&2
    exit 2
  }
fi
if [[ $MAIN_CACHE_CAPTURE == 1 ]]; then
  [[ $PROFILE == 8k && $INTERNAL_CAPTURE == 0 && \
     $PROMPT_CACHE_CAPTURE == 0 && $PREFILL_PROJECTION_CAPTURE == 0 && \
     $DECODE_PROJECTION_CAPTURE == 0 ]] || {
    echo "main-cache capture requires isolated accepted 8K oracle mode" >&2
    exit 2
  }
fi
if [[ $ATTENTION_PROJECTION_CAPTURE == 1 ]]; then
  [[ -r $ATTENTION_PROJECTION_INGREDIENTS_DIR/contract.json &&
     -r $ATTENTION_PROJECTION_INGREDIENTS_DIR/position_8155_ingredients.npz ]] || {
    echo "protected table-on attention ingredients are unavailable" >&2
    exit 2
  }
  [[ $(sha256sum "$ATTENTION_PROJECTION_INGREDIENTS_DIR/contract.json" | awk '{print $1}') == \
     "$ATTENTION_PROJECTION_INGREDIENTS_CONTRACT_SHA" &&
     $(sha256sum "$ATTENTION_PROJECTION_INGREDIENTS_DIR/position_8155_ingredients.npz" | awk '{print $1}') == \
     "$ATTENTION_PROJECTION_INGREDIENTS_TENSOR_SHA" ]] || {
    echo "protected table-on attention ingredient identity drifted" >&2
    exit 2
  }
fi
readonly INTERNAL_TARGET_POSITION
readonly TOKEN_ORACLE_DIR=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/$PROFILE/$TOKEN_ORACLE_TAG/oracle

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
HARNESS_PIN=$(git -C "$HARNESS_REPO" rev-parse HEAD)
HARNESS_SHORT=$(git -C "$HARNESS_REPO" rev-parse --short HEAD)
LEGACY_SHORT=$(git -C "$LEGACY_SOURCE_REPO" rev-parse --short HEAD)
ORACLE_SHORT=$(git -C "$ORACLE_REPO" rev-parse --short HEAD)
TAG=${GLM_GREENFIELD_SHORT_DSA_ORACLE_TAG:-${TAG_PREFIX}_$(date -u +%Y%m%dT%H%M%S%NZ)}
if [[ $LAYER1_RMS_INPUT_CAPTURE == 1 && ! $TAG =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "layer-1 RMS-input tag contains unsafe characters" >&2
  exit 2
fi
RUN_DIR=/home/gianl/glm-run/$TAG
SOURCE_DIR=$RUN_DIR/source_dumps
ORACLE_DIR=$RUN_DIR/oracle
REMOTE_PREFIX=${GLM_GREENFIELD_SHORT_DSA_REMOTE_PREFIX:-$APPROVED_BUCKET/oracles/greenfield/glm52/short_context_dsa/$PROFILE/$TAG}
DUMP_PREFIX=/tmp/$TAG/topk.npz
INTERNAL_DUMP_PREFIX=/tmp/$TAG/internals.npz
PROMPT_CACHE_DUMP_PREFIX=/tmp/$TAG/index_cache.npz
PROMPT_CACHE_RESULT_DIR=$RUN_DIR/prompt_index_cache
PROMPT_CACHE_COMPARISON_DIR=$RUN_DIR/prompt_index_cache_comparison
MAIN_CACHE_DUMP_PREFIX=/tmp/$TAG/main_cache.npz
MAIN_CACHE_RESULT_DIR=$RUN_DIR/layer0_main_cache_comparison
PREFILL_PROFILE_PREFIX=/tmp/$TAG/prefill_projection_profile
PREFILL_HLO_PREFIX=/tmp/$TAG/prefill_projection_hlo
PREFILL_PROJECTION_RESULT_DIR=$RUN_DIR/accepted_prompt_projection_lowering
DECODE_HLO_PREFIX=/tmp/$TAG/decode_projection_hlo_raw
DECODE_HLO_COMPACT_PREFIX=/tmp/$TAG/decode_projection_hlo
DECODE_PROJECTION_RESULT_DIR=$RUN_DIR/accepted_decode_projection_lowering
readonly VLLM_RUNTIME_ROOT=/tmp/glm_vllm_$TAG
readonly VLLM_RUNTIME_ARCHIVE_REMOTE=/tmp/glm_vllm_$TAG.tar.gz
VLLM_RUNTIME_ARCHIVE_SHA=
VLLM_RUNTIME_FILE_COUNT=0
readonly OBSERVER_BUNDLE_LOCAL=$RUN_DIR/observer_${LEGACY_PIN}.bundle
readonly OBSERVER_BUNDLE_REMOTE=/tmp/glm_observer_$TAG.bundle
readonly OBSERVER_BUNDLE_TEMP=/tmp/glm_observer_$TAG.tmp
OBSERVER_BUNDLE_SHA=
OBSERVER_TRACKED_FILE_COUNT=0
if [[ $INTERNAL_MODE == prompt_key_input ]]; then
  INTERNAL_RESULT_DIR=$RUN_DIR/prompt_projection_input_comparison
elif [[ $INTERNAL_MODE == prompt_key ]]; then
  INTERNAL_RESULT_DIR=$RUN_DIR/prompt_key_comparison
elif [[ $ATTENTION_OUTPUT_CAPTURE == 1 ]]; then
  INTERNAL_RESULT_DIR=$RUN_DIR/attention_output_capture
elif [[ $ATTENTION_PROJECTION_CAPTURE == 1 ]]; then
  INTERNAL_RESULT_DIR=$RUN_DIR/attention_projection_capture
elif [[ $ATTENTION_UPDATE_CAPTURE == 1 ]]; then
  INTERNAL_RESULT_DIR=$RUN_DIR/attention_update_capture
elif [[ $DENSE_BOUNDARY_CAPTURE == 1 ]]; then
  INTERNAL_RESULT_DIR=$RUN_DIR/dense_boundary_capture
elif [[ $DENSE_INPUT_CAPTURE == 1 ]]; then
  INTERNAL_RESULT_DIR=$RUN_DIR/dense_input_capture
elif [[ $DENSE_PARTIAL_CAPTURE == 1 ]]; then
  INTERNAL_RESULT_DIR=$RUN_DIR/dense_partials_capture
elif [[ $LAYER1_RMS_INPUT_CAPTURE == 1 ]]; then
  INTERNAL_RESULT_DIR=$RUN_DIR/layer1_rms_input_capture
elif [[ $INTERNAL_COMPARE_LAYER0 == 1 ]]; then
  INTERNAL_RESULT_DIR=$RUN_DIR/internal_comparison
else
  INTERNAL_RESULT_DIR=$RUN_DIR/internal_capture
fi
readonly INTERNAL_RESULT_DIR

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing DSA oracle outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing DSA oracle from a dirty greenfield worktree" >&2
  exit 2
}
[[ -z $(git -C "$HARNESS_REPO" status --porcelain --untracked-files=no) ]] || {
  echo "tracked harness files are dirty" >&2
  exit 2
}
for ray_source in \
  "$WORKTREE/scripts/launch_glm_32chip.sh:$RAY_LAUNCHER_SHA" \
  "$HARNESS_REPO/scripts/launch_glm_32chip.sh:$RAY_LAUNCHER_SHA" \
  "$WORKTREE/scripts/validate_ray_network.sh:$RAY_VALIDATOR_SHA" \
  "$HARNESS_REPO/scripts/validate_ray_network.sh:$RAY_VALIDATOR_SHA"; do
  IFS=: read -r ray_path ray_sha <<<"$ray_source"
  [[ -f $ray_path && $(sha256sum "$ray_path" | awk '{print $1}') == "$ray_sha" ]] || {
    echo "reviewed Ray protection source drifted: $ray_path" >&2
    exit 2
  }
done
[[ $(git -C "$LEGACY_SOURCE_REPO" rev-parse HEAD) == "$LEGACY_PIN" ]] || {
  echo "legacy repository pin changed" >&2
  exit 2
}
[[ -z $(git -C "$LEGACY_SOURCE_REPO" status --porcelain --untracked-files=no) ]] || {
  echo "tracked legacy files are dirty" >&2
  exit 2
}
if [[ $INTERNAL_CAPTURE == 1 || $PREFILL_PROJECTION_CAPTURE == 1 ||
      $DECODE_PROJECTION_CAPTURE == 1 || $MAIN_CACHE_CAPTURE == 1 ]]; then
  [[ $(git -C "$ORACLE_REPO" rev-parse HEAD) == "$ORACLE_PIN" ]] || {
    echo "accepted legacy oracle pin changed" >&2
    exit 2
  }
  [[ -z $(git -C "$ORACLE_REPO" status --porcelain --untracked-files=no) ]] || {
    echo "accepted legacy oracle worktree is dirty" >&2
    exit 2
  }
  git -C "$LEGACY_SOURCE_REPO" merge-base --is-ancestor \
    "$ORACLE_PIN" "$LEGACY_PIN" || {
      echo "legacy observer does not descend from the accepted oracle" >&2
      exit 2
    }
  [[ $(git -C "$LEGACY_SOURCE_REPO" rev-list --count \
    "$ORACLE_PIN..$LEGACY_PIN") -eq $OBSERVER_COMMIT_DISTANCE ]] || {
      echo "legacy observer commit distance drifted" >&2
      exit 2
    }
  if [[ $LAYER1_RMS_INPUT_CAPTURE == 1 ]]; then
    [[ $(git -C "$LEGACY_SOURCE_REPO" rev-parse "$OBSERVER_BRANCH") == \
       "$LEGACY_PIN" ]] || {
      echo "legacy layer-1 observer branch ref drifted" >&2
      exit 2
    }
  fi
fi
[[ -r $RESULTS_DB && -r $TOKEN_ORACLE_DIR/manifest.json ]] || {
  echo "source DB or sealed token oracle is unavailable" >&2
  exit 2
}
if [[ $MAIN_CACHE_CAPTURE == 1 ]]; then
  [[ -r $MAIN_CACHE_INGREDIENTS_DIR/contract.json &&
     -r $MAIN_CACHE_INGREDIENTS_DIR/position_8155_ingredients.npz ]] || {
    echo "protected PP8 layer-0 ingredients are unavailable" >&2
    exit 2
  }
  [[ $(sha256sum "$MAIN_CACHE_INGREDIENTS_DIR/contract.json" | awk '{print $1}') == \
     "$MAIN_CACHE_INGREDIENTS_CONTRACT_SHA" &&
     $(sha256sum "$MAIN_CACHE_INGREDIENTS_DIR/position_8155_ingredients.npz" | awk '{print $1}') == \
     "$MAIN_CACHE_INGREDIENTS_TENSOR_SHA" ]] || {
    echo "protected PP8 layer-0 ingredient identity drifted" >&2
    exit 2
  }
fi
if [[ $INTERNAL_CAPTURE == 1 || $PREFILL_PROJECTION_CAPTURE == 1 ||
      $DECODE_PROJECTION_CAPTURE == 1 || $MAIN_CACHE_CAPTURE == 1 ]]; then
  [[ -r $REFERENCE_8K_DSA_ORACLE_ROOT/SUCCESS &&
     -r $REFERENCE_8K_DSA_ORACLE/manifest.json &&
     -r $REFERENCE_8K_DSA_ORACLE/dsa_events.safetensors &&
     -r $REFERENCE_8K_DSA_ORACLE/source_row.json ]] || {
    echo "sealed DSA oracle comparison prerequisite is unavailable" >&2
    exit 2
  }
  [[ $(sha256sum "$REFERENCE_8K_DSA_ORACLE_ROOT/SUCCESS" | awk '{print $1}') == \
     "$REFERENCE_8K_DSA_ORACLE_SUCCESS_SHA" ]] || {
    echo "sealed DSA oracle SUCCESS identity drifted" >&2
    exit 2
  }
  JAX_PLATFORMS=cpu PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
    "$REFERENCE_8K_DSA_ORACLE" "$REFERENCE_8K_DSA_ORACLE_MANIFEST_SHA" <<'PY'
from pathlib import Path
import sys

from glm_tpu.greenfield.validation.short_context_dsa_oracle import (
    inspect_short_context_dsa_oracle,
)

manifest = inspect_short_context_dsa_oracle(Path(sys.argv[1]))
if manifest["manifest_sha256"] != sys.argv[2]:
    raise SystemExit("sealed DSA oracle manifest identity drifted")
PY
  if [[ $INTERNAL_CAPTURE == 1 && \
        ($INTERNAL_COMPARE_LAYER0 == 1 || $PROMPT_KEY_CAPTURE == 1) ]]; then
    [[ -r $LAYER0_INPUT_DIR/manifest.json &&
       -r $DISTRIBUTED_Q_A_DIR/manifest.json ]] || {
      echo "sealed layer-0 comparison prerequisites are unavailable" >&2
      exit 2
    }
  fi
  if [[ $PROMPT_CACHE_CAPTURE == 1 ]]; then
    # Sealing binds the prompt token identity through the sealed layer-0 DSA
    # input; verify it before any TPU work instead of failing after the run.
    PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
      "$LAYER0_INPUT_DIR" "$LAYER0_INPUT_MANIFEST_SHA" <<'PY' || {
import sys
from pathlib import Path

from glm_tpu.greenfield.validation.layer0_dsa_association import (
    inspect_layer0_dsa_association_input,
)

inspect_layer0_dsa_association_input(
    Path(sys.argv[1]), expected_manifest_sha256=sys.argv[2]
)
PY
      echo "sealed layer-0 DSA input required by prompt-cache sealing is unavailable" >&2
      exit 2
    }
  fi
fi
exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected pod workflow holds the global lease" >&2
  exit 1
}

# Refuse stale recreated-pod networking before creating the append-only run
# directory. The wrapper may inspect this rule but must never mutate it.
pod_contract=$(gcloud compute tpus tpu-vm describe "$POD" --zone "$ZONE" \
  --format='value(id,state,health)')
IFS=$'\t' read -r current_pod_id current_pod_state current_pod_health \
  <<<"$pod_contract"
[[ $current_pod_id =~ ^[0-9]+$ && $current_pod_state == READY &&
   $current_pod_health == HEALTHY ]] || {
  echo "TPU pod is not ready and healthy" >&2
  exit 2
}
current_pod_rule="tpu-pod-${POD}-${current_pod_id}"
current_ray_target=$(gcloud compute firewall-rules describe "$current_pod_rule" \
  --format='value(targetTags.list())')
ray_rule_contract=$(gcloud compute firewall-rules describe "$RAY_FIREWALL_RULE" \
  --format='value(network.basename(),direction,priority,sourceRanges.list(),allowed[].map().firewall_rule().list(),disabled,targetTags.list())')
printf '%s\n' "$ray_rule_contract" | \
  bash "$WORKTREE/scripts/validate_ray_network.sh" firewall \
    "$current_ray_target" "$current_pod_id" || {
  echo "stale or unsafe Ray firewall contract; refusing before tag burn" >&2
  exit 2
}

# Prove the exact worker-to-head data-plane path before creating the run
# directory. The short-lived listener accepts seven connections and touches
# neither Ray nor libtpu; the global lease prevents collision with real work.
head_ip=$(gcloud compute tpus tpu-vm describe "$POD" --zone "$ZONE" \
  --format='value(networkEndpoints[0].ipAddress)')
endpoint_ips=$(gcloud compute tpus tpu-vm describe "$POD" --zone "$ZONE" \
  --format='value(networkEndpoints.ipAddress)')
[[ $head_ip =~ ^[0-9]+([.][0-9]+){3}$ ]] || {
  echo "invalid worker-0 private IP" >&2
  exit 2
}
hostname -i | tr ' ' '\n' | grep -qxF -- "$head_ip" || {
  echo "worker-0 private IP is not local to this controller" >&2
  exit 2
}
expected_peer_ips=$(/home/gianl/vllm-env/bin/python -c \
  'import ipaddress,sys; head=sys.argv[1]; peers={x for x in sys.argv[2].split(";") if x and x != head}; print(",".join(sorted(peers, key=ipaddress.ip_address)))' \
  "$head_ip" "$endpoint_ips")
[[ $(tr ';' '\n' <<<"$endpoint_ips" | sort -u | wc -l) -eq 8 &&
   $(tr ',' '\n' <<<"$expected_peer_ips" | wc -l) -eq 7 ]] || {
  echo "TPU pod endpoint contract is not exactly eight unique hosts" >&2
  exit 2
}
expected_listener_receipt="PREFLIGHT_LISTENER_OK $expected_peer_ips"
listener_log=$(mktemp /tmp/glm_ray_preflight_listener.XXXXXXXX)
listener_pid=
cleanup_preflight_listener() {
  if [[ -n $listener_pid ]]; then
    kill "$listener_pid" >/dev/null 2>&1 || true
    wait "$listener_pid" >/dev/null 2>&1 || true
  fi
  rm -f -- "$listener_log"
}
trap cleanup_preflight_listener EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
/home/gianl/vllm-env/bin/python -c \
  'import ipaddress,socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind((sys.argv[1],6379)); s.listen(7); s.settimeout(15); peers=sorted({s.accept()[1][0] for _ in range(7)}, key=ipaddress.ip_address); print("PREFLIGHT_LISTENER_OK", ",".join(peers)); raise SystemExit(0 if len(peers)==7 else 1)' \
  "$head_ip" >"$listener_log" 2>&1 &
listener_pid=$!
sleep 1
pretag_probe='idx=${HOSTNAME##*-w-}; if timeout 3 bash -c "</dev/tcp/'"$head_ip"'/6379" >/dev/null 2>&1; then echo "PRETAG_TCP6379_OK $idx"; else echo "PRETAG_TCP6379_BAD $idx"; exit 1; fi'
set +e
pretag_probe_output=$(bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 60 \
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" \
  --worker=1,2,3,4,5,6,7 --command="$pretag_probe" 2>&1)
pretag_probe_rc=$?
wait "$listener_pid"
listener_rc=$?
listener_pid=
set -e
pretag_listener_output=$(<"$listener_log")
cleanup_preflight_listener
trap - EXIT INT TERM
[[ $pretag_probe_rc -eq 0 && $listener_rc -eq 0 &&
   $pretag_listener_output == "$expected_listener_receipt" ]] &&
  printf '%s\n' "$pretag_probe_output" | \
    bash "$WORKTREE/scripts/validate_ray_network.sh" receipts \
      PRETAG_TCP6379_OK 7 1,2,3,4,5,6,7 || {
  printf '%s\n%s\n' "$pretag_probe_output" "$pretag_listener_output" >&2
  echo "worker-to-head TCP/6379 preflight refused before tag burn" >&2
  exit 2
}

[[ ! -e $RUN_DIR ]] || {
  echo "append-only run directory exists: $RUN_DIR" >&2
  exit 2
}
mkdir -p "$RUN_DIR" "$SOURCE_DIR"
printf '%s\n' "$ray_rule_contract" >"$RUN_DIR/ray_firewall_preflight.txt"
ray_tcp6379_contract=$(printf '%s\n%s' \
  "$pretag_probe_output" "$pretag_listener_output")
printf '%s\n' "$ray_tcp6379_contract" >"$RUN_DIR/ray_tcp6379_preflight.txt"

say() {
  echo "[short-dsa-oracle $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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

if [[ $ATTENTION_UPDATE_CAPTURE == 1 ]]; then
  for spec in \
    "$PROJECTION_REDUCTION_RUNNER_SHA $PROJECTION_REDUCTION_REMOTE/runner.json" \
    "$PROJECTION_REDUCTION_TENSOR_SHA $PROJECTION_REDUCTION_REMOTE/projection_reduction.npz" \
    "$PROJECTION_REDUCTION_SUMMARY_SHA $PROJECTION_REDUCTION_REMOTE/summary.json" \
    "$PROJECTION_REDUCTION_SUCCESS_SHA $PROJECTION_REDUCTION_REMOTE/SUCCESS"; do
    read -r expected uri <<<"$spec"
    observed=$(gcloud storage cat "$uri" | sha256sum | awk '{print $1}')
    [[ $observed == "$expected" ]] || {
      say "ABORT: protected DB538 remote source hash drifted: $uri"
      exit 1
    }
  done
fi
if [[ $DENSE_PARTIAL_CAPTURE == 1 ]]; then
  for spec in \
    "$DENSE_PARTIAL_PROBE_RUNNER_SHA $DENSE_PARTIAL_PROBE_REMOTE/runner.json" \
    "$DENSE_PARTIAL_PROBE_TENSOR_SHA $DENSE_PARTIAL_PROBE_REMOTE/dense_partial_capture.npz" \
    "$DENSE_PARTIAL_PROBE_SUMMARY_SHA $DENSE_PARTIAL_PROBE_REMOTE/summary.json" \
    "$DENSE_PARTIAL_PROBE_SUCCESS_SHA $DENSE_PARTIAL_PROBE_REMOTE/SUCCESS"; do
    read -r expected uri <<<"$spec"
    observed=$(gcloud storage cat "$uri" | sha256sum | awk '{print $1}')
    [[ $observed == "$expected" ]] || {
      say "ABORT: protected DB548 remote source hash drifted: $uri"
      exit 1
    }
  done
fi
if [[ $DENSE_BOUNDARY_CAPTURE == 1 || $DENSE_INPUT_CAPTURE == 1 ]]; then
  for spec in \
    "$DENSE_CONVOLUTION_RUNNER_SHA $DENSE_CONVOLUTION_REMOTE/runner.json" \
    "$DENSE_CONVOLUTION_TENSOR_SHA $DENSE_CONVOLUTION_REMOTE/dense_convolution.npz" \
    "$DENSE_CONVOLUTION_SUMMARY_SHA $DENSE_CONVOLUTION_REMOTE/summary.json" \
    "$DENSE_CONVOLUTION_SUCCESS_SHA $DENSE_CONVOLUTION_REMOTE/SUCCESS"; do
    read -r expected uri <<<"$spec"
    observed=$(gcloud storage cat "$uri" | sha256sum | awk '{print $1}')
    [[ $observed == "$expected" ]] || {
      say "ABORT: protected DB540 remote source hash drifted: $uri"
      exit 1
    }
  done
fi

has_eight_unique_markers() {
  local file=$1 marker=$2 expected_fields=${3:-2}
  bash "$WORKTREE/scripts/validate_ray_network.sh" receipts \
    "$marker" 8 '' "$expected_fields" <"$file"
}

strict_census() {
  local label=$1
  local out="$RUN_DIR/census_${label}.txt"
  local carrier="${TAG}_${label}"
  local ray_enum command
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[r]ay start --address|[r]ay start --head|[g]lm_longctx[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu: $holders"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" \
    bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 120 \
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

stop_owned_runtime() {
  # Pre-census proves no pre-existing Ray/vLLM work. Everything matched here
  # was launched under this global lease.
  local command
  # shellcheck disable=SC2016
  command='timeout --signal=TERM --kill-after=5 -- 20 /home/gianl/vllm-env/bin/ray stop -f >/dev/null 2>&1 || true; sudo pkill -TERM -f "[r]ay start --address|[r]ay start --head" >/dev/null 2>&1 || true; sudo pkill -TERM -f "VLLM::[E]ngineCore" >/dev/null 2>&1 || true; sudo pkill -TERM -f "[R]ayWorkerWrapper" >/dev/null 2>&1 || true; sudo pkill -TERM -x raylet >/dev/null 2>&1 || true; sleep 2; sudo pkill -KILL -f "[r]ay start --address|[r]ay start --head" >/dev/null 2>&1 || true; sudo pkill -KILL -f "VLLM::[E]ngineCore" >/dev/null 2>&1 || true; sudo pkill -KILL -f "[R]ayWorkerWrapper" >/dev/null 2>&1 || true; sudo pkill -KILL -x raylet >/dev/null 2>&1 || true; sudo rm -f /tmp/libtpu_lockfile; echo STOP_OK $(hostname)'
  bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 120 \
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$command" >"$RUN_DIR/stop.txt" 2>&1 || return 1
  has_eight_unique_markers "$RUN_DIR/stop.txt" STOP_OK
}

cleanup_vllm_runtime() {
  local label=$1
  local out="$RUN_DIR/vllm_cleanup_${label}.txt"
  local command
  # Only the run-owned, validated tag path is removable.
  # shellcheck disable=SC2016
  command='target='"$VLLM_RUNTIME_ROOT"'; archive='"$VLLM_RUNTIME_ARCHIVE_REMOTE"'; if ! printf "%s\n" "$target" | grep -Eq "^/tmp/glm_vllm_[A-Za-z0-9_]+$" || [ "$archive" != "${target}.tar.gz" ]; then echo "VLLM_CLEAN_BAD $(hostname) unsafe_target"; exit 0; fi; rm -rf -- "$target"; rm -f -- "$archive"; if [ ! -e "$target" ] && [ ! -e "$archive" ]; then echo "VLLM_CLEAN_OK $(hostname)"; else echo "VLLM_CLEAN_BAD $(hostname) residual"; fi'
  bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 120 \
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$command" >"$out" \
    2>"$RUN_DIR/vllm_cleanup_${label}_ssh.txt" || return 1
  has_eight_unique_markers "$out" VLLM_CLEAN_OK
}

cleanup_observer_transport() {
  local label=$1
  local out="$RUN_DIR/observer_transport_cleanup_${label}.txt"
  local command
  # Remove only this validated tag's transient bundle and pre-install tree.
  # The exact pin-specific runtime checkout is intentionally reusable.
  # shellcheck disable=SC2016
  command='bundle='"$OBSERVER_BUNDLE_REMOTE"'; temp='"$OBSERVER_BUNDLE_TEMP"'; if ! printf "%s\n" "$bundle" | grep -Eq "^/tmp/glm_observer_[A-Za-z0-9_]+[.]bundle$" || ! printf "%s\n" "$temp" | grep -Eq "^/tmp/glm_observer_[A-Za-z0-9_]+[.]tmp$" || [ "${bundle%.bundle}" != "${temp%.tmp}" ]; then echo "OBSERVER_TRANSPORT_CLEAN_BAD $(hostname) unsafe_target"; exit 0; fi; rm -f -- "$bundle"; rm -rf -- "$temp"; if [ ! -e "$bundle" ] && [ ! -e "$temp" ]; then echo "OBSERVER_TRANSPORT_CLEAN_OK $(hostname)"; else echo "OBSERVER_TRANSPORT_CLEAN_BAD $(hostname) residual"; fi'
  bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 120 \
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$command" >"$out" \
    2>"$RUN_DIR/observer_transport_cleanup_${label}_ssh.txt" || return 1
  has_eight_unique_markers "$out" OBSERVER_TRANSPORT_CLEAN_OK
}

runtime_started=0
post_census_done=0
terminal_success_done=0
vllm_runtime_prepared=0
observer_transport_prepared=0

rollback_internal_capture_db() {
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
    "$RESULTS_DB" "$TAG" "$HARNESS_SHORT" "$ORACLE_SHORT" \
    "$LEGACY_PIN" "$LEGACY_SHORT" "$ORACLE_PIN" "$REMOTE_PREFIX" \
    "$DUMP_PREFIX" "$INTERNAL_DUMP_PREFIX" \
    "$INTERNAL_MODE" \
    >"$RUN_DIR/provisional_db_rollback.txt" 2>&1 <<'PY'
from pathlib import Path
import sys

from glm_tpu.greenfield.validation.dense_partials import (
    DensePartialsRollbackConfig,
    rollback_dense_partial_oracle_run,
)

print(rollback_dense_partial_oracle_run(DensePartialsRollbackConfig(
    results_db=Path(sys.argv[1]),
    run_tag=sys.argv[2],
    expected_harness_git=sys.argv[3],
    expected_fork_git=sys.argv[4],
    expected_legacy_pin=sys.argv[5],
    expected_legacy_short=sys.argv[6],
    expected_oracle_pin=sys.argv[7],
    expected_remote_prefix=sys.argv[8],
    expected_dump_prefix=sys.argv[9],
    expected_internal_dump_prefix=sys.argv[10],
    expected_internal_mode=sys.argv[11],
)))
PY
}

on_exit() {
  local status=$?
  if [[ $runtime_started -eq 1 ]]; then
    stop_owned_runtime || true
  fi
  if [[ $vllm_runtime_prepared -eq 1 ]]; then
    cleanup_vllm_runtime failure_exit || true
    vllm_runtime_prepared=0
  fi
  if [[ $observer_transport_prepared -eq 1 ]]; then
    cleanup_observer_transport failure_exit || true
    observer_transport_prepared=0
  fi
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || true
  fi
  if [[ $status -ne 0 && \
        $((DENSE_PARTIAL_CAPTURE + DENSE_BOUNDARY_CAPTURE)) -eq 1 && \
        $terminal_success_done -eq 0 ]]; then
    if rollback_internal_capture_db; then
      say "authenticated provisional internal-capture DB rollback complete"
    else
      say "WARNING: provisional internal-capture DB rollback refused; preserving row for diagnosis"
    fi
  fi
  if [[ $status -ne 0 ]]; then
    say "FAILED status=$status; preserving diagnostics"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic_local/" >/dev/null 2>&1 || true
  fi
  return "$status"
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR GREENFIELD_PIN=$PIN HARNESS_PIN=$HARNESS_PIN LEGACY_PIN=$LEGACY_PIN"
say "PROFILE=$PROFILE DUMP_PREFIX=$DUMP_PREFIX REMOTE_PREFIX=$REMOTE_PREFIX INTERNAL_LAYER=$INTERNAL_LAYER INTERNAL_MODE=$INTERNAL_MODE INTERNAL_POSITION=$INTERNAL_TARGET_POSITION PROMPT_CACHE_CAPTURE=$PROMPT_CACHE_CAPTURE PREFILL_PROJECTION_CAPTURE=$PREFILL_PROJECTION_CAPTURE DECODE_PROJECTION_CAPTURE=$DECODE_PROJECTION_CAPTURE MAIN_CACHE_CAPTURE=$MAIN_CACHE_CAPTURE"
say "PROMPT_CACHE_LAYER_ID=$PROMPT_CACHE_LAYER_ID PROMPT_CACHE_SLOT=$PROMPT_CACHE_SLOT"
if [[ $PROMPT_CACHE_CAPTURE == 1 && $PROFILE != 8k ]]; then
  say "ABORT: prompt index-cache capture is defined only for the sealed 8K profile"
  exit 2
fi
strict_census pre || {
  say "ABORT: fleet is not eight-host zero work"
  exit 1
}
MIN_FREE_GB="$DISK_MIN_FREE_GB" WARN_FREE_GB="$DISK_WARN_FREE_GB" \
  bash "$WORKTREE/scripts/disk_watchdog.sh" check \
  > >(tee "$RUN_DIR/disk_preflight.txt") 2>&1 || {
    say "ABORT: eight-host disk reserve is below ${DISK_MIN_FREE_GB} GiB"
    exit 1
  }

if [[ $INTERNAL_CAPTURE == 1 || $MAIN_CACHE_CAPTURE == 1 ]]; then
  if [[ $LAYER1_RMS_INPUT_CAPTURE == 1 ]]; then
    # The recreated pod has no accepted-oracle Git checkout on workers 1--7.
    # Build one authenticated self-contained bundle from the already reviewed
    # controller observer instead of fetching or assuming per-host Git state.
    git -C "$LEGACY_SOURCE_REPO" bundle create \
      "$OBSERVER_BUNDLE_LOCAL" "$OBSERVER_BRANCH"
    git -C "$LEGACY_SOURCE_REPO" bundle verify "$OBSERVER_BUNDLE_LOCAL" \
      >"$RUN_DIR/observer_bundle_verify.txt" 2>&1
    OBSERVER_BUNDLE_SHA=$(sha256sum "$OBSERVER_BUNDLE_LOCAL" | awk '{print $1}')
    OBSERVER_TRACKED_FILE_COUNT=$(git -C "$LEGACY_SOURCE_REPO" ls-tree -r \
      --name-only "$LEGACY_PIN" | wc -l)
    readonly OBSERVER_BUNDLE_SHA OBSERVER_TRACKED_FILE_COUNT
    [[ $OBSERVER_BUNDLE_SHA =~ ^[0-9a-f]{64}$ && \
       $OBSERVER_TRACKED_FILE_COUNT -eq $LAYER1_OBSERVER_TRACKED_FILE_COUNT ]] || {
      say "ABORT: exact legacy observer bundle identity is invalid"
      exit 1
    }
    printf 'pin=%s\noracle_pin=%s\nbranch=%s\nbundle_sha256=%s\ntracked_entries=%s\nsource_repository=%s\n' \
      "$LEGACY_PIN" "$ORACLE_PIN" "$OBSERVER_BRANCH" \
      "$OBSERVER_BUNDLE_SHA" "$OBSERVER_TRACKED_FILE_COUNT" \
      "$LEGACY_SOURCE_REPO" >"$RUN_DIR/observer_bundle_identity.txt"

    # Refuse rather than remove any transient path predating this append-only run.
    # shellcheck disable=SC2016
    observer_vacancy='bundle='"$OBSERVER_BUNDLE_REMOTE"'; temp='"$OBSERVER_BUNDLE_TEMP"'; if [ ! -e "$bundle" ] && [ ! -e "$temp" ]; then echo "OBSERVER_TRANSPORT_VACANT_OK $(hostname)"; else echo "OBSERVER_TRANSPORT_VACANT_BAD $(hostname)"; fi'
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
      --command="$observer_vacancy" \
      >"$RUN_DIR/observer_transport_vacancy.txt" \
      2>"$RUN_DIR/observer_transport_vacancy_ssh.txt"
    has_eight_unique_markers "$RUN_DIR/observer_transport_vacancy.txt" \
      OBSERVER_TRANSPORT_VACANT_OK || {
      say "ABORT: run-owned observer transport path is not vacant"
      exit 1
    }

    observer_transport_prepared=1
    : >"$RUN_DIR/observer_bundle_copy.txt"
    for worker in 0 1 2 3 4 5 6 7; do
      if gcloud compute tpus tpu-vm scp --zone "$ZONE" --worker="$worker" \
          "$OBSERVER_BUNDLE_LOCAL" \
          "$POD:$OBSERVER_BUNDLE_REMOTE" >/dev/null 2>&1; then
        echo "OBSERVER_BUNDLE_COPY_OK $worker" >>"$RUN_DIR/observer_bundle_copy.txt"
      else
        echo "OBSERVER_BUNDLE_COPY_BAD $worker" >>"$RUN_DIR/observer_bundle_copy.txt"
      fi
    done
    has_eight_unique_markers "$RUN_DIR/observer_bundle_copy.txt" \
      OBSERVER_BUNDLE_COPY_OK || {
      say "ABORT: exact legacy observer bundle did not reach all hosts"
      exit 1
    }

    # shellcheck disable=SC2016
    sync_observer='set -e; bundle='"$OBSERVER_BUNDLE_REMOTE"'; bundle_sha='"$OBSERVER_BUNDLE_SHA"'; dest='"$OBSERVER_RUNTIME_REPO"'; temp='"$OBSERVER_BUNDLE_TEMP"'; pin='"$LEGACY_PIN"'; oracle='"$ORACLE_PIN"'; distance='"$OBSERVER_COMMIT_DISTANCE"'; expected_files='"$OBSERVER_TRACKED_FILE_COUNT"'; actual_bundle=$(sha256sum "$bundle" | cut -d " " -f 1); disposition=existing; if [ "$actual_bundle" != "$bundle_sha" ]; then echo "OBSERVER_SYNC_BAD $(hostname) bundle_sha"; exit 0; fi; if git -C "$dest" rev-parse HEAD >/dev/null 2>&1; then git -C "$dest" bundle verify "$bundle" >/dev/null 2>&1 || { echo "OBSERVER_SYNC_BAD $(hostname) bundle_verify"; exit 0; }; elif [ -e "$dest" ]; then echo "OBSERVER_SYNC_BAD $(hostname) destination_exists"; exit 0; elif [ -e "$temp" ]; then echo "OBSERVER_SYNC_BAD $(hostname) temp_exists"; exit 0; else disposition=installed; git clone -q --no-checkout "$bundle" "$temp" || { echo "OBSERVER_SYNC_BAD $(hostname) clone"; exit 0; }; git -C "$temp" bundle verify "$bundle" >/dev/null 2>&1 || { echo "OBSERVER_SYNC_BAD $(hostname) bundle_verify"; exit 0; }; git -C "$temp" checkout -q --detach "$pin" || { echo "OBSERVER_SYNC_BAD $(hostname) checkout"; exit 0; }; temp_code=$(git -C "$temp" rev-parse HEAD); temp_dirty=$(git -C "$temp" status --porcelain | wc -l); if [ "$temp_code" != "$pin" ] || [ "$temp_dirty" -ne 0 ]; then echo "OBSERVER_SYNC_BAD $(hostname) temp_identity"; exit 0; fi; mv "$temp" "$dest"; fi; code=$(git -C "$dest" rev-parse HEAD); dirty=$(git -C "$dest" status --porcelain | wc -l); commits=$(git -C "$dest" rev-list --count "$oracle..$pin"); files=$(git -C "$dest" ls-tree -r --name-only "$pin" | wc -l); ancestor=0; git -C "$dest" merge-base --is-ancestor "$oracle" "$pin" && ancestor=1; if [ "$code" = "$pin" ] && [ "$dirty" -eq 0 ] && [ "$commits" -eq "$distance" ] && [ "$ancestor" -eq 1 ] && [ "$files" -eq "$expected_files" ]; then echo "OBSERVER_SYNC_OK $(hostname) bundle_sha256=$actual_bundle tracked_entries=$files disposition=$disposition"; else echo "OBSERVER_SYNC_BAD $(hostname) code=$code dirty=$dirty commits=$commits ancestor=$ancestor files=$files"; fi'
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
      --command="$sync_observer" >"$RUN_DIR/sync_observer.txt" \
      2>"$RUN_DIR/sync_observer_ssh.txt"
    has_eight_unique_markers "$RUN_DIR/sync_observer.txt" OBSERVER_SYNC_OK 5 || {
      say "ABORT: exact legacy observer bundle did not reconstruct on all hosts"
      exit 1
    }
    cleanup_observer_transport post_sync || {
      say "ABORT: run-owned observer transport cleanup failed"
      exit 1
    }
    observer_transport_prepared=0
  else
    # Historical observer modes retain their already proven worktree sync.
    # shellcheck disable=SC2016
    sync_observer='set -e; base='"$ORACLE_REPO"'; dest='"$OBSERVER_RUNTIME_REPO"'; pin='"$LEGACY_PIN"'; oracle='"$ORACLE_PIN"'; branch='"$OBSERVER_BRANCH"'; distance='"$OBSERVER_COMMIT_DISTANCE"'; if git -C "$dest" rev-parse HEAD >/dev/null 2>&1; then :; elif [ -e "$dest" ]; then echo "SYNC_BAD $(hostname) destination_exists"; exit 0; else git -C "$base" fetch origin "$branch" >/dev/null 2>&1 && git -C "$base" worktree add --detach "$dest" "$pin" >/dev/null 2>&1; fi; code=$(git -C "$dest" rev-parse HEAD); dirty=$(git -C "$dest" status --porcelain | wc -l); commits=$(git -C "$dest" rev-list --count "$oracle..$pin"); ancestor=0; git -C "$dest" merge-base --is-ancestor "$oracle" "$pin" && ancestor=1; if [ "$code" = "$pin" ] && [ "$dirty" -eq 0 ] && [ "$commits" -eq "$distance" ] && [ "$ancestor" -eq 1 ]; then echo "SYNC_OK $(hostname)"; else echo "SYNC_BAD $(hostname) code=$code dirty=$dirty commits=$commits ancestor=$ancestor"; fi'
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
      --command="$sync_observer" >"$RUN_DIR/sync_observer.txt" \
      2>"$RUN_DIR/sync_observer_ssh.txt"
    has_eight_unique_markers "$RUN_DIR/sync_observer.txt" SYNC_OK || {
      say "ABORT: exact legacy observer is unavailable on all hosts"
      exit 1
    }
  fi
fi
if [[ $LAYER1_RMS_INPUT_CAPTURE == 1 ]]; then
  # The new pod no longer has the old ~/vllm-build worktree targeted by the
  # editable install. Transfer one exact pin-derived source archive from the
  # controller instead of assuming every worker has a complete Git object DB.
  readonly VLLM_RUNTIME_ARCHIVE_LOCAL=$RUN_DIR/vllm_${VLLM_PIN}.tar.gz
  git -C "$VLLM_ARCHIVE_REPO" archive --format=tar.gz \
    --output "$VLLM_RUNTIME_ARCHIVE_LOCAL" "$VLLM_PIN"
  VLLM_RUNTIME_ARCHIVE_SHA=$(sha256sum "$VLLM_RUNTIME_ARCHIVE_LOCAL" | awk '{print $1}')
  VLLM_RUNTIME_FILE_COUNT=$(git -C "$VLLM_ARCHIVE_REPO" ls-tree -r \
    --name-only "$VLLM_PIN" | wc -l)
  readonly VLLM_RUNTIME_ARCHIVE_SHA VLLM_RUNTIME_FILE_COUNT
  [[ $VLLM_RUNTIME_ARCHIVE_SHA =~ ^[0-9a-f]{64}$ && \
     $VLLM_RUNTIME_FILE_COUNT -gt 0 ]] || {
    say "ABORT: exact accepted vLLM archive identity is invalid"
    exit 1
  }
  printf 'pin=%s\narchive_sha256=%s\ntracked_entries=%s\nsource_repository=%s\n' \
    "$VLLM_PIN" "$VLLM_RUNTIME_ARCHIVE_SHA" "$VLLM_RUNTIME_FILE_COUNT" \
    "$VLLM_ARCHIVE_REPO" >"$RUN_DIR/vllm_archive_identity.txt"

  # Refuse rather than remove any path that predates this append-only run.
  # shellcheck disable=SC2016
  vllm_vacancy='dest='"$VLLM_RUNTIME_ROOT"'; archive='"$VLLM_RUNTIME_ARCHIVE_REMOTE"'; if [ ! -e "$dest" ] && [ ! -e "$archive" ]; then echo "VLLM_VACANT_OK $(hostname)"; else echo "VLLM_VACANT_BAD $(hostname)"; fi'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$vllm_vacancy" >"$RUN_DIR/vllm_vacancy.txt" \
    2>"$RUN_DIR/vllm_vacancy_ssh.txt"
  has_eight_unique_markers "$RUN_DIR/vllm_vacancy.txt" VLLM_VACANT_OK || {
    say "ABORT: run-owned accepted vLLM path is not vacant on all hosts"
    exit 1
  }

  vllm_runtime_prepared=1
  : >"$RUN_DIR/vllm_archive_copy.txt"
  for worker in 0 1 2 3 4 5 6 7; do
    if gcloud compute tpus tpu-vm scp --zone "$ZONE" --worker="$worker" \
        "$VLLM_RUNTIME_ARCHIVE_LOCAL" \
        "$POD:$VLLM_RUNTIME_ARCHIVE_REMOTE" >/dev/null 2>&1; then
      echo "VLLM_ARCHIVE_COPY_OK $worker" >>"$RUN_DIR/vllm_archive_copy.txt"
    else
      echo "VLLM_ARCHIVE_COPY_BAD $worker" >>"$RUN_DIR/vllm_archive_copy.txt"
    fi
  done
  has_eight_unique_markers "$RUN_DIR/vllm_archive_copy.txt" \
    VLLM_ARCHIVE_COPY_OK || {
    say "ABORT: exact accepted vLLM archive did not reach all hosts"
    exit 1
  }

  # Authenticate archive bytes before extraction, then authenticate the two
  # semantic source files and complete tracked-entry count after extraction.
  # shellcheck disable=SC2016
  sync_vllm='set -e; archive='"$VLLM_RUNTIME_ARCHIVE_REMOTE"'; dest='"$VLLM_RUNTIME_ROOT"'; archive_sha='"$VLLM_RUNTIME_ARCHIVE_SHA"'; expected_files='"$VLLM_RUNTIME_FILE_COUNT"'; ir_sha='"$VLLM_IR_LAYERNORM_SHA"'; executor_sha='"$VLLM_EXECUTOR_LAYERNORM_SHA"'; actual_archive=$(sha256sum "$archive" | cut -d " " -f 1); if [ "$actual_archive" != "$archive_sha" ] || [ -e "$dest" ]; then echo "VLLM_SYNC_BAD $(hostname) archive_or_destination"; exit 0; fi; umask 077; mkdir "$dest"; tar -xzf "$archive" -C "$dest"; actual_files=$(find "$dest" \( -type f -o -type l \) | wc -l); actual_ir=$(sha256sum "$dest/vllm/ir/ops/layernorm.py" | cut -d " " -f 1); actual_executor=$(sha256sum "$dest/vllm/model_executor/layers/layernorm.py" | cut -d " " -f 1); rm -f -- "$archive"; if [ "$actual_files" -eq "$expected_files" ] && [ "$actual_ir" = "$ir_sha" ] && [ "$actual_executor" = "$executor_sha" ] && [ ! -e "$archive" ]; then echo "VLLM_SYNC_OK $(hostname) archive_sha256=$actual_archive tracked_entries=$actual_files"; else echo "VLLM_SYNC_BAD $(hostname) files=$actual_files ir=$actual_ir executor=$actual_executor"; fi'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$sync_vllm" >"$RUN_DIR/sync_vllm.txt" \
    2>"$RUN_DIR/sync_vllm_ssh.txt"
  has_eight_unique_markers "$RUN_DIR/sync_vllm.txt" VLLM_SYNC_OK 4 || {
    say "ABORT: exact accepted vLLM source authentication failed on the fleet"
    exit 1
  }
  [[ $(awk -v sha="$VLLM_RUNTIME_ARCHIVE_SHA" \
      -v count="$VLLM_RUNTIME_FILE_COUNT" \
      '$1 == "VLLM_SYNC_OK" && $3 == "archive_sha256=" sha && \
       $4 == "tracked_entries=" count {n++} END {print n + 0}' \
      "$RUN_DIR/sync_vllm.txt") -eq 8 ]] || {
    say "ABORT: accepted vLLM fleet receipts drifted"
    exit 1
  }
fi

# Recreated pods do not retain /tmp/golden.json. Rehydrate it from the
# approved read-only same-region manifest mirror, but never overwrite an
# existing non-identical file. All rank files are independently authenticated.
# shellcheck disable=SC2016
golden_sync='set -e; idx=${HOSTNAME##*-w-}; source='"$GOLDEN_SOURCE_DIR"'/golden.rank${idx}.json; dest=/tmp/golden.json; temp=/tmp/golden_sync_$$.tmp; expected_sha='"$GOLDEN_MANIFEST_SHA"'; expected_bytes='"$GOLDEN_MANIFEST_BYTES"'; disposition=existing; case "$idx" in 0|1|2|3|4|5|6|7) ;; *) echo "GOLDEN_SYNC_BAD $(hostname) rank"; exit 0 ;; esac; mount_source=$(findmnt -T "$source" -n -o SOURCE 2>/dev/null); mount_type=$(findmnt -T "$source" -n -o FSTYPE 2>/dev/null); [ "$mount_source" = driftbench-dsv4-uc ] && [ "$mount_type" = fuse.gcsfuse ] && [ -r "$source" ] || { echo "GOLDEN_SYNC_BAD $(hostname) source"; exit 0; }; source_sha=$(sha256sum "$source" | cut -d " " -f 1); source_bytes=$(stat -c %s "$source"); [ "$source_sha" = "$expected_sha" ] && [ "$source_bytes" -eq "$expected_bytes" ] || { echo "GOLDEN_SYNC_BAD $(hostname) source_identity"; exit 0; }; if [ -e "$dest" ]; then [ -r "$dest" ] || { echo "GOLDEN_SYNC_BAD $(hostname) unreadable_destination"; exit 0; }; else disposition=installed; trap '\''rm -f -- "$temp"'\'' EXIT; [ ! -e "$temp" ] || { echo "GOLDEN_SYNC_BAD $(hostname) temp_exists"; exit 0; }; umask 022; cp -- "$source" "$temp"; actual_temp=$(sha256sum "$temp" | cut -d " " -f 1); [ "$actual_temp" = "$expected_sha" ] || { echo "GOLDEN_SYNC_BAD $(hostname) copied_identity"; exit 0; }; chmod 0444 "$temp"; mv "$temp" "$dest"; trap - EXIT; fi; actual_sha=$(sha256sum "$dest" | cut -d " " -f 1); actual_bytes=$(stat -c %s "$dest"); if [ "$actual_sha" = "$expected_sha" ] && [ "$actual_bytes" -eq "$expected_bytes" ]; then echo "GOLDEN_SYNC_OK $(hostname) sha256=$actual_sha bytes=$actual_bytes disposition=$disposition"; else echo "GOLDEN_SYNC_BAD $(hostname) destination_identity"; fi'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$golden_sync" >"$RUN_DIR/golden_sync.txt" \
  2>"$RUN_DIR/golden_sync_ssh.txt"
has_eight_unique_markers "$RUN_DIR/golden_sync.txt" GOLDEN_SYNC_OK 5 || {
  say "ABORT: exact golden state manifest is unavailable on all hosts"
  exit 1
}

# All hosts must carry the exact clean legacy tree, golden state file, and
# approved read-only OOB checkpoint mirror used by the PWAL self-healer.
# shellcheck disable=SC2016
prereq='code=$(git -C '"$LEGACY_REPO"' rev-parse HEAD); dirty=$(git -C '"$LEGACY_REPO"' status --porcelain --untracked-files=no | wc -l); mount_source=$(findmnt -T '"$OOB_DIR"' -n -o SOURCE 2>/dev/null); mount_type=$(findmnt -T '"$OOB_DIR"' -n -o FSTYPE 2>/dev/null); if [ "$code" = '"$LEGACY_PIN"' ] && [ "$dirty" -eq 0 ] && [ -r /tmp/golden.json ] && [ -r '"$OOB_DIR"'/model.safetensors.index.json ] && [ "$mount_source" = driftbench-dsv4-uc ] && [ "$mount_type" = fuse.gcsfuse ] && [ ! -e /tmp/'"$TAG"' ]; then echo "PREREQ_OK $(hostname)"; else echo "PREREQ_BAD $(hostname) code=$code dirty=$dirty mount_source=$mount_source mount_type=$mount_type"; fi'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$prereq" >"$RUN_DIR/prereq.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/prereq.txt" PREREQ_OK || {
  say "ABORT: exact legacy/golden/OOB/dump prerequisite failed"
  exit 1
}
if [[ $INTERNAL_CAPTURE == 1 || $MAIN_CACHE_CAPTURE == 1 ]]; then
  if [[ $LAYER1_RMS_INPUT_CAPTURE == 1 ]]; then
    # The self-contained reviewed observer bundle carries and proves the
    # accepted ancestor; no separate per-host oracle checkout is required.
    # shellcheck disable=SC2016
    oracle_prereq='repo='"$LEGACY_REPO"'; pin='"$LEGACY_PIN"'; oracle='"$ORACLE_PIN"'; distance='"$OBSERVER_COMMIT_DISTANCE"'; commits=$(git -C "$repo" rev-list --count "$oracle..$pin" 2>/dev/null || echo invalid); ancestor=0; git -C "$repo" merge-base --is-ancestor "$oracle" "$pin" >/dev/null 2>&1 && ancestor=1; if [ "$commits" = "$distance" ] && [ "$ancestor" -eq 1 ]; then echo "ORACLE_OK $(hostname)"; else echo "ORACLE_BAD $(hostname) commits=$commits ancestor=$ancestor"; fi'
  else
    # shellcheck disable=SC2016
    oracle_prereq='code=$(git -C '"$ORACLE_REPO"' rev-parse HEAD); dirty=$(git -C '"$ORACLE_REPO"' status --porcelain --untracked-files=no | wc -l); if [ "$code" = '"$ORACLE_PIN"' ] && [ "$dirty" -eq 0 ]; then echo "ORACLE_OK $(hostname)"; else echo "ORACLE_BAD $(hostname) code=$code dirty=$dirty"; fi'
  fi
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$oracle_prereq" >"$RUN_DIR/oracle_prereq.txt" 2>&1
  has_eight_unique_markers "$RUN_DIR/oracle_prereq.txt" ORACLE_OK || {
    say "ABORT: accepted legacy oracle checkout drifted on the fleet"
    exit 1
  }
fi

COMMON_ENVS='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=1 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_DCP_SCATTER_IMPL=flat GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR='"$OOB_DIR"' GLM_WK_OOB_GOLDEN=/tmp/golden.json GLM_DSA_DUMP_TOPK='"$DUMP_PREFIX"' GLM_DSA_DUMP_TOPK_EVENTS=all GLM_DSA_DUMP_TOPK_SKIP_WARMUP=1 GLM_EXPECT_CODE_HASH='"$LEGACY_SHORT"
if [[ $PROMPT_CACHE_CAPTURE == 1 ]]; then
  COMMON_ENVS="$COMMON_ENVS GLM_DCP_CACHE_DUMP=$PROMPT_CACHE_DUMP_PREFIX GLM_DCP_CACHE_DUMP_LAYERS=$PROMPT_CACHE_SLOT"
fi
if [[ $MAIN_CACHE_CAPTURE == 1 ]]; then
  COMMON_ENVS="PYTHONPATH=$OBSERVER_RUNTIME_REPO $COMMON_ENVS GLM_DCP_CACHE_DUMP=$MAIN_CACHE_DUMP_PREFIX GLM_DCP_CACHE_DUMP_LAYERS=1 GLM_DCP_CACHE_DUMP_STEPS=4,5"
fi
if [[ $INTERNAL_CAPTURE == 1 ]]; then
  if [[ $LAYER1_RMS_INPUT_CAPTURE == 1 ]]; then
    INTERNAL_PYTHONPATH=$OBSERVER_RUNTIME_REPO:$VLLM_RUNTIME_ROOT
  else
    INTERNAL_PYTHONPATH=$OBSERVER_RUNTIME_REPO
  fi
  COMMON_ENVS="PYTHONPATH=$INTERNAL_PYTHONPATH $COMMON_ENVS GLM_DSA_DUMP_INTERNALS=$INTERNAL_DUMP_PREFIX GLM_DSA_DUMP_INTERNALS_MODE=$INTERNAL_MODE GLM_DSA_DUMP_INTERNALS_LAYER=$INTERNAL_LAYER GLM_DSA_DUMP_INTERNALS_POSITION=$INTERNAL_TARGET_POSITION GLM_DSA_DUMP_INTERNALS_RUN_TAG=$TAG GLM_DSA_DUMP_INTERNALS_CODE_HASH=$LEGACY_PIN GLM_DSA_DUMP_INTERNALS_ORACLE_PIN=$ORACLE_PIN GLM_DSA_DUMP_INTERNALS_MODEL_ID=$MODEL_ID"
fi
PREFILL_PROJECTION_ENVS=
if [[ $PREFILL_PROJECTION_CAPTURE == 1 ]]; then
  # Keep XLA_FLAGS out of DRIVER_ENVS: quotes introduced by variable expansion
  # do not protect its spaces. The TPU workers inherit this exact value from
  # the raylets, which is verified below before the protected request starts.
  PREFILL_PROJECTION_ENVS=" PHASED_PROFILING_DIR=$PREFILL_PROFILE_PREFIX PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR=1 PYTHON_TRACER_LEVEL=0 XLA_FLAGS=\"--xla_dump_to=$PREFILL_HLO_PREFIX --xla_dump_hlo_as_text --xla_dump_hlo_module_re=jit_step_fun_impl\""
fi
DECODE_PROJECTION_ENVS=
if [[ $DECODE_PROJECTION_CAPTURE == 1 ]]; then
  # Restrict extra pass dumps to the final-codegen selector and use short HLO
  # text for jit_step_fun_impl. The exact run-owned raw tree is compacted on
  # its producing host before fleet gather so multi-gigabyte buffer-assignment
  # files are never copied to worker 0.
  DECODE_PROJECTION_ENVS=" XLA_FLAGS=\"--xla_dump_to=$DECODE_HLO_PREFIX --xla_dump_hlo_as_text --xla_dump_hlo_as_long_text=false --xla_dump_hlo_module_re=jit_step_fun_impl --xla_dump_hlo_pass_re=after_codegen\""
fi
RAYLET_ENVS="$COMMON_ENVS$PREFILL_PROJECTION_ENVS$DECODE_PROJECTION_ENVS LIBTPU_INIT_ARGS=\"--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false\""
DRIVER_ENVS='NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 RUNAI_STREAMER_CONCURRENCY=32 RUNAI_STREAMER_MEMORY_LIMIT=34359738368 JAX_SHARE_BINARY_BETWEEN_HOSTS=1 JAX_SHARE_BINARY_BETWEEN_HOSTS_TIMEOUT_MS=120000 '"$COMMON_ENVS"

say "launching exact protected legacy runtime"
runtime_started=1
RAY_JOIN_TIMEOUT_SECONDS=120 EXTRA_ENVS="$RAYLET_ENVS" TPU_MIN_TOKEN_BUCKET=32 \
  bash "$HARNESS_REPO/scripts/launch_glm_32chip.sh" \
  >"$RUN_DIR/launch.log" 2>&1

# Ray hygiene must preserve the approved read-only model/oracle mirror. Re-run
# the exact prerequisite after launch and before any driver/model invocation;
# a future launcher regression therefore fails before loading 753B.
bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 120 \
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$prereq" >"$RUN_DIR/prereq_post_launch.txt" 2>&1 || {
  say "ABORT: bounded post-launch legacy/golden/OOB prerequisite failed"
  exit 1
}
has_eight_unique_markers "$RUN_DIR/prereq_post_launch.txt" PREREQ_OK || {
  say "ABORT: Ray launch drifted the legacy/golden/OOB prerequisite"
  exit 1
}

# Verify every raylet inherited every source-defining flag.
# shellcheck disable=SC2016
env_check='p=$(pgrep -x raylet | head -1); f=/tmp/dsa_oracle_env_$$; [ -n "$p" ] && tr "\0" "\n" < /proc/$p/environ > "$f"; if grep -qx "GLM_DCP=1" "$f" && grep -qx "GLM_DCP_SCATTER_IMPL=pageloop" "$f" && grep -qx "GLM_DSA_DCP_SCATTER_IMPL=flat" "$f" && grep -qx "GLM_DSA_DUMP_TOPK='"$DUMP_PREFIX"'" "$f" && grep -qx "GLM_DSA_DUMP_TOPK_EVENTS=all" "$f" && grep -qx "GLM_EXPECT_CODE_HASH='"$LEGACY_SHORT"'" "$f" && grep -qx "GLM_LOAD_CHECKSUM=1" "$f" && grep -qx "GLM_LOAD_NAN_CHECK=1" "$f" && grep -qx "GLM_PWAL_NAN_CHECK=1" "$f" && grep -qx "GLM_STATE_HASH_REF=/tmp/golden.json" "$f" && grep -qx "GLM_WK_OOB_DIR='"$OOB_DIR"'" "$f" && grep -qx "GLM_WK_OOB_GOLDEN=/tmp/golden.json" "$f"; then echo "ENV_OK $(hostname)"; else echo "ENV_BAD $(hostname)"; fi; rm -f "$f"'
if [[ $INTERNAL_CAPTURE == 1 ]]; then
  # shellcheck disable=SC2016
  env_check='p=$(pgrep -x raylet | head -1); f=/tmp/dsa_internal_env_$$; [ -n "$p" ] && tr "\0" "\n" < /proc/$p/environ > "$f"; if grep -qx "PYTHONPATH='"$INTERNAL_PYTHONPATH"'" "$f" && grep -qx "GLM_DSA_DUMP_TOPK='"$DUMP_PREFIX"'" "$f" && grep -qx "GLM_DSA_DUMP_INTERNALS='"$INTERNAL_DUMP_PREFIX"'" "$f" && grep -qx "GLM_DSA_DUMP_INTERNALS_MODE='"$INTERNAL_MODE"'" "$f" && grep -qx "GLM_DSA_DUMP_INTERNALS_LAYER='"$INTERNAL_LAYER"'" "$f" && grep -qx "GLM_DSA_DUMP_INTERNALS_POSITION='"$INTERNAL_TARGET_POSITION"'" "$f" && grep -qx "GLM_DSA_DUMP_INTERNALS_RUN_TAG='"$TAG"'" "$f" && grep -qx "GLM_DSA_DUMP_INTERNALS_CODE_HASH='"$LEGACY_PIN"'" "$f" && grep -qx "GLM_DSA_DUMP_INTERNALS_ORACLE_PIN='"$ORACLE_PIN"'" "$f" && grep -qx "GLM_EXPECT_CODE_HASH='"$LEGACY_SHORT"'" "$f" && grep -qx "GLM_LOAD_CHECKSUM=1" "$f" && grep -qx "GLM_STATE_HASH_REF=/tmp/golden.json" "$f"; then echo "ENV_OK $(hostname)"; else echo "ENV_BAD $(hostname)"; fi; rm -f "$f"'
elif [[ $MAIN_CACHE_CAPTURE == 1 ]]; then
  # shellcheck disable=SC2016
  env_check='p=$(pgrep -x raylet | head -1); f=/tmp/main_cache_env_$$; [ -n "$p" ] && tr "\0" "\n" < /proc/$p/environ > "$f"; if grep -qx "PYTHONPATH='"$OBSERVER_RUNTIME_REPO"'" "$f" && grep -qx "GLM_DSA_DUMP_TOPK='"$DUMP_PREFIX"'" "$f" && grep -qx "GLM_DCP_CACHE_DUMP='"$MAIN_CACHE_DUMP_PREFIX"'" "$f" && grep -qx "GLM_DCP_CACHE_DUMP_LAYERS=1" "$f" && grep -qx "GLM_DCP_CACHE_DUMP_STEPS=4,5" "$f" && grep -qx "GLM_EXPECT_CODE_HASH='"$LEGACY_SHORT"'" "$f" && grep -qx "GLM_LOAD_CHECKSUM=1" "$f" && grep -qx "GLM_STATE_HASH_REF=/tmp/golden.json" "$f"; then echo "ENV_OK $(hostname)"; else echo "ENV_BAD $(hostname)"; fi; rm -f "$f"'
fi
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$env_check" >"$RUN_DIR/raylet_env.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/raylet_env.txt" ENV_OK || {
  say "ABORT: eight-host raylet environment mismatch"
  exit 1
}
if [[ $PROMPT_CACHE_CAPTURE == 1 ]]; then
  # shellcheck disable=SC2016
  cache_env_check='p=$(pgrep -x raylet | head -1); f=/tmp/prompt_cache_env_$$; [ -n "$p" ] && tr "\0" "\n" < /proc/$p/environ > "$f"; if grep -qx "GLM_DCP_CACHE_DUMP='"$PROMPT_CACHE_DUMP_PREFIX"'" "$f" && grep -qx "GLM_DCP_CACHE_DUMP_LAYERS='"$PROMPT_CACHE_SLOT"'" "$f"; then echo "CACHE_ENV_OK $(hostname)"; else echo "CACHE_ENV_BAD $(hostname)"; fi; rm -f "$f"'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$cache_env_check" >"$RUN_DIR/raylet_cache_env.txt" 2>&1
  has_eight_unique_markers "$RUN_DIR/raylet_cache_env.txt" CACHE_ENV_OK || {
    say "ABORT: eight-host prompt-cache environment mismatch"
    exit 1
  }
fi
if [[ $PREFILL_PROJECTION_CAPTURE == 1 ]]; then
  # shellcheck disable=SC2016
  profile_env_check='p=$(pgrep -x raylet | head -1); f=/tmp/prefill_profile_env_$$; [ -n "$p" ] && tr "\0" "\n" < /proc/$p/environ > "$f"; if grep -qx "PHASED_PROFILING_DIR='"$PREFILL_PROFILE_PREFIX"'" "$f" && grep -qx "PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR=1" "$f" && grep -qx "PYTHON_TRACER_LEVEL=0" "$f" && grep -qx "XLA_FLAGS=--xla_dump_to='"$PREFILL_HLO_PREFIX"' --xla_dump_hlo_as_text --xla_dump_hlo_module_re=jit_step_fun_impl" "$f"; then echo "PROFILE_ENV_OK $(hostname)"; else echo "PROFILE_ENV_BAD $(hostname)"; fi; rm -f "$f"'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$profile_env_check" >"$RUN_DIR/raylet_prefill_profile_env.txt" 2>&1
  has_eight_unique_markers "$RUN_DIR/raylet_prefill_profile_env.txt" PROFILE_ENV_OK || {
    say "ABORT: eight-host prefill profile/HLO environment mismatch"
    exit 1
  }
fi
if [[ $DECODE_PROJECTION_CAPTURE == 1 ]]; then
  # shellcheck disable=SC2016
  decode_hlo_env_check='p=$(pgrep -x raylet | head -1); f=/tmp/decode_hlo_env_$$; [ -n "$p" ] && tr "\0" "\n" < /proc/$p/environ > "$f"; if grep -qx "XLA_FLAGS=--xla_dump_to='"$DECODE_HLO_PREFIX"' --xla_dump_hlo_as_text --xla_dump_hlo_as_long_text=false --xla_dump_hlo_module_re=jit_step_fun_impl --xla_dump_hlo_pass_re=after_codegen" "$f"; then echo "DECODE_HLO_ENV_OK $(hostname)"; else echo "DECODE_HLO_ENV_BAD $(hostname)"; fi; rm -f "$f"'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$decode_hlo_env_check" >"$RUN_DIR/raylet_decode_hlo_env.txt" 2>&1
  has_eight_unique_markers "$RUN_DIR/raylet_decode_hlo_env.txt" DECODE_HLO_ENV_OK || {
    say "ABORT: eight-host decode-HLO environment mismatch"
    exit 1
  }
fi

say "running one exact $PROFILE raw passkey item"
(
  cd "$HARNESS_REPO/bench"
  set -a
  # shellcheck disable=SC1091
  . "$HARNESS_REPO/.env"
  set +a
  # shellcheck disable=SC2086
  LIBTPU_INIT_ARGS='--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false' \
    env $DRIVER_ENVS setsid --wait /home/gianl/vllm-env/bin/python -u \
    glm_longctx.py --lengths "$BENCHMARK_LENGTH" --depths "$BENCHMARK_DEPTH" --trials 1 \
    --protocol raw --max-new "$EXPECTED_GENERATED_TOKENS" \
    --max-len "$BENCHMARK_MAX_LEN" --max-seqs 1 \
    --max-batched-tokens "$BENCHMARK_MAX_BATCHED_TOKENS" --gmu 0.90 \
    --num-gpu-blocks "$BENCHMARK_NUM_BLOCKS" \
    --seed 12345 --note "fresh flat all-event DSA oracle $TAG" \
    --out-json "$RUN_DIR/legacy_summary.json"
) >"$RUN_DIR/legacy.log" 2>&1
grep -q '\[longctx\].*correct=True' "$RUN_DIR/legacy.log" || {
  say "ABORT: fresh legacy item was not correct"
  exit 1
}

# Ray de-duplicates aggregate driver logs, so validate the load and final state
# from each host's own current-session worker logs. Record any repair messages,
# but do not require a corruption draw: exact final state is the acceptance gate.
# shellcheck disable=SC2016
integrity_check='logs=/tmp/ray/session_latest/logs; checksum=$(grep -Rhs --include="worker-*.out" -E "\[GLM_LOAD_CHECKSUM\].*SUMMARY verified=1882 mismatches=0 skipped=312$" "$logs" 2>/dev/null | tail -1); state=$(grep -Rhs --include="worker-*.out" -E "\[GLM_STATE_HASH\].*manifest VERIFIED leaves=2455 combined=371110325 ref=/tmp/golden.json$" "$logs" 2>/dev/null | tail -1); refusal=$(grep -Rhs --include="worker-*.out" -E "StateHashMismatchError|LoadNanCheckError|PwalNanCheckError|LoadChecksumError|CodeFingerprintMismatchError|GLM_LOAD_CHECKSUM.*diverged|GLM_WK_OOB_GOLDEN.*(no manifest key|does not describe|no checkpoint prefix|still != golden)|GLM_WK_OOB_DIR.*STILL.*all-zero" "$logs" 2>/dev/null | tail -1); repair=$(grep -Rhs --include="worker-*.out" -E "zero-fill repaired at PWAL|manifest-repair at PWAL" "$logs" 2>/dev/null || true); printf "%s\n%s\n" "$checksum" "$state"; [ -z "$repair" ] || printf "%s\n" "$repair"; if [ -n "$checksum" ] && [ -n "$state" ] && [ -z "$refusal" ]; then echo "INTEGRITY_OK $(hostname)"; else [ -z "$refusal" ] || printf "%s\n" "$refusal"; echo "INTEGRITY_BAD $(hostname)"; fi'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$integrity_check" >"$RUN_DIR/fleet_integrity.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/fleet_integrity.txt" INTEGRITY_OK || {
  say "ABORT: load/final-state integrity is not exact on all eight hosts"
  exit 1
}
if [[ $INTERNAL_CAPTURE == 1 ]]; then
  if [[ $DENSE_PARTIAL_CAPTURE == 1 ]]; then
    # Every host owns four physical model ranks. The device-side observer
    # writes one exact pre-psum BF16 row per rank for the requested position.
    # shellcheck disable=SC2016
    internal_integrity='logs=/tmp/ray/session_latest/logs; armed=$(grep -Rhs --include="worker-*.out" --include="worker-*.err" -F "[GLM_DSA_DUMP_INTERNALS] DENSE PARTIAL ARMED" "$logs" 2>/dev/null | tail -1); wrote=$(grep -Rhs --include="worker-*.out" --include="worker-*.err" -F "[GLM_DSA_DUMP_INTERNALS] first dense-partial file written" "$logs" 2>/dev/null | tail -1); files=$(find /tmp/'"$TAG"' -type f -name "internals.*.position'"$INTERNAL_TARGET_POSITION"'.proc*.rank*.npz" 2>/dev/null | wc -l); ranks=$(find /tmp/'"$TAG"' -type f -name "internals.*.position'"$INTERNAL_TARGET_POSITION"'.proc*.rank*.npz" -printf "%f\n" 2>/dev/null | sed -n "s/.*\.rank\([0-9][0-9]\)\.npz$/\1/p" | sort -u | wc -l); errors=$(find /tmp/'"$TAG"' -type f -name "*.INTERNAL.ERROR.*" 2>/dev/null | wc -l); printf "%s\n%s\nfiles=%s ranks=%s errors=%s\n" "$armed" "$wrote" "$files" "$ranks" "$errors"; if [ -n "$armed" ] && [ -n "$wrote" ] && [ "$files" -eq 4 ] && [ "$ranks" -eq 4 ] && [ "$errors" -eq 0 ]; then echo "INTERNAL_OK $(hostname)"; echo "INTERNAL_OWNER $(hostname)"; else echo "INTERNAL_BAD $(hostname)"; fi'
  elif [[ $PROMPT_KEY_CAPTURE == 1 ]]; then
    # Prompt prefill is replicated over the accepted model mesh. Permit one
    # independently produced file per JAX process and require every present
    # replica to be sealed bitwise by the post-run inspector.
    # shellcheck disable=SC2016
    internal_integrity='logs=/tmp/ray/session_latest/logs; armed=$(grep -Rhs --include="worker-*.out" --include="worker-*.err" -F "[GLM_DSA_DUMP_INTERNALS] ARMED" "$logs" 2>/dev/null | tail -1); wrote=$(grep -Rhs --include="worker-*.out" --include="worker-*.err" -F "[GLM_DSA_DUMP_INTERNALS] first state file written" "$logs" 2>/dev/null | tail -1); files=$(find /tmp/'"$TAG"' -type f -name "internals.*.position'"$INTERNAL_TARGET_POSITION"'.proc*.npz" 2>/dev/null | wc -l); errors=$(find /tmp/'"$TAG"' -type f -name "*.INTERNAL.ERROR.*" 2>/dev/null | wc -l); printf "%s\n%s\nfiles=%s errors=%s\n" "$armed" "$wrote" "$files" "$errors"; if [ -z "$armed" ] || [ "$errors" -ne 0 ] || [ "$files" -gt 1 ]; then echo "INTERNAL_BAD $(hostname)"; elif [ "$files" -eq 1 ] && [ -n "$wrote" ]; then echo "INTERNAL_OK $(hostname)"; echo "INTERNAL_OWNER $(hostname)"; elif [ "$files" -eq 0 ] && [ -z "$wrote" ]; then echo "INTERNAL_OK $(hostname)"; echo "INTERNAL_NONOWNER $(hostname)"; else echo "INTERNAL_BAD $(hostname)"; fi'
  else
    # The decode token row is DCP-sharded, so exactly process 0 owns it.
    # shellcheck disable=SC2016
    internal_integrity='logs=/tmp/ray/session_latest/logs; armed=$(grep -Rhs --include="worker-*.out" --include="worker-*.err" -F "[GLM_DSA_DUMP_INTERNALS] ARMED" "$logs" 2>/dev/null | tail -1); wrote=$(grep -Rhs --include="worker-*.out" --include="worker-*.err" -F "[GLM_DSA_DUMP_INTERNALS] first state file written" "$logs" 2>/dev/null | tail -1); files=$(find /tmp/'"$TAG"' -type f -name "internals.*.position'"$INTERNAL_TARGET_POSITION"'.proc*.npz" 2>/dev/null | wc -l); owner=$(find /tmp/'"$TAG"' -type f -name "internals.*.position'"$INTERNAL_TARGET_POSITION"'.proc0.npz" 2>/dev/null | head -1); errors=$(find /tmp/'"$TAG"' -type f -name "*.INTERNAL.ERROR.*" 2>/dev/null | wc -l); printf "%s\n%s\nfiles=%s errors=%s\n" "$armed" "$wrote" "$files" "$errors"; if [ -z "$armed" ] || [ "$errors" -ne 0 ] || [ "$files" -gt 1 ]; then echo "INTERNAL_BAD $(hostname)"; elif [ "$files" -eq 1 ] && [ -n "$wrote" ] && [ -n "$owner" ]; then echo "INTERNAL_OK $(hostname)"; echo "INTERNAL_OWNER $(hostname)"; elif [ "$files" -eq 0 ] && [ -z "$wrote" ]; then echo "INTERNAL_OK $(hostname)"; echo "INTERNAL_NONOWNER $(hostname)"; else echo "INTERNAL_BAD $(hostname)"; fi'
  fi
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$internal_integrity" >"$RUN_DIR/fleet_internal_integrity.txt" 2>&1
  has_eight_unique_markers "$RUN_DIR/fleet_internal_integrity.txt" INTERNAL_OK || {
    say "ABORT: layer-0 DSA internal callback did not arm cleanly fleet-wide"
    exit 1
  }
  internal_owner_count=$(grep -c '^INTERNAL_OWNER ' \
    "$RUN_DIR/fleet_internal_integrity.txt" || true)
  internal_nonowner_count=$(grep -c '^INTERNAL_NONOWNER ' \
    "$RUN_DIR/fleet_internal_integrity.txt" || true)
  if [[ $DENSE_PARTIAL_CAPTURE == 1 ]]; then
    [[ $internal_owner_count -eq 8 && $internal_nonowner_count -eq 0 ]] || {
      say "ABORT: dense-partial physical-rank host coverage drifted"
      exit 1
    }
  elif [[ $PROMPT_KEY_CAPTURE == 1 ]]; then
    ((internal_owner_count >= 1 && internal_owner_count <= 8 &&
      internal_owner_count + internal_nonowner_count == 8)) || {
      say "ABORT: prompt-key internal replica coverage drifted"
      exit 1
    }
  else
    [[ $internal_owner_count -eq 1 && $internal_nonowner_count -eq 7 ]] || {
      say "ABORT: layer-0 DSA internal owner coverage drifted"
      exit 1
    }
  fi
fi
if [[ $PROMPT_CACHE_CAPTURE == 1 ]]; then
  # The host-side hook is outside the model JIT. Each process must write one
  # post-forward snapshot for each of the four 2,048-token prefill chunks;
  # decode-only steps are intentionally skipped by the inherited observer.
  # shellcheck disable=SC2016
  cache_integrity='logs=/tmp/ray/session_latest/logs; armed=$(grep -Rhs --include="worker-*.out" --include="worker-*.err" -F "[GLM_DCP_CACHE_DUMP] ARMED" "$logs" 2>/dev/null | tail -1); failures=$(grep -Rhs --include="worker-*.out" --include="worker-*.err" -F "GLM_DCP_CACHE_DUMP failed" "$logs" 2>/dev/null | wc -l); files=$(find /tmp/'"$TAG"' -type f -name "index_cache.postfwd.step*.proc*.npz" 2>/dev/null | wc -l); final=$(find /tmp/'"$TAG"' -type f -name "index_cache.postfwd.step0004.proc*.npz" 2>/dev/null | wc -l); printf "%s\nfiles=%s final=%s failures=%s\n" "$armed" "$files" "$final" "$failures"; if [ -n "$armed" ] && [ "$files" -eq 4 ] && [ "$final" -eq 1 ] && [ "$failures" -eq 0 ]; then echo "CACHE_OK $(hostname)"; else echo "CACHE_BAD $(hostname)"; fi'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$cache_integrity" >"$RUN_DIR/fleet_prompt_cache_integrity.txt" 2>&1
  has_eight_unique_markers "$RUN_DIR/fleet_prompt_cache_integrity.txt" CACHE_OK || {
    say "ABORT: layer-$PROMPT_CACHE_LAYER_ID prompt-cache observer coverage is incomplete"
    exit 1
  }
fi
if [[ $MAIN_CACHE_CAPTURE == 1 ]]; then
  # The exact-step selector captures only the final prefill state and the
  # first decode-updated state. Each host owns one JAX process and must write
  # exactly those two fully replicated cache snapshots.
  # shellcheck disable=SC2016
  main_cache_integrity='logs=/tmp/ray/session_latest/logs; armed=$(grep -Rhs --include="worker-*.out" --include="worker-*.err" -F "[GLM_DCP_CACHE_DUMP] ARMED" "$logs" 2>/dev/null | tail -1); failures=$(grep -Rhs --include="worker-*.out" --include="worker-*.err" -F "GLM_DCP_CACHE_DUMP failed" "$logs" 2>/dev/null | wc -l); files=$(find /tmp/'"$TAG"' -type f -name "main_cache.postfwd.step*.proc*.npz" 2>/dev/null | wc -l); prefill=$(find /tmp/'"$TAG"' -type f -name "main_cache.postfwd.step0004.proc*.npz" 2>/dev/null | wc -l); decode=$(find /tmp/'"$TAG"' -type f -name "main_cache.postfwd.step0005.proc*.npz" 2>/dev/null | wc -l); printf "%s\nfiles=%s prefill=%s decode=%s failures=%s\n" "$armed" "$files" "$prefill" "$decode" "$failures"; if [ -n "$armed" ] && [ "$files" -eq 2 ] && [ "$prefill" -eq 1 ] && [ "$decode" -eq 1 ] && [ "$failures" -eq 0 ]; then echo "MAIN_CACHE_OK $(hostname)"; else echo "MAIN_CACHE_BAD $(hostname)"; fi'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$main_cache_integrity" >"$RUN_DIR/fleet_main_cache_integrity.txt" 2>&1
  has_eight_unique_markers "$RUN_DIR/fleet_main_cache_integrity.txt" MAIN_CACHE_OK || {
    say "ABORT: layer-0 main-cache observer coverage is incomplete"
    exit 1
  }
fi
if [[ $PREFILL_PROJECTION_CAPTURE == 1 ]]; then
  # One-step phase profiling writes one XPlane/JSON trace and two composition
  # records (start and stop) on every host. Binary sharing may leave the
  # module-filtered optimized HLO on only the compile leader.
  # shellcheck disable=SC2016
  profile_integrity='root='"$PREFILL_PROFILE_PREFIX"'; hlo_root='"$PREFILL_HLO_PREFIX"'; xplanes=$(find "$root" -type f -name "*.xplane.pb" 2>/dev/null | wc -l); traces=$(find "$root" -type f -name "*.trace.json.gz" 2>/dev/null | wc -l); stats=$(find "$root" -type f -name "batch_composition_stats_*.json" 2>/dev/null | wc -l); hlo=$(find "$hlo_root" -type f -name "*.txt" -exec grep -l "HloModule jit_step_fun_impl, is_scheduled=true" {} + 2>/dev/null | wc -l); printf "xplanes=%s traces=%s stats=%s hlo=%s\n" "$xplanes" "$traces" "$stats" "$hlo"; if [ "$xplanes" -eq 1 ] && [ "$traces" -eq 1 ] && [ "$stats" -eq 2 ]; then echo "PROFILE_OK $(hostname)"; else echo "PROFILE_BAD $(hostname)"; fi; if [ "$hlo" -ge 1 ]; then echo "HLO_OWNER $(hostname)"; fi'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$profile_integrity" >"$RUN_DIR/fleet_prefill_profile_integrity.txt" 2>&1
  has_eight_unique_markers "$RUN_DIR/fleet_prefill_profile_integrity.txt" PROFILE_OK || {
    say "ABORT: accepted one-step prefill profile coverage is incomplete"
    exit 1
  }
  [[ $(grep -c '^HLO_OWNER ' "$RUN_DIR/fleet_prefill_profile_integrity.txt" || true) -ge 1 ]] || {
    say "ABORT: no accepted scheduled jit_step_fun_impl HLO dump owner exists"
    exit 1
  }
fi
if [[ $DECODE_PROJECTION_CAPTURE == 1 ]]; then
  # Select the unique M32 decode module on each compile owner, gzip it
  # reproducibly, and reclaim only this run's raw XLA tree before scp.  A host
  # with no local compiler output is an admissible binary-sharing non-owner.
  decode_hlo_compactor=$WORKTREE/scripts/greenfield/compact_accepted_decode_hlo.sh
  decode_hlo_compactor_sha=$(sha256sum "$decode_hlo_compactor" | cut -d ' ' -f 1)
  for worker in 0 1 2 3 4 5 6 7; do
    gcloud compute tpus tpu-vm scp --zone "$ZONE" --worker="$worker" \
      "$decode_hlo_compactor" \
      "$POD:/tmp/$TAG/compact_accepted_decode_hlo.sh" >/dev/null 2>&1
  done
  # shellcheck disable=SC2016
  decode_hlo_integrity='helper=/tmp/'"$TAG"'/compact_accepted_decode_hlo.sh; actual=$(sha256sum "$helper" | cut -d " " -f 1); if [ "$actual" = '"$decode_hlo_compactor_sha"' ]; then bash "$helper" '"$DECODE_HLO_PREFIX"' '"$DECODE_HLO_COMPACT_PREFIX"'; else echo "DECODE_HLO_BAD $(hostname) helper_sha256=$actual"; fi'
  if ! gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$decode_hlo_integrity" \
    >"$RUN_DIR/fleet_decode_hlo_integrity.txt" 2>&1; then
    say "ABORT: accepted M32 decode HLO remote compaction failed"
    exit 1
  fi
  has_eight_unique_markers "$RUN_DIR/fleet_decode_hlo_integrity.txt" DECODE_HLO_OK || {
    say "ABORT: accepted M32 decode HLO selection/compaction failed"
    exit 1
  }
  decode_projection_owner_count=$(grep -c '^DECODE_HLO_OWNER ' \
    "$RUN_DIR/fleet_decode_hlo_integrity.txt" || true)
  [[ $decode_projection_owner_count -ge 1 ]] || {
    say "ABORT: no accepted M32 decode HLO compile owner exists"
    exit 1
  }
fi

run_id=$(sed -n 's/.*\[longctx\] run_id=\([0-9][0-9]*\).*/\1/p' \
  "$RUN_DIR/legacy.log" | tail -1)
[[ -n $run_id ]] || {
  say "ABORT: legacy run ID is unavailable"
  exit 1
}

say "gathering append-only legacy dump files for DB run $run_id"
for worker in 0 1 2 3 4 5 6 7; do
  if gcloud compute tpus tpu-vm scp --zone "$ZONE" --worker="$worker" \
      --recurse "$POD:/tmp/$TAG" "$SOURCE_DIR/w$worker" \
      >/dev/null 2>&1; then
    echo "$worker" >>"$RUN_DIR/dump_hosts.txt"
  fi
done
find "$SOURCE_DIR" -type f -name '*.ERROR*' -print -quit | grep -q . && {
  say "ABORT: armed DSA callback emitted an error sentinel"
  exit 1
}
dump_count=$(find "$SOURCE_DIR" -type f -name 'topk.step*.evt*.proc*.npz' | wc -l)
[[ $dump_count -ge 294 ]] || {
  say "ABORT: incomplete DSA source capture files=$dump_count"
  exit 1
}
internal_count=0
if [[ $INTERNAL_CAPTURE == 1 ]]; then
  if [[ $DENSE_PARTIAL_CAPTURE == 1 ]]; then
    internal_count=$(find "$SOURCE_DIR" -type f \
      -name "internals.*.position${INTERNAL_TARGET_POSITION}.proc*.rank*.npz" | wc -l)
    [[ $internal_count -eq 32 ]] || {
      say "ABORT: expected 32 physical dense-partial files, found $internal_count"
      exit 1
    }
  else
    internal_count=$(find "$SOURCE_DIR" -type f \
      -name "internals.*.position${INTERNAL_TARGET_POSITION}.proc*.npz" | wc -l)
  fi
  if [[ $DENSE_PARTIAL_CAPTURE == 1 ]]; then
    :
  elif [[ $PROMPT_KEY_CAPTURE == 1 ]]; then
    ((internal_count >= 1 && internal_count <= 8)) || {
      say "ABORT: expected 1..8 prompt-key replica files, found $internal_count"
      exit 1
    }
  else
    [[ $internal_count -eq 1 ]] || {
      say "ABORT: expected one DCP-owner DSA internal file, found $internal_count"
      exit 1
    }
  fi
fi
prompt_cache_source_count=0
if [[ $PROMPT_CACHE_CAPTURE == 1 ]]; then
  prompt_cache_source_count=$(find "$SOURCE_DIR" -type f \
    -name 'index_cache.postfwd.step*.proc*.npz' | wc -l)
  prompt_cache_final_count=$(find "$SOURCE_DIR" -type f \
    -name 'index_cache.postfwd.step0004.proc*.npz' | wc -l)
  [[ $prompt_cache_source_count -eq 32 && $prompt_cache_final_count -eq 8 ]] || {
    say "ABORT: prompt-cache source coverage drifted total=$prompt_cache_source_count final=$prompt_cache_final_count"
    exit 1
  }
fi
main_cache_source_count=0
if [[ $MAIN_CACHE_CAPTURE == 1 ]]; then
  main_cache_source_count=$(find "$SOURCE_DIR" -type f \
    -name 'main_cache.postfwd.step*.proc*.npz' | wc -l)
  main_cache_prefill_count=$(find "$SOURCE_DIR" -type f \
    -name 'main_cache.postfwd.step0004.proc*.npz' | wc -l)
  main_cache_decode_count=$(find "$SOURCE_DIR" -type f \
    -name 'main_cache.postfwd.step0005.proc*.npz' | wc -l)
  [[ $main_cache_source_count -eq 16 && $main_cache_prefill_count -eq 8 && \
     $main_cache_decode_count -eq 8 ]] || {
    say "ABORT: main-cache source coverage drifted total=$main_cache_source_count prefill=$main_cache_prefill_count decode=$main_cache_decode_count"
    exit 1
  }
fi
prefill_profile_xplane_count=0
prefill_profile_trace_count=0
prefill_profile_hlo_count=0
if [[ $PREFILL_PROJECTION_CAPTURE == 1 ]]; then
  prefill_profile_xplane_count=$(find "$SOURCE_DIR" -type f \
    -name '*.xplane.pb' | wc -l)
  prefill_profile_trace_count=$(find "$SOURCE_DIR" -type f \
    -name '*.trace.json.gz' | wc -l)
  prefill_profile_hlo_count=$(find "$SOURCE_DIR" -type f -name '*.txt' \
    -exec grep -l 'HloModule jit_step_fun_impl, is_scheduled=true' {} + | wc -l)
  [[ $prefill_profile_xplane_count -eq 8 && $prefill_profile_trace_count -eq 8 && \
     $prefill_profile_hlo_count -ge 1 ]] || {
    say "ABORT: gathered prefill profile/HLO coverage drifted xplanes=$prefill_profile_xplane_count traces=$prefill_profile_trace_count hlo=$prefill_profile_hlo_count"
    exit 1
  }
fi
decode_projection_hlo_count=0
if [[ $DECODE_PROJECTION_CAPTURE == 1 ]]; then
  decode_projection_hlo_count=$(find "$SOURCE_DIR" -type f \
    -name 'accepted_decode.after_codegen.txt.gz' | wc -l)
  ((decode_projection_hlo_count >= 1 && decode_projection_hlo_count <= 8)) || {
    say "ABORT: gathered decode HLO owner coverage drifted files=$decode_projection_hlo_count"
    exit 1
  }
  [[ $decode_projection_hlo_count -eq $decode_projection_owner_count ]] || {
    say "ABORT: gathered decode HLO owner set is incomplete markers=$decode_projection_owner_count files=$decode_projection_hlo_count"
    exit 1
  }
fi

/home/gianl/vllm-env/bin/python - "$RESULTS_DB" "$run_id" \
  "$EXPECTED_PROMPT_TOKENS" "$EXPECTED_GENERATED_TOKENS" \
  >"$RUN_DIR/source_identity.json" <<'PY'
import json
import sqlite3
import sys

connection = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
connection.row_factory = sqlite3.Row
run_id = int(sys.argv[2])
row = connection.execute(
    "SELECT r.harness_git,r.fork_git,i.id AS item_row_id,i.correct,"
    "i.n_prompt_tokens,i.n_gen_tokens,i.raw_output FROM runs r JOIN items i "
    "ON i.run_id=r.run_id WHERE r.run_id=? ORDER BY i.id", (run_id,)
).fetchall()
assert len(row) == 1, row
value = dict(row[0])
assert value["correct"] == 1
assert value["n_prompt_tokens"] == int(sys.argv[3])
assert value["n_gen_tokens"] == int(sys.argv[4])
print(json.dumps(value, indent=2, sort_keys=True))
PY
item_row_id=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["item_row_id"])' \
  "$RUN_DIR/source_identity.json")
source_harness=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["harness_git"])' \
  "$RUN_DIR/source_identity.json")
source_fork=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["fork_git"])' \
  "$RUN_DIR/source_identity.json")
[[ $source_harness == "$HARNESS_SHORT" && $source_fork == "$ORACLE_SHORT" ]] || {
  say "ABORT: DB code provenance drifted"
  exit 1
}

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/capture_short_context_dsa_oracle.py" \
  --results-db "$RESULTS_DB" \
  --token-oracle-dir "$TOKEN_ORACLE_DIR" \
  --source-dump-dir "$SOURCE_DIR" \
  --output "$ORACLE_DIR" \
  --expected-code-hash "$PIN" \
  --legacy-repository-pin "$LEGACY_PIN" \
  --token-oracle-manifest-sha256 "$TOKEN_ORACLE_SHA" \
  --run-id "$run_id" \
  --item-row-id "$item_row_id" \
  --expected-harness-git "$source_harness" \
  --expected-fork-git "$source_fork" \
  --expected-benchmark "$EXPECTED_BENCHMARK" \
  --expected-model-uri gs://driftbench-dsv4-uc/models/GLM-5.2-FP8 \
  --expected-prompt-tokens "$EXPECTED_PROMPT_TOKENS" \
  --expected-generated-tokens "$EXPECTED_GENERATED_TOKENS" \
  --expected-seed "$EXPECTED_SEED" \
  --expected-gold "$EXPECTED_GOLD" \
  --expected-oob-dir "$OOB_DIR" \
  --expected-dump-prefix "$DUMP_PREFIX" \
  --expected-process-count 8 \
  --first-source-step "$FIRST_SOURCE_STEP" \
  --decode-step-count 14 \
  --first-decode-position "$FIRST_DECODE_POSITION" \
  --selected-width 2048 >"$RUN_DIR/capture.json"

if [[ $INTERNAL_CAPTURE == 1 || $PREFILL_PROJECTION_CAPTURE == 1 || \
      $DECODE_PROJECTION_CAPTURE == 1 || \
      $MAIN_CACHE_CAPTURE == 1 ]]; then
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
    "$REFERENCE_8K_DSA_ORACLE" "$ORACLE_DIR" \
    >"$RUN_DIR/dsa_exact_comparison.json" <<'PY'
import json
from pathlib import Path
import sys

from glm_tpu.greenfield.validation import compare_short_context_dsa_oracles

value = compare_short_context_dsa_oracles(Path(sys.argv[1]), Path(sys.argv[2]))
print(json.dumps(value, indent=2, sort_keys=True))
PY
fi
if [[ $PROMPT_CACHE_CAPTURE == 1 ]]; then
  say "sealing accepted logical layer-$PROMPT_CACHE_LAYER_ID prompt index cache (legacy slot $PROMPT_CACHE_SLOT)"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/capture_legacy_prompt_index_cache.py" \
    --source-dump-dir "$SOURCE_DIR" \
    --output "$PROMPT_CACHE_RESULT_DIR" \
    --expected-code-hash "$PIN" \
    --legacy-repository-pin "$LEGACY_PIN" \
    --run-tag "$TAG" \
    --source-run-id "$run_id" \
    --source-item-row-id "$item_row_id" \
    --layer0-input-dir "$LAYER0_INPUT_DIR" \
    --layer0-input-manifest-sha256 "$LAYER0_INPUT_MANIFEST_SHA" \
    --layer-id "$PROMPT_CACHE_LAYER_ID" \
    >"$RUN_DIR/prompt_index_cache_capture.json"
fi

# Snapshot the append-only provenance DB at the exact source row.
/home/gianl/vllm-env/bin/python - "$RESULTS_DB" "$RUN_DIR/results_ckpt.db" <<'PY'
import sqlite3
import sys

source = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
destination = sqlite3.connect(sys.argv[2])
source.backup(destination)
assert destination.execute("pragma integrity_check").fetchone()[0] == "ok"
destination.close()
source.close()
PY

stop_owned_runtime
runtime_started=0
if [[ $MAIN_CACHE_CAPTURE == 1 ]]; then
  say "comparing legacy and protected PP8 layer-0 main-cache boundaries"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/compare_legacy_layer0_main_cache.py" \
    --source-dump-dir "$SOURCE_DIR" \
    --ingredients-dir "$MAIN_CACHE_INGREDIENTS_DIR" \
    --output "$MAIN_CACHE_RESULT_DIR" \
    --expected-code-hash "$PIN" \
    --legacy-repository-pin "$LEGACY_PIN" \
    --accepted-oracle-pin "$ORACLE_PIN" \
    --ingredients-code-hash "$MAIN_CACHE_INGREDIENTS_CODE_HASH" \
    --ingredients-contract-sha256 "$MAIN_CACHE_INGREDIENTS_CONTRACT_SHA" \
    --ingredients-tensor-sha256 "$MAIN_CACHE_INGREDIENTS_TENSOR_SHA" \
    --run-tag "$TAG" \
    --source-run-id "$run_id" \
    --source-item-row-id "$item_row_id" \
    >"$RUN_DIR/layer0_main_cache_comparison_summary.json"
fi
if [[ $PREFILL_PROJECTION_CAPTURE == 1 ]]; then
  say "sealing accepted M2048 projection XPlane and optimized-HLO association"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/inspect_accepted_prompt_projection_lowering.py" \
    --trace-root "$SOURCE_DIR" \
    --output "$PREFILL_PROJECTION_RESULT_DIR" \
    --expected-code-hash "$PIN" \
    --legacy-code-hash "$LEGACY_PIN" \
    --run-tag "$TAG" >"$RUN_DIR/accepted_prompt_projection_lowering_summary.json"
fi
if [[ $DECODE_PROJECTION_CAPTURE == 1 ]]; then
  say "sealing accepted M32 decode projection after-codegen lowering"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/inspect_accepted_decode_projection_lowering.py" \
    --source-dump-dir "$SOURCE_DIR" \
    --output "$DECODE_PROJECTION_RESULT_DIR" \
    --expected-code-hash "$PIN" \
    --legacy-code-hash "$LEGACY_PIN" \
    --run-tag "$TAG" >"$RUN_DIR/accepted_decode_projection_lowering_summary.json"
fi
if [[ $PROMPT_CACHE_CAPTURE == 1 ]]; then
  prompt_cache_manifest_sha=$(/home/gianl/vllm-env/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
    "$PROMPT_CACHE_RESULT_DIR/manifest.json")
  if [[ $PROMPT_CACHE_LAYER_ID != 0 ]]; then
    # The one-host production probe recomputes layer-0 keys from the sealed
    # layer-0 input; deeper layers are compared offline against the executed
    # DB518 greenfield capture (compare_layer1_prompt_index_cache_offline.py).
    say "layer-$PROMPT_CACHE_LAYER_ID prompt cache sealed; production comparison is offline only"
  elif [[ $PROMPT_KEY_CAPTURE != 1 ]]; then
    say "comparing production one-row layer-0 prompt keys on one local TPU host"
    env JAX_PLATFORMS=tpu \
      TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
      TPU_PROCESS_BOUNDS=1,1,1 \
      TPU_VISIBLE_DEVICES=0,1,2,3 \
      PYTHONPATH="$WORKTREE" \
      timeout --signal=TERM --kill-after=60 1800 \
      /home/gianl/vllm-env/bin/python \
      "$WORKTREE/scripts/greenfield/probe_layer0_prompt_index_cache.py" \
      --expected-code-hash "$PIN" \
      --run-tag "$TAG" \
      --input-dir "$LAYER0_INPUT_DIR" \
      --input-manifest-sha256 "$LAYER0_INPUT_MANIFEST_SHA" \
      --prompt-cache-dir "$PROMPT_CACHE_RESULT_DIR" \
      --prompt-cache-manifest-sha256 "$prompt_cache_manifest_sha" \
      --output "$PROMPT_CACHE_COMPARISON_DIR" \
      >"$RUN_DIR/prompt_index_cache_comparison_summary.json"
  fi
fi
if [[ $INTERNAL_CAPTURE == 1 && $LAYER1_RMS_INPUT_CAPTURE == 1 ]]; then
  say "sealing accepted layer-1 fused-add/RMSNorm FP32 input boundary"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/capture_accepted_layer1_rms_input.py" \
    --source-dump-dir "$SOURCE_DIR" \
    --output "$INTERNAL_RESULT_DIR" \
    --db550-boundary "$DB550_BOUNDARY" \
    --straddler-classification "$STRADDLER_CLASSIFICATION" \
    --vllm-repository "$VLLM_ARCHIVE_REPO" \
    --run-tag "$TAG" \
    --legacy-code-hash "$LEGACY_PIN" \
    --oracle-pin "$ORACLE_PIN" \
    --model-id "$MODEL_ID" \
    --layer-name "$INTERNAL_LAYER" \
    --position "$INTERNAL_TARGET_POSITION" \
    --process-count 8 \
    --capture-process-index 0 \
    --source-row 0 \
    >"$RUN_DIR/layer1_rms_input_capture_summary.json"
elif [[ $INTERNAL_CAPTURE == 1 && $DENSE_PARTIAL_CAPTURE == 1 ]]; then
  say "sealing accepted layer-0 pre-reduction dense partials and comparing DB548"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/capture_accepted_dense_partials.py" \
    --source-dump-dir "$SOURCE_DIR" \
    --db548-dir "$DENSE_PARTIAL_PROBE_DIR" \
    --output "$INTERNAL_RESULT_DIR" \
    --run-tag "$TAG" \
    --legacy-code-hash "$LEGACY_PIN" \
    --oracle-pin "$ORACLE_PIN" \
    --db548-runner-sha256 "$DENSE_PARTIAL_PROBE_RUNNER_SHA" \
    --db548-tensor-sha256 "$DENSE_PARTIAL_PROBE_TENSOR_SHA" \
    --db548-summary-sha256 "$DENSE_PARTIAL_PROBE_SUMMARY_SHA" \
    --db548-success-sha256 "$DENSE_PARTIAL_PROBE_SUCCESS_SHA" \
    --model-id "$MODEL_ID" \
    --layer-name "$INTERNAL_LAYER" \
    --position "$INTERNAL_TARGET_POSITION" \
    --process-count 8 \
    >"$RUN_DIR/dense_partials_comparison_summary.json"
elif [[ $INTERNAL_CAPTURE == 1 && $DENSE_INPUT_CAPTURE == 1 ]]; then
  say "sealing accepted layer-0 normalized dense-MLP input"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/capture_accepted_dense_input.py" \
    --source-dump-dir "$SOURCE_DIR" \
    --output "$INTERNAL_RESULT_DIR" \
    --run-tag "$TAG" \
    --legacy-code-hash "$LEGACY_PIN" \
    --oracle-pin "$ORACLE_PIN" \
    --model-id "$MODEL_ID" \
    --layer-name "$INTERNAL_LAYER" \
    --position "$INTERNAL_TARGET_POSITION" \
    --process-count 8 \
    --capture-process-indices 0 \
    >"$RUN_DIR/dense_input_capture_summary.json"
  accepted_dense_input_capture_sha=$(sha256sum \
    "$INTERNAL_RESULT_DIR/capture.json" | awk '{print $1}')
  say "comparing accepted dense input with protected DB540"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/compare_accepted_dense_input.py" \
    --accepted-capture-dir "$INTERNAL_RESULT_DIR" \
    --probe-dir "$DENSE_CONVOLUTION_DIR" \
    --output "$RUN_DIR/dense_input_comparison" \
    --accepted-capture-file-sha256 "$accepted_dense_input_capture_sha" \
    --probe-runner-sha256 "$DENSE_CONVOLUTION_RUNNER_SHA" \
    --probe-tensor-sha256 "$DENSE_CONVOLUTION_TENSOR_SHA" \
    --probe-summary-sha256 "$DENSE_CONVOLUTION_SUMMARY_SHA" \
    --probe-success-sha256 "$DENSE_CONVOLUTION_SUCCESS_SHA" \
    --probe-run-id "$DENSE_CONVOLUTION_RUN_ID" \
    --accepted-run-tag "$TAG" \
    --legacy-code-hash "$LEGACY_PIN" \
    --oracle-pin "$ORACLE_PIN" \
    --probe-code-hash "$DENSE_CONVOLUTION_CODE_HASH" \
    --probe-tag "$DENSE_CONVOLUTION_TAG" \
    --position "$INTERNAL_TARGET_POSITION" \
    >"$RUN_DIR/dense_input_comparison_summary.json"
elif [[ $INTERNAL_CAPTURE == 1 && $DENSE_BOUNDARY_CAPTURE == 1 ]]; then
  say "sealing accepted layer-0 dense output boundary"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/capture_accepted_dense_boundary.py" \
    --source-dump-dir "$SOURCE_DIR" \
    --output "$INTERNAL_RESULT_DIR" \
    --run-tag "$TAG" \
    --legacy-code-hash "$LEGACY_PIN" \
    --oracle-pin "$ORACLE_PIN" \
    --model-id "$MODEL_ID" \
    --layer-name "$INTERNAL_LAYER" \
    --position "$INTERNAL_TARGET_POSITION" \
    --process-count 8 \
    --capture-process-indices 0 \
    >"$RUN_DIR/dense_boundary_capture_summary.json"
  accepted_dense_boundary_capture_sha=$(sha256sum \
    "$INTERNAL_RESULT_DIR/capture.json" | awk '{print $1}')
  say "comparing accepted dense boundary with protected DB540"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/compare_accepted_dense_boundary.py" \
    --accepted-capture-dir "$INTERNAL_RESULT_DIR" \
    --probe-dir "$DENSE_CONVOLUTION_DIR" \
    --output "$RUN_DIR/dense_boundary_comparison" \
    --accepted-capture-file-sha256 "$accepted_dense_boundary_capture_sha" \
    --probe-runner-sha256 "$DENSE_CONVOLUTION_RUNNER_SHA" \
    --probe-tensor-sha256 "$DENSE_CONVOLUTION_TENSOR_SHA" \
    --probe-summary-sha256 "$DENSE_CONVOLUTION_SUMMARY_SHA" \
    --probe-success-sha256 "$DENSE_CONVOLUTION_SUCCESS_SHA" \
    --probe-run-id "$DENSE_CONVOLUTION_RUN_ID" \
    --accepted-run-tag "$TAG" \
    --legacy-code-hash "$LEGACY_PIN" \
    --oracle-pin "$ORACLE_PIN" \
    --probe-code-hash "$DENSE_CONVOLUTION_CODE_HASH" \
    --probe-tag "$DENSE_CONVOLUTION_TAG" \
    --position "$INTERNAL_TARGET_POSITION" \
    >"$RUN_DIR/dense_boundary_comparison_summary.json"
elif [[ $INTERNAL_CAPTURE == 1 && $ATTENTION_UPDATE_CAPTURE == 1 ]]; then
  say "sealing accepted layer-0 post-o_proj attention update"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/capture_accepted_attention_update.py" \
    --source-dump-dir "$SOURCE_DIR" \
    --output "$INTERNAL_RESULT_DIR" \
    --run-tag "$TAG" \
    --legacy-code-hash "$LEGACY_PIN" \
    --oracle-pin "$ORACLE_PIN" \
    --model-id "$MODEL_ID" \
    --layer-name "$INTERNAL_LAYER" \
    --position "$INTERNAL_TARGET_POSITION" \
    --process-count 8 \
    --capture-process-indices 0 \
    >"$RUN_DIR/attention_update_capture_summary.json"
  accepted_update_capture_sha=$(sha256sum \
    "$INTERNAL_RESULT_DIR/capture.json" | awk '{print $1}')
  say "comparing accepted attention update with protected DB538 candidates"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/compare_accepted_attention_update.py" \
    --accepted-capture-dir "$INTERNAL_RESULT_DIR" \
    --probe-dir "$PROJECTION_REDUCTION_DIR" \
    --output "$RUN_DIR/attention_update_comparison" \
    --accepted-capture-file-sha256 "$accepted_update_capture_sha" \
    --probe-runner-sha256 "$PROJECTION_REDUCTION_RUNNER_SHA" \
    --probe-tensor-sha256 "$PROJECTION_REDUCTION_TENSOR_SHA" \
    --probe-summary-sha256 "$PROJECTION_REDUCTION_SUMMARY_SHA" \
    --probe-success-sha256 "$PROJECTION_REDUCTION_SUCCESS_SHA" \
    --probe-run-id "$PROJECTION_REDUCTION_RUN_ID" \
    --accepted-run-tag "$TAG" \
    --legacy-code-hash "$LEGACY_PIN" \
    --oracle-pin "$ORACLE_PIN" \
    --probe-code-hash "$PROJECTION_REDUCTION_CODE_HASH" \
    --probe-tag "$PROJECTION_REDUCTION_TAG" \
    --position "$INTERNAL_TARGET_POSITION" \
    >"$RUN_DIR/attention_update_comparison_summary.json"
elif [[ $INTERNAL_CAPTURE == 1 && $ATTENTION_PROJECTION_CAPTURE == 1 ]]; then
  say "sealing accepted operands on both sides of layer-0 W_UV"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/capture_accepted_attention_projection_operands.py" \
    --source-dump-dir "$SOURCE_DIR" \
    --output "$INTERNAL_RESULT_DIR" \
    --run-tag "$TAG" \
    --legacy-code-hash "$LEGACY_PIN" \
    --oracle-pin "$ORACLE_PIN" \
    --model-id "$MODEL_ID" \
    --layer-name "$INTERNAL_LAYER" \
    --position "$INTERNAL_TARGET_POSITION" \
    --process-count 8 \
    --capture-process-indices 0 \
    >"$RUN_DIR/attention_projection_capture_summary.json"
  accepted_projection_capture_sha=$(sha256sum \
    "$INTERNAL_RESULT_DIR/capture.json" | awk '{print $1}')
  say "comparing accepted and preserved table-on PP8 attention projection operands"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/compare_attention_projection_operands.py" \
    --accepted-capture-dir "$INTERNAL_RESULT_DIR" \
    --greenfield-ingredients-dir "$ATTENTION_PROJECTION_INGREDIENTS_DIR" \
    --output "$RUN_DIR/attention_projection_comparison" \
    --accepted-capture-file-sha256 "$accepted_projection_capture_sha" \
    --accepted-run-tag "$TAG" \
    --ingredients-contract-sha256 "$ATTENTION_PROJECTION_INGREDIENTS_CONTRACT_SHA" \
    --ingredients-tensor-sha256 "$ATTENTION_PROJECTION_INGREDIENTS_TENSOR_SHA" \
    --greenfield-code-hash "$ATTENTION_PROJECTION_INGREDIENTS_CODE_HASH" \
    --greenfield-run-tag "$ATTENTION_PROJECTION_INGREDIENTS_TAG" \
    --legacy-code-hash "$LEGACY_PIN" \
    --oracle-pin "$ORACLE_PIN" \
    --main-rope-table-sha256 "$MAIN_ROPE_TABLE_SHA" \
    --position "$INTERNAL_TARGET_POSITION" \
    >"$RUN_DIR/attention_projection_comparison_summary.json"
elif [[ $INTERNAL_CAPTURE == 1 && $ATTENTION_OUTPUT_CAPTURE == 1 ]]; then
  say "sealing accepted layer-0 post-W_UV/pre-o_proj operand"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/capture_accepted_attention_output_operand.py" \
    --source-dump-dir "$SOURCE_DIR" \
    --output "$INTERNAL_RESULT_DIR" \
    --run-tag "$TAG" \
    --legacy-code-hash "$LEGACY_PIN" \
    --oracle-pin "$ORACLE_PIN" \
    --model-id "$MODEL_ID" \
    --layer-name "$INTERNAL_LAYER" \
    --position "$INTERNAL_TARGET_POSITION" \
    --process-count 8 \
    --capture-process-indices 0 \
    >"$RUN_DIR/attention_output_capture_summary.json"
elif [[ $INTERNAL_CAPTURE == 1 && $PROMPT_KEY_CAPTURE == 1 ]]; then
  say "comparing accepted prompt-key producer at position $INTERNAL_TARGET_POSITION"
  env JAX_PLATFORMS=tpu \
    TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES=0,1,2,3 \
    PYTHONPATH="$WORKTREE" \
    timeout --signal=TERM --kill-after=60 1800 \
    /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/compare_accepted_prompt_key_internals.py" \
    --source-dump-dir "$SOURCE_DIR" \
    --input-dir "$LAYER0_INPUT_DIR" \
    --input-manifest-sha256 "$LAYER0_INPUT_MANIFEST_SHA" \
    --prompt-cache-dir "$PROMPT_CACHE_RESULT_DIR" \
    --prompt-cache-manifest-sha256 "$prompt_cache_manifest_sha" \
    --output "$INTERNAL_RESULT_DIR" \
    --run-tag "$TAG" \
    --greenfield-code-hash "$PIN" \
    --legacy-code-hash "$LEGACY_PIN" \
    --oracle-pin "$ORACLE_PIN" \
    --model-id "$MODEL_ID" \
    --layer-name "$INTERNAL_LAYER" \
    --position "$INTERNAL_TARGET_POSITION" \
    --process-count 8 \
    --expected-accepted-cache-sha256 "$ACCEPTED_PROMPT_CACHE_SHA" \
    --expected-candidate-cache-sha256 "$DB512_PROMPT_CACHE_SHA" \
    --expected-cache-mismatch-count 45 \
    --expected-first-cache-mismatch-position 113 \
    --capture-mode "$INTERNAL_MODE" \
    >"$RUN_DIR/prompt_key_comparison_summary.json"
elif [[ $INTERNAL_CAPTURE == 1 && $INTERNAL_COMPARE_LAYER0 == 1 ]]; then
  say "comparing accepted layer-0 scorer state on one local TPU host"
  env JAX_PLATFORMS=tpu \
    TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES=0,1,2,3 \
    PYTHONPATH="$WORKTREE" \
    timeout --signal=TERM --kill-after=60 1800 \
    /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/compare_legacy_layer0_dsa_internals.py" \
    --source-dump-dir "$SOURCE_DIR" \
    --layer0-input-dir "$LAYER0_INPUT_DIR" \
    --distributed-q-a-norm-dir "$DISTRIBUTED_Q_A_DIR" \
    --output "$INTERNAL_RESULT_DIR" \
    --run-tag "$TAG" \
    --greenfield-code-hash "$PIN" \
    --legacy-code-hash "$LEGACY_PIN" \
    --oracle-pin "$ORACLE_PIN" \
    --input-manifest-sha256 "$LAYER0_INPUT_MANIFEST_SHA" \
    --q-a-manifest-sha256 "$DISTRIBUTED_Q_A_MANIFEST_SHA" \
    --q-a-code-hash "$DISTRIBUTED_Q_A_CODE_HASH" \
    --model-id "$MODEL_ID" \
    --layer-name "$INTERNAL_LAYER" \
    --position "$INTERNAL_TARGET_POSITION" \
    --process-count 8 \
    --capture-process-indices 0 >"$RUN_DIR/internal_comparison_summary.json"
elif [[ $INTERNAL_CAPTURE == 1 ]]; then
  say "sealing accepted $INTERNAL_LAYER scorer state"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/inspect_legacy_dsa_internals.py" \
    --source-dump-dir "$SOURCE_DIR" \
    --output "$INTERNAL_RESULT_DIR" \
    --run-tag "$TAG" \
    --legacy-code-hash "$LEGACY_PIN" \
    --oracle-pin "$ORACLE_PIN" \
    --model-id "$MODEL_ID" \
    --layer-name "$INTERNAL_LAYER" \
    --position "$INTERNAL_TARGET_POSITION" \
    --process-count 8 \
    --capture-process-indices 0 >"$RUN_DIR/internal_capture_summary.json"
fi
if [[ $vllm_runtime_prepared -eq 1 ]]; then
  cleanup_vllm_runtime post || {
    say "ABORT: run-owned accepted vLLM runtime cleanup failed"
    exit 1
  }
  vllm_runtime_prepared=0
fi
strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

# Revalidate the pre-tag network contract immediately before hashing and
# archiving the run. This prevents a stale/mutated receipt from being sealed
# merely because it exists in the append-only object ledger.
terminal_ray_rule_contract=$(gcloud compute firewall-rules describe \
  "$RAY_FIREWALL_RULE" \
  --format='value(network.basename(),direction,priority,sourceRanges.list(),allowed[].map().firewall_rule().list(),disabled,targetTags.list())')
[[ $terminal_ray_rule_contract == "$ray_rule_contract" ]] &&
  [[ $(<"$RUN_DIR/ray_firewall_preflight.txt") == "$ray_rule_contract" ]] &&
  printf '%s\n' "$terminal_ray_rule_contract" | \
    bash "$WORKTREE/scripts/validate_ray_network.sh" firewall \
      "$current_ray_target" "$current_pod_id" || {
  say "ABORT: terminal Ray firewall contract drifted"
  exit 1
}
terminal_tcp_receipts=$(<"$RUN_DIR/ray_tcp6379_preflight.txt")
[[ $terminal_tcp_receipts == "$ray_tcp6379_contract" ]] &&
  printf '%s\n' "$terminal_tcp_receipts" | \
    bash "$WORKTREE/scripts/validate_ray_network.sh" receipts \
    PRETAG_TCP6379_OK 7 1,2,3,4,5,6,7 || {
  say "ABORT: terminal pre-tag TCP/6379 receipts drifted"
  exit 1
}
[[ $(grep -c '^PREFLIGHT_LISTENER_OK ' \
  "$RUN_DIR/ray_tcp6379_preflight.txt") -eq 1 &&
   $(grep '^PREFLIGHT_LISTENER_OK ' \
     "$RUN_DIR/ray_tcp6379_preflight.txt") == "$expected_listener_receipt" ]] || {
  say "ABORT: terminal pre-tag listener receipt drifted"
  exit 1
}

say "freezing fresh DSA-oracle evidence"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
/home/gianl/vllm-env/bin/python - "$RUN_DIR" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
records = []
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root).as_posix()
    if (
        not path.is_file()
        or relative == "orchestrator.log"
        or path.name in {"SUCCESS", "evidence_sha256.json"}
    ):
        continue
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    records.append({
        "byte_count": path.stat().st_size,
        "path": relative,
        "sha256": digest.hexdigest(),
    })
(root / "evidence_sha256.json").write_text(
    json.dumps({"files": records}, indent=2, sort_keys=True) + "\n"
)
PY

say "uploading fresh DSA source and compact oracle append-only"
gcloud storage cp --recursive --no-clobber "$RUN_DIR"/* \
  "$REMOTE_PREFIX/" >/dev/null

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" <<'PY' \
  >"$RUN_DIR/remote_objects.json"
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import google_crc32c

root = Path(sys.argv[1])
prefix = sys.argv[2]
paths = []
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root).as_posix()
    if (
        not path.is_file()
        or relative == "orchestrator.log"
        or path.name in {"SUCCESS", "remote_objects.json"}
    ):
        continue
    paths.append((path, relative))

def local_crc32c(path):
    checksum = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode()

def describe(item):
    path, relative = item
    remote = json.loads(subprocess.run(
        ["gcloud", "storage", "objects", "describe", f"{prefix}/{relative}", "--format=json"],
        check=True, capture_output=True, text=True,
    ).stdout)
    crc32c = remote.get("crc32c_hash") or remote.get("crc32c")
    if (
        int(remote["size"]) != path.stat().st_size
        or crc32c != local_crc32c(path)
    ):
        raise SystemExit(f"remote object verification failed: {relative}")
    return {
        "crc32c": crc32c,
        "generation": remote["generation"],
        "path": relative,
        "size": int(remote["size"]),
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

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$REMOTE_PREFIX" "$PIN" \
  "$LEGACY_PIN" "$run_id" "$item_row_id" "$dump_count" \
  "$INTERNAL_CAPTURE" "$internal_count" "$INTERNAL_COMPARE_LAYER0" \
  "$PROMPT_CACHE_CAPTURE" "$prompt_cache_source_count" "$INTERNAL_MODE" \
  "$PREFILL_PROJECTION_CAPTURE" "$prefill_profile_xplane_count" \
  "$prefill_profile_trace_count" "$prefill_profile_hlo_count" \
  "$MAIN_CACHE_CAPTURE" "$main_cache_source_count" \
  "$DECODE_PROJECTION_CAPTURE" "$decode_projection_hlo_count" \
  "$TAG" "$DENSE_PARTIAL_PROBE_DIR" \
  "$DENSE_PARTIAL_PROBE_RUNNER_SHA" "$DENSE_PARTIAL_PROBE_TENSOR_SHA" \
  "$DENSE_PARTIAL_PROBE_SUMMARY_SHA" "$DENSE_PARTIAL_PROBE_SUCCESS_SHA" \
  "$ORACLE_PIN" "$VLLM_PIN" "$VLLM_RUNTIME_ARCHIVE_SHA" \
  "$VLLM_RUNTIME_FILE_COUNT" "$VLLM_ARCHIVE_REPO" \
  "$OBSERVER_BUNDLE_SHA" "$OBSERVER_TRACKED_FILE_COUNT" \
  "$GOLDEN_MANIFEST_SHA" "$GOLDEN_MANIFEST_BYTES" \
  "$LEGACY_SOURCE_REPO" "${OBSERVER_BRANCH:-none}" "$PROMPT_CACHE_LAYER_ID" <<'PY'
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
remote = sys.argv[2]
manifest = json.loads((root / "oracle" / "manifest.json").read_text())
lines = {
    "artifact_kind": manifest["artifact_kind"],
    "code_hash": sys.argv[3],
    "legacy_repository_pin": sys.argv[4],
    "manifest_sha256": manifest["manifest_sha256"],
    "source_run_id": sys.argv[5],
    "source_item_row_id": sys.argv[6],
    "source_dump_file_count": sys.argv[7],
    "evidence_sha256": sha256((root / "evidence_sha256.json").read_bytes()).hexdigest(),
    "remote_objects_sha256": sha256((root / "remote_objects.json").read_bytes()).hexdigest(),
    "remote_prefix": remote,
}

def manifest_sha256(value):
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()

def is_sha256(value):
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )

def valid_dense_boundary_numeric(value):
    expected_keys = {
        "elementwise_exact",
        "expected_sha256",
        "first_mismatch_index",
        "max_abs_error",
        "mean_abs_error",
        "mismatch_count",
        "observed_sha256",
        "shape",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or type(value["elementwise_exact"]) is not bool
        or not is_sha256(value["expected_sha256"])
        or not is_sha256(value["observed_sha256"])
        or type(value["mismatch_count"]) is not int
        or not 0 <= value["mismatch_count"] <= 6144
        or type(value["max_abs_error"]) is not float
        or type(value["mean_abs_error"]) is not float
        or not math.isfinite(value["max_abs_error"])
        or not math.isfinite(value["mean_abs_error"])
        or value["shape"] != [6144]
    ):
        return False
    if value["elementwise_exact"]:
        return (
            value["mismatch_count"] == 0
            and value["first_mismatch_index"] is None
            and value["expected_sha256"] == value["observed_sha256"]
            and value["max_abs_error"] == 0.0
            and value["mean_abs_error"] == 0.0
        )
    return (
        1 <= value["mismatch_count"] <= 6144
        and type(value["first_mismatch_index"]) is int
        and 0 <= value["first_mismatch_index"] < 6144
        and value["expected_sha256"] != value["observed_sha256"]
        and 0.0 < value["mean_abs_error"] <= value["max_abs_error"]
    )

if sys.argv[8] == "1":
    exact_dsa = json.loads((root / "dsa_exact_comparison.json").read_text())
    mode = sys.argv[13]
    if mode == "layer1_rms_input":
        from glm_tpu.greenfield.validation.layer1_rms_input import (
            Layer1RmsInputCaptureConfig,
            validate_layer1_rms_input_artifacts,
        )
        if not exact_dsa["exact"]:
            raise SystemExit("layer-1 RMS-input DSA event tensors drifted")
        capture_root = root / "layer1_rms_input_capture"
        result = validate_layer1_rms_input_artifacts(
            Layer1RmsInputCaptureConfig(
                source_dump_dir=root / "source_dumps",
                output_dir=capture_root,
                db550_boundary_path=Path(
                    "/home/gianl/gcs-models/results/"
                    "greenfield_layer0_dense_partial_capture_"
                    "20260813T200736889447458Z/dense_partial_capture.npz"
                ),
                straddler_classification_path=Path(
                    "/home/gianl/glm-tpu-topology-rewrite/docs/artifacts/"
                    "pp16-feature2-layer1-straddler-classification.json"
                ),
                vllm_repository=Path(sys.argv[32]),
                expected_run_tag=sys.argv[22],
                expected_legacy_code_hash=sys.argv[4],
                expected_oracle_pin=sys.argv[28],
            )
        )
        capture = result["capture"]
        comparison = result["comparison"]
        capture_path = capture_root / "capture.json"
        comparison_path = capture_root / "comparison.json"
        archive_path = root / f"vllm_{sys.argv[29]}.tar.gz"
        identity = dict(
            line.split("=", 1)
            for line in (root / "vllm_archive_identity.txt").read_text().splitlines()
        )
        source_repository = identity.pop("source_repository", None)
        expected_identity = {
            "archive_sha256": sys.argv[30],
            "pin": sys.argv[29],
            "tracked_entries": sys.argv[31],
        }
        if (
            identity != expected_identity
            or source_repository != sys.argv[32]
            or sys.argv[32] != "/home/gianl/vllm-build-a30addc"
        ):
            raise SystemExit("accepted vLLM archive identity drifted")
        if (
            sys.argv[29] != capture["vllm_source_contract"]["repository_pin"]
            or not is_sha256(sys.argv[30])
            or sys.argv[31] != "5493"
            or sha256(archive_path.read_bytes()).hexdigest() != sys.argv[30]
        ):
            raise SystemExit("accepted vLLM archive bytes drifted")

        observer_archive_path = root / f"observer_{sys.argv[4]}.bundle"
        observer_identity = dict(
            line.split("=", 1)
            for line in (root / "observer_bundle_identity.txt")
            .read_text()
            .splitlines()
        )
        expected_observer_identity = {
            "branch": sys.argv[38],
            "bundle_sha256": sys.argv[33],
            "oracle_pin": sys.argv[28],
            "pin": sys.argv[4],
            "source_repository": sys.argv[37],
            "tracked_entries": sys.argv[34],
        }
        if (
            observer_identity != expected_observer_identity
            or sys.argv[37]
            != "/home/gianl/tpu-inference-greenfield-layer1-rms-input-observer"
            or sys.argv[38] != "greenfield/legacy-layer1-rms-input-observer"
            or not is_sha256(sys.argv[33])
            or sys.argv[34] != "947"
            or sha256(observer_archive_path.read_bytes()).hexdigest()
            != sys.argv[33]
            or sys.argv[35]
            != "916d421a10de9495086c0ad52645c9937746c88bae87a5ca1a69483bfe60d45d"
            or sys.argv[36] != "321146"
        ):
            raise SystemExit("accepted observer/golden bootstrap identity drifted")

        def exact_receipts(name, marker, expected_tail=None, expected_owners=None):
            records = [
                line.split()
                for line in (root / name).read_text().splitlines()
                if line.strip()
            ]
            if (
                len(records) != 8
                or any(len(record) < 2 or record[0] != marker for record in records)
                or len({record[1] for record in records}) != 8
                or (
                    expected_tail is not None
                    and any(record[2:] != expected_tail for record in records)
                )
                or (
                    expected_owners is not None
                    and {record[1] for record in records} != expected_owners
                )
            ):
                raise SystemExit(f"accepted vLLM fleet receipt drifted: {name}")

        exact_receipts(
            "vllm_archive_copy.txt",
            "VLLM_ARCHIVE_COPY_OK",
            expected_owners={str(index) for index in range(8)},
        )
        exact_receipts("vllm_vacancy.txt", "VLLM_VACANT_OK", expected_tail=[])
        exact_receipts(
            "sync_vllm.txt",
            "VLLM_SYNC_OK",
            expected_tail=[
                f"archive_sha256={sys.argv[30]}",
                f"tracked_entries={sys.argv[31]}",
            ],
        )
        exact_receipts(
            "vllm_cleanup_post.txt", "VLLM_CLEAN_OK", expected_tail=[]
        )
        exact_receipts(
            "observer_bundle_copy.txt",
            "OBSERVER_BUNDLE_COPY_OK",
            expected_owners={str(index) for index in range(8)},
        )
        exact_receipts(
            "observer_transport_vacancy.txt",
            "OBSERVER_TRANSPORT_VACANT_OK",
            expected_tail=[],
        )
        exact_receipts(
            "observer_transport_cleanup_post_sync.txt",
            "OBSERVER_TRANSPORT_CLEAN_OK",
            expected_tail=[],
        )
        ssh_status_names = {
            "golden_sync_ssh.txt",
            "observer_transport_cleanup_post_sync_ssh.txt",
            "observer_transport_vacancy_ssh.txt",
            "sync_observer_ssh.txt",
            "sync_vllm_ssh.txt",
            "vllm_cleanup_post_ssh.txt",
            "vllm_vacancy_ssh.txt",
        }
        if any(not (root / name).is_file() for name in ssh_status_names):
            raise SystemExit("accepted gcloud SSH status separation drifted")

        observer_sync_records = [
            line.split()
            for line in (root / "sync_observer.txt").read_text().splitlines()
            if line.strip()
        ]
        if (
            len(observer_sync_records) != 8
            or len({record[1] for record in observer_sync_records}) != 8
            or any(
                len(record) != 5
                or record[0] != "OBSERVER_SYNC_OK"
                or record[2] != f"bundle_sha256={sys.argv[33]}"
                or record[3] != f"tracked_entries={sys.argv[34]}"
                or record[4]
                not in {"disposition=existing", "disposition=installed"}
                for record in observer_sync_records
            )
        ):
            raise SystemExit("accepted observer fleet receipt drifted")

        golden_records = [
            line.split()
            for line in (root / "golden_sync.txt").read_text().splitlines()
            if line.strip()
        ]
        if (
            len(golden_records) != 8
            or len({record[1] for record in golden_records}) != 8
            or any(
                len(record) != 5
                or record[0] != "GOLDEN_SYNC_OK"
                or record[2] != f"sha256={sys.argv[35]}"
                or record[3] != f"bytes={sys.argv[36]}"
                or record[4] not in {"disposition=existing", "disposition=installed"}
                for record in golden_records
            )
        ):
            raise SystemExit("accepted golden-manifest fleet receipt drifted")
        lines.update({
            "accepted_layer1_rms_input_capture": "true",
            "accepted_layer1_rms_input_capture_layout": capture[
                "capture_layout"
            ],
            "accepted_layer1_rms_input_capture_mode": mode,
            "accepted_layer1_rms_input_diagnostic_only": "true",
            "accepted_layer1_rms_input_dsa_event_tensors_exact": "true",
            "accepted_layer1_rms_input_manifest_file_sha256": sha256(
                capture_path.read_bytes()
            ).hexdigest(),
            "accepted_layer1_rms_input_manifest_sha256": capture[
                "manifest_sha256"
            ],
            "accepted_layer1_rms_input_source_file_count": sys.argv[9],
            "accepted_layer1_rms_input_comparison_file_sha256": sha256(
                comparison_path.read_bytes()
            ).hexdigest(),
            "accepted_layer1_rms_input_comparison_manifest_sha256": comparison[
                "manifest_sha256"
            ],
            "accepted_layer1_rms_input_classification": comparison[
                "classification"
            ],
            "accepted_layer1_rms_input_operands_match_db550": str(
                comparison["operands_match_db550"]
            ).lower(),
            "accepted_layer1_fused_add_float32_sha256": capture["tensors"][
                "fused_add_float32"
            ]["tensor_sha256"],
            "accepted_layer1_hidden_update_bfloat16_sha256": capture["tensors"][
                "hidden_update_bfloat16_bits"
            ]["tensor_sha256"],
            "accepted_layer1_carried_residual_bfloat16_sha256": capture[
                "tensors"
            ]["carried_residual_bfloat16_bits"]["tensor_sha256"],
            "accepted_oracle_pin": capture["oracle_pin"],
            "accepted_observer_bundle_sha256": sys.argv[33],
            "accepted_observer_runtime_fleet_receipts_exact": "true",
            "accepted_observer_runtime_tracked_entries": sys.argv[34],
            "accepted_gcloud_ssh_status_separated": "true",
            "accepted_vllm_pin": capture["vllm_source_contract"][
                "repository_pin"
            ],
            "accepted_vllm_runtime_archive_sha256": sys.argv[30],
            "accepted_vllm_runtime_fleet_receipts_exact": "true",
            "accepted_vllm_runtime_tracked_entries": sys.argv[31],
            "db550_boundary_sha256": comparison["db550"]["file_sha256"],
            "dsa_internal_capture": "true",
            "dsa_internal_capture_process_indices": "0",
            "dsa_internal_file_count": sys.argv[9],
            "dsa_internal_layer_name": capture["layer_name"],
            "dsa_event_tensors_exact": "true",
            "golden_manifest_fleet_receipts_exact": "true",
            "golden_manifest_sha256": sys.argv[35],
            "pp16_straddler_classification_sha256": comparison["straddler"][
                "file_sha256"
            ],
        })
    elif mode == "dense_partial":
        from glm_tpu.greenfield.validation.dense_partials import (
            DensePartialsCaptureConfig,
            validate_dense_partials_artifacts,
        )
        if not exact_dsa["exact"]:
            raise SystemExit("dense-partial DSA event tensors drifted")
        capture_root = root / "dense_partials_capture"
        comparison = validate_dense_partials_artifacts(
            DensePartialsCaptureConfig(
                source_dump_dir=root / "source_dumps",
                db548_dir=Path(sys.argv[23]),
                output_dir=capture_root,
                expected_run_tag=sys.argv[22],
                expected_legacy_code_hash=sys.argv[4],
                expected_oracle_pin=sys.argv[28],
                expected_db548_runner_sha256=sys.argv[24],
                expected_db548_tensor_sha256=sys.argv[25],
                expected_db548_summary_sha256=sys.argv[26],
                expected_db548_success_sha256=sys.argv[27],
            )
        )
        capture_path = capture_root / "capture.json"
        comparison_path = capture_root / "comparison.json"
        capture = json.loads(capture_path.read_text())
        numeric = comparison["comparison"]
        lines.update({
            "accepted_dense_partials_capture": "true",
            "accepted_dense_partials_capture_layout": capture["capture_layout"],
            "accepted_dense_partials_capture_manifest_file_sha256": sha256(
                capture_path.read_bytes()
            ).hexdigest(),
            "accepted_dense_partials_capture_manifest_sha256": capture[
                "manifest_sha256"
            ],
            "accepted_dense_partials_comparison_file_sha256": sha256(
                comparison_path.read_bytes()
            ).hexdigest(),
            "accepted_dense_partials_comparison_manifest_sha256": comparison[
                "manifest_sha256"
            ],
            "accepted_dense_partials_diagnostic_only": "true",
            "accepted_dense_partials_dsa_event_tensors_exact": "true",
            "accepted_dense_partials_elementwise_exact_db548": str(
                numeric["elementwise_exact"]
            ).lower(),
            "accepted_dense_partials_mismatch_count": str(
                numeric["mismatch_count"]
            ),
            "accepted_dense_partials_sha256": numeric["expected_sha256"],
            "accepted_dense_partials_source_file_count": sys.argv[9],
            "accepted_oracle_pin": capture["oracle_pin"],
            "db548_dense_partials_sha256": numeric["observed_sha256"],
            "dense_partials_classification": comparison["classification"],
            "dsa_internal_capture": "true",
            "dsa_internal_capture_process_indices": "0,1,2,3,4,5,6,7",
            "dsa_internal_file_count": sys.argv[9],
            "dsa_internal_layer_name": capture["layer_name"],
            "dsa_event_tensors_exact": "true",
        })
    elif mode == "dense_input":
        capture_path = root / "dense_input_capture" / "capture.json"
        comparison_path = (
            root / "dense_input_comparison" / "comparison.json"
        )
        capture = json.loads(capture_path.read_text())
        comparison = json.loads(comparison_path.read_text())
        normalized = comparison["normalized_mlp"]
        capture_manifest_sha = manifest_sha256(capture)
        comparison_manifest_sha = manifest_sha256(comparison)
        observed_sha = (
            "082125fead43b25f10686705c1b6473153f4092dd5bc476f8e01a86629f0758f"
        )
        if (
            not exact_dsa["exact"]
            or capture.get("manifest_sha256") != capture_manifest_sha
            or capture["artifact_kind"]
            != "glm52_accepted_dense_input_capture"
            or capture["capture_layout"] != "replicated_logical_live_row"
            or capture["capture_mode"] != mode
            or capture["capture_process_indices"] != [0]
            or capture["diagnostic_only"] is not True
            or capture["performance_claim"] is not False
            or capture["legacy_code_hash"] != sys.argv[4]
            or capture["oracle_pin"]
            != "b3c25df47ac98783912dc658878181ec0a8ae16d"
            or capture["position"] != 8155
            or capture["tensor"]["shape"] != [6144]
            or comparison["artifact_kind"]
            != "glm52_accepted_greenfield_dense_input_comparison"
            or comparison.get("manifest_sha256") != comparison_manifest_sha
            or comparison.get("accepted_capture_manifest_sha256")
            != capture_manifest_sha
            or comparison["status"] != "SUCCESS"
            or comparison["diagnostic_only"] is not True
            or comparison["performance_claim"] is not False
            or comparison["classification"] not in {
                "normalized_mlp_exact_dense_arithmetic_open",
                "normalized_mlp_nonexact",
            }
            or comparison["first_open_boundary"] not in {
                "dense_mlp_or_cross_layer_fusion",
                "post_attention_add_rmsnorm",
            }
            or normalized["shape"] != [6144]
            or normalized.get("observed_sha256") != observed_sha
            or normalized.get("expected_sha256")
            != capture["tensor"].get("tensor_sha256")
            or comparison["probe"] != {
                "code_hash": "2f63779309b25c71c1cc7d35ff97715ae4bf631e",
                "run_id": 540,
                "runner_sha256": "876353e2504d728343223f03be9092a08d2662924ceb4b88d6f09563101bad91",
                "success_sha256": "d4c01377daae55ea23b329b1b6dc819b9595dc80f7d35521d0cbdd7d28caa799",
                "summary_sha256": "9b277ca495d3b9e9ce497d2bf78a520a2672f133e68228d14a10937d4a4b1449",
                "tag": "greenfield_layer0_dense_convolution_20260813T005213127235575Z",
                "tensor_sha256": "2cdf128976eb7e04d5c84e012f066d1766361af57896cd22f83c01c7d704e1b9",
            }
        ):
            raise SystemExit("dense-input comparison evidence drifted")
        expected_classification = (
            "normalized_mlp_exact_dense_arithmetic_open"
            if normalized["elementwise_exact"]
            else "normalized_mlp_nonexact"
        )
        expected_boundary = (
            "dense_mlp_or_cross_layer_fusion"
            if normalized["elementwise_exact"]
            else "post_attention_add_rmsnorm"
        )
        mismatch_count = normalized.get("mismatch_count")
        first_mismatch = normalized.get("first_mismatch_index")
        mean_error = normalized.get("mean_abs_error")
        max_error = normalized.get("max_abs_error")
        numeric_types = (int, float)
        finite_errors = (
            isinstance(mean_error, numeric_types)
            and not isinstance(mean_error, bool)
            and isinstance(max_error, numeric_types)
            and not isinstance(max_error, bool)
            and math.isfinite(mean_error)
            and math.isfinite(max_error)
        )
        if normalized["elementwise_exact"] is True:
            numerical_contract = (
                type(mismatch_count) is int
                and mismatch_count == 0
                and first_mismatch is None
                and finite_errors
                and mean_error == 0.0
                and max_error == 0.0
                and normalized["expected_sha256"] == observed_sha
            )
        elif normalized["elementwise_exact"] is False:
            numerical_contract = (
                type(mismatch_count) is int
                and 1 <= mismatch_count <= 6144
                and type(first_mismatch) is int
                and 0 <= first_mismatch < 6144
                and finite_errors
                and 0.0 < mean_error <= max_error
                and normalized["expected_sha256"] != observed_sha
            )
        else:
            numerical_contract = False
        if (
            comparison["classification"] != expected_classification
            or comparison["first_open_boundary"] != expected_boundary
            or not numerical_contract
        ):
            raise SystemExit("dense-input numerical/classification contradiction")
        lines.update({
            "accepted_dense_input_capture": "true",
            "accepted_dense_input_capture_layout": capture[
                "capture_layout"
            ],
            "accepted_dense_input_capture_mode": mode,
            "accepted_dense_input_diagnostic_only": "true",
            "accepted_dense_input_dsa_event_tensors_exact": "true",
            "accepted_dense_input_manifest_file_sha256": sha256(
                capture_path.read_bytes()
            ).hexdigest(),
            "accepted_dense_input_manifest_sha256": capture[
                "manifest_sha256"
            ],
            "accepted_dense_input_source_file_count": sys.argv[9],
            "accepted_normalized_mlp_sha256": capture["tensor"][
                "tensor_sha256"
            ],
            "accepted_oracle_pin": capture["oracle_pin"],
            "dense_input_classification": comparison["classification"],
            "dense_input_comparison_file_sha256": sha256(
                comparison_path.read_bytes()
            ).hexdigest(),
            "dense_input_comparison_manifest_sha256": comparison[
                "manifest_sha256"
            ],
            "dense_input_first_open_boundary": comparison[
                "first_open_boundary"
            ],
            "dense_input_normalized_mlp_exact": str(
                normalized["elementwise_exact"]
            ).lower(),
            "dense_input_probe_runner_sha256": comparison["probe"][
                "runner_sha256"
            ],
            "dense_input_probe_success_sha256": comparison["probe"][
                "success_sha256"
            ],
            "dense_input_probe_summary_sha256": comparison["probe"][
                "summary_sha256"
            ],
            "dense_input_probe_tensor_sha256": comparison["probe"][
                "tensor_sha256"
            ],
            "dsa_internal_capture": "true",
            "dsa_internal_capture_process_indices": "0",
            "dsa_internal_file_count": sys.argv[9],
            "dsa_internal_layer_name": capture["layer_name"],
            "dsa_event_tensors_exact": "true",
        })
    elif mode == "dense_boundary":
        import numpy as np

        capture_path = root / "dense_boundary_capture" / "capture.json"
        comparison_path = (
            root / "dense_boundary_comparison" / "comparison.json"
        )
        capture = json.loads(capture_path.read_text())
        comparison = json.loads(comparison_path.read_text())
        residual = comparison["post_attention_residual"]
        expected_capture_keys = {
            "artifact_kind",
            "capture_layout",
            "capture_mode",
            "capture_process_indices",
            "diagnostic_only",
            "format_version",
            "layer_name",
            "legacy_code_hash",
            "manifest_sha256",
            "model_id",
            "oracle_pin",
            "performance_claim",
            "position",
            "process_count",
            "process_files",
            "run_tag",
            "tensor_file",
            "tensors",
        }
        expected_comparison_keys = {
            "accepted_capture_manifest_sha256",
            "artifact_kind",
            "classification",
            "diagnostic_only",
            "first_open_boundary",
            "format_version",
            "legacy_code_hash",
            "manifest_sha256",
            "oracle_pin",
            "performance_claim",
            "position",
            "post_attention_residual",
            "probe",
            "status",
        }
        if (
            not exact_dsa["exact"]
            or type(capture) is not dict
            or set(capture) != expected_capture_keys
            or capture["manifest_sha256"] != manifest_sha256(capture)
            or capture["artifact_kind"]
            != "glm52_accepted_dense_boundary_capture"
            or capture["capture_layout"]
            != "replicated_logical_live_rows"
            or capture["capture_mode"] != mode
            or capture["capture_process_indices"] != [0]
            or capture["diagnostic_only"] is not True
            or capture["performance_claim"] is not False
            or capture["format_version"] != 2
            or capture["legacy_code_hash"] != sys.argv[4]
            or capture["oracle_pin"]
            != "b3c25df47ac98783912dc658878181ec0a8ae16d"
            or capture["position"] != 8155
            or capture["process_count"] != 8
            or capture["run_tag"] != sys.argv[22]
            or capture["model_id"] != "zai-org/GLM-5.2-FP8"
            or capture["layer_name"] != "model.layers.0.self_attn.attn"
            or set(capture["tensors"]) != {"post_attention_residual"}
            or type(comparison) is not dict
            or set(comparison) != expected_comparison_keys
            or comparison["manifest_sha256"] != manifest_sha256(comparison)
            or comparison["accepted_capture_manifest_sha256"]
            != capture["manifest_sha256"]
            or comparison["artifact_kind"]
            != "glm52_accepted_greenfield_dense_boundary_comparison"
            or comparison["format_version"] != 2
            or comparison["legacy_code_hash"] != sys.argv[4]
            or comparison["oracle_pin"] != capture["oracle_pin"]
            or comparison["position"] != 8155
            or comparison["status"] != "SUCCESS"
            or comparison["diagnostic_only"] is not True
            or comparison["performance_claim"] is not False
            or comparison["classification"] not in {
                "post_attention_residual_nonexact",
                "post_attention_residual_exact_layer1_boundary_open",
            }
            or comparison["first_open_boundary"] not in {
                "layer0_post_attention_residual",
                "layer1_fused_add_rmsnorm",
            }
            or not valid_dense_boundary_numeric(residual)
            or comparison["probe"] != {
                "code_hash": "2f63779309b25c71c1cc7d35ff97715ae4bf631e",
                "run_id": 540,
                "runner_sha256": "876353e2504d728343223f03be9092a08d2662924ceb4b88d6f09563101bad91",
                "success_sha256": "d4c01377daae55ea23b329b1b6dc819b9595dc80f7d35521d0cbdd7d28caa799",
                "summary_sha256": "9b277ca495d3b9e9ce497d2bf78a520a2672f133e68228d14a10937d4a4b1449",
                "tag": "greenfield_layer0_dense_convolution_20260813T005213127235575Z",
                "tensor_sha256": "2cdf128976eb7e04d5c84e012f066d1766361af57896cd22f83c01c7d704e1b9",
            }
        ):
            raise SystemExit("dense-boundary comparison evidence drifted")
        tensor_record = capture["tensors"]["post_attention_residual"]
        tensor_file = capture["tensor_file"]
        process_files = capture["process_files"]
        source_relative = (
            process_files[0].get("path")
            if type(process_files) is list and len(process_files) == 1
            and type(process_files[0]) is dict
            else None
        )
        source_parts = (
            Path(source_relative).parts
            if type(source_relative) is str
            else ()
        )
        expected_source_name = (
            "internals.model_layers_0_self_attn_attn.position8155.proc0.npz"
        )
        if (
            type(tensor_record) is not dict
            or set(tensor_record) != {"shape", "sha256"}
            or tensor_record["shape"] != [6144]
            or not is_sha256(tensor_record["sha256"])
            or type(tensor_file) is not dict
            or set(tensor_file) != {"byte_count", "filename", "sha256"}
            or tensor_file["filename"] != "dense_boundary.npz"
            or type(tensor_file["byte_count"]) is not int
            or tensor_file["byte_count"] <= 0
            or not is_sha256(tensor_file["sha256"])
            or type(process_files) is not list
            or len(process_files) != 1
            or type(process_files[0]) is not dict
            or set(process_files[0])
            != {"byte_count", "path", "process_index", "sha256"}
            or process_files[0]["process_index"] != 0
            or type(process_files[0]["byte_count"]) is not int
            or process_files[0]["byte_count"] <= 0
            or not is_sha256(process_files[0]["sha256"])
            or len(source_parts) != 2
            or source_parts[0] not in {f"w{rank}" for rank in range(8)}
            or source_parts[1] != expected_source_name
        ):
            raise SystemExit("dense-boundary capture ledger drifted")
        tensor_path = capture_path.parent / tensor_file["filename"]
        source_path = root / "source_dumps" / process_files[0]["path"]
        if (
            not tensor_path.is_file()
            or tensor_path.stat().st_size != tensor_file["byte_count"]
            or sha256(tensor_path.read_bytes()).hexdigest()
            != tensor_file["sha256"]
            or not source_path.is_file()
            or source_path.stat().st_size != process_files[0]["byte_count"]
            or sha256(source_path.read_bytes()).hexdigest()
            != process_files[0]["sha256"]
        ):
            raise SystemExit("dense-boundary capture file drifted")
        with np.load(tensor_path, allow_pickle=False) as payload:
            if set(payload.files) != {"post_attention_residual_bfloat16_bits"}:
                raise SystemExit("dense-boundary tensor keys drifted")
            residual_bits = np.ascontiguousarray(
                payload["post_attention_residual_bfloat16_bits"]
            )
        with np.load(source_path, allow_pickle=False) as payload:
            expected_source_keys = {
                "artifact_kind",
                "format_version",
                "capture_mode",
                "process_index",
                "process_count",
                "layer_name",
                "position",
                "source_row",
                "run_tag",
                "code_hash",
                "oracle_pin",
                "model_id",
                "post_attention_residual",
                "post_attention_residual__dtype",
            }
            if set(payload.files) != expected_source_keys:
                raise SystemExit("dense-boundary raw source schema drifted")
            source_scalars = {
                name: payload[name].item()
                for name in expected_source_keys
                if name not in {"post_attention_residual"}
            }
            source_residual_bits = np.ascontiguousarray(
                payload["post_attention_residual"]
            )
        if source_scalars != {
            "artifact_kind": "glm52_legacy_dense_boundary",
            "format_version": 2,
            "capture_mode": "dense_boundary",
            "process_index": 0,
            "process_count": 8,
            "layer_name": "model.layers.0.self_attn.attn",
            "position": 8155,
            "source_row": 0,
            "run_tag": sys.argv[22],
            "code_hash": sys.argv[4],
            "oracle_pin": capture["oracle_pin"],
            "model_id": "zai-org/GLM-5.2-FP8",
            "post_attention_residual__dtype": "bfloat16",
        }:
            raise SystemExit("dense-boundary raw source identity drifted")
        residual_sha = sha256(residual_bits.tobytes(order="C")).hexdigest()
        if (
            residual_bits.shape != (6144,)
            or residual_bits.dtype != np.dtype(np.uint16)
            or not np.array_equal(source_residual_bits, residual_bits)
            or residual_sha != tensor_record["sha256"]
            or residual["expected_sha256"] != residual_sha
            or residual["observed_sha256"]
            != "a105fdbd429adb1d06a70bf71598a72a91d7b6faa83360005487ce11ce099f8e"
        ):
            raise SystemExit("dense-boundary residual identity drifted")
        residual_exact = residual["elementwise_exact"]
        expected_classification = (
            "post_attention_residual_exact_layer1_boundary_open"
            if residual_exact
            else "post_attention_residual_nonexact"
        )
        expected_boundary = (
            "layer1_fused_add_rmsnorm"
            if residual_exact
            else "layer0_post_attention_residual"
        )
        if (
            comparison["classification"] != expected_classification
            or comparison["first_open_boundary"] != expected_boundary
        ):
            raise SystemExit("dense-boundary classification contradiction")
        lines.update({
            "accepted_dense_boundary_capture": "true",
            "accepted_dense_boundary_capture_layout": capture[
                "capture_layout"
            ],
            "accepted_dense_boundary_capture_mode": mode,
            "accepted_dense_boundary_diagnostic_only": "true",
            "accepted_dense_boundary_dsa_event_tensors_exact": "true",
            "accepted_dense_boundary_manifest_file_sha256": sha256(
                capture_path.read_bytes()
            ).hexdigest(),
            "accepted_dense_boundary_manifest_sha256": capture[
                "manifest_sha256"
            ],
            "accepted_dense_boundary_source_file_count": sys.argv[9],
            "accepted_post_attention_residual_sha256": capture["tensors"][
                "post_attention_residual"
            ]["sha256"],
            "accepted_oracle_pin": capture["oracle_pin"],
            "dense_boundary_classification": comparison["classification"],
            "dense_boundary_comparison_file_sha256": sha256(
                comparison_path.read_bytes()
            ).hexdigest(),
            "dense_boundary_comparison_manifest_sha256": comparison[
                "manifest_sha256"
            ],
            "dense_boundary_first_open_boundary": comparison[
                "first_open_boundary"
            ],
            "dense_boundary_residual_exact": str(residual_exact).lower(),
            "dense_boundary_probe_runner_sha256": comparison["probe"][
                "runner_sha256"
            ],
            "dense_boundary_probe_success_sha256": comparison["probe"][
                "success_sha256"
            ],
            "dense_boundary_probe_summary_sha256": comparison["probe"][
                "summary_sha256"
            ],
            "dense_boundary_probe_tensor_sha256": comparison["probe"][
                "tensor_sha256"
            ],
            "dsa_internal_capture": "true",
            "dsa_internal_capture_process_indices": "0",
            "dsa_internal_file_count": sys.argv[9],
            "dsa_internal_layer_name": capture["layer_name"],
            "dsa_event_tensors_exact": "true",
        })
    elif mode == "attention_update":
        capture_path = root / "attention_update_capture" / "capture.json"
        comparison_path = (
            root / "attention_update_comparison" / "comparison.json"
        )
        capture = json.loads(capture_path.read_text())
        comparison = json.loads(comparison_path.read_text())
        candidates = comparison["candidate_comparisons"]
        if (
            not exact_dsa["exact"]
            or capture["artifact_kind"]
            != "glm52_accepted_attention_update_capture"
            or capture["capture_layout"] != "replicated_logical_live_row"
            or capture["capture_mode"] != mode
            or capture["capture_process_indices"] != [0]
            or capture["diagnostic_only"] is not True
            or capture["performance_claim"] is not False
            or capture["legacy_code_hash"] != sys.argv[4]
            or capture["oracle_pin"]
            != "b3c25df47ac98783912dc658878181ec0a8ae16d"
            or capture["position"] != 8155
            or capture["tensor"]["shape"] != [6144]
            or comparison["artifact_kind"]
            != "glm52_accepted_greenfield_attention_update_comparison"
            or comparison["status"] != "SUCCESS"
            or comparison["diagnostic_only"] is not True
            or comparison["performance_claim"] is not False
            or comparison["classification"] not in {
                "local_attention_projection_exact",
                "strategy_nd_attention_projection_exact",
                "both_attention_projections_exact",
                "attention_projection_arithmetic_unresolved",
            }
            or set(candidates) != {"local", "strategy_nd"}
            or any(value["shape"] != [6144] for value in candidates.values())
        ):
            raise SystemExit("attention-update comparison evidence drifted")
        lines.update({
            "accepted_attention_update_capture": "true",
            "accepted_attention_update_capture_layout": capture[
                "capture_layout"
            ],
            "accepted_attention_update_capture_mode": mode,
            "accepted_attention_update_diagnostic_only": "true",
            "accepted_attention_update_dsa_event_tensors_exact": "true",
            "accepted_attention_update_manifest_file_sha256": sha256(
                capture_path.read_bytes()
            ).hexdigest(),
            "accepted_attention_update_manifest_sha256": capture[
                "manifest_sha256"
            ],
            "accepted_attention_update_source_file_count": sys.argv[9],
            "accepted_attention_update_tensor_sha256": capture["tensor"][
                "tensor_sha256"
            ],
            "accepted_oracle_pin": capture["oracle_pin"],
            "attention_update_classification": comparison["classification"],
            "attention_update_comparison_file_sha256": sha256(
                comparison_path.read_bytes()
            ).hexdigest(),
            "attention_update_comparison_manifest_sha256": comparison[
                "manifest_sha256"
            ],
            "attention_update_exact_candidates": ",".join(
                comparison["exact_candidates"]
            ) or "none",
            "attention_update_first_open_boundary": comparison[
                "first_open_boundary"
            ],
            "dsa_internal_capture": "true",
            "dsa_internal_capture_process_indices": "0",
            "dsa_internal_file_count": sys.argv[9],
            "dsa_internal_layer_name": capture["layer_name"],
            "dsa_event_tensors_exact": "true",
        })
    elif mode == "attention_projection":
        capture_path = root / "attention_projection_capture" / "capture.json"
        comparison_path = (
            root / "attention_projection_comparison" / "comparison.json"
        )
        capture = json.loads(capture_path.read_text())
        comparison = json.loads(comparison_path.read_text())
        output = comparison["attention_output"]
        latent = comparison["attended_latent"]
        if (
            not exact_dsa["exact"]
            or capture["artifact_kind"]
            != "glm52_accepted_attention_projection_capture"
            or capture["capture_layout"] != "logical_head_order_live_row"
            or capture["capture_mode"] != mode
            or capture["capture_process_indices"] != [0]
            or capture["diagnostic_only"] is not True
            or capture["performance_claim"] is not False
            or capture["legacy_code_hash"] != sys.argv[4]
            or capture["oracle_pin"]
            != "b3c25df47ac98783912dc658878181ec0a8ae16d"
            or capture["position"] != 8155
            or capture["tensors"]["attended_latent_bfloat16_bits"]["shape"]
            != [64, 512]
            or capture["tensors"]["attention_output_bfloat16_bits"]["shape"]
            != [16384]
            or capture["tensors"]["attention_output_bfloat16_bits"]["sha256"]
            != "79a6e290274ef470b518a6de894b44d2929ad6861d1751f0915aa4eb20cf2e9d"
            or comparison["status"] != "SUCCESS"
            or comparison["artifact_kind"]
            != "glm52_accepted_greenfield_attention_projection_comparison"
            or comparison["classification"] not in {
                "attention_arithmetic_before_w_uv",
                "w_uv_projection_arithmetic",
            }
            or output["elementwise_exact"] is not False
            or output["mismatch_count"] != 5117
            or output["first_mismatch_index"] != 3
            or output["max_abs_error"] != 6.103515625e-05
            or output["expected_sha256"]
            != "79a6e290274ef470b518a6de894b44d2929ad6861d1751f0915aa4eb20cf2e9d"
            or output["observed_sha256"]
            != "0103e22c558d1390820bc9d39cc05b90f4555fa7c0be5de7cd33c34748a582ab"
            or latent["shape"] != [64, 512]
        ):
            raise SystemExit("attention-projection comparison evidence drifted")
        lines.update({
            "accepted_attention_projection_capture": "true",
            "accepted_attention_projection_capture_layout": capture[
                "capture_layout"
            ],
            "accepted_attention_projection_capture_mode": mode,
            "accepted_attention_projection_diagnostic_only": "true",
            "accepted_attention_projection_dsa_event_tensors_exact": "true",
            "accepted_attention_projection_manifest_file_sha256": sha256(
                capture_path.read_bytes()
            ).hexdigest(),
            "accepted_attention_projection_manifest_sha256": capture[
                "manifest_sha256"
            ],
            "accepted_attention_projection_source_file_count": sys.argv[9],
            "accepted_attention_projection_latent_sha256": capture["tensors"][
                "attended_latent_bfloat16_bits"
            ]["sha256"],
            "accepted_attention_projection_output_sha256": capture["tensors"][
                "attention_output_bfloat16_bits"
            ]["sha256"],
            "attention_projection_classification": comparison["classification"],
            "attention_projection_comparison_file_sha256": sha256(
                comparison_path.read_bytes()
            ).hexdigest(),
            "attention_projection_comparison_manifest_sha256": comparison[
                "manifest_sha256"
            ],
            "attention_projection_latent_mismatch_count": str(
                latent["mismatch_count"]
            ),
            "attention_projection_output_mismatch_count": str(
                output["mismatch_count"]
            ),
            "accepted_oracle_pin": capture["oracle_pin"],
            "dsa_internal_capture": "true",
            "dsa_internal_capture_process_indices": "0",
            "dsa_internal_file_count": sys.argv[9],
            "dsa_internal_layer_name": capture["layer_name"],
            "dsa_event_tensors_exact": "true",
        })
    elif mode == "attention_output":
        capture = json.loads(
            (root / "attention_output_capture" / "capture.json").read_text()
        )
        if (
            not exact_dsa["exact"]
            or capture["artifact_kind"]
            != "glm52_accepted_attention_output_operand_capture"
            or capture["capture_layout"] != "logical_head_order_live_row"
            or capture["capture_mode"] != mode
            or capture["capture_process_indices"] != [0]
            or capture["diagnostic_only"] is not True
            or capture["performance_claim"] is not False
            or capture["legacy_code_hash"] != sys.argv[4]
            or capture["oracle_pin"]
            != "b3c25df47ac98783912dc658878181ec0a8ae16d"
            or capture["position"] != 8155
        ):
            raise SystemExit("attention-output capture evidence drifted")
        capture_path = root / "attention_output_capture" / "capture.json"
        lines.update({
            "accepted_attention_output_capture": "true",
            "accepted_attention_output_capture_layout": capture[
                "capture_layout"
            ],
            "accepted_attention_output_capture_mode": mode,
            "accepted_attention_output_diagnostic_only": "true",
            "accepted_attention_output_dsa_event_tensors_exact": "true",
            "accepted_attention_output_manifest_file_sha256": sha256(
                capture_path.read_bytes()
            ).hexdigest(),
            "accepted_attention_output_manifest_sha256": capture[
                "manifest_sha256"
            ],
            "accepted_attention_output_source_file_count": sys.argv[9],
            "accepted_attention_output_tensor_sha256": capture["tensor"][
                "tensor_sha256"
            ],
            "accepted_oracle_pin": capture["oracle_pin"],
            "dsa_internal_capture": "true",
            "dsa_internal_capture_process_indices": "0",
            "dsa_internal_file_count": sys.argv[9],
            "dsa_internal_layer_name": capture["layer_name"],
            "dsa_event_tensors_exact": "true",
        })
    elif mode in ("prompt_key", "prompt_key_input"):
        result_name = (
            "prompt_projection_input_comparison"
            if mode == "prompt_key_input"
            else "prompt_key_comparison"
        )
        comparison = json.loads(
            (root / result_name / "comparison.json").read_text()
        )
        if comparison["status"] != "SUCCESS" or (
            not comparison["hlo"]["states"]["contract"]["passed"]
        ) or not comparison["hlo"]["cache"]["contract"]["passed"]:
            raise SystemExit("prompt-key internal comparison drifted")
        if mode == "prompt_key_input" and not comparison["hlo"][
            "projection_input"
        ]["contract"]["passed"]:
            raise SystemExit("projection-input HLO contract drifted")
        lines.update({
            "dsa_internal_capture": "true",
            "dsa_internal_capture_layout": (
                "prompt_key_input_producer_replicas"
                if mode == "prompt_key_input"
                else "prompt_key_producer_replicas"
            ),
            "dsa_internal_capture_mode": mode,
            "dsa_internal_capture_process_indices": ",".join(
                str(value)
                for value in comparison["accepted_capture"][
                    "capture_process_indices"
                ]
            ),
            "dsa_internal_file_count": sys.argv[9],
            "dsa_internal_layer_name": comparison["layer_name"],
            "dsa_internal_first_divergent_field": (
                comparison["conclusion"]["first_divergent_field"] or "none"
            ),
            "dsa_internal_classification": comparison["conclusion"][
                "classification"
            ],
            "dsa_internal_manifest_sha256": comparison["manifest_sha256"],
            "dsa_event_tensors_exact": str(exact_dsa["exact"]).lower(),
            "accepted_oracle_pin": comparison["oracle_pin"],
        })
    else:
        compare_layer0 = sys.argv[10] == "1"
        result_name = (
            "internal_comparison" if compare_layer0 else "internal_capture"
        )
        record_name = "comparison.json" if compare_layer0 else "capture.json"
        comparison = json.loads((root / result_name / record_name).read_text())
        if comparison["capture_layout"] != "topology_sharded_live_row_owner":
            raise SystemExit("DSA internal capture layout drifted")
        if comparison["capture_process_indices"] != [0]:
            raise SystemExit("DSA internal owner process drifted")
        lines.update({
            "dsa_internal_capture": "true",
            "dsa_internal_capture_layout": comparison["capture_layout"],
            "dsa_internal_capture_mode": mode,
            "dsa_internal_file_count": sys.argv[9],
            "dsa_internal_owner_actual_sha256": comparison[
                "owner_actual_sha256"
            ],
            "dsa_internal_layer_name": comparison["layer_name"],
            "dsa_internal_first_divergent_field": (
                (comparison["first_divergent_field"] or "none")
                if compare_layer0 else "not_compared"
            ),
            "dsa_event_tensors_exact": str(exact_dsa["exact"]).lower(),
            "accepted_oracle_pin": comparison["oracle_pin"],
        })
if sys.argv[11] == "1":
    prompt_cache = json.loads(
        (root / "prompt_index_cache" / "manifest.json").read_text()
    )
    lines.update({
        "prompt_index_cache_capture": "true",
        "prompt_index_cache_manifest_sha256": prompt_cache[
            "manifest_sha256"
        ],
        "prompt_index_cache_bfloat16_sha256": prompt_cache[
            "prompt_index_key_bfloat16_sha256"
        ],
        "prompt_index_cache_source_file_count": sys.argv[12],
    })
    if sys.argv[13] in ("prompt_key", "prompt_key_input"):
        result_name = (
            "prompt_projection_input_comparison"
            if sys.argv[13] == "prompt_key_input"
            else "prompt_key_comparison"
        )
        prompt_comparison = json.loads(
            (root / result_name / "comparison.json").read_text()
        )
        if prompt_comparison["accepted_cache"]["manifest_sha256"] != (
            prompt_cache["manifest_sha256"]
        ):
            raise SystemExit("prompt-key comparison cache identity drifted")
        lines.update({
            "prompt_index_cache_internal_comparison_manifest_sha256": (
                prompt_comparison["manifest_sha256"]
            ),
            "prompt_index_cache_internal_candidate_sha256": (
                prompt_comparison["greenfield_cache"][
                    "prompt_index_key_bfloat16_sha256"
                ]
            ),
            "prompt_index_cache_internal_mismatch_count": (
                prompt_comparison["cache_comparison"]["mismatch_count"]
            ),
        })
    elif sys.argv[39] != "0":
        # Deeper-layer capture: bind the sealed layer identity. The one-host
        # layer-0 production comparison does not exist for it; deeper layers
        # are compared offline against the executed DB518 greenfield capture.
        layer_id = int(sys.argv[39])
        if (
            prompt_cache.get("layer_id") != layer_id
            or prompt_cache.get("cache_slot") != 2 * layer_id
            or prompt_cache.get("layer_name")
            != f"model.layers.{layer_id}.self_attn.attn"
            or prompt_cache["artifact_kind"]
            != f"glm52_legacy_layer{layer_id}_prompt_index_cache"
        ):
            raise SystemExit("prompt index-cache layer binding drifted")
        lines.update({
            "prompt_index_cache_layer_id": str(layer_id),
            "prompt_index_cache_cache_slot": str(2 * layer_id),
            "prompt_index_cache_layer_name": prompt_cache["layer_name"],
            "prompt_index_cache_production_comparison": (
                "not_applicable_deeper_layer_offline_only"
            ),
        })
    else:
        prompt_comparison = json.loads(
            (
                root
                / "prompt_index_cache_comparison"
                / "comparison.json"
            ).read_text()
        )
        if (
            prompt_comparison["prompt_cache_manifest_sha256"]
            != prompt_cache["manifest_sha256"]
            or not prompt_comparison["hlo"]["contract"]["passed"]
            or prompt_comparison["status"] != "SUCCESS"
        ):
            raise SystemExit("prompt index-cache comparison identity drifted")
        lines.update({
            "prompt_index_cache_production_comparison_manifest_sha256": (
                prompt_comparison["manifest_sha256"]
            ),
            "prompt_index_cache_production_elementwise_exact": str(
                prompt_comparison["comparison"]["elementwise_exact"]
            ).lower(),
            "prompt_index_cache_production_first_mismatch_position": (
                "none"
                if prompt_comparison["comparison"][
                    "first_mismatch_position"
                ] is None
                else prompt_comparison["comparison"][
                    "first_mismatch_position"
                ]
            ),
            "prompt_index_cache_production_mismatch_count": (
                prompt_comparison["comparison"]["mismatch_count"]
            ),
            "prompt_index_cache_production_hlo_sha256": (
                prompt_comparison["hlo"]["optimized_hlo_sha256"]
            ),
        })
if sys.argv[14] == "1":
    exact_dsa = json.loads((root / "dsa_exact_comparison.json").read_text())
    lowering_root = root / "accepted_prompt_projection_lowering"
    lowering = json.loads((lowering_root / "summary.json").read_text())
    lowering_manifest = json.loads((lowering_root / "manifest.json").read_text())
    if (
        not exact_dsa["exact"]
        or lowering["status"] != "SUCCESS"
        or lowering["profile"]["file_count"] != 8
        or lowering["profile"]["core_count"] != 64
        or lowering["profile"]["steps_per_core"] != 1
        or lowering["profile"]["invocations_per_core"] != 21
        or lowering["hlo"]["convolution_count"] != 21
    ):
        raise SystemExit("accepted prompt projection lowering evidence drifted")
    lines.update({
        "accepted_prompt_projection_capture": "true",
        "accepted_prompt_projection_diagnostic_only": "true",
        "accepted_prompt_projection_dsa_event_tensors_exact": "true",
        "accepted_prompt_projection_emitter": lowering["hlo"]["emitter"],
        "accepted_prompt_projection_hlo_source_file_count": sys.argv[17],
        "accepted_prompt_projection_manifest_sha256": lowering_manifest[
            "manifest_sha256"
        ],
        "accepted_prompt_projection_source": lowering["hlo"]["source"],
        "accepted_prompt_projection_xplane_file_count": sys.argv[15],
        "accepted_prompt_projection_trace_json_file_count": sys.argv[16],
    })
if sys.argv[18] == "1":
    exact_dsa = json.loads((root / "dsa_exact_comparison.json").read_text())
    comparison = json.loads(
        (root / "layer0_main_cache_comparison" / "comparison.json").read_text()
    )
    if (
        not exact_dsa["exact"]
        or comparison["artifact_kind"]
        != "glm52_legacy_pp8_layer0_main_cache_comparison"
        or comparison["diagnostic_only"] is not True
        or comparison["performance_claim"] is not False
        or comparison["legacy"]["observer_pin"] != sys.argv[4]
        or len(comparison["source_dump_files"]) != int(sys.argv[19])
    ):
        raise SystemExit("layer-0 main-cache comparison evidence drifted")
    lines.update({
        "layer0_main_cache_capture": "true",
        "layer0_main_cache_classification": comparison["classification"],
        "layer0_main_cache_diagnostic_only": "true",
        "layer0_main_cache_dsa_event_tensors_exact": "true",
        "layer0_main_cache_first_divergent_primitive": (
            comparison["first_divergent_primitive"] or "none"
        ),
        "layer0_main_cache_manifest_sha256": comparison["manifest_sha256"],
        "layer0_main_cache_source_file_count": sys.argv[19],
    })
if sys.argv[20] == "1":
    exact_dsa = json.loads((root / "dsa_exact_comparison.json").read_text())
    lowering_root = root / "accepted_decode_projection_lowering"
    lowering = json.loads((lowering_root / "summary.json").read_text())
    lowering_manifest = json.loads((lowering_root / "manifest.json").read_text())
    if (
        not exact_dsa["exact"]
        or lowering["artifact_kind"]
        != "accepted_decode_projection_lowering_v1"
        or lowering["status"] != "SUCCESS"
        or lowering["diagnostic_only"] is not True
        or lowering["performance_claim"] is not False
        or lowering["protected_request_sequences"] != 1
        or lowering["compile_owner_count"] != int(sys.argv[21])
        or lowering["hlo"]["compile_bucket_rows"] != 32
        or lowering["hlo"]["partition_count"] != 32
        or lowering["hlo"]["collective_count"] != 156
        or lowering["hlo"]["category_counts"]
        != {"attention": 78, "dense_mlp": 3, "moe_tuple": 75}
        or lowering["hlo"]["replica_group"] != list(range(32))
        or lowering["hlo"]["reduction_dtype"] != "bf16"
    ):
        raise SystemExit("accepted decode projection lowering evidence drifted")
    algorithm = lowering["hlo"]["collective_algorithm_config"]
    lines.update({
        "accepted_decode_projection_capture": "true",
        "accepted_decode_projection_diagnostic_only": "true",
        "accepted_decode_projection_dsa_event_tensors_exact": "true",
        "accepted_decode_projection_emitter": algorithm["emitter"],
        "accepted_decode_projection_hlo_source_file_count": sys.argv[21],
        "accepted_decode_projection_manifest_sha256": lowering_manifest[
            "manifest_sha256"
        ],
        "accepted_decode_projection_reduction_dtype": "bf16",
        "accepted_decode_projection_strategy": algorithm["strategy"],
    })
(root / "SUCCESS").write_text(
    "".join(f"{key}={value}\n" for key, value in lines.items())
)
PY
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" \
  "$REMOTE_PREFIX/SUCCESS" >/dev/null
local_success_sha=$(sha256sum "$RUN_DIR/SUCCESS" | awk '{print $1}')
remote_success_sha=$(gcloud storage cat "$REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}')
[[ $local_success_sha == "$remote_success_sha" ]] || {
  say "ABORT: remote SUCCESS checksum mismatch"
  exit 1
}
terminal_success_done=1

manifest_sha=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
  "$ORACLE_DIR/manifest.json")
say "SUCCESS run=$run_id item=$item_row_id files=$dump_count manifest=$manifest_sha"
