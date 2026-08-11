#!/usr/bin/env bash
# Select and compact one run-owned accepted M32 after-codegen HLO.
set -euo pipefail

[[ $# -eq 2 ]] || {
  echo "usage: $0 RAW_ROOT COMPACT_ROOT" >&2
  exit 2
}
readonly RAW_ROOT=$1
readonly COMPACT_ROOT=$2
readonly RUN_ROOT=${RAW_ROOT%/decode_projection_hlo_raw}
readonly PATTERN='bf16\[32,6144\].*all-reduce\(.*VllmRowParallelLinear/shard_map/psum'

[[ $RAW_ROOT == "$RUN_ROOT/decode_projection_hlo_raw" && \
   $COMPACT_ROOT == "$RUN_ROOT/decode_projection_hlo" && \
   $(dirname "$RUN_ROOT") == /tmp && \
   $(basename "$RUN_ROOT") == greenfield_* ]] || {
  echo "DECODE_HLO_BAD $(hostname) unsafe_run_root=$RUN_ROOT"
  exit 1
}

mapfile -t candidates < <(
  find "$RAW_ROOT" -type f -name '*after_codegen.txt' \
    -exec grep -lE "$PATTERN" {} + 2>/dev/null || true
)
if [[ ${#candidates[@]} -eq 0 ]]; then
  if [[ ! -e $RAW_ROOT ]]; then
    echo "DECODE_HLO_OK $(hostname)"
    echo "DECODE_HLO_NONOWNER $(hostname)"
    exit 0
  fi
  RAW_FILE_COUNT=$(find "$RAW_ROOT" -type f | wc -l)
  readonly RAW_FILE_COUNT
  if [[ $RAW_FILE_COUNT -eq 0 ]]; then
    find "$RAW_ROOT" -depth -delete
    echo "DECODE_HLO_OK $(hostname)"
    echo "DECODE_HLO_NONOWNER $(hostname)"
    exit 0
  fi
  echo "DECODE_HLO_BAD $(hostname) candidates=0 raw_files=$RAW_FILE_COUNT"
  exit 1
fi
[[ ${#candidates[@]} -eq 1 ]] || {
  echo "DECODE_HLO_BAD $(hostname) candidates=${#candidates[@]}"
  exit 1
}

readonly CANDIDATE=${candidates[0]}
MATCHES=$(grep -cE "$PATTERN" "$CANDIDATE")
readonly MATCHES
HEADER=$(head -n 1 "$CANDIDATE")
readonly HEADER
[[ $MATCHES -eq 156 && \
   $HEADER == "HloModule jit_step_fun_impl, is_scheduled=true"* && \
   $HEADER == *"num_partitions=32"* ]] || {
  echo "DECODE_HLO_BAD $(hostname) candidates=1 matches=$MATCHES"
  exit 1
}

mkdir -p "$COMPACT_ROOT"
readonly OUTPUT=$COMPACT_ROOT/accepted_decode.after_codegen.txt.gz
readonly INVENTORY=$COMPACT_ROOT/raw_hlo_inventory.txt
[[ ! -e $OUTPUT && ! -e $INVENTORY ]] || {
  echo "DECODE_HLO_BAD $(hostname) append_only_output_exists"
  exit 1
}
# Preserve the exact reclaimed filename/size inventory without rereading the
# multi-gigabyte dump solely to hash non-selected compiler scratch files.
find "$RAW_ROOT" -type f -printf '%P\t%s\n' | LC_ALL=C sort >"$INVENTORY"
gzip -n -c "$CANDIDATE" >"$OUTPUT"
gzip -t "$OUTPUT"
SHA256=$(sha256sum "$OUTPUT" | cut -d ' ' -f 1)
readonly SHA256
RAW_FILE_COUNT=$(wc -l <"$INVENTORY")
readonly RAW_FILE_COUNT

# The selected gzip is durable before deleting only the validated run-owned
# raw subtree. The helper itself remains beside the compact evidence.
find "$RAW_ROOT" -depth -delete
echo "matches=$MATCHES raw_files=$RAW_FILE_COUNT sha256=$SHA256"
echo "DECODE_HLO_OK $(hostname)"
echo "DECODE_HLO_OWNER $(hostname)"
