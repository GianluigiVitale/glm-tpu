#!/usr/bin/env python
"""ground_truth_sum.py — compute the TRUE uint32 wraparound byte-sum of a
model-state leaf directly from the GLM-5.2-FP8 checkpoint on GCS, to
adjudicate disputed manifest leaves (streamer-corruption disputes).

Usage:
    ~/vllm-env/bin/python ground_truth_sum.py <leaf-name> [<leaf-name> ...]
    ~/vllm-env/bin/python ground_truth_sum.py --file leaves.txt

Leaf names are model-state names, e.g.
    vllm_model.model.layers.10.self_attn.indexer.wk_weights_proj.weight

GCS access is READ-ONLY via ranged `gcloud storage cat -r` reads: only the
safetensors headers plus the exact byte spans of the source tensors are
fetched (a few MB per leaf).

NAME-MAPPING RULES (discovered from vllm-build deepseek_v2.py load_weights +
tpu-inference glm_dsa_indexer.py; GlmMoeDsaForCausalLM loads via the
DeepseekV2 code path, leaves are the raw vLLM module params prefixed
'vllm_model.'):

  1. Strip the 'vllm_model.' prefix -> candidate checkpoint name.
  2. If the name exists in model.safetensors.index.json -> single source
     tensor, bytes taken verbatim (no transform).
  3. Fused (stacked) params: reverse-map via stacked_params_mapping:
        <p>.gate_up_proj.<s>      <- [gate_proj, up_proj]           (row concat)
        <p>.fused_qkv_a_proj.<s>  <- [q_a_proj, kv_a_proj_with_mqa] (row concat)
        <p>.qkv_proj.<s>          <- [q_proj, k_proj, v_proj]       (row concat)
        <p>.wk_weights_proj.weight<- [indexer wk, indexer weights_proj]
     where <s> is 'weight' or 'weight_scale_inv'. Shards are concatenated in
     shard-id order along dim 0, matching MergedColumnParallelLinear's
     narrow+copy weight_loader (byte-preserving when dtypes match).
  4. SPECIAL CASE (the primary dispute): indexer wk is stored F8_E4M3 with a
     separate F32 weight_scale_inv while weights_proj is BF16.
     deepseek_v2._try_load_fp8_indexer_wk dequantizes wk to BF16 at load:
         block = W.shape[1] // S.shape[1]            # 6144//48 = 128
         W_bf16 = (W_fp8.to(f32) * broadcast128x128(S.to(f32))).to(bf16)
     (scaled_dequantize, GroupShape(128,128), torch RNE f32->bf16 cast; runs
     on host CPU at load time, so the CPU replication here is bit-exact).
     The fused leaf bytes are concat(dequant_bf16(wk), raw_bf16(weights_proj)).

  NOT SUPPORTED (refused with an explanation rather than guessed): MoE
  expert-stacked params (w13/w2 experts fusion), derived/absorbed params
  (kv_b_proj W_UK_T/W_UV, glm_dsa_adapted_*), and any leaf whose on-device
  dtype differs from the reconstruction implied by the rules above.
"""

import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np
import torch

GCS_ROOT = "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# Reverse of deepseek_v2.load_weights stacked_params_mapping:
# fused-param-name -> list of checkpoint shard names in shard-id order.
FUSED_RULES = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "wk_weights_proj": ["wk", "weights_proj"],
}

UNSUPPORTED_MARKERS = [
    ("glm_dsa_adapted_", "derived param (precompute_indexer_params fp32 "
     "adapter output) — recompute requires full fp8 dequant + orientation "
     "logic; adjudicate the RAW module leaves instead"),
    (".experts.", "MoE expert-stacked param — expert fusion/permutation not "
     "replicated offline"),
    ("W_UK_T", "MLA absorption-derived param"),
    ("W_UV", "MLA absorption-derived param"),
]

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n"
                           f"{r.stderr.decode(errors='replace')[-2000:]}")
    return r.stdout


def ranged_read(gcs_path, start, length):
    """Inclusive-range read: gcloud storage cat -r start-(start+length-1)."""
    out = _run(["gcloud", "storage", "cat",
                "-r", f"{start}-{start + length - 1}", gcs_path])
    if len(out) != length:
        raise RuntimeError(f"ranged read of {gcs_path} [{start},+{length}) "
                           f"returned {len(out)} bytes")
    return out


def load_index():
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = os.path.join(CACHE_DIR, "model.safetensors.index.json")
    if not os.path.exists(p):
        with open(p, "wb") as f:
            f.write(_run(["gcloud", "storage", "cat",
                          f"{GCS_ROOT}/model.safetensors.index.json"]))
    with open(p) as f:
        return json.load(f)["weight_map"]


_HEADERS = {}


def shard_header(shard):
    """Return (header_dict, data_section_start) for a shard, cached."""
    if shard in _HEADERS:
        return _HEADERS[shard]
    p = os.path.join(CACHE_DIR, shard + ".header.json")
    gcs = f"{GCS_ROOT}/{shard}"
    if os.path.exists(p):
        with open(p) as f:
            saved = json.load(f)
        _HEADERS[shard] = (saved["header"], saved["data_start"])
        return _HEADERS[shard]
    hlen = struct.unpack("<Q", ranged_read(gcs, 0, 8))[0]
    hjson = json.loads(ranged_read(gcs, 8, hlen))
    data_start = 8 + hlen
    with open(p, "w") as f:
        json.dump({"header": hjson, "data_start": data_start}, f)
    _HEADERS[shard] = (hjson, data_start)
    return _HEADERS[shard]


def fetch_tensor_bytes(weight_map, ckpt_name):
    """Ranged-read one checkpoint tensor. Returns (bytes, dtype, shape)."""
    shard = weight_map[ckpt_name]
    hdr, data_start = shard_header(shard)
    meta = hdr[ckpt_name]
    a, b = meta["data_offsets"]
    nbytes = b - a
    exp = int(np.prod(meta["shape"], dtype=np.int64)) * DTYPE_BYTES[meta["dtype"]]
    assert nbytes == exp, (ckpt_name, nbytes, exp)
    raw = ranged_read(f"{GCS_ROOT}/{shard}", data_start + a, nbytes)
    return raw, meta["dtype"], meta["shape"]


def dequant_fp8_to_bf16_bytes(w_raw, w_shape, s_raw, s_shape):
    """Bit-exact replication of vllm scaled_dequantize(..., GroupShape(b,b),
    out_dtype=bf16) as used by deepseek_v2._try_load_fp8_indexer_wk."""
    wq = torch.frombuffer(bytearray(w_raw),
                          dtype=torch.float8_e4m3fn).reshape(w_shape)
    s = torch.frombuffer(bytearray(s_raw),
                         dtype=torch.float32).reshape(s_shape)
    block = w_shape[1] // s_shape[1]
    assert w_shape[0] // s_shape[0] == block, "non-square group shape"
    # group_broadcast: repeat each scale over its block along both dims
    # (size-1 dims broadcast implicitly, matching quant_utils.group_broadcast)
    sf = s.to(torch.float32)
    if sf.shape[0] != w_shape[0] and sf.shape[0] != 1:
        sf = sf.repeat_interleave(block, dim=0)
    if sf.shape[1] != w_shape[1] and sf.shape[1] != 1:
        sf = sf.repeat_interleave(block, dim=1)
    w_bf16 = (wq.to(torch.float32) * sf).to(torch.bfloat16)
    return w_bf16.contiguous().view(torch.uint8).numpy().tobytes()


def u32_sum(buf):
    return int(np.sum(np.frombuffer(buf, dtype=np.uint8),
                      dtype=np.uint32))


def resolve_sources(weight_map, leaf):
    """Map a model-state leaf to [(ckpt_name, transform)] in concat order.
    transform in {'raw', 'fp8_dequant_bf16'} ('fp8_dequant_bf16' entries
    consume the sibling weight_scale_inv)."""
    name = leaf
    if name.startswith("vllm_model."):
        name = name[len("vllm_model."):]

    for marker, why in UNSUPPORTED_MARKERS:
        if marker in name:
            raise ValueError(f"UNSUPPORTED leaf ({why}): {leaf}")

    # Rule 2: direct hit
    if name in weight_map:
        return name, [(name, "raw")]

    # Rule 3: fused params
    for fused, shards in FUSED_RULES.items():
        token = f".{fused}."
        if token not in name:
            continue
        sources = []
        for shard_name in shards:
            src = name.replace(token, f".{shard_name}.")
            if src not in weight_map:
                raise ValueError(
                    f"fused-source '{src}' not in checkpoint index "
                    f"(leaf {leaf}); mapping rules need extension")
            # Rule 4: fp8-stored shard destined for an unquantized fused
            # param -> dequantized to bf16 at load (indexer wk case).
            scale = src.rsplit(".", 1)[0] + ".weight_scale_inv"
            hdr, _ = shard_header(weight_map[src])
            if (fused == "wk_weights_proj" and src.endswith(".weight")
                    and hdr[src]["dtype"] == "F8_E4M3"):
                if scale not in weight_map:
                    raise ValueError(f"{src} is F8_E4M3 but no scale found")
                sources.append((src, "fp8_dequant_bf16"))
            else:
                sources.append((src, "raw"))
        return name, sources

    raise ValueError(f"cannot map leaf to checkpoint tensor(s): {leaf}")


def adjudicate(weight_map, leaf, expect=None):
    ckpt_name, sources = resolve_sources(weight_map, leaf)
    print(f"\nLEAF  {leaf}")
    print(f"  ckpt param name : {ckpt_name}")
    pieces = []
    for src, transform in sources:
        raw, dtype, shape = fetch_tensor_bytes(weight_map, src)
        if transform == "fp8_dequant_bf16":
            scale_name = src.rsplit(".", 1)[0] + ".weight_scale_inv"
            s_raw, s_dtype, s_shape = fetch_tensor_bytes(weight_map,
                                                         scale_name)
            assert s_dtype == "F32", s_dtype
            out = dequant_fp8_to_bf16_bytes(raw, shape, s_raw, s_shape)
            note = (f"{dtype}{shape} x scale F32{s_shape} "
                    f"-> scaled_dequantize(block "
                    f"{shape[1] // s_shape[1]}) -> BF16 bytes")
        else:
            out = raw
            note = f"{dtype}{shape} raw bytes"
        pieces.append(out)
        print(f"  source {src}")
        print(f"    transform  : {note}")
        print(f"    bytes      : {len(out)}   u32-sum {u32_sum(out)}")
    total = b"".join(pieces)
    s = u32_sum(total)
    print(f"  TOTAL bytes {len(total)}  TRUE u32 wraparound sum = {s}")
    if expect:
        for label, cand in expect.items():
            mark = "MATCH" if cand == s else "no"
            print(f"    candidate {label:>8} = {cand:>12}  -> {mark}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("leaves", nargs="*", help="model-state leaf names")
    ap.add_argument("--file", help="file with one leaf name per line")
    ap.add_argument("--expect", action="append", default=[],
                    metavar="LABEL=SUM",
                    help="candidate sums to compare (repeatable)")
    args = ap.parse_args()
    leaves = list(args.leaves)
    if args.file:
        with open(args.file) as f:
            leaves += [l.strip() for l in f if l.strip()]
    if not leaves:
        ap.error("no leaves given")
    expect = {}
    for e in args.expect:
        k, v = e.split("=", 1)
        expect[k] = int(v)

    weight_map = load_index()
    failures = 0
    for leaf in leaves:
        try:
            adjudicate(weight_map, leaf, expect or None)
        except (ValueError, RuntimeError) as exc:
            failures += 1
            print(f"\nLEAF  {leaf}\n  FAILED: {exc}", file=sys.stderr)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
