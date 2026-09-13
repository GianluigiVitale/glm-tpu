"""Pinned HF-card campaign, with explicit uncertainty and no legacy execution.

The private request payload contains dataset questions/golds and prompt IDs.
Only its digest and compact protocol belong in Git. No model call, paid judge,
checkpoint download or score is performed by registration.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "ws32_native_hf_campaign_v1"
MAX_NEW = 163840
# Benchmark-only cache envelope: longest pinned prompt2796 + full163840
# generation cap, rounded UP to a512-token page. No question/cap is shortened.
CAPACITY = 166912
EOS = (154820, 154827, 154829)
VOCAB = 154880
PAYLOAD_CAP = 16 << 20
TARGETS = {"gpqa_diamond": 0.912, "aime_2026": 0.992}
COUNTS = {"gpqa_diamond": 198, "aime_2026": 30}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def benchmark_registry(repo: Path = REPO) -> Any:
    """Load only pinned pure builders/extractors; NEVER bench/run_bench.py."""
    pins = json.loads((repo / "docs/artifacts/native-benchmark-dataset-access-20260912.json").read_text())
    for name, digest in pins["builder_files"].items():
        if sha256((repo / name).read_bytes()).hexdigest() != digest:
            raise ValueError("registered benchmark builder/template changed")
    # The original registry imports `extract` absolutely. Refuse a different
    # already-imported module rather than accidentally scoring with another one.
    for module, relative in (("extract", "bench/extract.py"),
                             ("glm_native_card_registry", "bench/benchmarks.py")):
        path = repo / relative
        if module in sys.modules:
            if Path(sys.modules[module].__file__).resolve() != path.resolve():
                raise ValueError("benchmark module resolved outside this repository")
        else:
            spec = importlib.util.spec_from_file_location(module, path)
            obj = importlib.util.module_from_spec(spec)
            sys.modules[module] = obj
            spec.loader.exec_module(obj)
    return sys.modules["glm_native_card_registry"]


def make_payload(rows: Mapping[str, Sequence[Mapping]], tokenizer: Any, *,
                 repo: Path = REPO) -> dict:
    registry = benchmark_registry(repo)
    pins = json.loads((repo / "docs/artifacts/native-benchmark-dataset-access-20260912.json").read_text())
    template = (repo / "reference/hf-repo/chat_template.jinja").read_text()
    requests = []
    for dataset in pins["datasets"]:
        name = dataset["name"]
        spec = registry.REGISTRY[name]
        items = [asdict((spec.card.build or spec.build)(row, i)) for i, row in enumerate(rows[name])]
        if (len(items) != COUNTS[name]
                or sha256(canonical(items)).hexdigest() != dataset["ordered_card_items_sha256"]):
            raise ValueError("complete ordered card dataset differs from registered source")
        for i, item in enumerate(items):
            messages = ([] if item["system_prompt"] is None else
                        [dict(role="system", content=item["system_prompt"])])
            messages.append(dict(role="user", content=item["prompt"]))
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                tokenize=True, return_dict=False, chat_template=template)
            if (not isinstance(ids, list) or not ids
                    or any(type(x) is not int or not 0 <= x < VOCAB for x in ids)
                    or len(ids) + MAX_NEW > CAPACITY):
                raise ValueError("prompt cannot use the complete card generation cap")
            requests.append(dict(dataset=name, item=item, request_id=f"{name}/{item['item_id']}/sample0",
                seed=20260912 + i, prompt_ids=ids,
                prompt_ids_sha256=sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()))
    payload = dict(schema=SCHEMA, requests=requests)
    if len(canonical(payload)) > PAYLOAD_CAP:
        raise ValueError("private benchmark payload exceeds registered size")
    return payload


def protocol(payload: Mapping, tokenizer_files: Mapping[str, str]) -> dict:
    """Fixed before candidate outputs; one sample/item, not invented card avg@k."""
    result = dict(schema=SCHEMA, payload_sha256=sha256(canonical(payload)).hexdigest(),
        card=dict(url="https://huggingface.co/zai-org/GLM-5.2-FP8", column="GLM-5.2",
            revision="f33c6dc501ee5a2c7e35155653b1b1abbc320951",
            readme_sha256="522afffe4e9bb3b3054a68df39b1796417a745aaed689511e7c0a11b449c4e9d"),
        counts=COUNTS, targets=TARGETS, samples_per_item=1, temperature=1.0,
        top_p=0.95, max_new_tokens=MAX_NEW, context_capacity=CAPACITY,
        vocab_size=VOCAB, eos_ids=list(EOS), tokenizer_files=dict(tokenizer_files),
        campaign_order=list(COUNTS), selection="all items, pinned source order; no answer-based selection",
        aggregation="arithmetic mean of all item correctness values, one draw/item",
        uncertainty="95% Wilson interval over items; not repeated-draw/model-card uncertainty",
        material_deficit_absolute=0.05,
        deficit_rule="investigate if complete-set Wilson upper < card target minus 0.05; otherwise report point gap and uncertainty, not equivalence",
        incomplete_rule="unfinished/unjudged set is INCONCLUSIVE; retain all partials, misses and truncations; never score completed-only subset as full set",
        tranche_wall_seconds=86400, runtime_rule="24h operational tranche; preserve partial request and stop at deadline; full campaign stays incomplete, no reduced token cap",
        scorer=dict(gpqa_diamond="pinned bench/extract.py MC extraction/exact letter match",
                    aime_2026="UNSCORED pending explicit GPT-5.5 medium judge approval/protocol; no substitute"),
        caveats=["GPQA prompt, choice order and extraction are declared harness choices; card does not specify them.",
                 "One sample/item and seeds are harness choices; card does not publish repetition/aggregation.",
                 "No matched-card parity or equivalence claim from this protocol.",
                 "GPQA extraction only consumes text after </think>; unfinished reasoning is not a final answer and scores wrong on terminal truncation.",
                 "Thinking ON / Reasoning Effort Max follows the pinned model chat template."])
    validate(payload, result)
    return result


def validate(payload: Mapping, plan: Mapping) -> None:
    if (payload.get("schema") != SCHEMA or plan.get("schema") != SCHEMA
            or sha256(canonical(payload)).hexdigest() != plan["payload_sha256"]
            or len(canonical(payload)) > PAYLOAD_CAP
            or plan["counts"] != COUNTS or plan["targets"] != TARGETS
            or plan["samples_per_item"] != 1 or plan["temperature"] != 1.0
            or plan["top_p"] != 0.95 or plan["max_new_tokens"] != MAX_NEW
            or plan["context_capacity"] != CAPACITY or plan["vocab_size"] != VOCAB
            or plan["eos_ids"] != list(EOS) or plan["material_deficit_absolute"] != 0.05
            or plan["tranche_wall_seconds"] != 86400):
        raise ValueError("native campaign registered policy/payload differs")
    requests = payload["requests"]
    if len(requests) != sum(COUNTS.values()) or len({r["request_id"] for r in requests}) != len(requests):
        raise ValueError("native campaign is not the complete unique request set")
    expected = [(d, i) for d, count in COUNTS.items() for i in range(count)]
    for row, (dataset, i) in zip(requests, expected, strict=True):
        prefix = "gpqa" if dataset == "gpqa_diamond" else "aime"
        item_id = f"{prefix}_{i}"
        ids = row["prompt_ids"]
        if (row["dataset"] != dataset or row["item"]["item_id"] != item_id
                or row["request_id"] != f"{dataset}/{item_id}/sample0"
                or type(row["seed"]) is not int or row["seed"] != 20260912 + i
                or not isinstance(ids, list) or not ids or len(ids) + MAX_NEW > CAPACITY
                or any(type(x) is not int or not 0 <= x < VOCAB for x in ids)
                or sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest() != row["prompt_ids_sha256"]):
            raise ValueError("native campaign order/seed/prompt token identity differs")


def wilson(correct: int, total: int) -> tuple[float, float]:
    if type(correct) is not int or type(total) is not int or not 0 <= correct <= total or total <= 0:
        raise ValueError("invalid full-set score counts")
    z, p = 1.959963984540054, correct / total
    denominator = 1 + z*z/total
    middle = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return max(0.0, middle-half), min(1.0, middle+half)


def main() -> None:
    """Materialize small pinned requests once; credentials never enter output."""
    import argparse
    import csv
    import io
    import os
    import requests
    import pyarrow.parquet as parquet
    from dotenv import dotenv_values
    from transformers import AutoTokenizer
    from scripts.greenfield.ws32_history_preflight import _plain_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    _plain_path(root)
    if root.exists():
        raise ValueError("preserve existing registration; use its pinned payload")
    credentials = dotenv_values("/home/gianl/glm-tpu/.env")
    token = os.environ.get("HF_TOKEN") or credentials.get("HF_TOKEN")
    pins = json.loads((REPO / "docs/artifacts/native-benchmark-dataset-access-20260912.json").read_text())
    rows = {}
    for dataset in pins["datasets"]:
        url = f"https://huggingface.co/datasets/{dataset['repo']}/resolve/{dataset['revision']}/{dataset['file']}"
        # Bounded data access only. Do not include authentication headers or a
        # possibly signed redirect URL in any error/report.
        with requests.get(url, headers={"Authorization": "Bearer " + token} if token else {},
                          timeout=60, stream=True) as response:
            if response.status_code != 200:
                raise RuntimeError(f"pinned dataset fetch failed: {dataset['name']} HTTP {response.status_code}")
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > dataset["bytes"]:
                    raise ValueError("pinned dataset exceeds registered bytes")
                chunks.append(chunk)
        raw = b"".join(chunks)
        if len(raw) != dataset["bytes"] or sha256(raw).hexdigest() != dataset["raw_sha256"]:
            raise ValueError("pinned dataset payload changed")
        rows[dataset["name"]] = (list(csv.DictReader(io.StringIO(raw.decode())))
            if dataset["file"].endswith(".csv") else parquet.read_table(io.BytesIO(raw)).to_pylist())
    tokenizer_root = Path("/home/gianl/gcs-models/models/GLM-5.2-FP8")
    tokenizer_files = {name: sha256((tokenizer_root / name).read_bytes()).hexdigest()
                       for name in ("tokenizer.json", "tokenizer_config.json")}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True, trust_remote_code=False)
    payload = make_payload(rows, tokenizer)
    plan = protocol(payload, tokenizer_files)
    root.mkdir(exist_ok=False)
    for name, data in (("requests.json", canonical(payload)), ("protocol.json", canonical(plan)+b"\n")):
        with (root / name).open("xb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
    print(json.dumps(dict(root=str(root), requests=len(payload["requests"]),
        prompt_tokens=sum(len(r["prompt_ids"]) for r in payload["requests"]),
        payload_bytes=len(canonical(payload)), payload_sha256=plan["payload_sha256"],
        protocol_file_sha256=sha256(canonical(plan)+b"\n").hexdigest(),
        model_outputs=0), sort_keys=True))


if __name__ == "__main__":
    main()
