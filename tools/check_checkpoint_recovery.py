"""Read-only retained checkpoint metadata audit; never reads weight payloads.

Authenticates the sealed WS32 recipe and current canonical source generations.
Metadata consistency is not a new pack/load test or an overlay payload rehash.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

BUCKET = "driftbench-dsv4-uc"
LINEAGE = "results/greenfield_ws32_runtime_pack_20260815T214050854386790Z"
OVERLAY = "checkpoints/greenfield/glm52/overlays/WS32_2D/greenfield_ws32_strategy_nd_dense_overlay_pack_20260827T002508229552699Z"
MODEL = "models/GLM-5.2-FP8"
PINS = {
    "manifest": (LINEAGE + "/checkpoint_manifest/manifest.json", "88df414301e0d303506163c078484ec14cdb15ff03972789df58960b091a7cbf", 8 << 20),
    "success": (LINEAGE + "/checkpoint_SUCCESS.json", "12703932637a9330f97ae0282b26dd7105d68793b3ae6620e109a10c7b0c7ca2", 4096),
    "preflight": (LINEAGE + "/remote_preflight.json", None, 128 << 10),
    "overlay": (OVERLAY + "/manifest.json", "c17194b6dbf2a41c5ae869c6184d6c88f04eb5adfe2105b94965e8434c2b5c8c", 1 << 20),
    "overlay_success": (OVERLAY + "/SUCCESS", "166566b9b066aa4803dacf769c51a890c5e79690770a9582e9dbdccb085332a6", 4096),
}


def authenticated_metadata(bucket: Any) -> tuple[dict, list[dict]]:
    if bucket.name != BUCKET or bucket.location.upper() != "US-CENTRAL2":
        raise ValueError("recovery audit requires the approved regional bucket")
    values, receipts = {}, []
    for key, (name, expected, cap) in PINS.items():
        blob = bucket.get_blob(name)
        if blob is None or not 0 < int(blob.size) <= cap or not blob.generation:
            raise ValueError("sealed recovery metadata missing or oversized: " + key)
        raw = blob.download_as_bytes(if_generation_match=int(blob.generation), timeout=60)
        if len(raw) != int(blob.size):
            raise ValueError("sealed recovery metadata byte count drifted: " + key)
        digest = sha256(raw).hexdigest()
        if key == "preflight":
            expected = values["success"]["remote_preflight_sha256"]
        if digest != expected:
            raise ValueError("sealed recovery metadata hash drifted: " + key)
        values[key] = json.loads(raw)
        receipts.append(dict(object=name, bytes=len(raw), generation=int(blob.generation), sha256=digest))
    return values, receipts


def reconcile_sources(manifest: dict, preflight: dict, observed: list[Any]) -> list[dict]:
    expected = {row["filename"]: row for row in manifest["source"]["files"]}
    sealed = {row["filename"]: row for row in preflight["source_files"]}
    if (manifest["source"]["uri"] != "gs://" + BUCKET + "/" + MODEL
            or len(expected) != 141 or len(sealed) != 141
            or len(manifest["source"]["files"]) != 141 or len(preflight["source_files"]) != 141
            or set(expected) != set(sealed)):
        raise ValueError("sealed canonical source membership differs")
    remote = {blob.name: blob for blob in observed}
    rows = []
    for name, source in sorted(expected.items()):
        if Path(name).name != name:
            raise ValueError("canonical source name escaped prefix")
        prior, blob = sealed[name], remote.get(MODEL + "/" + name)
        if (prior["file_bytes"] != source["file_bytes"] or prior["sha256"] != source["sha256"]
                or blob is None or int(blob.size) != source["file_bytes"]
                or blob.crc32c != prior["crc32c"] or int(blob.generation) != prior["generation"]):
            raise ValueError("canonical source no longer matches sealed generation/size/CRC: " + name)
        rows.append(dict(object=blob.name, bytes=int(blob.size), generation=int(blob.generation),
                         crc32c=blob.crc32c, sealed_sha256=source["sha256"]))
    return rows


def reconcile_overlay(manifest: dict, observed: list[Any]) -> list[dict]:
    remote = {blob.name: blob for blob in observed}
    files = manifest["files"]
    if len(files) != 96 or len({row["filename"] for row in files}) != 96:
        raise ValueError("overlay membership differs")
    rows = []
    for row in files:
        name = row["filename"]
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("overlay name escaped prefix")
        blob = remote.get(OVERLAY + "/" + name)
        if blob is None or int(blob.size) != row["byte_count"] or not blob.generation:
            raise ValueError("retained overlay file missing or resized: " + name)
        rows.append(dict(object=blob.name, bytes=int(blob.size), generation=int(blob.generation),
                         crc32c=blob.crc32c, sealed_sha256=row["sha256"], payload_rehashed=False))
    return rows


def main() -> None:
    from google.cloud import storage
    bucket = storage.Client().bucket(BUCKET)
    bucket.reload(timeout=60)
    values, metadata = authenticated_metadata(bucket)
    sources = reconcile_sources(values["manifest"], values["preflight"],
                                list(bucket.list_blobs(prefix=MODEL + "/")))
    overlays = reconcile_overlay(values["overlay"], list(bucket.list_blobs(prefix=OVERLAY + "/")))
    print(json.dumps(dict(schema="glm_release_checkpoint_recovery_metadata_v1",
        checked_utc=datetime.now(timezone.utc).isoformat(), bucket=BUCKET, location=bucket.location,
        sealed_metadata=metadata, canonical_sources=sources, dense_overlay=overlays,
        canonical_weight_bytes=sum(row["bytes"] for row in sources),
        overlay_weight_bytes=sum(row["bytes"] for row in overlays),
        reconstructed_ram_file_bytes=values["manifest"]["packed_file_bytes"],
        canonical_source_generation_size_crc_match=True, overlay_membership_sizes_match=True,
        limits=["No weight payload read, new pack, load, device access or writes",
                "Overlay generations observed now; payload SHA values inherited, not freshly verified",
                "Not full retained-bucket inventory or a complete recovery/deployment proof"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
