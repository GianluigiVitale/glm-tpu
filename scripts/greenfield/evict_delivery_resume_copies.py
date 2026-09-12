"""Exact archived local compiler/trace copies; retain cloud/weights.

The existing leased, generation-checked eviction engine performs every unlink.
Review is read-only except for its compact manifest; apply needs that SHA.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from scripts.greenfield import evict_reviewed_local_copies as engine
from scripts.greenfield.microbench_fp8_matmul import _atomic_json

TAG = "greenfield_fp8_ws32_capture_barrier_prefill_compile_20260912T025701169016006Z"
GRAPHS = {"prefill_256k_capture_barrier.stablehlo.mlir": 21130653,
          "prefill_256k_capture_barrier.optimized_hlo.txt": 103234144}
SNAPSHOTS = {
    "2k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_cd1_20260909T214620492938535Z": 188583936,
    "2k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_20260909T080827165939618Z": 172482560,
    "2k_numerical_c17_hrope_bp1_ps1_20260908T233558169096679Z": 151019520,
    "2k_numerical_c17_hrope_bp1_20260908T090441883274696Z": 49897472,
    "128k_d0_05_numerical_cap131072_hrope_20260907T132148212467000Z": 34230272,
    "128k_d0_0_numerical_cap131072_hrope_20260907T064941550123130Z": 33640448,
    "128k_d1_0_numerical_cap131072_hrope_20260907T012156230341652Z": 33046528,
    "8k_numerical_cap262656_hrope_20260906T211233754818996Z": 32456704,
    "8k_numerical_cap131072_hrope_20260906T191737720789480Z": 31723520,
    "8k_numerical_hrope_20260906T162720142039604Z": 31072256,
    "8k_numerical_c512_20260905T182725949766820Z": 30416896,
    "8k_numerical_20260905T163437339905065Z": 29769728,
    "8k_numerical_20260905T085534575653049Z": 29122560,
}
KIND = "ws32_delivery_resume_local_copy_review_v1"
TRACE_KIND = "ws32_delivery_archived_trace_copy_review_v1"
DELIVERY_TAG = "greenfield_ws32_short_decoder_128k_d1_0_numerical_c128_cap131072_hrope_bp1_ps1_rp1_ep1_lm1_cd1_s26long_20260912T093538603389142Z"
OLD_TRACES = {
    "8k_numerical_cap131072_hrope_20260906T182243839824117Z": ("2026_09_06_19_09_56", 299899619),
    "8k_numerical_20260826T213125786075567Z": ("2026_08_26_22_07_57", 294273314),
    "8k_numerical_hrope_20260906T151844646077531Z": ("2026_09_06_16_04_10", 298881762),
    "8k_numerical_20260827T011711674195301Z": ("2026_08_27_01_54_13", 298119150),
    "8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_live32_20260909T103047459508942Z": ("2026_09_09_11_11_08", 298307837),
    "8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_20260909T093335570726655Z": ("2026_09_09_10_08_07", 298146521),
    "8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_cd1_20260909T225859315457683Z": ("2026_09_09_23_33_41", 298519697),
}
PEER_TRACE_SIZES = (298876966, 298877403, 300258992, 300258564,
                    300257995, 298877763, 300258755)
DB617_KIND = "ws32_delivery_db617_pp8_trace_copy_review_v1"
DB617_TAG = "greenfield_ws32_short_decoder_128k_d0_0_numerical_c128_cap131072_hrope_bp1_ps1_rp1_ep1_lm1_cd1_s26long_20260912T111318376524388Z"
DB617_PEER_SIZES = (298868187, 298868649, 300412591, 300412743,
                   300412603, 298868112, 300412357)
PP8_RECOVERY = "greenfield_short_decoder_pp8_2k_gate_d_recovery_20260827T203000000000000Z"
PP8_ORIGINAL = "greenfield_short_decoder_compile_pp8_2k_pallas_feature_linear_ot256_downf32_token_splitres_queryexact_oracle_dsa_metaparent_trace2_20260827T194450473110366Z"
PP8_SIZES = (80039393, 78475829, 77775666, 80011173, 85311310,
             77586754, 80030683, 78459008)
PROFILE_KINDS = {"compiler": KIND, "archived-traces": TRACE_KIND,
                 "db617-pp8-traces": DB617_KIND}


def scope(profile: str = "compiler") -> dict[str, tuple[str, int]]:
    if profile == "db617-pp8-traces":
        pairs = {
            f"{DB617_TAG}/traces/trace.rank{rank}.xplane.pb":
            (f"results/{DB617_TAG}/traces/trace.rank{rank}.xplane.pb", size)
            for rank, size in enumerate(DB617_PEER_SIZES, 1)
        }
        pairs[f"{DB617_TAG}/results.db"] = (
            f"results/{DB617_TAG}/orchestrator/results.db", 197140480)
        for tag in (PP8_RECOVERY, PP8_ORIGINAL):
            for rank, size in enumerate(PP8_SIZES):
                # Both local copies must match the sealed recovery generation.
                # The original failed-run prefix has no cloud trace objects.
                pairs[f"{tag}/traces/trace.rank{rank}.xplane.pb"] = (
                    f"results/{PP8_RECOVERY}/traces/trace.rank{rank}.xplane.pb", size)
        return {str(engine.LOCAL_ROOT / p): v for p, v in pairs.items()}
    if profile == "archived-traces":
        pairs = {}
        for suffix, (stamp, size) in OLD_TRACES.items():
            tag = "greenfield_ws32_short_decoder_" + suffix
            local = f"{tag}/trace/plugins/profile/{stamp}/t1v-n-ae271d05-w-0.xplane.pb"
            pairs[local] = (f"results/{tag}/traces/trace.rank0.xplane.pb", size)
        for rank, size in enumerate(PEER_TRACE_SIZES, 1):
            local = f"{DELIVERY_TAG}/traces/trace.rank{rank}.xplane.pb"
            pairs[local] = (f"results/{local}", size)
        pairs[f"{DELIVERY_TAG}/results.db"] = (
            f"results/{DELIVERY_TAG}/orchestrator/results.db", 195305472)
        return {str(engine.LOCAL_ROOT / p): v for p, v in pairs.items()}
    if profile != "compiler":
        raise ValueError("unreviewed local copy profile")
    pairs = {}
    for rank in range(1, 8):
        for name, size in GRAPHS.items():
            pairs[f"{TAG}/fleet/rank{rank}/{name}"] = (f"results/{TAG}/workers/rank{rank}/{name}", size)
    for name, size in GRAPHS.items():
        pairs[f"{TAG}/rank0/{name}"] = (f"results/{TAG}/workers/rank0/{name}", size)
    pairs[f"{TAG}/hlo/candidate.optimized_hlo.txt"] = (f"results/{TAG}/hlo/candidate.optimized_hlo.txt", 103234144)
    pairs[f"{TAG}/results_ckpt.db"] = (f"results/{TAG}/results_ckpt.db", 193470464)
    for suffix, size in SNAPSHOTS.items():
        tag = "greenfield_ws32_short_decoder_" + suffix
        pairs[f"{tag}/results.db"] = (f"results/{tag}/orchestrator/results.db", size)
    return {str(engine.LOCAL_ROOT / p): v for p, v in pairs.items()}


def validate_manifest(value: dict) -> list[dict]:
    kind = value.get("artifact_kind")
    profiles = {value: key for key, value in PROFILE_KINDS.items()}
    if kind not in profiles:
        raise ValueError("unreviewed local copy manifest kind")
    expected = scope(profiles[kind])
    entries = value.get("files", [])
    if (value.get("status") != "VERIFIED_NOT_DELETED"
            or value.get("cloud_objects_deleted") != 0
            or value.get("bucket_location") != "US-CENTRAL2"
            or len(entries) != len(expected) or {r["path"] for r in entries} != set(expected)
            or value.get("total_bytes") != sum(size for _, size in expected.values())):
        raise ValueError("requires the exact reviewed local copy scope")
    for r in entries:
        if ((r["name"], r["size"]) != expected[r["path"]] or r["nlink"] != 1
                or r["restore_uri"] != f"gs://{engine.BUCKET}/{r['name']}#{r['generation']}"):
            raise ValueError("copy scope/size/recovery differs")
    return entries


def review(output: Path, profile: str = "compiler") -> None:
    from google.cloud import storage
    if output.exists():
        raise FileExistsError(output)
    with engine._leases():
        bucket = storage.Client().get_bucket(engine.BUCKET)
        if bucket.location != "US-CENTRAL2":
            raise ValueError("wrong region")
        rows = []
        for local, (name, size) in scope(profile).items():
            path = Path(local)
            st = path.lstat()
            blob = bucket.get_blob(name)
            if blob is None or st.st_size != size or st.st_nlink != 1 or int(blob.size) != size:
                raise ValueError(f"copy missing/size/link differs: {local}")
            # SHA and CRC are streamed again by the existing verifier; CRC must
            # match this exact cloud generation, not merely the current name.
            digest = sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(8 << 20):
                    digest.update(chunk)
            r = dict(path=local, name=name, size=size, generation=str(blob.generation),
                crc32c=blob.crc32c, sha256=digest.hexdigest(), inode=st.st_ino,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns, device=st.st_dev,
                nlink=st.st_nlink, restore_uri=f"gs://{engine.BUCKET}/{name}#{blob.generation}")
            with engine._verified_file(r, bucket):
                pass
            if path.name in GRAPHS:
                retained = engine.LOCAL_ROOT / TAG / "fleet/rank0" / path.name
                if retained.is_symlink() or sha256(retained.read_bytes()).hexdigest() != r["sha256"]:
                    raise ValueError("retained original graph differs")
            rows.append(r)
        kind = PROFILE_KINDS[profile]
        value = dict(artifact_kind=kind, status="VERIFIED_NOT_DELETED", bucket_location=bucket.location,
                     cloud_objects_deleted=0, total_bytes=sum(r["size"] for r in rows), files=rows)
        validate_manifest(value)
        _atomic_json(output, value)
        print(json.dumps(dict(files=len(rows), bytes=value["total_bytes"], sha256=sha256(output.read_bytes()).hexdigest())))


def main() -> int:
    if sys.argv[1:2] == ["review"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("output", type=Path)
        parser.add_argument("--profile", choices=tuple(PROFILE_KINDS), default="compiler")
        args = parser.parse_args(sys.argv[2:])
        review(args.output, args.profile)
        return 0
    return engine.main(validate_manifest=validate_manifest)


if __name__ == "__main__":
    raise SystemExit(main())
