"""Fail-closed contract for the original exact DB518 prompt-key producer."""

from __future__ import annotations

from hashlib import sha256
import re
from typing import Any

from .prompt_index_cache import validate_prompt_index_key_association_hlo

ORIGINAL_DB518_CODE_HASH = "86243115452920fe4244bb77a9bbf4c44110aeab"
ORIGINAL_DB518_TAG = (
    "greenfield_layer0_prompt_key_norm_m64_20260809T122010714691723Z")
ORIGINAL_DB518_CACHE_SHA256 = (
    "3808d502f3ea1829bf12ab7585d66f15dd83bf640657a17c35daabf5ab1859d1")
ORIGINAL_DB518_COMPARISON_MANIFEST_SHA256 = (
    "1d80d088561181a63e734a91cd0124c4011cfc4151198488c052740050d66fe5")
ORIGINAL_DB518_OPTIMIZED_HLO_SHA256 = (
    "ecb08493357ada2fa0452b3d5cee7abb8c980d33fbbf5a12679808b3696a7251")
ORIGINAL_DB518_CANONICAL_HLO_SHA256 = (
    "de48587500e5f3a83f06c17fc3a49613ff295186559117c568c7d2a5491b700a")
ORIGINAL_DB518_CANDIDATE = (
    "accepted_xla_m64_projection_keynorm_lax_map_gather_cache_write_"
    "fp32_weight_divide_sqrt_source_rope")

_ENTRY_LAYOUT = ("entry_computation_layout={(bf16[24,16,32,128]"
                 "{3,2,1,0:T(8,128)(2,1)}, s32[16]{0:T(128)}, "
                 "bf16[37,6144]{1,0:T(8,128)(2,1)}, s32[2048]{0:T(1024)}, "
                 "s32[2048]{0:T(1024)}, /*index=5*/bf16[6144]"
                 "{0:T(1024)(128)(2,1)}, f32[128,6144]{1,0:T(8,128)}, "
                 "bf16[128]{0:T(256)(128)(2,1)}, "
                 "bf16[128]{0:T(256)(128)(2,1)})->"
                 "bf16[24,16,32,128]{3,2,1,0:T(8,128)(2,1)}}")
_ENTRY_RE = re.compile(
    r"^ENTRY %[^ ]+ \([^\n]+\) -> bf16\[24,16,32,128\] \{$",
    re.MULTILINE,
)
_ROOT_RE = re.compile(r"^  ROOT %[\w.-]+ = bf16\[24,16,32,128\][^\n]*$",
                      re.MULTILINE)
_STABLE_SIGNATURES = (
    "tensor<24x16x32x128xbf16>",
    "tensor<16xi32>",
    "tensor<37x6144xbf16>",
    "tensor<2048xi32>",
    "tensor<6144xbf16>",
    "tensor<128x6144xf32>",
    "tensor<128xbf16>",
)
_FORBIDDEN_STABLE = (
    "stablehlo.all_gather",
    "stablehlo.all_reduce",
    "stablehlo.collective_permute",
    "stablehlo.recv",
    "stablehlo.send",
    "stablehlo.custom_call @xla_python",
    "host_callback",
    "python_callback",
)


def _strip_braced_field(line: str, marker: str) -> str:
    """Remove one or more balanced ``marker={...}`` fields from a HLO line."""

    while marker in line:
        start = line.index(marker)
        brace = start + len(marker) - 1
        depth = 0
        quoted = False
        escaped = False
        end = None
        for index in range(brace, len(line)):
            character = line[index]
            if quoted:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
            elif character == '"':
                quoted = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None or depth != 0 or quoted:
            raise ValueError(f"malformed optimized-HLO field {marker!r}")
        line = line[:start] + line[end:]
    return line


def canonicalize_original_db518_hlo(text: str) -> str:
    """Drop source-location tables/fields while retaining the exact graph."""

    if not isinstance(text, str) or not text.startswith("HloModule "):
        raise ValueError("optimized HLO is missing its module header")
    table_start = text.find("\nFileNames\n")
    if table_start < 0:
        raise ValueError("optimized HLO is missing source tables")
    body_start = text.find("\n\n%", table_start)
    if body_start < 0:
        raise ValueError("optimized HLO source tables are unterminated")
    text = text[:table_start] + text[body_start + 1:]
    canonical_lines = []
    for raw_line in text.splitlines():
        line = raw_line
        for marker in (", metadata={", ", frontend_attributes={"):
            line = _strip_braced_field(line, marker)
        canonical_lines.append(line.rstrip())
    return "\n".join(canonical_lines).strip() + "\n"


def validate_original_db518_prompt_key_hlo(text: str) -> dict[str, Any]:
    """Require the exact original DB518 graph, not semantic lookalikes."""

    historical = validate_prompt_index_key_association_hlo(
        text,
        candidate=ORIGINAL_DB518_CANDIDATE,
        prompt_token_count=8155,
        unique_token_count=37,
    )
    canonical = canonicalize_original_db518_hlo(text)
    canonical_sha256 = sha256(canonical.encode("utf-8")).hexdigest()
    violations = list(historical.get("violations", ()))
    if not historical.get("passed"):
        violations.append("historical DB518 HLO contract failed")
    if _ENTRY_LAYOUT not in text.splitlines()[0]:
        violations.append("exact DB518 entry layout is absent")
    if len(_ENTRY_RE.findall(text)) != 1:
        violations.append("DB518 executable must have exactly one cache entry")
    if len(_ROOT_RE.findall(text)) != 1:
        violations.append(
            "DB518 executable must have exactly one BF16 cache root")
    if canonical_sha256 != ORIGINAL_DB518_CANONICAL_HLO_SHA256:
        violations.append("DB518 canonical optimized-HLO graph drifted")
    return {
        "canonical_sha256": canonical_sha256,
        "expected_canonical_sha256": ORIGINAL_DB518_CANONICAL_HLO_SHA256,
        "historical_contract": historical,
        "passed": not violations,
        "violations": violations,
    }


def validate_original_db518_prompt_key_stablehlo(text: str) -> dict[str, Any]:
    """Bound the separately archived portable IR before TPU invocation."""

    violations = []
    if not isinstance(
            text, str
    ) or "module @jit_layer0_prompt_index_key_gather_cache_chunk" not in text:
        violations.append("StableHLO module identity drifted")
    missing = [
        signature for signature in _STABLE_SIGNATURES if signature not in text
    ]
    if missing:
        violations.append(
            f"StableHLO parameter/result signatures absent: {missing}")
    forbidden = [token for token in _FORBIDDEN_STABLE if token in text]
    if forbidden:
        violations.append(
            f"StableHLO contains forbidden operations: {forbidden}")
    if text.count("stablehlo.while") != 1:
        violations.append("StableHLO must retain exactly one M64 map loop")
    if text.count("stablehlo.scatter") != 1:
        violations.append("StableHLO must retain exactly one cache scatter")
    if text.count("stablehlo.cosine") != 1 or text.count(
            "stablehlo.sine") != 1:
        violations.append("StableHLO must retain literal source RoPE")
    return {
        "forbidden": forbidden,
        "missing_signatures": missing,
        "passed": not violations,
        "violations": violations,
    }
