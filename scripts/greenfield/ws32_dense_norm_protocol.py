"""Fixed same-job capture/replay scope; not standalone launch authorization."""

from __future__ import annotations

import re

PROTOCOL = "ws32-dense01-norm-db605-retained-reproduction-v1"
KERNEL = "ws32_dense_norm_boundary"
PROFILE = "ws32-dense01-norm-capture-completed-suffix-v1"
PROGRAMS = ("wk_decode", "wk_promote", "dense01_norm", "dense_suffix")
CAPTURES = (("wide_final", 128, 0),) + tuple(
    (f"narrow_{start + 32}", 32, start) for start in (0, 32, 64, 96)
)
CALLS = (
    *((f"layer{layer}/{name}", name) for layer in (0, 1) for name in PROGRAMS[:2]),
    *((f"norm/capture/{label}", "dense01_norm") for label, _, _ in CAPTURES),
    *((f"norm/own/{label}", "dense_suffix") for label, _, _ in CAPTURES),
    *((f"norm/cross/{start}", "dense_suffix") for start in (0, 32, 64, 96)),
)
MODEL_ORIGINALS_LIMIT = 128 << 20
RESERVE = 1 << 30
RAW = {
    "wk_decode": (
        10725,
        "8eeefbb0cbc3518ac223b49e1bade70dc1aa78c965c983ebef83c0140284c362",
    ),
    "wk_promote": (
        802,
        "7b277bb821af372bd03687010b1db3630533dc9db46747863dd08cc0742006e5",
    ),
    "dense01_norm": (
        663350,
        "65379b5f5a87c2aef25687ce846e149749857a831924123a228ff267976f617c",
    ),
    "dense_suffix": (
        33495,
        "b895cadecf1571b351e41bebdd440bfe14f5ead52cf69562aac92a2fb1defb23",
    ),
}


def is_tag(tag: str) -> bool:
    return (
        isinstance(tag, str)
        and re.fullmatch(r"greenfield_fp8_ws32_dense_norm_d01_[0-9]{8}T[0-9]+Z", tag)
        is not None
    )
