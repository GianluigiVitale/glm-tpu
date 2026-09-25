"""Tests of :mod:`glm_tpu.utils.json_utils`: the canonical JSON bytes and the hash contract.

``canonical`` is the wire and identity form of requests and resident commands (``request_sha256`` hashes
it): sorted keys, no whitespace, UTF-8 text. ``fingerprint`` is the hash contract of the recorded
geometry and topology digests: the same form, ASCII-escaped. Both refuse ``NaN`` and infinities.
"""

from __future__ import annotations

from hashlib import sha256
import math

import pytest

from glm_tpu.utils.json_utils import canonical, fingerprint

VALUE = {"b": [1, 2.5, None, True], "a": {"z": "x", "y": -3}}


def test_canonical_bytes_are_sorted_compact_utf8():
    assert canonical(VALUE) == b'{"a":{"y":-3,"z":"x"},"b":[1,2.5,null,true]}'
    assert canonical({"name": "café →"}) == '{"name":"café →"}'.encode()
    assert canonical("é") == b'"\xc3\xa9"'


def test_canonical_bytes_do_not_depend_on_key_order():
    assert canonical({"b": 1, "a": 2}) == canonical({"a": 2, "b": 1}) == b'{"a":2,"b":1}'


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf])
def test_canonical_refuses_non_finite_numbers(number):
    with pytest.raises(ValueError):
        canonical({"x": number})


def test_fingerprint_is_the_sha256_of_the_ascii_canonical_form():
    assert fingerprint(VALUE) == sha256(b'{"a":{"y":-3,"z":"x"},"b":[1,2.5,null,true]}').hexdigest()
    assert fingerprint(VALUE) == "9faf1a7598c85bf2573c8524f91c5527746dfea00deec23a5e9f603ef9e84131"
    assert fingerprint({"b": 1, "a": 2}) == fingerprint({"a": 2, "b": 1})


def test_fingerprint_escapes_non_ascii_text_unlike_canonical():
    value = {"city": "Zürich"}
    assert fingerprint(value) == sha256(b'{"city":"Z\\u00fcrich"}').hexdigest()
    assert fingerprint(value) != sha256(canonical(value)).hexdigest()
    assert fingerprint({"city": "Zurich"}) == sha256(canonical({"city": "Zurich"})).hexdigest()


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf])
def test_fingerprint_refuses_non_finite_numbers(number):
    with pytest.raises(ValueError):
        fingerprint({"x": number})
