# Security and private data

This repository remains private. There is no supported public HTTP endpoint.
Operation assumes a trusted operator on the existing authenticated TPU hosts;
it is not a hardened multi-tenant service or a sandbox for untrusted Python code.

## Keep outside Git

Never commit credentials, service-account keys, local environment files, weights,
runtime databases, private benchmark questions/golds, user prompts or raw user
responses. Token IDs can reveal prompt contents too. Prepare requests outside
the checkout; the preparation command creates owner-only output and refuses
overwrites. `.gitignore` and the release content check add defense in depth, not
permission to commit sensitive files under different names.

Keep credentials in the existing operator-managed authentication environment.
Do not paste tokens into issues, commit messages, command arguments or logs.
If a real credential is found, notify the owner privately without reproducing
its value. Removing a current file does not remove it from history; rotation and
any history rewrite require explicit coordination. Do not rewrite history as
an automatic cleanup step.

## Release checks

```bash
JAX_PLATFORMS=cpu python tools/check_release.py
```

This runs whitespace checks, installed dependency metadata checks, the tracked
content audit, unchanged model-source admission, Python syntax, selected CPU
runtime/release tests, and isolated offline wheel installation without dependencies.
It does not initialize TPU, download weights, call the model, run cloud commands,
upgrade the active environment or authorize deployment.

For a one-time history audit when needed:

```bash
python tools/audit_release_content.py --history
```

The audit reports locations/object IDs only, not matched credential contents.
It checks nine known credential formats and sensitive/payload filenames. History
mode includes unreachable local Git blobs and refuses more than2GiB total or
reports individual blobs over2MiB as unscanned errors. It cannot detect every
secret, private dataset, misleading file extension or security vulnerability.
Do not repeatedly rescan unchanged history: retain the object-set receipt and
review new content through release checks.

## Current audit scope

The2026-09-14 audit of10,281 local Git blobs (1,496,886,163 uncompressed bytes)
found no matches for the nine credential rules. The tracked snapshot had1,940
files, no detected credential/payload filename violations and no scan errors.
This is a bounded heuristic check, **not complete security clearance**.

A separate structural inspection of tracked JSON flagged eight `gold` fields
in a single historical passkey assessment. All belong to the explicitly labeled
synthetic passkey task, not GPQA/AIME questions. That compact scientific receipt
is preserved. This inspection is not a universal detector of private text in
Markdown, source code or binary formats.

Before any future public release, revisit Git history, third-party provenance,
operator/site-specific information and benchmark data permissions. Keeping this
repo private does not make credentials safe to store in it. Do not change
visibility as part of release engineering.
