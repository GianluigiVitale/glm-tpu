# Security and private data

The source is public; a deployment is not. There is no supported public HTTP
endpoint: the chat UI and the `/v1` API listen on loopback only and are reached
through a private tunnel ([UI](docs/UI.md), [API](docs/API.md)). Operation
assumes a trusted operator on the existing authenticated TPU hosts; the engine is
not a hardened multi-tenant service or a sandbox for untrusted Python code.

## Reporting a vulnerability

Report a vulnerability, or a credential you find in the repository or its
history, privately: through GitHub's private vulnerability reporting (the
repository's **Security** tab, **Report a vulnerability**) or by contacting the
maintainer, Gianluigi Vitale, directly. Do not open a public issue for it and do
not reproduce a credential's value. Problems that are not security-sensitive go
to ordinary public issues.

## Keep outside Git

Never commit credentials, service-account keys, local environment files, the
site file, weights, runtime databases, private benchmark questions or references,
user prompts or raw user responses. Token IDs can reveal prompt contents too.
Prepare requests outside the checkout: the preparation commands create output
readable only by you (mode 600) and refuse to overwrite. `.gitignore` adds defense in depth; it is not
permission to commit sensitive files under other names.

Keep credentials in the existing operator-managed authentication environment. Do
not paste tokens into issues, commit messages, command arguments or logs. If a
real credential is found, report it as [above](#reporting-a-vulnerability),
without reproducing its value. Removing a current file does not remove it from history; rotation and any
history rewrite need explicit coordination and are never an automatic cleanup
step.

## Release checks

Run before a release commit, on the CPU, from the repository root, in the
[documented environment](docs/release/INSTALLATION.md) plus the developer tools of
[CONTRIBUTING](CONTRIBUTING.md). None of them opens a TPU, downloads weights or
contacts a cloud service.

```bash
ruff check .
ruff format --check .
codespell --skip='*/hf_config' README.md CONTRIBUTING.md SECURITY.md THIRD_PARTY_NOTICES.md docs glm_tpu tests tools
git diff --check HEAD~1 HEAD
JAX_PLATFORMS=cpu python -m pytest -q -p no:cacheprovider -rs tests --ignore=tests/golden
JAX_PLATFORMS=cpu python -m pytest -q -p no:cacheprovider -rs tests/golden -m "not cpu32"
JAX_PLATFORMS=cpu python -m tools.equivalence check
JAX_PLATFORMS=cpu python -m tools.equivalence check --tier production
```

Also build the wheel from an exported copy of the commit
([INSTALLATION](docs/release/INSTALLATION.md#the-wheel)) and inspect what it
contains. `tests/golden/test_data_contract.py` (part of the second pytest run)
refuses home-directory paths, private addresses, private run names and bucket
URIs in the equivalence harness and its records, and the same literals and cloud
zone names in the governance files (`AGENTS.md`, `docs/development/HANDOFF.md`,
`docs/development/goal.md`, `docs/release/STATUS.md`) and the release receipts, where a receipt may name a
run only under its `run` or `run_id` key. These checks cover the software
contracts; they do not authorize a deployment, which needs the hardware steps of
[OPERATIONS](docs/release/OPERATIONS.md).

The research release had its own release check and a Git-history content audit
(`tools/check_release.py`, `tools/audit_release_content.py`). Both are preserved
at the tag `archive/research-20260922` and check the research tree of that tag.
The history audit scans the local Git objects of the repository it lives in, so
run it from a worktree of the tag inside this clone when it needs renewing:

```bash
git worktree add --detach ../glm-tpu-archive archive/research-20260922
python ../glm-tpu-archive/tools/audit_release_content.py --history
git worktree remove ../glm-tpu-archive
```

It reports locations and object IDs only, never matched credential contents. It
checks nine known credential formats and sensitive or payload file names. History
mode includes unreachable local Git blobs, refuses more than 2 GiB in total and
reports individual blobs over 2 MiB as unscanned errors. It cannot detect every
secret, private dataset, misleading file extension or vulnerability. Do not
rescan unchanged history repeatedly: keep the object-set receipt and review new
content through the release checks.

## Audit scope on record

The 2026-09-14 audit of 10,281 local Git blobs (1,496,886,163 uncompressed bytes)
found no match for the nine credential rules. The tracked snapshot of the research
tree then had 1,940 files, no detected credential or payload file-name violation
and no scan error. This is a bounded heuristic check, **not complete security
clearance**, and it predates this tree.

A separate structural inspection of tracked JSON flagged eight `gold` fields in a
single historical passkey assessment. All belong to the explicitly labelled
synthetic passkey task, not to GPQA or AIME questions; that compact scientific
receipt is preserved at the tag.

## History scan

Before the repository was made public, the whole Git history was scanned on
2026-09-29: all 35 branches, 5 tags and 2,177 commits.

- gitleaks 8.28.0 reported 293 findings, all false positives: SHA-256 digests
  labelled as a token or key, and prose such as "92 tokens".
- A pattern scan of all 13,100 blob versions and of every commit message for
  private keys, SSH keys, Hugging Face, GitHub, Google, OAuth, AWS, Anthropic,
  OpenAI and Slack tokens, `Bearer` headers and hard-coded passwords found none.
  The only `"private_key":` matches are detection patterns in the repository's
  own audit tools (`tools/audit_release_content.py` and
  `tools/release_inventory.py`, both kept in the history).
- No credential-like file name (`id_rsa`, `.pem`, `.key`, `.env`,
  `credentials`, service-account JSON, `.netrc`, API-key or Hugging Face token
  files) was ever committed.

Earlier commits keep infrastructure names (bucket, node, zone and host names,
internal addresses, home paths). They grant no access and stay in the history,
which is not rewritten; new commits keep them out (see [above](#keep-outside-git)).
Like the earlier audit, the scan is heuristic, not complete security clearance,
and it looked for credentials only: it is no review of third-party provenance
([notices](THIRD_PARTY_NOTICES.md)) or of benchmark data permissions.
