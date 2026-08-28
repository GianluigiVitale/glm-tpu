# Owner audit — PR 3 exact GLM-5.2 model contract and CI

This document covers PR 3 only. Official upstream is not authorized and has not
been mutated. Review PRs 1-2 first because this patch is stacked on both.

## Immutable review range

- Parent: `d837832ab41f947ee9ff759e65ea8417ba1bd5c9` (PR 2)
- Head: `101ec506d76a3ecb0b688315e432ae7e8d0ab37a`
- Branch: private `pr/glm-dsa-model-ci-v3`
- Dependencies: PR 1 and PR 2.
- Size: 2 files, +302/-0; two DCO-signed commits.

Inspect the exact patch:

```bash
git -C /home/gianl/tpu-inference-glm-baseline \
  diff --find-renames d837832ab41f947ee9ff759e65ea8417ba1bd5c9..101ec506d76a3ecb0b688315e432ae7e8d0ab37a
```

## File-by-file checklist

| File | Change | Owner question | Direct proof |
|---|---|---|---|
| `tests/models/vllm/test_glm_moe_dsa.py` | Resolves the current vLLM GLM class without weights, constructs mocked layers, and checks shared top-k wiring/IndexShare | Does the test prove current registration and scheduling instead of duplicating them? | Registry resolves `GlmMoeDsaForCausalLM`; seven layers share one buffer; prefix is `full,full,full,shared,shared,shared,full` |
| `.buildkite/models/zai-org_GLM-5.2-FP8.yml` | Adds a real TPU unit step for the model test plus PR1/PR2 regressions | Is this honest about what CI proves? | Accuracy and performance are literal `unverified`; no checkpoint is loaded for those stages |

No production registration is added: the pinned vLLM source already registers
`GlmMoeDsaForCausalLM`. Adding a second TPU-side registration would create the
semantic fork this series is designed to avoid.

CODEOWNER surfaces are model-test owners (`@kyuyeunk @lk-chen @jrplatin`) and
Buildkite owners (`@QiliangCui @yiw-wang @CienetStingLin @yunyao-gg
@theminghuang`).

## CI contract audit

- PyYAML parses six execution/recording steps with six unique keys.
- Every `depends_on` resolves within the file; every referenced pytest file
  exists; all result-recording steps carry complete `CI_TPU_VERSION`,
  `CI_TARGET`, `CI_STAGE`, and `CI_CATEGORY` metadata.
- `CI_TARGET=zai-org/GLM-5.2-FP8` is unique across `.buildkite/models`.
- The repository validator currently misparses its own yq-generated literal
  `\t` as a delimiter and reports every queue as empty, including plain `cpu`.
  Unchanged `zai-org_GLM-5.yml` and `deepseek-ai_DeepSeek-V3_2.yml` fail
  identically. This is an upstream baseline validator defect, not a GLM YAML
  exception; it should not be bundled into this model PR.
- The default execution queue is v6e. Protected numerical/performance evidence
  in this series is v4-only, so the CI step claims portable unit correctness,
  not v6e performance or full-753B fit. The TP2 regression skips if fewer than
  two devices are addressable.

## Evidence to accept or reject

- Exact-head protected model-contract test: 3/3 in 9.43 seconds;
  local/remote tag `upstream_glm_dsa_pr3_rebase_20260828T085938Z`;
  evidence-manifest SHA-256
  `a3cbb2b461d8329c8dc48f083671ac0cc02732ca1f4f31fcf7cdfaefc6a09df4`.
- The run used the exact PR3 head and ended with authenticated 8/8 pre/post
  cleanup. Focused CPU tests and all-file pre-commit also pass.
- This proves registry/constructor/IndexShare wiring only. It is not a
  full-checkpoint load, quality run, HBM proof, or latency result.

## Compatibility and rollback decision

- Production code is unchanged. The only runtime cost is model-specific CI.
- `soft_fail: true` matches current model-YAML convention while results remain
  explicitly unverified; maintainers can request promotion after supported
  hardware evidence exists.
- Rollback is the two PR3 commits and removes only the test/YAML.

Owner decision: approve after PRs 1-2, request narrower CI coverage, or defer
this final enablement without blocking the first two PRs.
