# Preserved real-weight comparison measurements

## Optimized ordinary and MTP research suite

These are the completed historical suite at `dc0479335d57504bd643b6b80a266c5ea3bf49a4`, two repeats each. They used component profiling and ordinary-first order. The later alternating/profiling-off answer run was cancelled before any answer case completed. None of these rows is relabelled as that later protocol.

[Original suite](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/tpu-real-native-suite-20260920T031727Z.json), [derived exact rows](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/mtp-preserved-suite-rows-20260920.json). Rates use timed delivered tokens divided by maximum-host decode wall. The first prefill-produced output token is excluded from that numerator. Cold load/compile, prefill and network transport are excluded. Required host votes and rank0 write/flush are included.

Displayed rates are rounded to six decimal places; the archived receipts retain full precision.

## Aggregate over all fixed prompts and repeats

| Mode | Timed output tokens | Summed decode wall s | Wall tok/s | Relative to ordinary |
|---|---:|---:|---:|---:|
| R2 | 31378 | 2314.939322 | 13.554567 | 0.946712x |
| R3 | 25910 | 1773.525489 | 14.609319 | 1.020381x |
| ordinary | 24898 | 1738.988529 | 14.317518 | 1.000000x |


No qualified speculative gain: all MTP modes diverge on long outputs. The aggregate includes all six fixed cases per mode, not a selected favorable subset.

## Per-request rates and outcomes

| Prompt | Repeat | Mode | Prompt / output / timed tokens | Decode wall s | Delivered tok/s | Paired ratio | First mismatch | Termination | Answer outcome |
|---|---:|---|---|---:|---:|---:|---|---|---|
| prose | 1 | ordinary | 204 / 2655 / 2654 | 183.763925 | 14.442443 | 1.000000x | none | eos | completed_answer_failed_manual_review |
| prose | 2 | ordinary | 204 / 2655 / 2654 | 185.713577 | 14.290824 | 1.000000x | none | eos | completed_answer_failed_manual_review |
| code | 1 | ordinary | 347 / 7168 / 7167 | 502.101793 | 14.273998 | 1.000000x | none | length | incomplete_reasoning_at_token_cap |
| code | 2 | ordinary | 347 / 7168 / 7167 | 501.092727 | 14.302742 | 1.000000x | none | length | incomplete_reasoning_at_token_cap |
| structured | 1 | ordinary | 696 / 2629 / 2628 | 183.437057 | 14.326440 | 1.000000x | none | eos | values_correct_format_failed |
| structured | 2 | ordinary | 696 / 2629 / 2628 | 182.879451 | 14.370122 | 1.000000x | none | eos | values_correct_format_failed |
| prose | 1 | R2 | 204 / 6134 / 6133 | 461.494157 | 13.289442 | 0.920166x | 6 | eos | completed_answer_failed_manual_review |
| prose | 2 | R2 | 204 / 6134 / 6133 | 461.124886 | 13.300085 | 0.930673x | 6 | eos | completed_answer_failed_manual_review |
| code | 1 | R2 | 347 / 7168 / 7167 | 526.812093 | 13.604471 | 0.953095x | 5 | length | incomplete_reasoning_at_token_cap |
| code | 2 | R2 | 347 / 7168 / 7167 | 527.422483 | 13.588727 | 0.950078x | 5 | length | incomplete_reasoning_at_token_cap |
| structured | 1 | R2 | 696 / 2390 / 2389 | 169.211202 | 14.118451 | 0.985482x | 816 | eos | values_correct_format_failed |
| structured | 2 | R2 | 696 / 2390 / 2389 | 168.874501 | 14.146600 | 0.984445x | 816 | eos | values_correct_format_failed |
| prose | 1 | R3 | 204 / 3745 / 3744 | 277.554850 | 13.489226 | 0.933999x | 6 | eos | completed_answer_failed_manual_review |
| prose | 2 | R3 | 204 / 3745 / 3744 | 278.355662 | 13.450418 | 0.941193x | 6 | eos | completed_answer_failed_manual_review |
| code | 1 | R3 | 347 / 7168 / 7167 | 479.978277 | 14.931926 | 1.046093x | 5 | length | incomplete_reasoning_at_token_cap |
| code | 2 | R3 | 347 / 7168 / 7167 | 480.416572 | 14.918303 | 1.043038x | 5 | length | incomplete_reasoning_at_token_cap |
| structured | 1 | R3 | 696 / 2045 / 2044 | 128.689328 | 15.883213 | 1.108664x | 823 | eos | values_correct_format_failed |
| structured | 2 | R3 | 696 / 2045 / 2044 | 128.530800 | 15.902803 | 1.106657x | 823 | eos | values_correct_format_failed |


## Prefill, TTFT, verification and memory

Ranges are minimum–maximum across hosts, not confidence intervals or independent trials. Peak HBM is the entire run high-water mark; a separate ordinary per-case peak was unavailable. Component-profiled verifier times are unavailable for ordinary decode.

| Prompt / repeat / mode | Prefill tok/s (host range) | Warm TTFT s (host range) | Accepted / proposed drafts | Accepted by draft position | Rounds | Mean verifier ms (max host) | Peak HBM bytes/chip |
|---|---|---|---|---|---:|---:|---:|
| prose / 1 / ordinary | 83.150270–94.575219 | 2.408432–2.716471 | None / None | null | None | n/a | 28228678144 |
| prose / 2 / ordinary | 90.054946–101.458328 | 2.157045–2.406178 | None / None | null | None | n/a | 28228678144 |
| code / 1 / ordinary | 99.106352–108.738325 | 3.453170–3.773294 | None / None | null | None | n/a | 28228678144 |
| code / 2 / ordinary | 101.541529–114.694923 | 3.176578–3.563337 | None / None | null | None | n/a | 28228678144 |
| structured / 1 / ordinary | 111.323327–114.345116 | 6.354438–6.539520 | None / None | null | None | n/a | 28228678144 |
| structured / 2 / ordinary | 111.180661–117.164403 | 6.078821–6.405735 | None / None | null | None | n/a | 28228678144 |
| prose / 1 / R2 | 97.364527–102.608189 | 2.557187–2.661675 | 2801 / 3332 | [{"accepted": 2801, "proposed": 3332, "rate": 0.8406362545018007}] | 3332 | 120.323950 | 28228678144 |
| prose / 2 / R2 | 96.517959–101.896149 | 2.568492–2.684036 | 2801 / 3332 | [{"accepted": 2801, "proposed": 3332, "rate": 0.8406362545018007}] | 3332 | 120.130718 | 28228678144 |
| code / 1 / R2 | 109.394597–115.189195 | 3.881620–4.049351 | 3385 / 3782 | [{"accepted": 3385, "proposed": 3782, "rate": 0.8950290851401375}] | 3782 | 120.850232 | 28228678144 |
| code / 2 / R2 | 112.017829–114.814159 | 3.868125–3.946394 | 3385 / 3782 | [{"accepted": 3385, "proposed": 3782, "rate": 0.8950290851401375}] | 3782 | 120.850472 | 28228678144 |
| structured / 1 / R2 | 115.379883–117.525559 | 7.503073–7.612469 | 1189 / 1201 | [{"accepted": 1189, "proposed": 1201, "rate": 0.9900083263946711}] | 1201 | 122.194483 | 28228678144 |
| structured / 2 / R2 | 115.241429–117.128010 | 7.493574–7.594517 | 1189 / 1201 | [{"accepted": 1189, "proposed": 1201, "rate": 0.9900083263946711}] | 1201 | 122.220645 | 28228678144 |
| prose / 1 / R3 | 95.250405–100.915297 | 2.615283–2.725822 | 2200 / 3090 | [{"accepted": 1246, "proposed": 1545, "rate": 0.8064724919093851}, {"accepted": 954, "proposed": 1545, "rate": 0.6174757281553398}] | 1545 | 159.382440 | 28228678144 |
| prose / 2 / R3 | 96.239874–101.736359 | 2.578916–2.693786 | 2200 / 3090 | [{"accepted": 1246, "proposed": 1545, "rate": 0.8064724919093851}, {"accepted": 954, "proposed": 1545, "rate": 0.6174757281553398}] | 1545 | 159.509388 | 28228678144 |
| code / 1 / R3 | 109.106445–114.519722 | 3.899826–4.049744 | 4533 / 5268 | [{"accepted": 2425, "proposed": 2634, "rate": 0.9206529992406985}, {"accepted": 2108, "proposed": 2634, "rate": 0.8003037205770691}] | 2634 | 161.395119 | 28228678144 |
| code / 2 / R3 | 106.186307–114.393149 | 3.911743–4.145277 | 4533 / 5268 | [{"accepted": 2425, "proposed": 2634, "rate": 0.9206529992406985}, {"accepted": 2108, "proposed": 2634, "rate": 0.8003037205770691}] | 2634 | 161.480092 | 28228678144 |
| structured / 1 / R3 | 113.243413–116.964044 | 7.502899–7.703024 | 1346 / 1398 | [{"accepted": 687, "proposed": 699, "rate": 0.9828326180257511}, {"accepted": 659, "proposed": 699, "rate": 0.9427753934191703}] | 699 | 163.364039 | 28228678144 |
| structured / 2 / R3 | 86.647256–116.683797 | 7.529285–9.597168 | 1346 / 1398 | [{"accepted": 687, "proposed": 699, "rate": 0.9828326180257511}, {"accepted": 659, "proposed": 699, "rate": 0.9427753934191703}] | 699 | 163.077442 | 28228678144 |


## Exact output identities and quality-check scope

Prompt text, generated text and token arrays remain private. The hashes below bind the retained output evidence. Passing a final-data check does not certify generated Python or prose; unfinished reasoning is not a completed answer.

### prose / repeat 1 / ordinary

Prompt SHA-256: `a85120228ba51e9a8d97433893be21fddcd72f84c928cf211ecf17cbe14d0d71`. Output-token SHA-256: `7ca25839bb2dbc2f5f21859da72944dd308f920ec5458f35f24e3122cba2a5f1`. Ordinary agreement: **True**. Recorded qualification: preserved baseline.


- **automated_status**: manual_review_required

- **correctness_established**: False

- **finish_reason**: eos

- **independent_review**: False

- **manual_checks**: {"kv_cache_mechanics": false, "native_mtp_description": true, "prefill_decode_qualification": true, "rejected_cache_rollback": false, "request_completeness": true, "speculative_acceptance_description": false, "ttft_vs_throughput": true, "wall_time_arithmetic": true}

- **manual_review_sha256**: 7c34f668cb37d5dbcf92683941586675c4f858c52e75042a7bd687e579ac5907

- **nonempty_answer**: True

- **oracle_sha256**: aced4fee7796b00fcc8dc7332c5173639255570fd4b4892c7ddff79f77fbcc2d

- **reasoning_closed**: True

- **response_sha256**: 9c49f26c29d18fe6ab182d4f7007b676c2493aea77e32b4363f6a45c85373000

- **scope**: Assistant self-review of named mechanism, arithmetic and latency checks in this completed answer; no independent review or model-wide quality score.

- **status**: completed_answer_failed_manual_review

- **structured_answer_present**: False

### prose / repeat 2 / ordinary

Prompt SHA-256: `a85120228ba51e9a8d97433893be21fddcd72f84c928cf211ecf17cbe14d0d71`. Output-token SHA-256: `7ca25839bb2dbc2f5f21859da72944dd308f920ec5458f35f24e3122cba2a5f1`. Ordinary agreement: **True**. Recorded qualification: preserved baseline.


- **automated_status**: manual_review_required

- **correctness_established**: False

- **finish_reason**: eos

- **independent_review**: False

- **manual_checks**: {"kv_cache_mechanics": false, "native_mtp_description": true, "prefill_decode_qualification": true, "rejected_cache_rollback": false, "request_completeness": true, "speculative_acceptance_description": false, "ttft_vs_throughput": true, "wall_time_arithmetic": true}

- **manual_review_sha256**: f81eca0b301bc1d1431846fbd4e4939474e0bc3e5f74404ecef54f7e62e6eba9

- **nonempty_answer**: True

- **oracle_sha256**: aced4fee7796b00fcc8dc7332c5173639255570fd4b4892c7ddff79f77fbcc2d

- **reasoning_closed**: True

- **response_sha256**: 9c49f26c29d18fe6ab182d4f7007b676c2493aea77e32b4363f6a45c85373000

- **scope**: Assistant self-review of named mechanism, arithmetic and latency checks in this completed answer; no independent review or model-wide quality score.

- **status**: completed_answer_failed_manual_review

- **structured_answer_present**: False

### code / repeat 1 / ordinary

Prompt SHA-256: `59d5983c346baa131a2607a801a306241ab46d498b7fe55983039b0b9aef33c1`. Output-token SHA-256: `bbef8eae595ac68736b5e0d130473f7c1e40fc02790d802379ba9090a6e3058f`. Ordinary agreement: **True**. Recorded qualification: preserved baseline.


- **correctness_established**: False

- **finish_reason**: length

- **oracle_sha256**: 39dededfa688a1ac6bce39e6f1d09856eaaa01c42b5803655823eb118f2295c1

- **reasoning_closed**: False

- **response_sha256**: dc4e6ae865761260495210f6ef7028263fc5a38d31e5ef84db4f9c5f9c1ea8bb

- **scope**: Final schedule compatibility and exact optimal value; generated Python and explanatory proof not executed or fully graded

- **status**: incomplete_reasoning_at_token_cap

- **structured_answer_present**: False

### code / repeat 2 / ordinary

Prompt SHA-256: `59d5983c346baa131a2607a801a306241ab46d498b7fe55983039b0b9aef33c1`. Output-token SHA-256: `bbef8eae595ac68736b5e0d130473f7c1e40fc02790d802379ba9090a6e3058f`. Ordinary agreement: **True**. Recorded qualification: preserved baseline.


- **correctness_established**: False

- **finish_reason**: length

- **oracle_sha256**: 39dededfa688a1ac6bce39e6f1d09856eaaa01c42b5803655823eb118f2295c1

- **reasoning_closed**: False

- **response_sha256**: dc4e6ae865761260495210f6ef7028263fc5a38d31e5ef84db4f9c5f9c1ea8bb

- **scope**: Final schedule compatibility and exact optimal value; generated Python and explanatory proof not executed or fully graded

- **status**: incomplete_reasoning_at_token_cap

- **structured_answer_present**: False

### structured / repeat 1 / ordinary

Prompt SHA-256: `0959c0c3666fcd2709fcb03fde2dd49272bd89fd7df23b6985dab668bc04cc62`. Output-token SHA-256: `3afd78211c7587cf62c494704096305bd0a28ab0ead5eb97681ee9d47e81c74f`. Ordinary agreement: **True**. Recorded qualification: preserved baseline.


- **correctness_established**: False

- **finish_reason**: eos

- **oracle_sha256**: 3a9d228938f6fd1fe7279c5e9f1ec6e7efdd7968c64b912dc935e15deb62cc43

- **reasoning_closed**: True

- **response_sha256**: d4b4808e0087f3f07c2c422126dd7881dd35bb995e3983a336730ad42180e839

- **scope**: Exact final JSON values, types, ordering and requested standalone format

- **standalone_json**: False

- **status**: values_correct_format_failed

- **structured_answer_present**: True

- **values_match_oracle**: True

### structured / repeat 2 / ordinary

Prompt SHA-256: `0959c0c3666fcd2709fcb03fde2dd49272bd89fd7df23b6985dab668bc04cc62`. Output-token SHA-256: `3afd78211c7587cf62c494704096305bd0a28ab0ead5eb97681ee9d47e81c74f`. Ordinary agreement: **True**. Recorded qualification: preserved baseline.


- **correctness_established**: False

- **finish_reason**: eos

- **oracle_sha256**: 3a9d228938f6fd1fe7279c5e9f1ec6e7efdd7968c64b912dc935e15deb62cc43

- **reasoning_closed**: True

- **response_sha256**: d4b4808e0087f3f07c2c422126dd7881dd35bb995e3983a336730ad42180e839

- **scope**: Exact final JSON values, types, ordering and requested standalone format

- **standalone_json**: False

- **status**: values_correct_format_failed

- **structured_answer_present**: True

- **values_match_oracle**: True

### prose / repeat 1 / R2

Prompt SHA-256: `a85120228ba51e9a8d97433893be21fddcd72f84c928cf211ecf17cbe14d0d71`. Output-token SHA-256: `2401af85cf374ffc163d4032ee24fa18c73944a7beb8bde5d5af754cf11e1ef1`. Ordinary agreement: **False**. Recorded qualification: rejected: ordinary token divergence.


- **automated_status**: manual_review_required

- **correctness_established**: False

- **finish_reason**: eos

- **independent_review**: False

- **manual_checks**: {"kv_cache_mechanics": true, "native_mtp_description": false, "prefill_decode_qualification": true, "rejected_cache_rollback": true, "request_completeness": true, "speculative_acceptance_description": false, "ttft_vs_throughput": true, "wall_time_arithmetic": true}

- **manual_review_sha256**: f51cc2dcd20517bf55fc4034a050b489203a697f08f853cd751307f8c9d21f29

- **nonempty_answer**: True

- **oracle_sha256**: aced4fee7796b00fcc8dc7332c5173639255570fd4b4892c7ddff79f77fbcc2d

- **reasoning_closed**: True

- **response_sha256**: 054f4e43bbec6485176db732f7ecf14ff5ce0ffd4668a21163723f60e85ddd69

- **scope**: Assistant self-review of named mechanism, arithmetic and latency checks in this completed answer; no independent review or model-wide quality score.

- **status**: completed_answer_failed_manual_review

- **structured_answer_present**: False

### prose / repeat 2 / R2

Prompt SHA-256: `a85120228ba51e9a8d97433893be21fddcd72f84c928cf211ecf17cbe14d0d71`. Output-token SHA-256: `2401af85cf374ffc163d4032ee24fa18c73944a7beb8bde5d5af754cf11e1ef1`. Ordinary agreement: **False**. Recorded qualification: rejected: ordinary token divergence.


- **automated_status**: manual_review_required

- **correctness_established**: False

- **finish_reason**: eos

- **independent_review**: False

- **manual_checks**: {"kv_cache_mechanics": true, "native_mtp_description": false, "prefill_decode_qualification": true, "rejected_cache_rollback": true, "request_completeness": true, "speculative_acceptance_description": false, "ttft_vs_throughput": true, "wall_time_arithmetic": true}

- **manual_review_sha256**: 12e7c38c7d899ec5a601c6a8b0fe7b29503181d06623919c7da310ae40bdfea9

- **nonempty_answer**: True

- **oracle_sha256**: aced4fee7796b00fcc8dc7332c5173639255570fd4b4892c7ddff79f77fbcc2d

- **reasoning_closed**: True

- **response_sha256**: 054f4e43bbec6485176db732f7ecf14ff5ce0ffd4668a21163723f60e85ddd69

- **scope**: Assistant self-review of named mechanism, arithmetic and latency checks in this completed answer; no independent review or model-wide quality score.

- **status**: completed_answer_failed_manual_review

- **structured_answer_present**: False

### code / repeat 1 / R2

Prompt SHA-256: `59d5983c346baa131a2607a801a306241ab46d498b7fe55983039b0b9aef33c1`. Output-token SHA-256: `09313b612cf604ebe2008f49bdc9a9aae64c2aa189de248360a3b9e21053bb0c`. Ordinary agreement: **False**. Recorded qualification: rejected: ordinary token divergence.


- **correctness_established**: False

- **finish_reason**: length

- **oracle_sha256**: 39dededfa688a1ac6bce39e6f1d09856eaaa01c42b5803655823eb118f2295c1

- **reasoning_closed**: False

- **response_sha256**: 8da00714a2f5f18e266abc6daaa9c25a1fdfcad3027aef75bb8fb344814636bd

- **scope**: Final schedule compatibility and exact optimal value; generated Python and explanatory proof not executed or fully graded

- **status**: incomplete_reasoning_at_token_cap

- **structured_answer_present**: False

### code / repeat 2 / R2

Prompt SHA-256: `59d5983c346baa131a2607a801a306241ab46d498b7fe55983039b0b9aef33c1`. Output-token SHA-256: `09313b612cf604ebe2008f49bdc9a9aae64c2aa189de248360a3b9e21053bb0c`. Ordinary agreement: **False**. Recorded qualification: rejected: ordinary token divergence.


- **correctness_established**: False

- **finish_reason**: length

- **oracle_sha256**: 39dededfa688a1ac6bce39e6f1d09856eaaa01c42b5803655823eb118f2295c1

- **reasoning_closed**: False

- **response_sha256**: 8da00714a2f5f18e266abc6daaa9c25a1fdfcad3027aef75bb8fb344814636bd

- **scope**: Final schedule compatibility and exact optimal value; generated Python and explanatory proof not executed or fully graded

- **status**: incomplete_reasoning_at_token_cap

- **structured_answer_present**: False

### structured / repeat 1 / R2

Prompt SHA-256: `0959c0c3666fcd2709fcb03fde2dd49272bd89fd7df23b6985dab668bc04cc62`. Output-token SHA-256: `26917f4e566442addbf5291df87a9eee949a56aa55e12980c4ca6e696a95fb60`. Ordinary agreement: **False**. Recorded qualification: rejected: ordinary token divergence.


- **correctness_established**: False

- **finish_reason**: eos

- **oracle_sha256**: 3a9d228938f6fd1fe7279c5e9f1ec6e7efdd7968c64b912dc935e15deb62cc43

- **reasoning_closed**: True

- **response_sha256**: f1127519d02d222f16b76444b73f7267f1dc82e12f4ac3bdee9fed3a05938149

- **scope**: Exact final JSON values, types, ordering and requested standalone format

- **standalone_json**: False

- **status**: values_correct_format_failed

- **structured_answer_present**: True

- **values_match_oracle**: True

### structured / repeat 2 / R2

Prompt SHA-256: `0959c0c3666fcd2709fcb03fde2dd49272bd89fd7df23b6985dab668bc04cc62`. Output-token SHA-256: `26917f4e566442addbf5291df87a9eee949a56aa55e12980c4ca6e696a95fb60`. Ordinary agreement: **False**. Recorded qualification: rejected: ordinary token divergence.


- **correctness_established**: False

- **finish_reason**: eos

- **oracle_sha256**: 3a9d228938f6fd1fe7279c5e9f1ec6e7efdd7968c64b912dc935e15deb62cc43

- **reasoning_closed**: True

- **response_sha256**: f1127519d02d222f16b76444b73f7267f1dc82e12f4ac3bdee9fed3a05938149

- **scope**: Exact final JSON values, types, ordering and requested standalone format

- **standalone_json**: False

- **status**: values_correct_format_failed

- **structured_answer_present**: True

- **values_match_oracle**: True

### prose / repeat 1 / R3

Prompt SHA-256: `a85120228ba51e9a8d97433893be21fddcd72f84c928cf211ecf17cbe14d0d71`. Output-token SHA-256: `7aaa7bf5dc8288127b99282746b999867f038b5094474753192301703870b343`. Ordinary agreement: **False**. Recorded qualification: rejected: ordinary token divergence.


- **automated_status**: manual_review_required

- **correctness_established**: False

- **finish_reason**: eos

- **independent_review**: False

- **manual_checks**: {"kv_cache_mechanics": true, "native_mtp_description": true, "prefill_decode_qualification": true, "rejected_cache_rollback": true, "request_completeness": true, "speculative_acceptance_description": false, "ttft_vs_throughput": true, "wall_time_arithmetic": false}

- **manual_review_sha256**: 2fc115576bb8246db05c75be9c6dbd6d4f0ecae83cc564c1e4461a96b52ed7cd

- **nonempty_answer**: True

- **oracle_sha256**: aced4fee7796b00fcc8dc7332c5173639255570fd4b4892c7ddff79f77fbcc2d

- **reasoning_closed**: True

- **response_sha256**: 6d3104a01c93205d02a1b0f822cf0bf832db56f9eda8e83faf23f18b2968ebe9

- **scope**: Assistant self-review of named mechanism, arithmetic and latency checks in this completed answer; no independent review or model-wide quality score.

- **status**: completed_answer_failed_manual_review

- **structured_answer_present**: False

### prose / repeat 2 / R3

Prompt SHA-256: `a85120228ba51e9a8d97433893be21fddcd72f84c928cf211ecf17cbe14d0d71`. Output-token SHA-256: `7aaa7bf5dc8288127b99282746b999867f038b5094474753192301703870b343`. Ordinary agreement: **False**. Recorded qualification: rejected: ordinary token divergence.


- **automated_status**: manual_review_required

- **correctness_established**: False

- **finish_reason**: eos

- **independent_review**: False

- **manual_checks**: {"kv_cache_mechanics": true, "native_mtp_description": true, "prefill_decode_qualification": true, "rejected_cache_rollback": true, "request_completeness": true, "speculative_acceptance_description": false, "ttft_vs_throughput": true, "wall_time_arithmetic": false}

- **manual_review_sha256**: 9206c2c5ff569e14dda66a55b82cd91860e5fd6f3d5df59adef139af5e45bf16

- **nonempty_answer**: True

- **oracle_sha256**: aced4fee7796b00fcc8dc7332c5173639255570fd4b4892c7ddff79f77fbcc2d

- **reasoning_closed**: True

- **response_sha256**: 6d3104a01c93205d02a1b0f822cf0bf832db56f9eda8e83faf23f18b2968ebe9

- **scope**: Assistant self-review of named mechanism, arithmetic and latency checks in this completed answer; no independent review or model-wide quality score.

- **status**: completed_answer_failed_manual_review

- **structured_answer_present**: False

### code / repeat 1 / R3

Prompt SHA-256: `59d5983c346baa131a2607a801a306241ab46d498b7fe55983039b0b9aef33c1`. Output-token SHA-256: `d7d66fdf315d74166ed5ed1245364b7394569a7d500aba40c6c920870fde8a4a`. Ordinary agreement: **False**. Recorded qualification: rejected: ordinary token divergence.


- **correctness_established**: False

- **finish_reason**: length

- **oracle_sha256**: 39dededfa688a1ac6bce39e6f1d09856eaaa01c42b5803655823eb118f2295c1

- **reasoning_closed**: False

- **response_sha256**: 2299f55b1080b8483b5b950ced00690e243401b061dacf8ae3455390241f7c62

- **scope**: Final schedule compatibility and exact optimal value; generated Python and explanatory proof not executed or fully graded

- **status**: incomplete_reasoning_at_token_cap

- **structured_answer_present**: False

### code / repeat 2 / R3

Prompt SHA-256: `59d5983c346baa131a2607a801a306241ab46d498b7fe55983039b0b9aef33c1`. Output-token SHA-256: `d7d66fdf315d74166ed5ed1245364b7394569a7d500aba40c6c920870fde8a4a`. Ordinary agreement: **False**. Recorded qualification: rejected: ordinary token divergence.


- **correctness_established**: False

- **finish_reason**: length

- **oracle_sha256**: 39dededfa688a1ac6bce39e6f1d09856eaaa01c42b5803655823eb118f2295c1

- **reasoning_closed**: False

- **response_sha256**: 2299f55b1080b8483b5b950ced00690e243401b061dacf8ae3455390241f7c62

- **scope**: Final schedule compatibility and exact optimal value; generated Python and explanatory proof not executed or fully graded

- **status**: incomplete_reasoning_at_token_cap

- **structured_answer_present**: False

### structured / repeat 1 / R3

Prompt SHA-256: `0959c0c3666fcd2709fcb03fde2dd49272bd89fd7df23b6985dab668bc04cc62`. Output-token SHA-256: `c90035bb6829432c1b24460930b863adda3e72c4917002159b39b37ccc6474ff`. Ordinary agreement: **False**. Recorded qualification: rejected: ordinary token divergence.


- **correctness_established**: False

- **finish_reason**: eos

- **oracle_sha256**: 3a9d228938f6fd1fe7279c5e9f1ec6e7efdd7968c64b912dc935e15deb62cc43

- **reasoning_closed**: True

- **response_sha256**: acd9bfe3d2570b9b336eaa744c2e641a172e85bd72782723dd94ccd3d420835b

- **scope**: Exact final JSON values, types, ordering and requested standalone format

- **standalone_json**: False

- **status**: values_correct_format_failed

- **structured_answer_present**: True

- **values_match_oracle**: True

### structured / repeat 2 / R3

Prompt SHA-256: `0959c0c3666fcd2709fcb03fde2dd49272bd89fd7df23b6985dab668bc04cc62`. Output-token SHA-256: `c90035bb6829432c1b24460930b863adda3e72c4917002159b39b37ccc6474ff`. Ordinary agreement: **False**. Recorded qualification: rejected: ordinary token divergence.


- **correctness_established**: False

- **finish_reason**: eos

- **oracle_sha256**: 3a9d228938f6fd1fe7279c5e9f1ec6e7efdd7968c64b912dc935e15deb62cc43

- **reasoning_closed**: True

- **response_sha256**: acd9bfe3d2570b9b336eaa744c2e641a172e85bd72782723dd94ccd3d420835b

- **scope**: Exact final JSON values, types, ordering and requested standalone format

- **standalone_json**: False

- **status**: values_correct_format_failed

- **structured_answer_present**: True

- **values_match_oracle**: True

## Historical release-engine baselines

These earlier results belong to the supported release engine, before the
optimized ordinary research path. They are preserved history, not additional
measurements of the approximately 14.3 tok/s research implementation.

| Workload | Prefill tok/s | Decode wall tok/s | Evidence and scope |
|---|---:|---:|---|
| 2,034-token prompt | 62.761 | 7.660 | [DB610](../../artifacts/prefill-canonical-short-db610-sealed-20260909.json): short-context numerical checks |
| 127,363-token passkey, depth 0.95 | 45.459 | 6.935 | [DB619](../../artifacts/prefill-delivery-db619-sealed-20260912.json): retrieval on this protected prompt |
| 262,144-token E0 | 32.157 | 6.148 | [DB620](../../artifacts/prefill-delivery-db620-sealed-20260912.json): capacity/structural checks; no correctness oracle |

Prefill excludes cold loading/compilation and later decode preparation.
All four protected 128K retrieval depths completed (DB616–619). The 256K run
measured 29.930 GB peak HBM per chip and 3.084 GB minimum headroom (decimal GB).

The [DB621 release smoke test](../../release/user-response-db621-sealed-20260914.json)
used a 19-token prompt, generated 71 tokens including reasoning, returned READY
and ended at EOS. Cold loading/compilation took 2,283.431 seconds; subsequent
local first-token delivery took 9.368 seconds. Its 50.241-second request wall
was instrumented. [Full qualifications](../../release/STATUS.md).
