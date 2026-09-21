# Goal — Private PhD application release

The owner explicitly approved merging the tested four-conversation implementation
into private main and requested a clear, clean main branch. This supersedes the
earlier instruction to hold the candidate after one capped answer. No new model
run is required or authorized by this merge.

Deliver the actual implementation, a readable README, honest measurements and
limits, preserved history, and a compact source/docs archive tied to the final
commit. The public concurrent interface is limited to four conversations with
32,768 total slots each. Four short questions ran concurrently: three completed
correctly and one exhausted its output budget. Do not relabel that as four
correct answers or full32K input validation.

Reuse existing numerical/hardware checks. Run affected host/interface, content,
curation and package checks; perform self-review, verify regional backup, then
commit, push and fast-forward private main. Keep all research refs, originals,
attribution, licenses, frozen MODEL_SOURCE and DB616–621. Do not force-push.

The existing eight-host fleet is already cleaned up. No TPU workload, retry,
resource lifecycle change, environment upgrade, new engine or research campaign.
Any pytest uses JAX_PLATFORMS=cpu. Only gs://driftbench-dsv4-uc, US-CENTRAL2;
respect workload and sync locks. Keep weights, secrets, private prompts, raw
outputs and databases out of Git and the reviewer package.

The slash goal remains paused. No external publication or messages. The current
status and final promotion record are linked in docs/release/STATUS.md.
Earlier goal text is exactly recoverable with
`git show f249469a5ff53c14366ba0a9e14b29144d34e636:goal.md`.
