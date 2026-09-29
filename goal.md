# Goal: the GLM-5.3 TPU engine as a package others can read, test and extend

The GLM-5.3-FP8 inference release (eight hosts, 32 TPU v4 chips; resident and
four-chat modes; measured speed and answers) was finished on 2026-09-22. Its
results are in the [release status](docs/release/STATUS.md); its goal, with the
authorization it carried, is preserved as
`git show archive/research-20260922:goal.md`.

The public-structure release that followed had one goal: **turn the released
engine into a package that a reader can navigate, test and extend, without
changing what runs on the TPU.** It started from the research tree at
`181c013e` (tag `archive/research-20260922`), was finished on 2026-09-27 on
the branch `release/public-structure-20260922` and was released on `main` as
`v1.0.0` on 2026-09-29.

## Acceptance

Every criterion is met; the evidence is in the [handoff](HANDOFF.md#evidence)
and the [release status](docs/release/STATUS.md).

1. **Layout and names.** One package, `glm_tpu/`, laid out like the vLLM and
   tpu-inference projects (entry points, executor, worker, engine, runner,
   models, layers, kernels, model loader, distributed, config), with tests that
   mirror it; public module, class, function and Pallas kernel names; research
   and phase labels removed from docstrings, help texts and error messages,
   except the six messages and the checkpoint-format identifiers that the frozen
   records hold (a few comments and one `info` value are in the backlog). The
   research layer and the migration tools are gone from the tree and kept in the
   history.
2. **Configuration and launch.** Every deployment value comes from one
   untracked, validated site file, whose launch policy admits only a clean
   checkout, on an allowed branch, of a commit pushed to the expected origin.
3. **Equivalence on the CPU.** Every restructuring stage ended with every
   graph-equivalence gate passing against records of `181c013e`: the lowered TPU
   device programs, the CPU execution goldens, the checkpoint identities, the
   import closures, the executed functions and the wire formats. The commit
   messages list the gates run, with exceptions: a few early commits deferred
   the heavy gates to their stage's close (as they state), some commits that
   built the harness name no complete gate run, and the merge of the reference
   model names only its own suite. The device-program, numerics and identity
   records have not changed since the harness was finished (`d54572a3`, before
   the first restructuring commit) and were only ever recorded from the
   `181c013e` production paths; after that the other records changed only in
   separate, reviewed re-baseline commits.
4. **Equivalence on the TPU.** On the same fleet, with the same prepared
   requests, the finished tree and `181c013e` produce identical tokens in
   sequential, concurrent and resident operation, and the programs the fleet
   compiled equal the ones the harness lowers on the CPU.
5. **Declared behaviour changes only.** Each one is stated and tested in its
   commit: the resident stop that no longer overwrites collected records, the
   site file and the launch policy, the remote helper modules, the
   `checkpoint inventory`, `checkpoint verify` and `collect-env` commands, the
   refusal of a directory as request input, the wording of names, help and
   messages.
6. **Documentation and license.** The documentation, these governance files and
   the tests that check them describe the tree as it is; the project's own work is
   under the Apache License 2.0.

## Release

`main` was fast-forwarded to the release branch at `6fa21301`, which the
annotated tag `v1.0.0` marks; the pre-refactor release keeps the tag
`pre-refactor-main-20260922`. The owner decided to make the repository public
after a full-history secret scan ([SECURITY](SECURITY.md#history-scan)).

## Left to the owner

- Schedule the [post-release backlog](HANDOFF.md#post-release-backlog).

## Direction for later work

Keep the equivalence discipline of [AGENTS](AGENTS.md#changing-the-code): a
restructuring change leaves every record equal. A change meant to alter the
device programs, the numerics or host behaviour (a performance, model or
operations change) is declared as such, carries its own tests, needs the owner's
decision on a new baseline where it changes the frozen records (they hold the
`181c013e` programs), and is confirmed by a TPU comparison before it is released.
Prefer a working, evidenced release over further research.
