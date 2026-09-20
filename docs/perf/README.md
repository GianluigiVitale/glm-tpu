# Frozen performance research

The owner ended the campaign on 2026-09-20. No experiment or old queue should resume.

Latest retained ordinary research result: **14.3175 wall decode tok/s** across
the fixed suite; **138.85 prompt tok/s** at 2,034 tokens in the short trial.
Those numbers describe the frozen research suite. The subsequently extracted
ordinary implementation and its completed answer are documented in the
[main README](../../README.md#release-results) and
[release integration history](ordinary-release-20260920.md).

- [Results and decisions](frozen-20260920/RESULTS_AND_DECISIONS.md): what worked, failed, or remained untested.
- [Measurements](frozen-20260920/MEASUREMENTS.md): all preserved paired speed and answer-check rows.
- [Historical notebooks](frozen-20260920/history/README.md): detailed development and public-source reviews.
- [Artifact register](frozen-20260920/EXPERIMENT_REGISTER.md): exact recovery pins and hashes.
- [Freeze operations](frozen-20260920/FREEZE_OPERATIONS.md): stop, archive, cleanup and publication boundary.

Main includes the justified ordinary implementation extracted for the release.
Rejected experimental variants and acquisition source-audit originals stay on
preserved research branches. Protected DB616–621 evidence is unchanged.
