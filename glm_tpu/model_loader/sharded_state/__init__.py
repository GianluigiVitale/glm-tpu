"""The sharded runtime checkpoint: one file per device slot behind a fixed header, the manifest and
seal that bind the files, verification and loading."""
