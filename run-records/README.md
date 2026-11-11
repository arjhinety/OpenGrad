# Run records

One JSON summary per recorded run, written by `python -m opengrad.runrecords export` from the git-ignored
`.run-records/` folder. Each holds the run's command (redacted), commit and whether tracked files differed from it,
the installed library versions, the GPU, the exit status, duration and the sha256 of its full output log. The log
itself is never committed. With the owner's word, `python -m opengrad.runrecords upload` keeps a durable copy, and
the summary records where.

A run that did not pass carries a disposition: the `docs/upstream/` entry it is logged under, or a reason it needs
none. `tests/runrecords/test_committed.py` fails CI on a summary without one, or one that holds an absolute path or
a secret.

`opengrad train` and `opengrad evaluate` record themselves, in process: the log captures what is written through
`sys.stdout`, `sys.stderr` and their `.buffer`, but not output that C extensions (CUDA, NCCL) or child processes
write straight to the file descriptors. Launch GPU runs through the launcher instead
(`python -m opengrad.runrecords run --kind training -- opengrad train ...`): it captures everything, and the
in-process record then steps aside, so each run has one record. Any other command can be recorded with
`python -m opengrad.runrecords run --kind <kind> -- <command>`.

**Remote GPU runs are not yet verified.** For Modal, the plan is to record the local launch, for example
`python -m opengrad.runrecords run --kind training -- modal run scripts/modal/<script>.py`. The launch command
streams the remote run's output to this console, so the record captures it. The library versions and GPU in
such a record describe this machine, not the remote one, until the remote side writes its own snapshot. No
authorised GPU run has exercised any of this.
