# Run records

One JSON summary per recorded run, written by `python -m opengrad.runrecords export` from the git-ignored
`.run-records/` folder. Each holds the run's command (redacted), commit and whether tracked files differed from it,
the installed library versions, the GPU, the exit status, duration and the sha256 of its full output log. Scripts
record themselves through `opengrad.runrecords.scripts`: a non-zero code a script returns by design records as a
pass with the code kept. The log itself is never committed. With the owner's word, `python -m opengrad.runrecords
upload` keeps a durable copy of each log it may send in a dataset repository, and refuses any repository the Hub
does not report as private; that check is
the real safety boundary. A log holding a credential or a home folder in a form the tests cover, and every
annotation log, goes up only as a redacted copy; any other log goes up byte for byte, and one that is not clean
UTF-8 text stays local. Upload is an allowlist: only a log whose entry point was reviewed and approved is sent
(`opengrad train`, `opengrad evaluate`, the labelling runner and fifteen reviewed scripts), and only when the run
started from reviewed code: a commit pushed to `origin/master` (checked at upload), with nothing it loads and no
other `configs/` file edited, untracked or ignored at the start, and each config it consumes (named in its
arguments; for `opengrad evaluate`, the suite and benchmark configs) present, exactly as the run read it, in
`origin/master`'s history at upload. A run made with a tweaked config uploads once that config is pushed. Any other
stays local and is listed. Even then, a log is
not sent when its run read held-out data, as far as the code shows: its command
runs an entry point that names a held-out file or loads code that does, or an argument (or a `configs/` file it
names) names one, `behavioral-heldout` included outside keys reviewed as only documenting or excluding the set. The exception is a short list of entry points whose console a test pins to status lines
(`PINNED_CONSOLE`): the labelling runner, and fifteen that touch held-out data only to exclude it. Nor is a log
sent when the command cannot be traced to a repository entry point (a shell, pytest, an outside tool): fail
closed, listed for a mapping to be added. The committed summary of a run whose log stays local, or of a pinned
one, keeps only the exception type of its error. The summary records which copy was sent, why, and the sha256 of
both. Redaction is best
effort: the skill `opengrad-experiments-readiness` lists the forms it covers and its known misses, so a run
should not print credentials.

A run that did not pass carries a disposition: the `docs/upstream/` entry it is logged under, or a reason it needs
none. `tests/runrecords/test_committed.py` fails CI on a summary without one, or one that holds an absolute path or
a secret.

`opengrad train` and `opengrad evaluate` record themselves, in process: the log captures what is written through
`sys.stdout`, `sys.stderr` and their `.buffer`, but not output that C extensions (CUDA, NCCL) or child processes
write straight to the file descriptors. Launch GPU runs through the launcher instead
(`python -m opengrad.runrecords run --kind training -- opengrad train ...`): it captures everything, and the
in-process record then steps aside, so each run has one record. Any other command can be recorded with
`python -m opengrad.runrecords run --kind <kind> -- <command>`.

**Remote GPU runs are not yet verified.** A Modal launch (`modal run scripts/modal/<script>.py`) records itself
from its local entrypoint, like any other entry point; if the modal CLI's Python cannot load the recorder from
`src/`, the launch runs unrecorded with a warning. The launch command streams the remote run's output to this
console, and the record captures what reaches `sys.stdout` and `sys.stderr`. The library versions and GPU in
such a record describe this machine, not the remote one, until the remote side writes its own snapshot. No
authorised GPU run has exercised any of this.
