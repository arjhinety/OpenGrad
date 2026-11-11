"""Run records: a full, redacted record of every recorded run, kept apart from the experiment store.

See ``opengrad.runrecords.record`` for the record and ``python -m opengrad.runrecords --help`` for the commands.
"""

from opengrad.runrecords.record import RECORD_SCHEMA_VERSION, recorded_call, run_child

__all__ = ["RECORD_SCHEMA_VERSION", "recorded_call", "run_child"]
