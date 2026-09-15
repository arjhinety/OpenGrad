"""Local, config-driven human annotation.

The engine is generic: a task configuration (``configs/annotation/*.yaml``) says what is annotated, how
items are displayed, which labels exist and how results are exported. Source datasets are only ever read;
working state lives in SQLite and research output is written as new, hash-pinned JSONL artifacts.

Nothing in this package predicts, suggests or pre-fills a label. Gold labels come from people.
"""

from opengrad.annotation.config import APP_VERSION, TaskConfig, TaskConfigError, load_task_config

__all__ = ["APP_VERSION", "TaskConfig", "TaskConfigError", "load_task_config"]
