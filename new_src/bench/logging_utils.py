"""
Per-run logging: a readable transcript plus machine-readable episode records.

    logs_authmem/<model>/<module>/<condition>/<pair_id>_<variant>_<timestamp>.log
    logs_authmem/<model>/<module>/<condition>/episodes.jsonl

The .jsonl is the file every summary is computed from, so a suite can be re-scored (by
category, by base, by condition) without re-parsing prose, and a partial run is never lost.
"""

from __future__ import annotations

import contextlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from new_src.config import LOGS_ROOT


def _clean(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", value)


def run_dir(model: str, module: str, condition: str) -> Path:
    path = Path(LOGS_ROOT) / _clean(model) / _clean(module) / _clean(condition)
    path.mkdir(parents=True, exist_ok=True)
    return path


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def transcript(model: str, module: str, condition: str, name: str, echo: bool = True):
    """Capture everything an episode prints into its own log file.

    `echo` controls the CONSOLE only. The log file always gets the full transcript — quiet
    mode is about not flooding the terminal during a long sweep, and a run that produced
    empty log files because it was quiet is a run whose evidence was thrown away.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = run_dir(model, module, condition) / f"{_clean(name)}_{stamp}.log"
    original = sys.stdout
    with open(path, "w", encoding="utf-8") as handle:
        sys.stdout = _Tee(original, handle) if echo else handle
        try:
            yield path
        finally:
            sys.stdout = original
            handle.flush()


def write_summary(model: str, module: str, condition: str, lines: list[str]) -> Path:
    """Persist a run's printed summary next to its episode records, so the headline numbers
    survive the terminal scrollback."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = run_dir(model, module, condition) / f"summary_{stamp}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
