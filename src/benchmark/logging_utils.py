"""
Shared per-run logging — domain-agnostic, like engine.py/metrics.py.

Every run_session() call gets its own transcript (.log, a mirror of everything printed
during the run) and a matching structured summary (.json: model/policy/scenario/metrics),
filed under logs/<model>/<policy>/ so results from different (model, defense-policy)
combinations — gate, baseline, and whatever future conditions (always-ask, coarse-flag,
scoped-ask-once, ...) get built — can be aggregated and compared later without re-parsing
free-text transcripts.
"""

import contextlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def _sanitize(name: str) -> str:
    # Model names carry ":" (e.g. "qwen2.5:14b") which is awkward in a path component.
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def log_paths(model: str, policy: str, scenario_label: str, logs_root: str = "logs") -> tuple[Path, Path]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(logs_root) / _sanitize(model) / _sanitize(policy)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{_sanitize(scenario_label)}_{ts}"
    return directory / f"{stem}.log", directory / f"{stem}.json"


class Tee:
    """Duplicates writes to multiple streams. Used to mirror the existing print()-based
    transcript to both the real console and a per-run log file without having to touch
    every print() call site across engine.py/safe_run.py/baseline_run.py."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def capture_run(model: str, policy: str, scenario_label: str, logs_root: str = "logs"):
    """
    Mirrors everything printed inside the `with` block to a fresh log file (in addition to
    the real console), and yields a `finish(metrics: dict)` callback the caller invokes once
    the run is done to write the matching structured JSON summary.
    """
    log_path, json_path = log_paths(model, policy, scenario_label, logs_root)
    real_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(real_stdout, log_file)
        try:
            def finish(metrics: dict) -> None:
                payload = {
                    "model": model,
                    "policy": policy,
                    "scenario": scenario_label,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **metrics,
                }
                with open(json_path, "w", encoding="utf-8") as json_file:
                    json.dump(payload, json_file, indent=2, ensure_ascii=False)

            yield finish
        finally:
            sys.stdout = real_stdout
