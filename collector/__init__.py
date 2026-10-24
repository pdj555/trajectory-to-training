"""Collector package: trajectory schema, collection API, and the live recorder."""

from .recorder import Recorder, scrub_step, scrub_text
from .trajectory import Collector, Trajectory, ToolCall, Verification

__all__ = [
    "Collector",
    "Recorder",
    "Trajectory",
    "ToolCall",
    "Verification",
    "scrub_step",
    "scrub_text",
]
