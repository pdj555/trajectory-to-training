"""Dashboard package: data-quality metrics and reports."""

from .metrics import verification_by_task, length_stats, report

__all__ = ["verification_by_task", "length_stats", "report"]
