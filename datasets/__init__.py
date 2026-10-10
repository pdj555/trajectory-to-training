"""Dataset package: SFT, preference pairs, and splits."""

from .sft import emit_sft, to_sft_row, save_jsonl as save_sft_jsonl
from .preference import build_pairs, save_jsonl as save_pref_jsonl
from .split import stratified_split

__all__ = [
    "emit_sft",
    "to_sft_row",
    "save_sft_jsonl",
    "build_pairs",
    "save_pref_jsonl",
    "stratified_split",
]
