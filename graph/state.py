"""
graph/state.py
==============
Re-exports ReviewState from the canonical schemas module.

All graph nodes import ReviewState from here — never directly from schemas.py —
so that if the import path changes, only this file needs updating.
"""

from schemas import ReviewState, ReviewConfig

__all__ = ["ReviewState", "ReviewConfig"]
