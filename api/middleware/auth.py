"""
api/middleware/auth.py
======================
API key authentication utilities.

Note: Primary auth enforcement is in api/dependencies.py via verify_api_key().
This module contains helper utilities for auth-related tasks.
"""

from __future__ import annotations

import secrets


def constant_time_compare(val1: str, val2: str) -> bool:
    """
    Compare two strings in constant time to prevent timing attacks.
    Used internally by verify_api_key.
    """
    return secrets.compare_digest(val1.encode(), val2.encode())
