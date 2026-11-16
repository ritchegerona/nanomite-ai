"""Verify security triage answers.

Checks that the output contains a valid severity and category from the allowed
sets, and optionally checks against expected values.
"""
import json
import re
from typing import Any


def extract_json(text: str) -> dict | None:
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


def verify(output: str, meta: dict) -> tuple[bool, str]:
    obj = extract_json(output)
    if not obj:
        return False, "no_json"

    category = obj.get("category", "")
    severity = obj.get("severity", "")
    confidence = obj.get("confidence", 0)

    allowed_categories = meta.get("allowed_categories", [])
    allowed_severities = meta.get("allowed_severities", ["high", "medium", "low"])

    # Check category
    if allowed_categories and category not in allowed_categories:
        return False, f"bad_category:{category}"

    # Check severity
    if severity not in allowed_severities:
        return False, f"bad_severity:{severity}"

    # Check confidence is a number between 0 and 1
    try:
        conf = float(confidence)
        if not (0.0 <= conf <= 1.0):
            return False, f"bad_confidence:{confidence}"
    except (TypeError, ValueError):
        return False, f"bad_confidence:{confidence}"

    # Check expected values if provided
    expected = meta.get("expected", {})
    if expected:
        if "category" in expected and category != expected["category"]:
            return False, f"category_mismatch:{category}!={expected['category']}"
        if "severity" in expected and severity != expected["severity"]:
            return False, f"severity_mismatch:{severity}!={expected['severity']}"

    return True, "ok"