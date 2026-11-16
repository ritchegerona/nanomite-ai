"""Verify a tool-call JSON object against a schema.

Verifies:
- Valid JSON with 'tool' and 'arguments' keys
- Tool name is in allowed list
- Required arguments are present
- Argument types match expected types (if specified in meta)
"""
import json
import re
from typing import Any


def extract_json(text: str) -> dict | None:
    """Extract the first valid JSON object from text."""
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


def _type_matches(value: Any, expected: str) -> bool:
    """Check if a value matches the expected type string."""
    type_map = {
        "str": str,
        "int": int,
        "float": (int, float),
        "bool": bool,
        "list": list,
        "dict": dict,
        "str_or_none": (str, type(None)),
        "int_or_none": (int, type(None)),
    }
    expected_types = type_map.get(expected)
    if expected_types is None:
        return True  # Unknown type, accept
    return isinstance(value, expected_types)


def verify(output: str, meta: dict) -> tuple[bool, str]:
    obj = extract_json(output)
    if not obj:
        return False, "no_json"

    name = obj.get("tool") or obj.get("name") or obj.get("function")
    if isinstance(name, dict):
        name = name.get("name")
    if not name or not isinstance(name, str):
        return False, "no_tool_name"

    args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return False, "args_not_json"
    if not isinstance(args, dict):
        return False, "args_not_object"

    allowed = meta.get("allowed_tools")
    if allowed and name not in allowed:
        return False, f"tool_not_allowed:{name}"

    # Check required args
    for key in meta.get("required_args", []):
        if key not in args:
            return False, f"missing_arg:{key}"

    # Check argument types if schema provided
    arg_schema = meta.get("arg_schema", {})
    for key, expected_type in arg_schema.items():
        if key in args and not _type_matches(args[key], expected_type):
            return False, f"bad_arg_type:{key}:{expected_type}"

    return True, "ok"