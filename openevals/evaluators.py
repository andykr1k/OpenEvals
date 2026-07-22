from __future__ import annotations

import json
import re
from typing import Any


def evaluate_checks(text: str, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for check in checks:
        kind = check.get("type")
        passed = False
        detail = ""
        try:
            if kind == "exact":
                expected = str(check.get("value", ""))
                actual = text.strip()
                if not check.get("case_sensitive", True):
                    expected, actual = expected.casefold(), actual.casefold()
                passed = actual == expected
            elif kind == "contains":
                needle, haystack = str(check.get("value", "")), text
                if not check.get("case_sensitive", False):
                    needle, haystack = needle.casefold(), haystack.casefold()
                passed = needle in haystack
            elif kind == "regex":
                passed = re.search(str(check["pattern"]), text, re.MULTILINE) is not None
            elif kind == "json":
                json.loads(text)
                passed = True
            elif kind == "json_keys":
                value = json.loads(text)
                keys = check.get("keys", [])
                passed = isinstance(value, dict) and all(key in value for key in keys)
            elif kind == "max_length":
                passed = len(text) <= int(check["characters"])
            else:
                detail = f"Unknown check type: {kind}"
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, re.error) as exc:
            detail = str(exc)
        results.append({"type": kind, "passed": passed, "detail": detail})
    return results
