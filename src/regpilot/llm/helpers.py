"""Small JSON / prompt helpers shared across LLM backends.

Public names (no leading underscore) because they're imported across module
boundaries inside the package. They stay internal to ``regpilot.llm``; the
package ``__init__`` does not re-export them.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel


def wrap_with_schema(prompt: str, schema: type[BaseModel]) -> str:
    """Append a strict JSON-only instruction with the Pydantic schema.

    Used by the base ``LLMClient.generate_structured`` fallback when a
    backend has no native structured-output API. Providers that do have
    one (Ollama 0.5+, OpenAI, Anthropic) skip this entirely.
    """

    schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    return (
        f"{prompt}\n\n"
        f"Reply with STRICT JSON only matching this schema (no commentary, "
        f"no markdown fence):\n{schema_json}"
    )


def safe_json_obj(raw: str) -> dict[str, Any]:
    """Extract a JSON object from raw model output.

    Two-pass: try the whole string as JSON (the happy path when the model
    behaves and we used a native structured-output API), then fall back to
    finding the first ``{...}`` block (for chatty models that prepend a
    paragraph before their JSON).
    """

    raw = raw.strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract_after(text: str, marker: str) -> str:
    """Return the text immediately after ``marker``, stopping at the next
    blank line.

    The stub LLM uses this to isolate the user's system description from
    the surrounding prompt boilerplate. Without the blank-line stop, the
    stub's keyword scan would match Article 5 / GPAI tokens that appear in
    the prompt's tier definitions and decision rules, not in the actual
    description.
    """

    idx = text.find(marker)
    if idx < 0:
        return ""
    tail = text[idx + len(marker):].lstrip()
    sep = tail.find("\n\n")
    if sep > 0:
        tail = tail[:sep]
    return tail.strip()
