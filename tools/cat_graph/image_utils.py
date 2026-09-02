"""Pure helpers for parsing and normalizing image-extraction responses."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import json5


IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

EDGE_ENDPOINT_KEYS = ("source_id", "target_id", "source", "target")


def image_media_type(image_path: Path) -> str:
    """Return the media type supported by the image extraction pipeline."""

    try:
        return IMAGE_MEDIA_TYPES[image_path.suffix.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported image extension: {image_path.suffix or '<none>'}") from exc


def response_content_to_text(content: Any) -> str:
    """Normalize common LangChain/OpenAI response content shapes to text."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                value = part.get("text") if part.get("type") in {None, "text"} else ""
            else:
                value = getattr(part, "text", "")
            if value:
                text_parts.append(str(value))
        return "".join(text_parts)
    return str(content or "")


def _extract_balanced_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    quote_char = ""
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote_char:
                in_string = False
                quote_char = ""
            continue

        if char in {'"', "'"}:
            in_string = True
            quote_char = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def parse_graph_response(content: Any) -> dict[str, Any]:
    """Parse direct, fenced, or prose-wrapped JSON/JSON5 graph output."""

    response_text = response_content_to_text(content).strip()
    if not response_text:
        raise ValueError("Model returned an empty response")

    candidates = [response_text]
    fenced = re.search(r"```(?:json|json5)?\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE)
    if fenced and fenced.group(1):
        candidates.append(fenced.group(1).strip())
    balanced = _extract_balanced_object(response_text)
    if balanced:
        candidates.append(balanced)

    last_error: Exception | None = None
    for candidate in dict.fromkeys(candidates):
        try:
            payload = json5.loads(candidate)
        except Exception as exc:  # json5 raises several parser exception types
            last_error = exc
            continue
        if not isinstance(payload, dict):
            last_error = ValueError("Graph response root must be a JSON object")
            continue

        nodes = payload.get("nodes", [])
        edges = payload.get("edges", payload.get("relationships", []))
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("Graph response 'nodes' and 'edges' must be arrays")
        payload["nodes"] = nodes
        payload["edges"] = edges
        return payload

    raise ValueError(f"Could not parse graph JSON: {last_error or 'no JSON object found'}")


def namespace_generated_node_ids(graph_data: dict[str, Any], namespace: str) -> dict[str, str]:
    """Namespace generic node IDs and update every matching edge endpoint."""

    id_map: dict[str, str] = {}
    for node in graph_data.get("nodes", []):
        if not isinstance(node, dict):
            continue
        original_id = node.get("id")
        if original_id is None:
            continue
        original_text = str(original_id)
        if "node" in original_text.lower() or original_text.isdigit():
            namespaced_id = f"{namespace}_{original_text}"
            node["id"] = namespaced_id
            id_map[original_text] = namespaced_id

    if not id_map:
        return id_map

    for edge in graph_data.get("edges", []):
        if not isinstance(edge, dict):
            continue
        for endpoint_key in EDGE_ENDPOINT_KEYS:
            endpoint = edge.get(endpoint_key)
            replacement = id_map.get(str(endpoint)) if endpoint is not None else None
            if replacement:
                edge[endpoint_key] = replacement
    return id_map
