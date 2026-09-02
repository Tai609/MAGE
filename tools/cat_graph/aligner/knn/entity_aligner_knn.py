import json
import json5
import re
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

# Alignment dependencies
from tools.cat_graph.aligner.knn.clip_knn import get_mutual_knn_pairs, center_embeddings
from tools.cat_graph.entity_embedder import ImageTextEmbedder
from models.models import get_model
from prompts.image_prompts import entity_resolution_prompt
from tqdm import tqdm

# Model used for the LLM alignment decision step
ALIGNER_MODEL_CONFIG = 'google_gemini-2.5-flash'
LLM_ALIGNMENT_MAX_ATTEMPTS = 3  # initial attempt + 2 repair retries
LLM_ALIGNMENT_RETRY_BACKOFF_SECONDS = float(os.getenv("CATGRAPH_ALIGNMENT_RETRY_BACKOFF_SECONDS", "3"))
LLM_ALIGNMENT_PROMPT_MAX_CHARS = int(os.getenv("CATGRAPH_ALIGNMENT_PROMPT_MAX_CHARS", "0"))
LLM_ALIGNMENT_PROMPT_LIMIT_ENABLED = LLM_ALIGNMENT_PROMPT_MAX_CHARS > 0
LLM_ALIGNMENT_ENABLE_PROMPT_COMPACTION = str(
    os.getenv("CATGRAPH_ALIGNMENT_ENABLE_PROMPT_COMPACTION", "0")
).strip().lower() in {"1", "true", "yes", "on"}
LLM_ALIGNMENT_MAX_TEXT_ENTITIES = int(os.getenv("CATGRAPH_ALIGNMENT_MAX_TEXT_ENTITIES", "80"))
LLM_ALIGNMENT_MAX_IMAGE_ENTITIES = int(os.getenv("CATGRAPH_ALIGNMENT_MAX_IMAGE_ENTITIES", "80"))
LLM_ALIGNMENT_MAX_SUGGESTIONS = int(os.getenv("CATGRAPH_ALIGNMENT_MAX_SUGGESTIONS", "120"))
LLM_ALIGNMENT_ENTITY_FIELD_MAX_CHARS = int(os.getenv("CATGRAPH_ALIGNMENT_ENTITY_FIELD_MAX_CHARS", "140"))
SUGGESTION_FALLBACK_MIN_SCORE = 0.15

logger = logging.getLogger(__name__)

# Lazily initialized global embedder
_CLIP_EMBEDDER = None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _get_ablation_flags() -> Dict[str, bool]:
    return {
        "disable_structural_hints": _env_flag("CATGRAPH_ABLATION_DISABLE_STRUCTURAL_HINTS", False),
        "disable_figure_grounding": _env_flag("CATGRAPH_ABLATION_DISABLE_FIGURE_GROUNDING", False),
        "disable_retry": _env_flag("CATGRAPH_ABLATION_DISABLE_RETRY", False),
        "disable_fallback": _env_flag("CATGRAPH_ABLATION_DISABLE_FALLBACK", False),
    }


def _build_task_instructions(figure_grounding_enabled: bool) -> str:
    if figure_grounding_enabled:
        return (
            "### Task Instructions:\n"
            "1. Match 'Text Entities' to 'Image Entities' using 'method', 'summary', and 'detected_figure_refs'.\n"
            "2. PRIORITIZE 'detected_figure_refs':\n"
            "   - I have strictly filtered 'detected_figure_refs' to ONLY include Main Figures (e.g., 'Figure 1a').\n"
            "   - Match these refs to Image Entity 'filenames' or 'names'.\n"
            "3. HANDLING MIXED REFERENCES:\n"
            "   - The original text snippet might still mention 'Figure S...' (Supporting Info) in 'evidence_context'.\n"
            "   - IGNORE ALL 'Figure S...' references. The Image Entity list DOES NOT contain any Supporting Info images.\n"
            "   - If a Text Node refers to 'Figure 1e' AND 'Figure S4', align it ONLY to the 'Figure 1' Image Node. Ignore the 'S4' part.\n"
            "4. Return the JSON alignment_map.\n"
            "5. Output contract (strict): Return ONLY one valid JSON object, no markdown fences, no comments, no explanation."
        )
    return (
        "### Task Instructions:\n"
        "1. Match 'Text Entities' to 'Image Entities' using 'method', 'summary', 'evidence_context', and semantic consistency.\n"
        "2. Do NOT rely on figure-reference grounding; judge the match only from entity meaning and local textual context.\n"
        "3. Return the JSON alignment_map.\n"
        "4. Output contract (strict): Return ONLY one valid JSON object, no markdown fences, no comments, no explanation."
    )

def get_clip_embedder():
    global _CLIP_EMBEDDER
    if _CLIP_EMBEDDER is None:
        _CLIP_EMBEDDER = ImageTextEmbedder()
    return _CLIP_EMBEDDER

def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)

def _extract_balanced_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    quote_char = ""
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == quote_char:
                in_string = False
                quote_char = ""
            continue

        if ch in ('"', "'"):
            in_string = True
            quote_char = ch
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None

def _collect_json_candidates(response_text: str) -> List[str]:
    candidates: List[str] = []
    fenced = re.findall(r"```(?:json|json5)?\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE)
    candidates.extend(fenced)

    balanced = _extract_balanced_json_object(response_text)
    if balanced:
        candidates.append(balanced)

    stripped = response_text.strip()
    if stripped:
        candidates.append(stripped)

    deduped: List[str] = []
    seen: Set[str] = set()
    for c in candidates:
        c_norm = c.strip()
        if c_norm and c_norm not in seen:
            seen.add(c_norm)
            deduped.append(c_norm)
    return deduped

def _normalize_alignment_map(
    raw_map: Any,
    valid_image_ids: Set[str],
    valid_text_ids: Set[str],
) -> Dict[str, Optional[str]]:
    if not isinstance(raw_map, dict):
        return {}

    normalized: Dict[str, Optional[str]] = {}
    for raw_img_id, raw_text_id in raw_map.items():
        img_id = str(raw_img_id).strip()
        if not img_id or img_id not in valid_image_ids:
            continue

        if raw_text_id is None:
            normalized[img_id] = None
            continue

        text_id = str(raw_text_id).strip()
        if not text_id or text_id.lower() in {"none", "null"}:
            normalized[img_id] = None
            continue

        if text_id in valid_text_ids:
            normalized[img_id] = text_id

    return normalized

def _parse_alignment_mapping(
    response_text: str,
    valid_image_ids: Set[str],
    valid_text_ids: Set[str],
) -> Tuple[bool, Dict[str, Optional[str]]]:
    for candidate in _collect_json_candidates(response_text):
        try:
            parsed = json5.loads(candidate)
        except Exception:
            continue

        if isinstance(parsed, dict):
            if "alignment_map" in parsed:
                if isinstance(parsed["alignment_map"], dict):
                    mapping = _normalize_alignment_map(parsed["alignment_map"], valid_image_ids, valid_text_ids)
                    return True, mapping
                continue

            direct_map = _normalize_alignment_map(parsed, valid_image_ids, valid_text_ids)
            if direct_map:
                return True, direct_map

    # Best-effort recovery for truncated responses: extract completed key/value pairs
    # from "alignment_map" block even if the closing braces are missing.
    anchor = response_text.find('"alignment_map"')
    segment = response_text[anchor:] if anchor >= 0 else response_text
    pair_pattern = re.compile(r'"([^"\\]+)"\s*:\s*(null|"([^"\\]*)")', re.IGNORECASE)
    recovered_raw: Dict[str, Optional[str]] = {}
    for m in pair_pattern.finditer(segment):
        img_id = (m.group(1) or "").strip()
        raw_token = (m.group(2) or "").strip().lower()
        if raw_token == "null":
            recovered_raw[img_id] = None
        else:
            text_id = (m.group(3) or "").strip()
            recovered_raw[img_id] = text_id

    recovered_map = _normalize_alignment_map(recovered_raw, valid_image_ids, valid_text_ids)
    if recovered_map:
        logger.warning(
            "[LLM Merge] Recovered partial alignment_map from truncated response "
            f"(pairs={len(recovered_map)})."
        )
        return True, recovered_map

    return False, {}

def _build_retry_prompt(base_prompt: str, previous_response: str, attempt: int) -> str:
    return (
        f"{base_prompt}\n\n"
        f"### Retry Correction ({attempt})\n"
        "Your previous output could not be parsed.\n"
        "Return ONLY one valid JSON object with this exact top-level shape:\n"
        '{"alignment_map": {"<image_entity_id>": "<text_entity_id>", "<image_entity_id_2>": null}}\n'
        "Rules:\n"
        "- No markdown code fences.\n"
        "- No explanations.\n"
        "- No comments.\n"
        "- Keys must be image_entity_id values from input.\n"
        "- Values must be text_entity_id from input or null.\n"
        "Previous output (for debugging, do not repeat):\n"
        f"{previous_response[:1200]}"
    )

def _get_alignment_debug_dir() -> Path:
    configured = os.getenv("CATGRAPH_ALIGNMENT_DEBUG_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.cwd() / "output_extract" / "alignment_debug").resolve()

def _save_alignment_debug_response(response_text: str, reason: str, attempt: int) -> Optional[Path]:
    try:
        debug_dir = _get_alignment_debug_dir()
        debug_dir.mkdir(parents=True, exist_ok=True)
        safe_reason = re.sub(r"[^a-zA-Z0-9_-]+", "_", reason).strip("_") or "unknown"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = debug_dir / f"alignment_raw_{stamp}_pid{os.getpid()}_a{attempt}_{safe_reason}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response_text)
        return file_path
    except Exception as e:
        logger.warning(f"[LLM Merge] Failed to save raw alignment response: {e}")
        return None

def _build_fallback_mapping_from_suggestions(
    suggestions: List[Dict[str, Any]],
    valid_image_ids: Set[str],
    valid_text_ids: Set[str],
    min_score: float = SUGGESTION_FALLBACK_MIN_SCORE,
) -> Dict[str, str]:
    ranked: List[Tuple[float, str, str]] = []
    for s in suggestions:
        img_id = str(s.get("image_entity_id", "")).strip()
        text_id = str(s.get("suggested_text_id", "")).strip()
        if not img_id or not text_id:
            continue
        if img_id not in valid_image_ids or text_id not in valid_text_ids:
            continue
        try:
            score = float(s.get("similarity_score", 0.0))
        except Exception:
            continue
        if score < min_score:
            continue
        ranked.append((score, img_id, text_id))

    ranked.sort(key=lambda x: x[0], reverse=True)

    mapping: Dict[str, str] = {}
    used_text_ids: Set[str] = set()
    for score, img_id, text_id in ranked:
        if img_id in mapping:
            continue
        if text_id in used_text_ids:
            continue
        mapping[img_id] = text_id
        used_text_ids.add(text_id)

    return mapping

def _choose_fallback_min_score(suggestions_count: int) -> float:
    # Lower threshold when suggestions are scarce to reduce zero-merge cases.
    if suggestions_count <= 10:
        return 0.12
    if suggestions_count <= 30:
        return 0.14
    return SUGGESTION_FALLBACK_MIN_SCORE

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

def _truncate_text(value: Any, max_chars: int) -> Any:
    if not isinstance(value, str):
        return value
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3] + "..."

def _compact_item_strings(item: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    compacted: Dict[str, Any] = {}
    for k, v in item.items():
        if isinstance(v, str):
            compacted[k] = _truncate_text(v, max_chars)
        elif isinstance(v, list):
            new_list = []
            for x in v:
                if isinstance(x, str):
                    new_list.append(_truncate_text(x, max_chars))
                else:
                    new_list.append(x)
            compacted[k] = new_list
        else:
            compacted[k] = v
    return compacted

def _build_priority_ids_from_suggestions(
    suggestions: List[Dict[str, Any]],
    image_limit: int,
    text_limit: int,
) -> Tuple[List[str], List[str]]:
    ranked = sorted(
        suggestions,
        key=lambda s: _safe_float(s.get("similarity_score", 0.0), 0.0),
        reverse=True,
    )
    image_ids: List[str] = []
    text_ids: List[str] = []
    seen_img: Set[str] = set()
    seen_txt: Set[str] = set()

    for s in ranked:
        img_id = str(s.get("image_entity_id", "")).strip()
        txt_id = str(s.get("suggested_text_id", "")).strip()
        if img_id and img_id not in seen_img and len(image_ids) < image_limit:
            seen_img.add(img_id)
            image_ids.append(img_id)
        if txt_id and txt_id not in seen_txt and len(text_ids) < text_limit:
            seen_txt.add(txt_id)
            text_ids.append(txt_id)
        if len(image_ids) >= image_limit and len(text_ids) >= text_limit:
            break

    return image_ids, text_ids

def _select_with_priority(
    items: List[Dict[str, Any]],
    priority_ids: List[str],
    limit: int,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []

    id_map = {str(it.get("id", "")).strip(): it for it in items if isinstance(it, dict)}
    selected: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for pid in priority_ids:
        if pid in id_map and pid not in seen:
            selected.append(id_map[pid])
            seen.add(pid)
            if len(selected) >= limit:
                return selected

    for it in items:
        iid = str(it.get("id", "")).strip()
        if iid in seen:
            continue
        selected.append(it)
        if len(selected) >= limit:
            break
    return selected

# ==========================================
# 1. Node types eligible for alignment
# ==========================================
ALIGN_TARGET_TYPES = [
    "characterization", 
    "characterization_data", 
    "image", 
    "figure", 
    "spectrum", 
    "microscopy",
    "plot",
    "data",
    "chemical",
    "material",
    "sample",
    "test"
]

def _is_target_type(node):
    """Return whether a node belongs to an alignable type."""
    n_type = node.get("type", "").lower()
    return any(k in n_type for k in ALIGN_TARGET_TYPES)

def _get_node_text_representation(node):
    """Build the most informative text representation for CLIP."""
    parts = []
    if node.get("method_name"): parts.append(str(node.get("method_name"))) 
    if node.get("name"): parts.append(str(node.get("name")))
    if node.get("characterization_summary"): 
        parts.append(str(node.get("characterization_summary")))
    if node.get("aliases"):
        parts.extend(node.get("aliases"))
        
    full_text = " ".join(parts)
    return full_text.strip() if full_text.strip() else node.get("id", "")

def align_and_merge_graphs(
    text_graph: Dict[str, Any], 
    image_graph: Dict[str, Any], 
    model: Any
) -> Dict[str, Any]:
    """Align entities with mutual kNN hints and a context-aware LLM decision."""
    ablation_flags = _get_ablation_flags()
    disable_structural_hints = ablation_flags["disable_structural_hints"]
    disable_figure_grounding = ablation_flags["disable_figure_grounding"]
    disable_retry = ablation_flags["disable_retry"]
    disable_fallback = ablation_flags["disable_fallback"]

    logger.info(
        "[LLM Merge] Ablation flags: structural_hints=%s figure_grounding=%s retry=%s fallback=%s",
        "off" if disable_structural_hints else "on",
        "off" if disable_figure_grounding else "on",
        "off" if disable_retry else "on",
        "off" if disable_fallback else "on",
    )

    # ----------------------------------------------------
    # 2. Select candidate nodes for alignment
    # ----------------------------------------------------
    
    # Retain all text nodes with target types. A snippet may mention both a
    # supplementary figure and a main-text figure, so do not filter here.
    text_nodes = [n for n in text_graph.get("nodes", []) if _is_target_type(n)]
    valid_text_ids: Set[str] = {str(n.get("id")).strip() for n in text_nodes if n.get("id")}
    
    # Image nodes are already limited to the user-supplied figure set.
    image_nodes = [n for n in image_graph.get("nodes", []) if _is_target_type(n)]
    valid_image_ids: Set[str] = {str(n.get("id")).strip() for n in image_nodes if n.get("id")}
    
    if not image_nodes:
        logger.info("No alignable image nodes found. Skipping alignment.")
        return _simple_merge(text_graph, image_graph)

    # ----------------------------------------------------
    # 3. Generate structural suggestions with CLIP mutual kNN
    # ----------------------------------------------------
    suggestions = []
    if disable_structural_hints:
        logger.info("[LLM Merge] Structural suggestion generation disabled by ablation flag.")
    else:
        try:
            embedder = get_clip_embedder()
            logger.info("Preparing text representations for alignment...")
            text_inputs = [_get_node_text_representation(n) for n in tqdm(text_nodes, desc="Preparing text nodes", unit="node")]

            img_paths = []
            for n in tqdm(image_nodes, desc="Preparing image paths", unit="img"):
                path = n.get("_full_image_path", n.get("source_image_file"))
                img_paths.append(path)

            if text_inputs and img_paths:
                logger.info(f"Computing embeddings for {len(text_inputs)} texts and {len(img_paths)} images...")
                text_feats, img_feats = embedder.get_embeddings(texts=text_inputs, image_paths=img_paths)

                if text_feats is not None and img_feats is not None:
                    text_feats = center_embeddings(text_feats)
                    img_feats = center_embeddings(img_feats)

                    mutual_pairs = get_mutual_knn_pairs(text_feats, img_feats, topk=2, similarity_threshold=0.1)

                    for pair in tqdm(mutual_pairs, desc="Generating alignment suggestions", unit="pair"):
                        t_node = text_nodes[pair['text_idx']]
                        i_node = image_nodes[pair['img_idx']]
                        t_display = t_node.get("method_name", t_node.get("name", t_node["id"]))

                        suggestions.append({
                            "image_entity_id": i_node["id"],
                            "suggested_text_match": t_display,
                            "suggested_text_id": t_node["id"],
                            "confidence_type": f"Mutual-kNN (Score={pair['score']:.2f})",
                            "similarity_score": round(float(pair['score']), 3)
                        })
                    logger.info(f"Generated {len(suggestions)} structural suggestions.")

        except Exception as e:
            logger.error(f"Structure alignment calculation failed: {e}. Proceeding with LLM only.", exc_info=True)

    # ----------------------------------------------------
    # 4. Resolve candidates with an LLM using the available context
    # ----------------------------------------------------
    mapping = {}
    alignment_pairs_from_llm = 0
    alignment_pairs_from_fallback = 0
    alignment_retry_count = 0
    llm_attempts_used = 0

    # Prefer the dedicated alignment model; fall back to the supplied model.
    aligner_model = None
    try:
        logger.info(f"Initializing aligner model for LLM decision: {ALIGNER_MODEL_CONFIG}")
        aligner_model = get_model(model=ALIGNER_MODEL_CONFIG, temperature=0)
    except Exception as e:
        logger.warning(f"Failed to initialize aligner model '{ALIGNER_MODEL_CONFIG}': {e}. Falling back to passed model.")
        aligner_model = model

    if not aligner_model:
        logger.warning("No aligner model available for LLM decision. Skipping LLM alignment, using simple merge.")
        return _simple_merge(text_graph, image_graph)

    if suggestions or (text_nodes and image_nodes):
        try:
            # A. Build a compact text-entity summary with figure grounding.
            text_summary_enhanced = []
            for n in text_nodes:
                item = {
                    "id": n["id"], 
                    "type": n.get("type", "unknown")
                }
                
                if n.get("method_name") or "characterization" in n.get("type", "").lower():
                    item["method"] = n.get("method_name", "Unknown Method")
                    if n.get("characterization_summary"):
                        item["summary"] = n.get("characterization_summary")[:200]

                    snippet = n.get("evidence_snippet", "")
                    if snippet:
                        item["evidence_context"] = snippet[:150]
                        if not disable_figure_grounding:
                            raw_refs = re.findall(r"(Fig(?:ure)?\.?\s?S?\d+[a-z]?)", snippet, re.IGNORECASE)
                            if raw_refs:
                                valid_refs = [r for r in raw_refs if not re.search(r"S\d", r, re.IGNORECASE)]
                                if valid_refs:
                                    item["detected_figure_refs"] = list(set(valid_refs))
                            
                else:
                    item["name"] = n.get("name", "Unnamed")
                    if n.get("aliases"):
                        item["aliases"] = n.get("aliases")
                
                text_summary_enhanced.append(item)

            # B. Build an image-entity summary.
            image_summary = []
            for n in image_nodes:
                img_item = { 
                    "id": n["id"], 
                    "name": n.get("name"), 
                    "type": n.get("type")
                }
                
                # Retain the basename of the source image.
                src = n.get("source_image_file", "")
                if src:
                    # Normalize either POSIX or Windows path separators.
                    img_item["filename"] = src.split("/")[-1].split("\\")[-1]

                # Include a concise visual description when available.
                if n.get("properties", {}).get("description"):
                    img_item["ocr_desc"] = n.get("properties", {}).get("description")[:150]
                elif n.get("properties", {}).get("technique"):
                    img_item["technique"] = n.get("properties", {}).get("technique")
                    
                image_summary.append(img_item)
            
            suggestions_text = json.dumps(suggestions, indent=2) if suggestions else "None"

            logger.info(f"[LLM Merge] Text Entities: {len(text_summary_enhanced)}, Image Entities: {len(image_summary)}, Structural Suggestions: {len(suggestions)}")
            logger.debug(f"[LLM Merge] Text Summary: {json.dumps(text_summary_enhanced, ensure_ascii=False, indent=2)}")
            logger.debug(f"[LLM Merge] Image Summary: {json.dumps(image_summary, ensure_ascii=False, indent=2)}")

            # C. Build the context-rich resolver prompt.
            final_prompt = (
                f"{entity_resolution_prompt.format(TEXT_ENTITIES_JSON=json.dumps(text_summary_enhanced), IMAGE_ENTITIES_JSON=json.dumps(image_summary))}\n\n"
                f"### Structural Alignment Hints (High Confidence):\n"
                f"{suggestions_text}\n\n"
                f"{_build_task_instructions(not disable_figure_grounding)}"
            )

            if (
                LLM_ALIGNMENT_PROMPT_LIMIT_ENABLED
                and LLM_ALIGNMENT_ENABLE_PROMPT_COMPACTION
                and len(final_prompt) > LLM_ALIGNMENT_PROMPT_MAX_CHARS
            ):
                logger.warning(
                    f"[LLM Merge] Prompt length {len(final_prompt)} exceeds limit "
                    f"{LLM_ALIGNMENT_PROMPT_MAX_CHARS}; applying compact prompt strategy."
                )

                text_priority_ids, image_priority_ids = _build_priority_ids_from_suggestions(
                    suggestions=suggestions,
                    image_limit=max(0, LLM_ALIGNMENT_MAX_IMAGE_ENTITIES),
                    text_limit=max(0, LLM_ALIGNMENT_MAX_TEXT_ENTITIES),
                )
                prompt_text_items = _select_with_priority(
                    text_summary_enhanced,
                    text_priority_ids,
                    max(1, LLM_ALIGNMENT_MAX_TEXT_ENTITIES),
                )
                prompt_image_items = _select_with_priority(
                    image_summary,
                    image_priority_ids,
                    max(1, LLM_ALIGNMENT_MAX_IMAGE_ENTITIES),
                )

                allowed_img_ids = {
                    str(x.get("id", "")).strip()
                    for x in prompt_image_items
                    if isinstance(x, dict) and x.get("id")
                }
                allowed_txt_ids = {
                    str(x.get("id", "")).strip()
                    for x in prompt_text_items
                    if isinstance(x, dict) and x.get("id")
                }
                ranked_suggestions = sorted(
                    suggestions,
                    key=lambda s: _safe_float(s.get("similarity_score", 0.0), 0.0),
                    reverse=True,
                )
                prompt_suggestions = [
                    s
                    for s in ranked_suggestions
                    if str(s.get("image_entity_id", "")).strip() in allowed_img_ids
                    and str(s.get("suggested_text_id", "")).strip() in allowed_txt_ids
                ][: max(0, LLM_ALIGNMENT_MAX_SUGGESTIONS)]

                prompt_text_items = [
                    _compact_item_strings(x, LLM_ALIGNMENT_ENTITY_FIELD_MAX_CHARS)
                    for x in prompt_text_items
                ]
                prompt_image_items = [
                    _compact_item_strings(x, LLM_ALIGNMENT_ENTITY_FIELD_MAX_CHARS)
                    for x in prompt_image_items
                ]
                prompt_suggestions = [
                    _compact_item_strings(x, LLM_ALIGNMENT_ENTITY_FIELD_MAX_CHARS)
                    for x in prompt_suggestions
                ]

                def _render_alignment_prompt(
                    text_items: List[Dict[str, Any]],
                    image_items: List[Dict[str, Any]],
                    suggestion_items: List[Dict[str, Any]],
                ) -> str:
                    text_json = json.dumps(text_items, ensure_ascii=False, separators=(",", ":"))
                    image_json = json.dumps(image_items, ensure_ascii=False, separators=(",", ":"))
                    suggestions_json = (
                        json.dumps(suggestion_items, ensure_ascii=False, separators=(",", ":"))
                        if suggestion_items else "None"
                    )
                    return (
                        f"{entity_resolution_prompt.format(TEXT_ENTITIES_JSON=text_json, IMAGE_ENTITIES_JSON=image_json)}\n\n"
                        f"### Structural Alignment Hints (High Confidence):\n"
                        f"{suggestions_json}\n\n"
                        f"{_build_task_instructions(not disable_figure_grounding)}"
                    )

                final_prompt = _render_alignment_prompt(prompt_text_items, prompt_image_items, prompt_suggestions)
                if LLM_ALIGNMENT_PROMPT_LIMIT_ENABLED and len(final_prompt) > LLM_ALIGNMENT_PROMPT_MAX_CHARS:
                    prompt_suggestions = []
                    final_prompt = _render_alignment_prompt(prompt_text_items, prompt_image_items, prompt_suggestions)
                    while (
                        LLM_ALIGNMENT_PROMPT_LIMIT_ENABLED
                        and len(final_prompt) > LLM_ALIGNMENT_PROMPT_MAX_CHARS
                        and (len(prompt_image_items) > 20 or len(prompt_text_items) > 20)
                    ):
                        if len(prompt_image_items) >= len(prompt_text_items) and len(prompt_image_items) > 20:
                            prompt_image_items = prompt_image_items[: max(20, len(prompt_image_items) - 10)]
                        elif len(prompt_text_items) > 20:
                            prompt_text_items = prompt_text_items[: max(20, len(prompt_text_items) - 10)]
                        final_prompt = _render_alignment_prompt(
                            prompt_text_items,
                            prompt_image_items,
                            prompt_suggestions,
                        )

                logger.warning(
                    f"[LLM Merge] Final prompt length={len(final_prompt)}, "
                    f"text_entities={len(prompt_text_items)}, image_entities={len(prompt_image_items)}"
                )
            elif LLM_ALIGNMENT_PROMPT_LIMIT_ENABLED and len(final_prompt) > LLM_ALIGNMENT_PROMPT_MAX_CHARS:
                logger.warning(
                    f"[LLM Merge] Prompt length {len(final_prompt)} exceeds limit "
                    f"{LLM_ALIGNMENT_PROMPT_MAX_CHARS}, but prompt compaction is disabled."
                )

            logger.debug(f"[LLM Merge] Full prompt:\n{final_prompt}")

            llm_parsed_ok = False
            last_response_text = ""
            last_invoke_error: Optional[Exception] = None

            max_attempts = 1 if disable_retry else max(1, int(LLM_ALIGNMENT_MAX_ATTEMPTS))
            for attempt in range(1, max_attempts + 1):
                llm_attempts_used = attempt
                prompt_for_attempt = final_prompt
                if attempt > 1:
                    prompt_for_attempt = _build_retry_prompt(final_prompt, last_response_text, attempt - 1)

                logger.info(
                    f"[LLM Merge] Sending prompt to LLM (attempt {attempt}/{max_attempts}, "
                    f"length: {len(prompt_for_attempt)} chars)"
                )
                try:
                    response = aligner_model.invoke(prompt_for_attempt)
                    response_text = _content_to_text(getattr(response, "content", ""))
                    last_response_text = response_text
                    last_invoke_error = None
                except Exception as invoke_err:
                    last_invoke_error = invoke_err
                    logger.warning(
                        f"[LLM Merge] LLM invoke failed (attempt {attempt}/{max_attempts}): {invoke_err}"
                    )
                    debug_path = _save_alignment_debug_response(str(invoke_err), "invoke_error", attempt)
                    if debug_path:
                        logger.warning(f"[LLM Merge] Invoke error saved to: {debug_path}")
                    if attempt < max_attempts:
                        time.sleep(max(0.0, LLM_ALIGNMENT_RETRY_BACKOFF_SECONDS) * attempt)
                        continue
                    break

                logger.info(
                    f"[LLM Merge] Received LLM response (attempt {attempt}/{max_attempts}, "
                    f"length: {len(response_text)} chars)"
                )
                logger.debug(f"[LLM Merge] Full LLM response (attempt {attempt}):\n{response_text}")

                parsed_ok, parsed_mapping = _parse_alignment_mapping(response_text, valid_image_ids, valid_text_ids)
                if parsed_ok:
                    if len(parsed_mapping) == 0:
                        debug_path = _save_alignment_debug_response(response_text, "parsed_empty", attempt)
                        if debug_path:
                            logger.warning(
                                f"[LLM Merge] Parsed JSON but alignment_map is empty after ID validation "
                                f"(attempt {attempt}/{max_attempts}). "
                                f"Raw response saved to: {debug_path}"
                            )
                        else:
                            logger.warning(
                                f"[LLM Merge] Parsed JSON but alignment_map is empty after ID validation "
                                f"(attempt {attempt}/{max_attempts})."
                            )
                        if attempt < max_attempts:
                            continue
                    else:
                        mapping = parsed_mapping
                        llm_parsed_ok = True
                        mapped_non_null = sum(1 for v in mapping.values() if v)
                        alignment_pairs_from_llm = mapped_non_null
                        logger.info(
                            f"[LLM Merge] Parsed alignment_map successfully: "
                            f"{mapped_non_null} mapped pairs, {len(mapping)} total entries."
                        )
                        break

                debug_path = _save_alignment_debug_response(response_text, "parse_failed", attempt)
                if debug_path:
                    logger.warning(
                        f"[LLM Merge] Could not parse Alignment JSON (attempt {attempt}/{max_attempts}). "
                        f"Raw response saved to: {debug_path}"
                    )
                else:
                    logger.warning(
                        f"[LLM Merge] Could not parse Alignment JSON (attempt {attempt}/{max_attempts})."
                    )

            if not llm_parsed_ok:
                if last_invoke_error is not None:
                    logger.warning(
                        "[LLM Merge] All invoke attempts failed; mapping will rely on fallback strategy if available."
                    )
                else:
                    logger.warning(
                        "[LLM Merge] All JSON parse attempts failed; mapping will rely on fallback strategy if available."
                    )

            alignment_retry_count = max(0, llm_attempts_used - 1)
            mapped_non_null = sum(1 for v in mapping.values() if v)
            if mapped_non_null == 0 and suggestions and not disable_fallback:
                fallback_min_score = _choose_fallback_min_score(len(suggestions))
                fallback_mapping = _build_fallback_mapping_from_suggestions(
                    suggestions=suggestions,
                    valid_image_ids=valid_image_ids,
                    valid_text_ids=valid_text_ids,
                    min_score=fallback_min_score,
                )
                if fallback_mapping:
                    mapping = fallback_mapping
                    alignment_pairs_from_fallback = len(fallback_mapping)
                    debug_path = _save_alignment_debug_response(last_response_text, "fallback_applied", 0)
                    if debug_path:
                        logger.warning(
                            f"[LLM Merge] Applied suggestion fallback mapping: {len(fallback_mapping)} pairs "
                            f"(min_similarity={fallback_min_score}). "
                            f"Last raw response saved to: {debug_path}"
                        )
                    else:
                        logger.warning(
                            f"[LLM Merge] Applied suggestion fallback mapping: {len(fallback_mapping)} pairs "
                            f"(min_similarity={fallback_min_score})."
                        )
                else:
                    debug_path = _save_alignment_debug_response(last_response_text, "empty_or_no_fallback", 0)
                    if debug_path:
                        logger.warning(
                            f"[LLM Merge] No mapped pairs from LLM and no usable fallback pairs. "
                            f"Last raw response saved to: {debug_path}"
                        )
                    else:
                        logger.warning("[LLM Merge] No mapped pairs from LLM and no usable fallback pairs.")
            elif mapped_non_null == 0 and suggestions and disable_fallback:
                logger.warning("[LLM Merge] Suggestion fallback disabled by ablation flag.")

        except Exception as e:
            logger.error(f"LLM Entity Resolution failed: {e}")

    # ----------------------------------------------------
    # 5. Merge aligned image evidence into text nodes.
    # ----------------------------------------------------
    final_graph = _deep_copy_graph(text_graph)
    text_node_map = {n["id"]: n for n in final_graph["nodes"]}

    merged_count = 0
    added_count = 0
    for img_node in tqdm(image_graph.get("nodes", []), desc="Merging aligned nodes", unit="node"):
        original_id = img_node["id"]

        if original_id in mapping and mapping[original_id]:
            target_text_id = mapping[original_id]
            target_node = text_node_map.get(target_text_id)

            if target_node:
                # Record the image provenance path.
                if "related_images" not in target_node: target_node["related_images"] = []
                if "source_image_file" in img_node:
                     if img_node["source_image_file"] not in target_node["related_images"]:
                        target_node["related_images"].append(img_node["source_image_file"])

                # Mount the full image evidence on the matched text node.
                if "visual_evidence" not in target_node: target_node["visual_evidence"] = []

                evidence_data = img_node.copy()
                evidence_data.pop("id", None)
                evidence_data.pop("_full_image_path", None)

                target_node["visual_evidence"].append(evidence_data)

                logger.debug(f"Merged Image Node '{original_id}' into Text Node '{target_text_id}' (Deep Mount)")
                merged_count += 1
            else:
                final_graph["nodes"].append(img_node)
                added_count += 1
        else:
            final_graph["nodes"].append(img_node)
            added_count += 1

    logger.info(f"Merge complete: {merged_count} nodes merged, {added_count} nodes added as new entities")

    # 6. Merge image edges after applying the alignment mapping.
    for edge in image_graph.get("edges", []):
        src = edge.get("source_id")
        tgt = edge.get("target_id")
        if src in mapping and mapping[src]: edge["source_id"] = mapping[src]
        if tgt in mapping and mapping[tgt]: edge["target_id"] = mapping[tgt]
        final_graph["edges"].append(edge)

    final_graph["_alignment_summary"] = {
        "alignment_pairs_from_llm": int(alignment_pairs_from_llm),
        "alignment_pairs_from_fallback": int(alignment_pairs_from_fallback),
        "alignment_retry_count": int(alignment_retry_count),
        "ablation_flags": ablation_flags,
    }

    return final_graph

def _deep_copy_graph(g):
    import copy
    res = copy.deepcopy(g)
    if "nodes" not in res: res["nodes"] = []
    if "edges" not in res: res["edges"] = []
    return res

def _simple_merge(g1, g2):
    import copy
    result = copy.deepcopy(g1)
    if "nodes" not in result: result["nodes"] = []
    if "edges" not in result: result["edges"] = []
    result["nodes"].extend(g2.get("nodes", []))
    result["edges"].extend(g2.get("edges", []))
    return result
