import os
import json
import re
import json5
import logging
import time
from json_repair import repair_json
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks.usage import get_usage_metadata_callback
from prompts.extract_HER_prompt_v3_2 import (
    synthesis_graph_prompt,
    synthesis_missing_check_prompt,
    testing_graph_prompt,
    testing_missing_check_prompt,
    characterization_graph_prompt,
    characterization_missing_check_prompt,
)


from models.models import get_model

# Vision model configuration
VISION_MODEL_CONFIG = 'google_gemini-2.5-flash' 

# Entity-alignment strategy ('knn' or 'llm')
ALIGNMENT_STRATEGY = 'knn' 

# Control characters removed from model output
CONTROL_SYMBOLS_TO_REMOVE = [
    "",
    "\u2605",  # black star
    "\u2606",  # white star
    "\u200b",  # zero-width space
    "\ufeff"   # byte order mark
]

logger = logging.getLogger(__name__)
INVOKE_MAX_ATTEMPTS = max(1, int(os.getenv("CATGRAPH_INVOKE_MAX_ATTEMPTS", "2")))
INVOKE_RETRY_BACKOFF_SECONDS = float(os.getenv("CATGRAPH_INVOKE_RETRY_BACKOFF_SECONDS", "0"))
RETRY_PROMPT_MAX_CHARS = {
    1: int(os.getenv("CATGRAPH_RETRY_PROMPT_CHARS_ATTEMPT1", "120000")),
    2: int(os.getenv("CATGRAPH_RETRY_PROMPT_CHARS_ATTEMPT2", "90000")),
    3: int(os.getenv("CATGRAPH_RETRY_PROMPT_CHARS_ATTEMPT3", "70000")),
}
VISION_MODEL_DISABLE_VALUES = {"", "disabled", "disable", "off", "none", "false", "0", "no"}
VISION_MODEL_EXACT_ALLOWLIST = {
    "deepseek-v3",
    "deepseek-v3.2",
    "grok-4-1-fast",
    "gpt-4.1-mini",
    "gpt-5",
    "gpt-5-mini",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "qwen3.5-flash-2026-02-23",
    "glm-4.7",
    "claude-haiku-4-5-20251001",
    "kimi-k2.5",
}
VISION_MODEL_KEYWORD_ALLOWLIST = (
    "vision",
    "gemini",
    "claude",
    "grok",
    "deepseek",
    "qwen",
    "glm",
    "kimi",
)
OPENAI_VISION_PREFIX_ALLOWLIST = ("gpt-4", "gpt-4.1", "gpt-4o", "gpt-5", "o3", "o4")
KNOWN_PROVIDER_PREFIXES = (
    "openai_",
    "google_",
    "deepseek_",
    "anthropic_",
    "xai_",
    "zhipu_",
    "moonshot_",
    "qwen_",
    "kimi_",
    "glm_",
)


def _parse_model_list_env(var_name: str) -> set:
    raw = os.getenv(var_name, "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


VISION_MODEL_FORCE_ALLOW = _parse_model_list_env("CATGRAPH_FORCE_VISION_MODEL_ALLOW")
VISION_MODEL_FORCE_DENY = _parse_model_list_env("CATGRAPH_FORCE_VISION_MODEL_DENY")

# --- Helper: Dynamic Aligner Import ---
def get_aligner_function(strategy: str):
    """Load aligner function dynamically based on alignment strategy."""
    try:
        if strategy.lower() == "knn":
            from tools.cat_graph.aligner.knn.entity_aligner_knn import align_and_merge_graphs
            logger.info("Using Alignment Strategy: KNN (CLIP Embeddings + Mutual kNN)")
            return align_and_merge_graphs
        if strategy.lower() == "llm":
            from tools.cat_graph.aligner.entity_aligner_llm import align_and_merge_graphs
            logger.info("Using Alignment Strategy: LLM (Text-based Prompting)")
            return align_and_merge_graphs
        raise ValueError(f"Unknown alignment strategy: {strategy}")
    except ImportError as e:
        logger.error(f"Failed to import alignment module: {e}")

        def no_op_align(text_g, img_g, model):
            return text_g

        return no_op_align

def _round_seconds(seconds_value: Optional[float]) -> Optional[float]:
    if seconds_value is None:
        return None
    return round(float(seconds_value), 3)

def get_alignment_model_name(strategy: str, fallback_model_name: str = "") -> str:
    strategy_normalized = (strategy or "").lower()
    try:
        if strategy_normalized == "knn":
            align_mod = importlib.import_module("tools.cat_graph.aligner.knn.entity_aligner_knn")
            llm_name = getattr(align_mod, "ALIGNER_MODEL_CONFIG", "")
            if llm_name:
                return f"knn_clip + llm:{llm_name}"
            return "knn_clip"
        if strategy_normalized == "llm":
            align_mod = importlib.import_module("tools.cat_graph.aligner.entity_aligner_llm")
            llm_name = getattr(align_mod, "ALIGNER_MODEL_CONFIG", "")
            if llm_name:
                return llm_name
    except Exception:
        pass
    return fallback_model_name or "unknown"


def normalize_model_token(model_name: str) -> str:
    return (model_name or "").strip().lower()


def strip_provider_prefix(model_name: str) -> str:
    token = normalize_model_token(model_name)
    for prefix in KNOWN_PROVIDER_PREFIXES:
        if token.startswith(prefix):
            return token[len(prefix):]
    return token


def is_vision_model_enabled(model_name: str) -> bool:
    """Heuristic gate for deciding whether multimodal image extraction should run."""
    token = normalize_model_token(model_name)
    stripped = strip_provider_prefix(token)

    if token in VISION_MODEL_DISABLE_VALUES or stripped in VISION_MODEL_DISABLE_VALUES:
        return False

    if token in VISION_MODEL_FORCE_DENY or stripped in VISION_MODEL_FORCE_DENY:
        return False

    if token in VISION_MODEL_FORCE_ALLOW or stripped in VISION_MODEL_FORCE_ALLOW:
        return True

    if token in VISION_MODEL_EXACT_ALLOWLIST or stripped in VISION_MODEL_EXACT_ALLOWLIST:
        return True

    # OpenAI family: treat GPT-4/GPT-5 variants as vision-capable by default.
    if stripped.startswith(OPENAI_VISION_PREFIX_ALLOWLIST):
        return True

    for keyword in VISION_MODEL_KEYWORD_ALLOWLIST:
        if keyword in token or keyword in stripped:
            return True

    return False

def _apply_graph_changes(initial_graph: Dict[str, Any], changes: Dict[str, Any], graph_type: str):
    """Applies changes (add, update, delete) to nodes and edges."""
    if not initial_graph or not isinstance(initial_graph, dict) or not changes: return
    
    if 'nodes' not in initial_graph: initial_graph['nodes'] = []
    if 'edges' not in initial_graph: initial_graph['edges'] = []

    # Deletions
    nodes_to_del = set(changes.get('node_ids_to_delete', []))
    if nodes_to_del:
        initial_graph['nodes'] = [n for n in initial_graph['nodes'] if n.get('id') not in nodes_to_del]
    
    edges_to_del = set(changes.get('edge_ids_to_delete', []))
    if edges_to_del:
        initial_graph['edges'] = [e for e in initial_graph['edges'] if e.get('id') not in edges_to_del]

    # Updates
    node_map = {n['id']: n for n in changes.get('nodes_to_update', []) if n.get('id')}
    for i, node in enumerate(initial_graph['nodes']):
        if node.get('id') in node_map: initial_graph['nodes'][i] = node_map[node['id']]
            
    edge_map = {e['id']: e for e in changes.get('edges_to_update', []) if e.get('id')}
    for i, edge in enumerate(initial_graph['edges']):
        if edge.get('id') in edge_map: initial_graph['edges'][i] = edge_map[edge['id']]

    # Additions
    initial_graph['nodes'].extend(changes.get('nodes_to_add', []))
    initial_graph['edges'].extend(changes.get('edges_to_add', []))

def remove_control_symbols(text: str) -> str:
    for s in CONTROL_SYMBOLS_TO_REMOVE: text = text.replace(s, '')
    return text.strip()

def safe_json_load(json_str: str) -> Dict[str, Any]:
    
    if not json_str or not isinstance(json_str, str):
        return {}
    
    try:
        return repair_json(json_str, return_objects=True)
    except Exception:
        pass

    try:
        return json5.loads(json_str)
    except Exception:
        pass

    try:
        return json.loads(json_str)
    except Exception as e:
        logging.error(f"safe_json_load failed to parse: {str(e)[:100]}...")
        return {}

def _extract_balanced_json_object(text: str) -> str:
    """Extract first balanced JSON object from free-form text."""
    if not text or not isinstance(text, str):
        return ""

    start = text.find("{")
    if start < 0:
        return ""

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

        if ch in ("\"", "'"):
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
    return ""

def extract_json_payload_text(response_text: str) -> str:
    """
    Extract JSON payload from LLM response robustly.
    Priority:
    1) fenced code block (json/json5/any)
    2) first balanced {...} object
    3) original text
    """
    if not response_text or not isinstance(response_text, str):
        return ""

    fenced = re.search(r"```(?:json|json5)?\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE)
    if fenced and fenced.group(1):
        return fenced.group(1)

    balanced = _extract_balanced_json_object(response_text)
    if balanced:
        return balanced

    return response_text

def normalize_graph_payload(payload: Any, stage_name: str = "") -> Dict[str, Any]:
    """Normalize model output to a graph dict with list-valued nodes/edges."""
    if isinstance(payload, str):
        parsed = safe_json_load(payload)
        payload = parsed if parsed else {}

    if isinstance(payload, list):
        logger.warning(
            "%s payload is list; wrapping as graph object.",
            stage_name or "Graph",
        )
        payload = {"nodes": payload, "edges": []}

    if not isinstance(payload, dict):
        if payload:
            logger.warning(
                "%s payload type %s is unsupported; using empty graph.",
                stage_name or "Graph",
                type(payload).__name__,
            )
        return {}

    normalized = dict(payload)
    nodes = normalized.get("nodes", [])
    edges = normalized.get("edges", normalized.get("relationships", []))

    if not isinstance(nodes, list):
        logger.warning("%s.nodes is not list; replaced with [].", stage_name or "Graph")
        nodes = []
    if not isinstance(edges, list):
        logger.warning("%s.edges is not list; replaced with [].", stage_name or "Graph")
        edges = []

    normalized["nodes"] = nodes
    normalized["edges"] = edges

    catalyst_ids = normalized.get("catalyst_tested_ids")
    if catalyst_ids is not None and not isinstance(catalyst_ids, list):
        logger.warning("%s.catalyst_tested_ids is not list; replaced with [].", stage_name or "Graph")
        normalized["catalyst_tested_ids"] = []

    return normalized


def _count_node_types(nodes: List[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        n_type = str(node.get("type", "")).strip().lower()
        if not n_type:
            continue
        counts[n_type] = counts.get(n_type, 0) + 1
    return counts


def build_stage_quality_checklist(stage_name: str, graph_payload: Any) -> str:
    """
    Build targeted checklist text for reflection prompts when graph structure looks incomplete.
    Returns empty string when no obvious structural issues are detected.
    """
    if not isinstance(graph_payload, dict):
        return ""

    nodes = graph_payload.get("nodes", [])
    edges = graph_payload.get("edges", [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    counts = _count_node_types(nodes)
    stage = (stage_name or "").strip().lower()
    issues: List[str] = []

    if stage == "synthesis":
        synth_nodes = counts.get("synthesis", 0)
        chem_nodes = counts.get("chemical", 0)
        if chem_nodes > 0 and synth_nodes == 0:
            issues.append(
                "Missing synthesis step nodes (type='synthesis'). Add explicit synthesis step nodes when procedures exist."
            )
        if (synth_nodes > 0 or chem_nodes > 0) and len(edges) == 0:
            issues.append(
                "Missing synthesis edges. Add synthesis_input/synthesis_output edges to connect chemicals and synthesis steps."
            )
    elif stage == "testing":
        testing_nodes = counts.get("testing", 0)
        if testing_nodes == 0:
            issues.append(
                "Missing testing nodes (type='testing'). Create testing nodes for each distinct testing setup/performance report."
            )
        if testing_nodes > 0 and len(edges) == 0:
            issues.append(
                "Missing tested_in edges. Each testing node must be linked from the corresponding catalyst chemical node."
            )
    elif stage == "characterization":
        char_nodes = counts.get("characterization", 0)
        if char_nodes == 0:
            issues.append(
                "Missing characterization nodes (type='characterization'). Create nodes for catalyst-focused methods reported in text."
            )
        if char_nodes > 0 and len(edges) == 0:
            issues.append(
                "Missing characterized_in edges. Each characterization node must be linked from at least one catalyst chemical node."
            )

    return " ".join(issues).strip()

def sanitize_prompt_template(text: str) -> str:
    """
    Keep prompt templates ASCII-only to avoid proxy resets caused by mojibake
    bullet characters from legacy encoded prompt files.
    """
    if not text:
        return ""
    return text.encode("ascii", "ignore").decode("ascii")

def to_text_content(content: Union[str, List[Any], Any]) -> str:
    """Normalize model response content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                chunks.append(str(item.get("text", "")))
            else:
                chunks.append(str(item))
        return "".join(chunks)
    if content is None:
        return ""
    return str(content)

def _compact_text_keep_head_tail(text: str, max_chars: int) -> str:
    """Compact long text by keeping the head and tail."""
    if not isinstance(text, str):
        return str(text)
    if len(text) <= max_chars or max_chars <= 0:
        return text
    marker = "\n\n[...content truncated for retry stability...]\n\n"
    head_chars = max(1, int(max_chars * 0.75))
    tail_chars = max(1, max_chars - head_chars - len(marker))
    return f"{text[:head_chars]}{marker}{text[-tail_chars:]}"

def _compact_payload_for_attempt(payload: Any, attempt: int) -> Any:
    """Reduce prompt size on later retries to improve connection stability."""
    max_chars = RETRY_PROMPT_MAX_CHARS.get(attempt, RETRY_PROMPT_MAX_CHARS[INVOKE_MAX_ATTEMPTS])
    if isinstance(payload, str):
        return _compact_text_keep_head_tail(payload, max_chars)

    if isinstance(payload, list):
        compacted_msgs = []
        for msg in payload:
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                compacted_content = _compact_text_keep_head_tail(content, max_chars)
                if compacted_content != content:
                    try:
                        msg = msg.__class__(content=compacted_content)
                    except Exception:
                        pass
            compacted_msgs.append(msg)
        return compacted_msgs

    return payload

def _payload_char_len(payload: Any) -> int:
    if isinstance(payload, str):
        return len(payload)
    if isinstance(payload, list):
        total = 0
        for msg in payload:
            total += len(to_text_content(getattr(msg, "content", "")))
        return total
    return len(str(payload))

def invoke_with_retries(model, payload: Any, stage_name: str):
    """Invoke model with bounded retries and explicit duration logging."""
    pid = os.getpid()
    last_err = None
    for attempt in range(1, INVOKE_MAX_ATTEMPTS + 1):
        started_at = time.time()
        attempt_payload = _compact_payload_for_attempt(payload, attempt)
        payload_len = _payload_char_len(attempt_payload)
        try:
            logger.info(
                f"[{pid}] {stage_name} invoke attempt {attempt}/{INVOKE_MAX_ATTEMPTS} "
                f"(payload_chars={payload_len})..."
            )
            resp = model.invoke(attempt_payload)
            elapsed = time.time() - started_at
            content_len = len(to_text_content(getattr(resp, "content", "")))
            logger.info(f"[{pid}] {stage_name} completed in {elapsed:.1f}s (content_len={content_len}).")
            return resp
        except Exception as e:
            elapsed = time.time() - started_at
            last_err = e
            logger.warning(
                f"[{pid}] {stage_name} failed in {elapsed:.1f}s "
                f"(attempt {attempt}/{INVOKE_MAX_ATTEMPTS}, payload_chars={payload_len}, "
                f"error_type={type(e).__name__}): {e}"
            )
            if attempt < INVOKE_MAX_ATTEMPTS:
                time.sleep(INVOKE_RETRY_BACKOFF_SECONDS * attempt)
    raise last_err


def _merge_usage_metadata(target: Dict[str, Any], usage: Any) -> None:
    """Merge usage metadata dictionaries by summing numeric fields."""
    if not isinstance(usage, dict):
        return
    for model_name, model_usage in usage.items():
        if not isinstance(model_usage, dict):
            continue
        entry = target.setdefault(model_name, {})
        if not isinstance(entry, dict):
            entry = {}
            target[model_name] = entry
        for key, value in model_usage.items():
            if isinstance(value, (int, float)):
                entry[key] = entry.get(key, 0) + value
            elif isinstance(value, dict):
                sub_entry = entry.get(key, {})
                if not isinstance(sub_entry, dict):
                    sub_entry = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        sub_entry[sub_key] = sub_entry.get(sub_key, 0) + sub_value
                    else:
                        sub_entry[sub_key] = sub_value
                entry[key] = sub_entry
            else:
                entry[key] = value
 

def invoke_with_usage_tracking(
    model,
    payload: Any,
    stage_name: str,
    usage_bucket: Dict[str, Any],
):
    """Invoke model with retries and always merge callback usage into usage_bucket."""
    with get_usage_metadata_callback() as usage_cb:
        try:
            return invoke_with_retries(model, payload, stage_name)
        finally:
            _merge_usage_metadata(usage_bucket, getattr(usage_cb, "usage_metadata", {}))


def run_with_usage_tracking(callable_fn, usage_bucket: Dict[str, Any]):
    """Run arbitrary callable and merge callback usage (for image/alignment helpers)."""
    with get_usage_metadata_callback() as usage_cb:
        try:
            return callable_fn()
        finally:
            _merge_usage_metadata(usage_bucket, getattr(usage_cb, "usage_metadata", {}))

# --- Main functionality ---

def extract_catgraph(file_path: Path, output_dir: Path, model, model_name: str) -> Dict:
    logger.info(f"Starting CatGraph extraction for: {file_path}")
    run_id = f"{file_path.stem}"
    output_file = output_dir / "graph" / f"{run_id}_output.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total_started_at = time.time()
    text_stage_started_at = time.time()
    text_extraction_duration_sec = None
    image_extraction_duration_sec = None
    alignment_duration_sec = None
    alignment_pairs_from_llm = 0
    alignment_pairs_from_fallback = 0
    alignment_retry_count = 0
    image_extraction_model = VISION_MODEL_CONFIG
    alignment_model = get_alignment_model_name(ALIGNMENT_STRATEGY, VISION_MODEL_CONFIG)
    text_usage_metadata: Dict[str, Any] = {}
    image_usage_metadata: Dict[str, Any] = {}
    alignment_usage_metadata: Dict[str, Any] = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_for_llm = remove_control_symbols(f.read())

        # ---------------------------------------------------------
        # 1. Synthesis Extraction
        # ---------------------------------------------------------
        prompt = sanitize_prompt_template(synthesis_graph_prompt).format(ARTICLE_TEXT=text_for_llm)
        msgs = [HumanMessage(content=prompt)]
        
        logger.info(f"[{os.getpid()}] Running Synthesis Extraction...")
        resp = invoke_with_usage_tracking(
            model=model,
            payload=prompt,
            stage_name="Synthesis Extraction",
            usage_bucket=text_usage_metadata,
        )
        resp_text = to_text_content(resp.content)
        
        synthesis_data = None
        try:
            json_str = extract_json_payload_text(resp_text)
            synthesis_data = normalize_graph_payload(safe_json_load(json_str), "Synthesis")
        except Exception as e:
            logger.error(f"Synthesis Parse Error: {e}")
            # Continue so the caller receives a structured extraction result.
        # --- Synthesis Validation (reflection and correction) ---
        if synthesis_data:
            try:
                synthesis_checklist = build_stage_quality_checklist("synthesis", synthesis_data)
                if synthesis_checklist:
                    logger.warning(
                        "[%s] Synthesis quality gate triggered: %s",
                        os.getpid(),
                        synthesis_checklist,
                    )
                validation_prompt = sanitize_prompt_template(synthesis_missing_check_prompt).format(
                    HUMAN_CHECKLIST=synthesis_checklist
                )
                msgs = [
                    HumanMessage(content=prompt),
                    AIMessage(content=resp_text),
                    HumanMessage(content=validation_prompt),
                ]
                logger.info(f"[{os.getpid()}] Running Synthesis Validation (Reflection)...")
                
                check_resp = invoke_with_usage_tracking(
                    model=model,
                    payload=msgs,
                    stage_name="Synthesis Validation",
                    usage_bucket=text_usage_metadata,
                )
                check_resp_text = to_text_content(check_resp.content)
                json_str = extract_json_payload_text(check_resp_text)
                if json_str:
                    changes = safe_json_load(json_str)
                    _apply_graph_changes(synthesis_data, changes, "Synthesis")
                    logger.info(f"[{os.getpid()}] Synthesis Validation Applied.")
            except Exception as e:
                logger.warning(f"Synthesis Validation Failed: {e}")
        else:
             logger.warning(f"[{os.getpid()}] Skipping Synthesis Validation (No data extracted).")

        # ---------------------------------------------------------
        # 2. Testing Extraction
        # ---------------------------------------------------------
        testing_data = None
        catalyst_ids = json.dumps(synthesis_data.get("catalyst_tested_ids", [])) if synthesis_data else "[]"
        
        if synthesis_data:
            prompt = sanitize_prompt_template(testing_graph_prompt).format(
                CATALYST_IDS_FROM_SYNTHESIS=catalyst_ids,
                ARTICLE_TEXT=text_for_llm
            )
            msgs = [HumanMessage(content=prompt)]
            
            logger.info(f"[{os.getpid()}] Running Testing Extraction...")
            resp = invoke_with_usage_tracking(
                model=model,
                payload=prompt,
                stage_name="Testing Extraction",
                usage_bucket=text_usage_metadata,
            )
            resp_text = to_text_content(resp.content)
            
            try:
                json_str = extract_json_payload_text(resp_text)
                testing_data = normalize_graph_payload(safe_json_load(json_str), "Testing")
                
                # --- Testing Validation (reflection and correction) ---
                if testing_data:
                    testing_checklist = build_stage_quality_checklist("testing", testing_data)
                    if testing_checklist:
                        logger.warning(
                            "[%s] Testing quality gate triggered: %s",
                            os.getpid(),
                            testing_checklist,
                        )
                    validation_prompt = sanitize_prompt_template(testing_missing_check_prompt).format(
                        CATALYST_IDS_FROM_SYNTHESIS=catalyst_ids,
                        HUMAN_CHECKLIST=testing_checklist,
                    )
                    msgs = [
                        HumanMessage(content=prompt),
                        AIMessage(content=resp_text),
                        HumanMessage(content=validation_prompt),
                    ]
                    logger.info(f"[{os.getpid()}] Running Testing Validation (Reflection)...")
                    
                    check_resp = invoke_with_usage_tracking(
                        model=model,
                        payload=msgs,
                        stage_name="Testing Validation",
                        usage_bucket=text_usage_metadata,
                    )
                    check_resp_text = to_text_content(check_resp.content)
                    json_str = extract_json_payload_text(check_resp_text)
                    if json_str:
                        changes = safe_json_load(json_str)
                        _apply_graph_changes(testing_data, changes, "Testing")
                        logger.info(f"[{os.getpid()}] Testing Validation Applied.")
            except Exception as e:
                 logger.warning(f"Testing Extraction/Validation Failed: {e}")

        # Merge Testing Nodes
        testing_data = normalize_graph_payload(testing_data, "Testing")
        if synthesis_data:
            syn_nodes = synthesis_data.setdefault("nodes", [])
            syn_ids = {n.get("id") for n in syn_nodes if isinstance(n, dict)}
            tested = set(synthesis_data.setdefault("catalyst_tested_ids", []))
            for n in testing_data.get("nodes", []):
                if not isinstance(n, dict):
                    continue
                nid = n.get("id")
                if n.get("type") == "chemical" and nid and nid not in syn_ids:
                    syn_nodes.append(n)
                    syn_ids.add(nid)
                    tested.add(nid)
            synthesis_data["catalyst_tested_ids"] = list(tested)

        # ---------------------------------------------------------
        # 3. Characterization Extraction
        # ---------------------------------------------------------
        char_data = None
        if synthesis_data:
            prompt = sanitize_prompt_template(characterization_graph_prompt).format(
                CATALYST_IDS_FROM_SYNTHESIS=catalyst_ids,
                ARTICLE_TEXT=text_for_llm
            )
            msgs = [HumanMessage(content=prompt)]

            logger.info(f"[{os.getpid()}] Running Characterization Extraction...")
            try:
                resp = invoke_with_usage_tracking(
                    model=model,
                    payload=prompt,
                    stage_name="Characterization Extraction",
                    usage_bucket=text_usage_metadata,
                )
                resp_text = to_text_content(resp.content)

                json_str = extract_json_payload_text(resp_text)
                char_data = normalize_graph_payload(safe_json_load(json_str), "Characterization")
                
                # --- Characterization Validation (reflection and correction) ---
                if char_data:
                    try:
                        char_checklist = build_stage_quality_checklist("characterization", char_data)
                        if char_checklist:
                            logger.warning(
                                "[%s] Characterization quality gate triggered: %s",
                                os.getpid(),
                                char_checklist,
                            )
                        validation_prompt = sanitize_prompt_template(characterization_missing_check_prompt).format(
                            CATALYST_IDS_FROM_SYNTHESIS=catalyst_ids,
                            HUMAN_CHECKLIST=char_checklist,
                        )
                        msgs = [
                            HumanMessage(content=prompt),
                            AIMessage(content=resp_text),
                            HumanMessage(content=validation_prompt),
                        ]
                        logger.info(f"[{os.getpid()}] Running Characterization Validation (Reflection)...")
                        
                        check_resp = invoke_with_usage_tracking(
                            model=model,
                            payload=msgs,
                            stage_name="Characterization Validation",
                            usage_bucket=text_usage_metadata,
                        )
                        check_resp_text = to_text_content(check_resp.content)
                        json_str = extract_json_payload_text(check_resp_text)
                        if json_str:
                            changes = safe_json_load(json_str)
                            _apply_graph_changes(char_data, changes, "Characterization")
                            logger.info(f"[{os.getpid()}] Characterization Validation Applied.")
                    except Exception as e:
                        logger.warning(f"Characterization Validation Warning: {e}")
                else:
                    logger.warning(f"[{os.getpid()}] Skipping Characterization Validation (No data extracted).")

                # Merge characterization chemical nodes into the synthesis graph.
                if char_data:
                    syn_nodes = synthesis_data.setdefault("nodes", [])
                    syn_ids = {n.get("id") for n in syn_nodes if isinstance(n, dict)}
                    for n in char_data.get("nodes", []):
                        if isinstance(n, dict) and n.get("type") == "chemical" and n.get("id") not in syn_ids:
                            syn_nodes.append(n)
                            syn_ids.add(n["id"])
            except Exception as e: 
                logger.warning(f"Characterization Extraction Error: {e}")

        # Continue with multimodal alignment and persistence.
        text_extraction_duration_sec = time.time() - text_stage_started_at

        # 4. Multimodal & Alignment
        final_graph = {}
        if synthesis_data: final_graph["synthesis"] = synthesis_data
        if testing_data: final_graph["testing"] = testing_data
        if char_data: final_graph["characterization"] = char_data
        
        try:
            text_raw_file = output_dir / "graph" / f"{run_id}_text_raw.json"
            with open(text_raw_file, "w", encoding="utf-8") as f:
                json.dump(final_graph, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved raw text graph to: {text_raw_file}")
        except Exception as e:
            logger.error(f"Failed to save raw text graph: {e}")


        image_dir = output_dir / "images"
        if not image_dir.exists(): image_dir = file_path.parent.parent / "images"

        if not image_dir.exists():
            logger.info(f"[{os.getpid()}] Skipping Multimodal Step: image directory not found -> {image_dir}")
        elif not is_vision_model_enabled(VISION_MODEL_CONFIG):
            logger.info(
                "[%s] Skipping Multimodal Step: vision model '%s' is not recognized as multimodal.",
                os.getpid(),
                VISION_MODEL_CONFIG,
            )
        else:
            logger.info(f"[{os.getpid()}] Starting Multimodal Step with {VISION_MODEL_CONFIG}...")
            
            try:
                mod = importlib.import_module("tools.cat_graph.image_extractor")
                extract_img = mod.extract_graph_from_images_via_api
                
                vision_model = get_model(model=VISION_MODEL_CONFIG, temperature=0)
                if vision_model:
                    image_extraction_started_at = time.time()
                    img_data = run_with_usage_tracking(
                        lambda: extract_img(file_path, vision_model, image_dir, VISION_MODEL_CONFIG),
                        image_usage_metadata,
                    )
                    image_extraction_duration_sec = time.time() - image_extraction_started_at
                    
                    try:
                        image_raw_file = output_dir / "graph" / f"{run_id}_image_raw.json"
                        with open(image_raw_file, "w", encoding="utf-8") as f:
                            json.dump(img_data, f, indent=2, ensure_ascii=False)
                        logger.info(f"Saved raw image graph to: {image_raw_file}")
                    except Exception as e:
                        logger.error(f"Failed to save raw image graph: {e}")


                    if img_data.get("nodes"):
                        logger.info(f"Extracted {len(img_data['nodes'])} image nodes. Aligning...")
                        
                        for node in img_data["nodes"]:
                            if "source_image_file" in node:
                                node["_full_image_path"] = str(image_dir / node["source_image_file"])

                        align_func = get_aligner_function(ALIGNMENT_STRATEGY)
                        
                        text_nodes = []
                        for sec in ["synthesis", "testing", "characterization"]:
                            if sec in final_graph: text_nodes.extend(final_graph[sec].get("nodes", []))
                        
                        alignment_started_at = time.time()
                        aligned_graph = run_with_usage_tracking(
                            lambda: align_func({"nodes": text_nodes, "edges": []}, img_data, vision_model),
                            alignment_usage_metadata,
                        )
                        alignment_duration_sec = time.time() - alignment_started_at
                        alignment_summary = {}
                        if isinstance(aligned_graph, dict):
                            maybe_summary = aligned_graph.pop("_alignment_summary", None)
                            if isinstance(maybe_summary, dict):
                                alignment_summary = maybe_summary
                        alignment_pairs_from_llm = int(alignment_summary.get("alignment_pairs_from_llm", 0) or 0)
                        alignment_pairs_from_fallback = int(alignment_summary.get("alignment_pairs_from_fallback", 0) or 0)
                        alignment_retry_count = int(alignment_summary.get("alignment_retry_count", 0) or 0)
                        
                        aligned_map = {n["id"]: n for n in aligned_graph["nodes"]}
                        
                        for sec in ["synthesis", "testing", "characterization"]:
                            if sec in final_graph:
                                for i, n in enumerate(final_graph[sec]["nodes"]):
                                    if n["id"] in aligned_map:
                                        updated_node = aligned_map[n["id"]]
                                        updated_node.pop("_full_image_path", None)
                                        final_graph[sec]["nodes"][i] = updated_node
                                        del aligned_map[n["id"]]
                        
                        if "characterization" not in final_graph: 
                            final_graph["characterization"] = {"nodes": [], "edges": []}
                        
                        for n in aligned_map.values():
                            n.pop("_full_image_path", None)
                            final_graph["characterization"]["nodes"].append(n)
                            
                        final_graph["characterization"]["edges"].extend(aligned_graph.get("edges", []))
                        
            except Exception as e:
                logger.error(f"Multimodal Error: {e}", exc_info=True)

        # 5. Save
        status = 'success' if any(k in final_graph for k in ["synthesis", "testing", "characterization"]) else 'error_no_data'
        if status == 'success':
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_graph, f, indent=2, ensure_ascii=False)

        total_extraction_duration_sec = time.time() - total_started_at

        return {
            'file': str(file_path),
            'status': status,
            'run_id': run_id,
            'output_graph_file': str(output_file) if status == 'success' else None,
            'model_name': model_name,
            'text_extraction_model': model_name,
            'text_extraction_duration_sec': _round_seconds(text_extraction_duration_sec),
            'image_extraction_model': image_extraction_model,
            'image_extraction_duration_sec': _round_seconds(image_extraction_duration_sec),
            'alignment_model': alignment_model,
            'alignment_duration_sec': _round_seconds(alignment_duration_sec),
            'alignment_pairs_from_llm': alignment_pairs_from_llm,
            'alignment_pairs_from_fallback': alignment_pairs_from_fallback,
            'alignment_retry_count': alignment_retry_count,
            'usage_metadata_text_stage': str(text_usage_metadata),
            'usage_metadata_image_stage': str(image_usage_metadata),
            'usage_metadata_alignment_stage': str(alignment_usage_metadata),
            'total_extraction_duration_sec': _round_seconds(total_extraction_duration_sec),
        }

    except Exception as e:
        logger.error(f"Processing Error {file_path}: {e}", exc_info=True)
        total_extraction_duration_sec = time.time() - total_started_at
        return {
            'file': str(file_path),
            'status': 'error_unknown',
            'error_message': str(e),
            'model_name': model_name,
            'text_extraction_model': model_name,
            'text_extraction_duration_sec': _round_seconds(text_extraction_duration_sec),
            'image_extraction_model': image_extraction_model,
            'image_extraction_duration_sec': _round_seconds(image_extraction_duration_sec),
            'alignment_model': alignment_model,
            'alignment_duration_sec': _round_seconds(alignment_duration_sec),
            'alignment_pairs_from_llm': alignment_pairs_from_llm,
            'alignment_pairs_from_fallback': alignment_pairs_from_fallback,
            'alignment_retry_count': alignment_retry_count,
            'usage_metadata_text_stage': str(text_usage_metadata),
            'usage_metadata_image_stage': str(image_usage_metadata),
            'usage_metadata_alignment_stage': str(alignment_usage_metadata),
            'total_extraction_duration_sec': _round_seconds(total_extraction_duration_sec),
        }


