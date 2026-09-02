import json
import json5
import re
import logging
from typing import Dict, Any, List
from models.models import get_model
from prompts.image_prompts import entity_resolution_prompt

logger = logging.getLogger(__name__)


ALIGNER_MODEL_CONFIG = 'openai_gpt-5-mini' 

def align_and_merge_graphs(
    text_graph: Dict[str, Any], 
    image_graph: Dict[str, Any], 
    model: Any = None 
) -> Dict[str, Any]:
    """
    使用 LLM (Prompt) 进行实体对齐和图合并。
    
    Args:
        text_graph: 包含文本提取节点的图数据
        image_graph: 包含图片提取节点的图数据
        model: (可选) 外部传入的模型对象。如果 ALIGNER_MODEL_CONFIG 有效，则优先使用内部初始化的模型。
    
    Returns:
        Dict[str, Any]: 合并后的图数据
    """
    
    # 1. 尝试初始化独立的对齐模型
    aligner_model = None
    try:
        # 使用 temperature=0 以获得确定性的对齐结果
        logger.info(f"Initializing independent aligner model: {ALIGNER_MODEL_CONFIG}")
        aligner_model = get_model(model=ALIGNER_MODEL_CONFIG, temperature=0)
    except Exception as e:
        logger.warning(f"Failed to initialize custom aligner model '{ALIGNER_MODEL_CONFIG}': {e}. Falling back to passed model.")
        aligner_model = model

    if not aligner_model:
        logger.error("No valid model available for alignment. Skipping alignment.")
        return _simple_merge(text_graph, image_graph)

    # 2. 提取实体准备 Prompt
    text_nodes = [n for n in text_graph.get("nodes", []) if n.get("type") == "chemical"]
    image_nodes = [n for n in image_graph.get("nodes", []) if n.get("type") == "chemical"]
    
    if not image_nodes or not text_nodes:
        logger.info("Insufficient nodes for alignment (missing text or image chemicals). Skipping.")
        return _simple_merge(text_graph, image_graph)

    mapping = {}
    try:
        # 构造精简的实体列表供 LLM 决策
        text_summary = [{ "id": n["id"], "name": n.get("name"), "aliases": n.get("aliases", []) } for n in text_nodes]
        image_summary = [{ "id": n["id"], "name": n.get("name"), "source_img": n.get("source_image_file") } for n in image_nodes]
        
        text_summary_json = json.dumps(text_summary, indent=2, ensure_ascii=False)
        image_summary_json = json.dumps(image_summary, indent=2, ensure_ascii=False)
        
        # 格式化 Prompt
        final_prompt = entity_resolution_prompt.format(
            TEXT_ENTITIES_JSON=text_summary_json,
            IMAGE_ENTITIES_JSON=image_summary_json
        )
        
        logger.info(f"Sending {len(text_nodes)} text nodes and {len(image_nodes)} image nodes to LLM for alignment...")
        
        # 调用 LLM
        response = aligner_model.invoke(final_prompt)
        content = response.content.strip()
        
        # --- [修复核心] 更稳健的 JSON 提取逻辑 ---
        json_str = ""
        # 1. 优先尝试匹配 Markdown 代码块
        code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            # 2. 如果没有代码块，寻找字符串中最外层的 {}
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx : end_idx + 1]
            else:
                logger.warning(f"No JSON object found in LLM response. Content snippet: {content[:100]}...")
        
        if json_str:
            alignment_result = json5.loads(json_str)
            mapping = alignment_result.get("alignment_map", {})
            logger.info(f"Alignment successful. Found {len(mapping)} mappings.")
        # ---------------------------------------------
            
    except Exception as e:
        logger.error(f"Error during LLM alignment: {e}", exc_info=True)
        # 出错时回退到简单合并
        return _simple_merge(text_graph, image_graph)

    # 3. 执行合并 (Merging Logic)
    final_graph = _deep_copy_graph(text_graph)
    
    # 创建快速查找表
    final_nodes_map = {n["id"]: n for n in final_graph["nodes"]}
    
    # 处理图片节点
    for img_node in image_graph.get("nodes", []):
        original_id = img_node["id"]
        is_chemical = img_node.get("type") == "chemical"
        
        # Case A: 这是一个化学节点，且在 Mapping 中找到了对应的文本节点 -> 合并
        if is_chemical and original_id in mapping and mapping[original_id]:
            target_text_id = mapping[original_id]
            
            if target_text_id in final_nodes_map:
                target_node = final_nodes_map[target_text_id]
                
                # A-1. 合并属性 (Properties)
                if "properties" not in target_node: target_node["properties"] = {}
                for k, v in img_node.get("properties", {}).items():
                    # 只有当目标没有该属性，或者想要覆盖时才写入。这里选择保留文本信息优先，仅补充缺失的。
                    if k not in target_node["properties"]:
                        target_node["properties"][k] = v
                
                # A-2. 记录图片来源 (Related Images)
                if "related_images" not in target_node: target_node["related_images"] = []
                if "source_image_file" in img_node:
                    # 避免重复添加
                    if img_node["source_image_file"] not in target_node["related_images"]:
                        target_node["related_images"].append(img_node["source_image_file"])
                
                logger.debug(f"Merged Image Node {original_id} -> Text Node {target_text_id}")
            else:
                # 映射的目标ID不存在（罕见情况），作为新节点添加
                final_graph["nodes"].append(img_node)
                final_nodes_map[img_node["id"]] = img_node
        else:
            # Case B: 没有映射或非化学节点 -> 直接添加
            # 注意避免 ID 冲突 (虽然理论上 ID 命名空间不同)
            if img_node["id"] not in final_nodes_map:
                final_graph["nodes"].append(img_node)
                final_nodes_map[img_node["id"]] = img_node
    
    # 4. 处理边 (Edges)
    # 图片图中的边，其 source/target ID 如果被合并了，需要更新指向新的 Text ID
    for edge in image_graph.get("edges", []):
        new_edge = edge.copy()
        src = new_edge.get("source_id")
        tgt = new_edge.get("target_id")
        
        # 如果源节点被映射了，更新 ID
        if src in mapping and mapping[src]:
            new_edge["source_id"] = mapping[src]
        
        # 如果目标节点被映射了，更新 ID
        if tgt in mapping and mapping[tgt]:
            new_edge["target_id"] = mapping[tgt]
            
        final_graph["edges"].append(new_edge)

    return final_graph

def _simple_merge(g1, g2):
    """简单合并两个图，不进行去重或对齐"""
    import copy
    result = copy.deepcopy(g1)
    if "nodes" not in result: result["nodes"] = []
    if "edges" not in result: result["edges"] = []
    
    # 简单的 ID 查重，避免完全重复的节点
    existing_ids = {n["id"] for n in result["nodes"]}
    
    for n in g2.get("nodes", []):
        if n["id"] not in existing_ids:
            result["nodes"].append(n)
            
    result["edges"].extend(g2.get("edges", []))
    return result

def _deep_copy_graph(g):
    import copy
    res = copy.deepcopy(g)
    if "nodes" not in res: res["nodes"] = []
    if "edges" not in res: res["edges"] = []
    return res