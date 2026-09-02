import os
import json
import re
import base64
import logging
import json5
from pathlib import Path
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage
from tqdm import tqdm

# 尝试导入 Prompt，如果找不到则使用默认兜底
try:
    from prompts.image_prompts import image_extraction_prompt
except ImportError:
    # 兜底 Prompt (以防 prompts 模块未更新)
    image_extraction_prompt = """
    You are an expert chemist. Analyze the provided image from a scientific paper.
    Identify any chemical entities, structures, or characterization data (like XRD patterns, SEM images, charts).
    Extract them into a Knowledge Graph JSON format with "nodes" and "edges".
    
    Nodes should have: "id", "type" (chemical, characterization, etc.), "name", "properties".
    Edges should have: "source_id", "target_id", "type".
    
    Return ONLY valid JSON inside ```json ... ``` block.
    """

logger = logging.getLogger(__name__)

def encode_image(image_path: Path) -> str:
    """将图片文件编码为 base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_graph_from_images_via_api(
    source_file_path: Path, 
    model: Any, 
    image_dir: Path, 
    model_name_str: str = ""
) -> Dict[str, Any]:
    """
    遍历指定目录下的图片，使用传入的视觉模型对象提取结构化图数据。
    
    Args:
        source_file_path: 源文件路径 (用于定位)
        model: 已经初始化好的 LangChain Chat Model 对象 (支持 Vision)
        image_dir: 图片文件夹路径
        model_name_str: 模型名称 (用于日志)
    """
    
    combined_image_graph = {"nodes": [], "edges": []}
    
    # 1. 检查目录
    if not image_dir.exists():
        # 尝试回退到上级目录查找 (兼容性处理)
        if (source_file_path.parent.parent / "images").exists():
             image_dir = source_file_path.parent.parent / "images"
        else:
             logger.warning(f"Image directory not found: {image_dir}")
             return combined_image_graph

    # 2. 获取图片文件
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in valid_extensions]
    
    if not image_files:
        logger.info(f"No images found in {image_dir}")
        return combined_image_graph

    logger.info(f"Found {len(image_files)} images. Processing with model: {model_name_str}")

    # 3. 遍历处理每张图片
    for img_path in tqdm(image_files, desc="Extracting entities from images", unit="img"):
        try:
            logger.info(f"Processing image: {img_path.name}")
            base64_image = encode_image(img_path)
            
            # --- 核心：构造多模态消息 (LangChain 格式) ---
            # 这与您原来的 OpenAI 格式类似，但使用了 HumanMessage 类
            message = HumanMessage(
                content=[
                    {
                        "type": "text", 
                        "text": image_extraction_prompt.replace("{IMAGE_FILENAME}", img_path.name)
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ]
            )
            
            # --- 核心：调用模型 ---
            # 直接使用传入的 model 对象，不需要再初始化 client
            response = model.invoke([message])
            content = response.content
            
            # 4. 解析返回的 JSON
            match = re.search(r"```json\n(.*?)\n```|(\{.*?\})", content, re.DOTALL | re.IGNORECASE)
            if match:
                json_str = next(g for g in match.groups() if g)
                try:
                    graph_data = json5.loads(json_str)
                    
                    # 后处理：为节点添加 source_image_file 和唯一 ID
                    nodes = graph_data.get("nodes", [])
                    for node in nodes:
                        original_id = node.get("id", "unknown")
                        # 加上图片名前缀防止 ID 冲突
                        if "node" in str(original_id).lower() or str(original_id).isdigit():
                            node["id"] = f"{img_path.stem}_{original_id}"
                        
                        node["source_image_file"] = img_path.name
                        
                        if "properties" not in node: node["properties"] = {}
                        node["properties"]["extracted_from_image"] = img_path.name

                    combined_image_graph["nodes"].extend(nodes)
                    combined_image_graph["edges"].extend(graph_data.get("edges", []))
                    
                except Exception as e:
                    logger.warning(f"Failed to parse JSON from image {img_path.name}: {e}")
            else:
                logger.warning(f"No JSON block found in response for {img_path.name}")
                
        except Exception as e:
            logger.error(f"Error processing image {img_path.name}: {e}")
            continue

    logger.info(f"Total nodes extracted from images: {len(combined_image_graph['nodes'])}")
    return combined_image_graph