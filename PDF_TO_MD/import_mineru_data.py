import os
import shutil
import json
import re
import argparse
from pathlib import Path

def extract_captions_from_content_list(content_list_path):
    """
    针对 MinerU 新版格式优化：优先读取内部 image_caption 字段。
    """
    if not os.path.exists(content_list_path):
        print(f"[Warning] Content list not found: {content_list_path}")
        return {}

    try:
        with open(content_list_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load content list: {e}")
        return {}

    image_caption_map = {}
    print(f"DEBUG: Processing {len(content_list)} items in content list...")
    
    for i, item in enumerate(content_list):
        if item.get('type') == 'image':
            # 1. 获取图片文件名
            img_path = item.get('img_path') or item.get('image_path') or ''
            if not img_path:
                continue
            image_filename = os.path.basename(img_path)
            
            caption = ""

            # 策略 A: 直接读取 image_caption
            caption_list = item.get('image_caption', [])
            if caption_list and isinstance(caption_list, list):
                caption = " ".join(caption_list).strip()
            
            # 策略 B: 向后查找 text (兜底)
            if not caption:
                for offset in range(1, 4): 
                    if i + offset >= len(content_list): break
                    next_item = content_list[i + offset]
                    if next_item.get('type') == 'image': break
                    if next_item.get('type') == 'text':
                        text = next_item.get('text', '').strip()
                        if text and len(text) < 500: 
                            caption = text
                            break 
            
            if caption:
                image_caption_map[image_filename] = caption

    return image_caption_map

def rename_images_and_fix_references(target_img_dir, target_txt_path, caption_map):
    """
    根据 Caption 重命名图片，并更新 Markdown 中的引用。
    处理逻辑：
    1. 从 Caption 提取 'Figure 1', 'Table 2' 等标识。
    2. 生成新文件名 (Fig_1.jpg)。
    3. 如果重名 (大图分割)，则追加后缀 (Fig_1_p1.jpg, Fig_1_p2.jpg)。
    4. 替换 Markdown 中的旧链接。
    5. 更新 caption_map 的键名为新文件名。
    """
    if not target_txt_path.exists():
        return caption_map

    with open(target_txt_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    new_caption_map = {}
    name_counters = {} # 记录每个图号出现的次数，用于处理分割图
    
    # 获取目录下所有图片文件
    existing_images = list(target_img_dir.glob("*"))
    # 建立文件名到路径的映射，方便查找
    img_lookup = {p.name: p for p in existing_images}

    print(f"--- Renaming Images ({len(caption_map)} captions found) ---")

    for old_filename, caption in caption_map.items():
        if old_filename not in img_lookup:
            continue
            
        old_file_path = img_lookup[old_filename]
        ext = old_file_path.suffix

        # 1. 尝试提取图号
        # 匹配: "Fig. 1", "Figure 2", "Table 3", "Scheme 4", "图 5", "表 6"
        # 忽略大小写，允许中间有点或空格
        match = re.search(r'(?:Fig(?:ure)?|Scheme|Table|图|表)\.?\s*(\d+)', caption, re.IGNORECASE)
        
        if match:
            # 提取前缀 (Fig/Table) 和 数字
            label_raw = match.group(0) # e.g., "Fig. 1"
            number = match.group(1)    # e.g., "1"
            
            # 标准化前缀
            if "tab" in label_raw.lower() or "表" in label_raw:
                prefix = "Table"
            elif "scheme" in label_raw.lower():
                prefix = "Scheme"
            else:
                prefix = "Figure"
            
            base_name = f"{prefix}_{number}"
        else:
            # 如果没找到图号，跳过重命名，或者可以保留原名
            # 为了保持一致性，这里选择不重命名那些无法识别图号的图片
            new_caption_map[old_filename] = caption
            continue

        # 2. 处理命名冲突 (分割图逻辑)
        if base_name in name_counters:
            name_counters[base_name] += 1
            # 采用 _p2, _p3 ... 命名分割部分
            new_filename = f"{base_name}_part{name_counters[base_name]}{ext}"
        else:
            name_counters[base_name] = 1
            # 第一个遇到的，先命名为 Figure_1.jpg
            # 思考：如果后面发现有 part2，是否应该回头把这个改成 part1？
            # 简单起见，第一个叫 Figure_1.jpg，第二个叫 Figure_1_part2.jpg 也是合理的
            # 或者统一加后缀。为了美观，我们采用：如果是唯一的，就不加后缀；如果有多个，全部加后缀？
            # 由于是流式处理，很难预知。
            # 方案：第一个叫 Fig_1.jpg，如果遇到第二个，新的叫 Fig_1_p2.jpg。用户能看懂。
            new_filename = f"{base_name}{ext}"

            # 稍微优化：如果已经存在 Fig_1.jpg (可能是上一轮运行残留，或者是真正的冲突)，
            # 这里的 name_counters 是内存里的计数，能处理当前批次的冲突。
            # 但要小心物理文件是否已存在。
            while (target_img_dir / new_filename).exists() and (target_img_dir / new_filename) != old_file_path:
                # 如果文件已存在（且不是自己），说明发生了意外的命名冲突
                name_counters[base_name] += 1
                new_filename = f"{base_name}_part{name_counters[base_name]}{ext}"

        # 3. 执行重命名
        new_file_path = target_img_dir / new_filename
        try:
            os.rename(old_file_path, new_file_path)
            # print(f"  [Rename] {old_filename} -> {new_filename}")
        except OSError as e:
            print(f"  [Error] Could not rename {old_filename}: {e}")
            new_caption_map[old_filename] = caption
            continue

        # 4. 更新 Markdown 内容
        # 替换 ![...](images/old_filename) 为 ![...](images/new_filename)
        # 或者是 MinerU 的格式 ![](images/...)
        md_content = md_content.replace(old_filename, new_filename)
        
        # 5. 更新 metadata map
        new_caption_map[new_filename] = caption
    
    # 将未重命名的图片也加回 map (保持完整性)
    for old, cap in caption_map.items():
        if old not in new_caption_map.values() and old not in img_lookup: 
             # 这里逻辑稍微有点绕，只要不在 new_caption_map keys 里说明没变
             # 但因为我们遍历的是 caption_map，如果上面 continue 了，这里需要补上
             # 简单判断：如果 old_filename 对应的文件还在（说明没被 rename），就加回去
             if (target_img_dir / old).exists():
                 if old not in new_caption_map:
                     new_caption_map[old] = cap

    # 写回 Markdown
    with open(target_txt_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"  [Info] Updated Markdown references and renamed files.")
    return new_caption_map

def process_mineru_output_folder(mineru_output_folder, project_root):
    """
    处理单个 MinerU 客户端输出文件夹，将其标准化导入到项目数据目录。
    """
    source_path = Path(mineru_output_folder).resolve()
    if not source_path.exists():
        print(f"[Error] Source folder not found: {source_path}")
        return

    # 1. 解析论文名称
    folder_name = source_path.name
    match = re.match(r"(.+?)\.pdf-[a-f0-9]+", folder_name)
    if match:
        paper_name = match.group(1) 
    else:
        paper_name = folder_name.replace(" ", "_")
    
    print(f"--- Importing Paper: {paper_name} ---")

    # 2. 定义目标路径
    target_base_dir = Path(project_root) / "data" / "processed_papers" / paper_name
    target_txt_dir = target_base_dir / "txt"
    target_img_dir = target_base_dir / "images"

    # 清理旧数据
    if target_base_dir.exists():
        shutil.rmtree(target_base_dir)
    
    target_txt_dir.mkdir(parents=True, exist_ok=True)
    target_img_dir.mkdir(parents=True, exist_ok=True)

    # 3. 搬运 Markdown 文件
    md_candidates = list(source_path.glob("*.md"))
    target_md_path = target_txt_dir / "output.md"
    if md_candidates:
        src_md = next((f for f in md_candidates if f.name == 'full.md'), md_candidates[0])
        shutil.copy2(src_md, target_md_path)
    else:
        print(f"[Error] No markdown (.md) file found.")
        return # 没有 MD 文件没法做后续的替换，直接退出

    # 4. 搬运图片
    src_img_dir = source_path / "images"
    if src_img_dir.exists():
        for img_file in src_img_dir.glob("*"):
            shutil.copy2(img_file, target_img_dir / img_file.name)
    
    # 5. 生成 Image Metadata 并执行重命名
    json_candidates = list(source_path.glob("*_content_list.json"))
    caption_map = {}
    if json_candidates:
        content_list_path = json_candidates[0]
        # 提取原始 Caption 映射
        caption_map = extract_captions_from_content_list(content_list_path)
        
        # --- 新增步骤：重命名图片并修复 Markdown 引用 ---
        if caption_map:
            caption_map = rename_images_and_fix_references(target_img_dir, target_md_path, caption_map)
        
        # 保存新的 metadata (包含新的文件名)
        metadata_path = target_img_dir / "image_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(caption_map, f, indent=2, ensure_ascii=False)
        print(f"[Success] Metadata saved with {len(caption_map)} items.")
    else:
        print(f"[Warning] No content_list.json found.")

    print(f"Done. Data ready at: {target_base_dir}\n")

if __name__ == "__main__":
    print(
        "This module exposes process_mineru_output_folder(...). "
        "Pass your local input path from a separate script; no path is hard-coded."
    )
