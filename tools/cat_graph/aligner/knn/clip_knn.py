import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# --- 来自 mlbio-epfl/structure/src/utils/utils.py 的核心数学工具 ---
# 保持引入这些经过验证的预处理函数，它们能显著提升特征质量

def are_embeddings_normalized(embeddings: torch.Tensor, epsilon: float = 1e-6) -> bool:
    if embeddings is None or embeddings.numel() == 0:
        return False
    norms = torch.norm(embeddings, p=2, dim=-1)
    return torch.all((norms - 1.0).abs() < epsilon).item()

def safe_normalize(
    embeddings: torch.Tensor, p: int = 2, dim: int = -1, epsilon: float = 1e-12
) -> torch.Tensor:
    if are_embeddings_normalized(embeddings):
        return embeddings
    else:
        return F.normalize(embeddings, p=p, dim=dim, eps=epsilon)

def center_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Mean Centering: 消除模态间的分布偏移，这是 STRUCTURE 提升效果的关键预处理。
    """
    if embeddings is None or embeddings.numel() == 0:
        return embeddings
    return embeddings - embeddings.mean(0, keepdim=True)

# --- 核心对齐逻辑 (保留您的双向奔赴逻辑，升级为矩阵运算) ---

def get_mutual_knn_pairs(
    text_feats: torch.Tensor, 
    img_feats: torch.Tensor, 
    topk: int = 2,  # 坚持使用您设定的 Top-2，保证鲁棒性
    similarity_threshold: float = 0.0
) -> list:
    """
    计算文本特征和图像特征之间的互近邻 (Mutual kNN)。
    
    逻辑说明 (您原有的逻辑):
    1. Text -> Image 找 Top-K
    2. Image -> Text 找 Top-K
    3. 只有当 (Text_i, Img_j) 在彼此的 Top-K 列表中时，才算匹配。
    
    Args:
        topk: 默认为 2，正如您指出的，比 Top-1 更鲁棒。
    """
    if text_feats is None or img_feats is None:
        return []
    
    # 1. 归一化 (mlbio 推荐的标准操作)
    text_feats = safe_normalize(text_feats)
    img_feats = safe_normalize(img_feats)
    
    device = text_feats.device
    n_text = text_feats.shape[0]
    n_img = img_feats.shape[0]
    
    # 2. 计算相似度矩阵 [N_text, N_img]
    sim_matrix = text_feats @ img_feats.T
    
    curr_topk = min(topk, n_img, n_text)
    if curr_topk <= 0:
        return []

    # 3. 寻找 Text -> Image 的 TopK
    # indices_t2i: [N_text, k] -> 每行包含该 Text 最相似的 k 个 Image 的索引
    vals_t2i, indices_t2i = torch.topk(sim_matrix, k=curr_topk, dim=1)
    
    # 4. 寻找 Image -> Text 的 TopK
    # indices_i2t: [N_img, k] -> 每行包含该 Image 最相似的 k 个 Text 的索引
    vals_i2t, indices_i2t = torch.topk(sim_matrix.T, k=curr_topk, dim=1)
    
    # 5. 交叉验证 (Intersection) - 使用矩阵加速代替 For 循环
    # 您的原逻辑是: if t_idx in best_texts_for_this_img AND i_idx in best_imgs_for_this_text
    
    # 构建 Text->Image 的布尔掩码
    # mask_t2i[i, j] = True 表示 Image j 在 Text i 的 TopK 中
    mask_t2i = torch.zeros((n_text, n_img), device=device, dtype=torch.bool)
    mask_t2i.scatter_(1, indices_t2i, True)
    
    # 构建 Image->Text 的布尔掩码 (注意维度转置)
    # mask_i2t_transposed[j, i] = True 表示 Text i 在 Image j 的 TopK 中
    mask_i2t_transposed = torch.zeros((n_img, n_text), device=device, dtype=torch.bool)
    mask_i2t_transposed.scatter_(1, indices_i2t, True)
    mask_i2t = mask_i2t_transposed.T  # 转置回 [N_text, N_img]
    
    # 取交集：双向奔赴
    mutual_mask = mask_t2i & mask_i2t
    
    # 提取结果
    mutual_pairs = []
    # nonzero 返回所有为 True 的坐标 (text_idx, img_idx)
    matches = torch.nonzero(mutual_mask, as_tuple=False)
    
    for match in matches:
        t_idx = match[0].item()
        i_idx = match[1].item()
        score = sim_matrix[t_idx, i_idx].item()
        
        if score >= similarity_threshold:
            mutual_pairs.append({
                "text_idx": t_idx,
                "img_idx": i_idx,
                "score": score
            })
            
    # 按分数排序
    mutual_pairs.sort(key=lambda x: x["score"], reverse=True)
    
    logger.debug(f"Mutual kNN (k={topk}) found {len(mutual_pairs)} pairs.")
    return mutual_pairs