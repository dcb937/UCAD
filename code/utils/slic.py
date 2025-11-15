from skimage.segmentation import slic
from skimage import graph
from collections import deque
from tqdm import tqdm
import torch
import numpy as np

def slic_superpixels(image_np, n_segments=100, compactness=10):
    segments = slic(image_np, n_segments=n_segments, compactness=compactness, start_label=1, channel_axis=None)
    return segments

def build_adjacency_graph(segments):
    rag = graph.rag_mean_color(np.zeros_like(segments), segments)
    adjacency = {label: set() for label in np.unique(segments)}
    for edge in rag.edges:
        n1, n2 = edge
        adjacency[n1].add(n2)
        adjacency[n2].add(n1)
    return adjacency

def bfs_expand(segments, start_label, max_regions=30):
    adjacency = build_adjacency_graph(segments)
    visited = set()
    queue = deque([start_label])
    visited.add(start_label)

    while queue and len(visited) < max_regions:
        current = queue.popleft()
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
            if len(visited) >= max_regions:
                break
    return visited

def generate_mask_slic_based(img_tensor, n_segments=100, compactness=10, num_superpixels=30):
    """
    img_tensor: torch.Tensor with shape (B, C, H, W)
    Returns: mask (B, H, W), loss_mask (B, H, W)
    """
    B, C, H, W = img_tensor.shape
    mask_batch = torch.ones((B, H, W), device=img_tensor.device)
    loss_mask_batch = torch.ones((B, H, W), device=img_tensor.device)

    for i in range(B):
        img_np = img_tensor[i, 0].cpu().numpy()  # use 1st channel
        segments = slic_superpixels(img_np, n_segments=n_segments, compactness=compactness)

        unique_labels = np.unique(segments)
        start_label = np.random.choice(unique_labels)
        selected_labels = bfs_expand(segments, start_label, max_regions=num_superpixels)

        # Create mask where selected labels are 0
        selected_mask = np.isin(segments, list(selected_labels))
        selected_mask_tensor = torch.from_numpy(selected_mask).long().to(img_tensor.device)

        mask_batch[i][selected_mask_tensor == 1] = 0
        loss_mask_batch[i][selected_mask_tensor == 1] = 0

    return mask_batch.long(), loss_mask_batch.long()

def generate_single_mask_slic_based(img_tensor, n_segments=100, compactness=10, num_superpixels=30):
        """
        生成单张图像的 SLIC 超像素掩码
        img_tensor: torch.Tensor with shape (1, C, H, W)
        Returns: mask (1, H, W)
        """
        # 确保输入是单张图像
        assert img_tensor.shape[0] == 1, "输入必须是单张图像 (批次大小为1)"
        
        # 提取第一个通道并转换为 numpy
        img_np = img_tensor[0, 0].cpu().numpy()  # 使用第一个通道
        
        # 生成 SLIC 超像素
        segments = slic_superpixels(img_np, n_segments=n_segments, compactness=compactness)
        
        # 随机选择起始标签并进行区域扩展
        unique_labels = np.unique(segments)
        start_label = np.random.choice(unique_labels)
        selected_labels = bfs_expand(segments, start_label, max_regions=num_superpixels)
        
        # 创建掩码：选中的区域为0，其余为1
        selected_mask = np.isin(segments, list(selected_labels))
        selected_mask_tensor = torch.from_numpy(selected_mask).long().to(img_tensor.device)
        
        # 创建掩码张量
        mask = torch.ones_like(selected_mask_tensor)
        loss_mask = torch.ones_like(selected_mask_tensor)
        mask[selected_mask_tensor == 1] = 0
        loss_mask[selected_mask_tensor == 1] = 0
        
        return mask.unsqueeze(0), loss_mask.unsqueeze(0)  # 返回 (1, H, W) 形状的掩码

def generate_sgements_slic_based(img_tensor, n_segments=100, compactness=10):
        """
        生成单张图像的 SLIC 超像素掩码
        img_tensor: torch.Tensor with shape (1, C, H, W)
        Returns: mask (1, H, W)
        """
        # 确保输入是单张图像
        assert img_tensor.shape[0] == 1, "输入必须是单张图像 (批次大小为1)"
        assert img_tensor.shape[1] == 1, "输入必须是单张图像 (批次大小为1)"
        
        # 提取第一个通道并转换为 numpy
        img_np = img_tensor[0, 0].cpu().numpy()  # 使用第一个通道
        
        # 生成 SLIC 超像素
        segments = slic_superpixels(img_np, n_segments=n_segments, compactness=compactness)
        # segments = slic(image_np, n_segments=n_segments, compactness=compactness, start_label=1, channel_axis=None)

        segments = torch.from_numpy(segments).long().to(img_tensor.device)
        return segments.unsqueeze(0)


def generate_3d_segments_slic_based(img_tensor, n_segments=100, compactness=10):
    """
    生成单张3D图像的 SLIC 超像素掩码
    img_tensor: torch.Tensor with shape (1, 1, D, H, W)
    Returns: mask (1, D, H, W) with long type
    """
    assert img_tensor.shape[0] == 1, "输入必须是单张图像 (批次大小为1)"
    assert img_tensor.shape[1] == 1, "输入必须是单通道图像 (通道数为1)"

    # 提取图像并转为 numpy 格式 (D, H, W)
    img_np = img_tensor[0, 0].cpu().numpy()

    # 生成 SLIC 超像素
    segments = slic(
        img_np,
        n_segments=n_segments,
        compactness=compactness,
        channel_axis=None,
        start_label=1
    )

    # 转为 torch.Tensor，保持设备一致
    segments_tensor = torch.from_numpy(segments).long().to(img_tensor.device)

    return segments_tensor.unsqueeze(0)  # (1, D, H, W)
