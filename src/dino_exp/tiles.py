"""图像切块（tiling）：大图切成重叠小块，训练/推理同尺度，提升小缺陷分辨率。

训练与推理必须共用同一网格与重叠率（记忆库特征与推理特征同尺度），
网格信息持久化在版本 config.yaml（tile_mode/tile_grid/tile_overlap）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image

CLASSIC_GRIDS = [(2, 2), (3, 3), (4, 4), (6, 6), (8, 8)]
TILE_MODES = ["off", "auto"] + [f"{x[0]}x{x[1]}" for x in CLASSIC_GRIDS]


@dataclass(frozen=True)
class TileSpec:
    grid: tuple[int, int]   # (nx, ny)，(1,1) 表示不切
    overlap: float = 0.1

    @property
    def enabled(self) -> bool:
        return self.grid != (1, 1)


def parse_tile_mode(mode: str) -> tuple[int, int] | str:
    """'off' → (1,1)；'auto' → 'auto'；'3x3' → (3,3)。其他抛 ValueError。"""
    if mode == "off":
        return (1, 1)
    if mode == "auto":
        return "auto"
    if mode in TILE_MODES:
        n = int(mode.split("x")[0])
        return (n, n)
    raise ValueError(f"未知 tile 模式 '{mode}'，可选: {TILE_MODES}")


def auto_grid(img_w: int, img_h: int, image_size: int, target_patch_px: int = 20) -> tuple[int, int]:
    """按目标 patch 覆盖（原图像素）自动选最小的经典网格。

    网格 (n,n) 下每块约 (w/n, h/n) 像素，缩到 image_size 后 patch 覆盖 ≈ (w/n)/patch_per_side。
    选满足覆盖 ≤ target_patch_px 的最小网格；都达不到则用最大网格 (8,8)。
    """
    patch_per_side = 16  # image_size=224、patch14 → 16×16 patch 网格（通用：image_size/patch）
    patch_per_side = image_size // 14  # DINOv2 patch=14
    for n, _ in CLASSIC_GRIDS:
        cover_x = (img_w / n) / patch_per_side
        cover_y = (img_h / n) / patch_per_side
        if max(cover_x, cover_y) <= target_patch_px:
            return (n, n)
    return CLASSIC_GRIDS[-1]


def split_image(image: Image.Image, grid: tuple[int, int], overlap: float = 0.1):
    """按网格+重叠率切图。返回 [(tile_image, (x0, y0, x1, y1)), ...]（原图坐标）。"""
    nx, ny = grid
    w, h = image.size
    tiles = []
    for iy in range(ny):
        for ix in range(nx):
            cw, ch = w / nx, h / ny
            ox, oy = cw * overlap, ch * overlap
            x0 = max(0, int(ix * cw - ox))
            y0 = max(0, int(iy * ch - oy))
            x1 = min(w, int((ix + 1) * cw + ox))
            y1 = min(h, int((iy + 1) * ch + oy))
            tiles.append((image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)))
    return tiles


def stitch_anomaly_maps(tile_amaps, boxes, grid: tuple[int, int], out_size: tuple[int, int]):
    """把各块的 anomaly map 拼回原图坐标（核心区归属，避免边缘重复计分）。

    每块只贡献其名义网格单元对应的子区域（重叠外延仅提供上下文）：
    每个像素恰好只有一个来源块，切块边缘的假热点不会被重复放大。
    tile_amaps: 与 boxes 对应的 amap 张量列表（顺序为先行后列）。
    boxes: split_image 返回的带重叠裁切框。grid: (nx, ny)。out_size: 原图 (W, H)。
    返回 torch.Tensor (1, 1, H, W)。
    """
    import torch
    import torch.nn.functional as F

    nx, ny = grid
    w, h = out_size
    merged = torch.zeros(h, w)
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            amap = tile_amaps[idx].squeeze()  # 统一为 2D (H', W')
            x0, y0, x1, y1 = boxes[idx]
            tw, th = x1 - x0, y1 - y0
            resized = F.interpolate(amap.unsqueeze(0).unsqueeze(0).float(),
                                    size=(th, tw), mode="bilinear", align_corners=False).squeeze()
            # 名义核心单元（整数网格线）
            cx0, cy0 = int(ix * w / nx), int(iy * h / ny)
            cx1, cy1 = int((ix + 1) * w / nx), int((iy + 1) * h / ny)
            sx0, sy0 = cx0 - x0, cy0 - y0
            merged[cy0:cy1, cx0:cx1] = resized[sy0:sy0 + (cy1 - cy0), sx0:sx0 + (cx1 - cx0)]
            idx += 1
    return merged.unsqueeze(0).unsqueeze(0)


def should_tile(img_w: int, img_h: int, image_size: int, grid: tuple[int, int]) -> bool:
    """只有图片大于模型输入尺寸时才切块（小图直接推理）。"""
    return grid != (1, 1) and (img_w > image_size or img_h > image_size)


def clamp_grid(grid: tuple[int, int], img_w: int, img_h: int, image_size: int) -> tuple[int, int]:
    """限制网格密度：每块不小于模型输入的一半，避免小图被过度上采样失真。"""
    if grid == (1, 1):
        return grid
    min_tile = max(1, image_size // 2)
    nx = max(1, min(grid[0], img_w // min_tile or 1))
    ny = max(1, min(grid[1], img_h // min_tile or 1))
    return (nx, ny)
