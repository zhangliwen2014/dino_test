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


# ---------- 固定尺寸切块（tile_scheme="size"，设计文档 2026-07-23-fixed-size-tiling） ----------
# 与上方比例切块（grid）并存：grid 仅为旧版本兼容保留，新训练（tile_mode=auto）走固定 T 方案。


def compute_tile_size(target_patch_px: int, image_size: int, patch_size: int) -> int:
    """T = P × (image_size / patch_size)：原图 T×T 块缩到 image_size 后每 patch 覆盖 P 原图像素。

    T 是类别级常数（patch14@224 → 16 patch/边 → T=20×16=320；patch16@224 → 14 → T=280）。
    """
    return target_patch_px * (image_size // patch_size)


def upscale_small(im: Image.Image, tile_size: int, margin: int) -> Image.Image:
    """小图（两轴都 < T+margin）各向同性放大到短边 ≥ T（与记忆库尺度对齐）。

    长条图（任一轴 ≥ T+margin，另一轴走 pad）与短边已 ≥ T 的图原样返回。
    """
    w, h = im.size
    if w >= tile_size + margin or h >= tile_size + margin:
        return im
    short = min(w, h)
    if short >= tile_size:
        return im
    scale = tile_size / short
    return im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)


def _reflect_pad(im: Image.Image, pad_w: int, pad_h: int) -> Image.Image:
    """右/下 reflect（symmetric）补边：长条图短边 < T 时补到 T，保持块为正方形。"""
    import numpy as np

    arr = np.array(im)
    arr = np.pad(arr, ((0, pad_h), (0, pad_w)) + ((0, 0),) * (arr.ndim - 2), mode="symmetric")
    return Image.fromarray(arr)


def _axis_segments(axis_len: int, tile_size: int, stride: int, margin: int):
    """单轴切分。返回 (segments, pad)：segments=[(start, length), ...]。

    轴长 ≥ T+margin → 滑窗（0, stride, …, 末块贴边锚定 axis_len-T，块长恒为 T）；
    T ≤ 轴长 < T+margin → 单段整轴；轴长 < T → 单段 T 并需 pad（pad = T - axis_len）。
    """
    if axis_len >= tile_size + margin:
        last = axis_len - tile_size
        starts = list(range(0, last, stride))
        if not starts or starts[-1] != last:
            starts.append(last)  # 末块贴边锚定
        return [(s, tile_size) for s in starts], 0
    if axis_len >= tile_size:
        return [(0, axis_len)], 0
    return [(0, tile_size)], tile_size - axis_len


def split_fixed(im: Image.Image, tile_size: int, overlap: float = 0.15, margin: int | None = None):
    """固定尺寸 T×T 滑窗切块（tile_scheme="size"）。返回 (tiles, pad)。

    tiles = [(tile_image, (x0, y0, x1, y1)), ...]（先行后列，坐标在 pad 后的图上）；
    pad = (pad_w, pad_h) 为短边 < T 时的 reflect 补边量（stitch_fixed 拼完据此裁回）。
    按轴独立判定：轴长 ≥ T+margin 才在该轴滑窗；margin 缺省 = stride（调用方按
    设计文档传 2P，P=tile_target_patch_px）。两轴都不满足滑窗条件时返回单块
    （整图，可能经 pad）——由调用方决定后续处理（score_one/训练对小图先
    upscale_small 再调本函数）。
    """
    stride = max(1, round(tile_size * (1 - overlap)))
    if margin is None:
        margin = stride
    w, h = im.size
    x_seg, pad_w = _axis_segments(w, tile_size, stride, margin)
    y_seg, pad_h = _axis_segments(h, tile_size, stride, margin)
    if pad_w or pad_h:
        im = _reflect_pad(im, pad_w, pad_h)
    tiles = []
    for y0, lh in y_seg:
        for x0, lw in x_seg:
            tiles.append((im.crop((x0, y0, x0 + lw, y0 + lh)), (x0, y0, x0 + lw, y0 + lh)))
    return tiles, (pad_w, pad_h)


def stitch_fixed(tile_amaps, boxes, out_size: tuple[int, int]):
    """固定尺寸切块的 anomaly map 拼接：重叠区按相邻块中心连线的中垂线归属。

    每轴归属线 = 相邻起点块中心的中点（非比例切块的 w/nx 网格线），每个像素
    恰好归属一个块。boxes 为 split_fixed 返回的块框（pad 后图坐标，先行后列）；
    先按 pad 后尺寸拼，再裁回 out_size（原图/放大图的未 pad 尺寸 (W, H)）。
    返回 torch.Tensor (1, 1, H, W)。
    """
    import torch
    import torch.nn.functional as F

    def _axis_bounds(starts: list[int], lens: dict[int, int], axis_end: int) -> list[int]:
        centers = [s + lens[s] / 2 for s in starts]
        bounds = [0] + [int((centers[i - 1] + centers[i]) / 2) for i in range(1, len(starts))]
        return bounds + [axis_end]

    w, h = out_size
    x0s = sorted({b[0] for b in boxes})
    y0s = sorted({b[1] for b in boxes})
    xlen = {x0: next(b[2] - b[0] for b in boxes if b[0] == x0) for x0 in x0s}
    ylen = {y0: next(b[3] - b[1] for b in boxes if b[1] == y0) for y0 in y0s}
    w_pad = max(b[2] for b in boxes)
    h_pad = max(b[3] for b in boxes)
    xb = _axis_bounds(x0s, xlen, w_pad)
    yb = _axis_bounds(y0s, ylen, h_pad)
    merged = torch.zeros(h_pad, w_pad)
    for amap, (x0, y0, x1, y1) in zip(tile_amaps, boxes):
        ix, iy = x0s.index(x0), y0s.index(y0)
        cx0, cx1, cy0, cy1 = xb[ix], xb[ix + 1], yb[iy], yb[iy + 1]
        resized = F.interpolate(amap.squeeze().unsqueeze(0).unsqueeze(0).float(),
                                size=(y1 - y0, x1 - x0), mode="bilinear",
                                align_corners=False).squeeze()
        merged[cy0:cy1, cx0:cx1] = resized[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    return merged[:h, :w].unsqueeze(0).unsqueeze(0)
