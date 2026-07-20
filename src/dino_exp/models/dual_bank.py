"""DualBankPatchcore：Patchcore 子类 + 缺陷原型库（设计文档 §3.2）。

分层：
- 纯函数（apply_defect_boost / coreset_cap / merge_pinned / calibrate_threshold）：
  全部业务规则，单元测试只覆盖这层，不依赖真实骨干。
- DualBankPatchcoreModel(PatchcoreModel)：defect_bank buffer、钉住区管理、
  nearest_neighbors 注入加分（仅 n_neighbors==1 的 patch 打分支路，不影响
  compute_anomaly_score 内部的 support-sample 查询）。
- DualBankPatchcore(Patchcore)：__init__ 替换 self.model；fit() 直接用
  KCenterGreedy（PatchcoreModel.subsample_embedding 自 anomalib 2.1 起 deprecated）；
  apply_threshold() 把 mean+3σ 注入 PostProcessor（ManualThreshold），使指标、
  判定、导出 metadata 共用同一阈值。
"""

from __future__ import annotations

import statistics
from pathlib import Path

import torch
from anomalib.metrics import ManualThreshold
from anomalib.models.components import KCenterGreedy
from anomalib.models.image.patchcore.lightning_model import Patchcore
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

from dino_exp.errors import DinoError

# ---------------- 纯函数（单测目标） ----------------


def apply_defect_boost(patch_scores: torch.Tensor, defect_dists: torch.Tensor, w: float) -> torch.Tensor:
    """缺陷库加分：d_d < d_n 的 patch 分数 ×(1+w)；只加分不减分。"""
    boost = (defect_dists < patch_scores).to(patch_scores.dtype) * w
    return patch_scores * (1.0 + boost)


def coreset_cap(base_size: int, cap_ratio: float, pinned: int) -> int:
    """非钉住区容量 = 基础版本库大小 × 固定倍数 - 钉住数（至少 1）。"""
    return max(1, int(base_size * cap_ratio) - pinned)


def merge_pinned(
    memory_bank: torch.Tensor, pinned_count: int, new_feats: torch.Tensor, pinned: bool
) -> tuple[torch.Tensor, int]:
    """合并新特征，维持 [钉住区..., 非钉住区...] 布局；返回 (新库, 新钉住数)。"""
    if pinned:
        merged = torch.cat([memory_bank[:pinned_count], new_feats, memory_bank[pinned_count:]])
        return merged, pinned_count + len(new_feats)
    return torch.cat([memory_bank, new_feats]), pinned_count


def calibrate_threshold(ok_scores: list[float], sigma: float = 3.0) -> float:
    """mean + sigma × 样本标准差；单样本时 std=0。"""
    if not ok_scores:
        raise DinoError("校准集为空，无法计算阈值。请检查数据集 test/good（或 train/good 切分）是否存在图片。")
    mean = statistics.fmean(ok_scores)
    std = statistics.stdev(ok_scores) if len(ok_scores) > 1 else 0.0
    return mean + sigma * std


# ---------------- 模型层（薄封装） ----------------


class DualBankPatchcoreModel(PatchcoreModel):
    def __init__(
        self,
        *,
        layers,
        backbone,
        pre_trained: bool = True,
        num_neighbors: int = 9,
        fusion_weight: float = 0.5,
        bank_cap_ratio: float = 1.5,
        coreset_sampling_ratio: float = 0.1,
    ) -> None:
        super().__init__(layers=layers, backbone=backbone, pre_trained=pre_trained, num_neighbors=num_neighbors)
        self.fusion_weight = fusion_weight
        self.bank_cap_ratio = bank_cap_ratio
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.register_buffer("defect_bank", torch.empty(0))
        self.register_buffer("pinned_count", torch.zeros(1, dtype=torch.long))
        self.base_bank_size: int | None = None  # 基础版本库大小；持久化在 meta.json

    def nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int):
        patch_scores, locations = super().nearest_neighbors(embedding, n_neighbors)
        # 仅 patch 打分支路（forward 固定 n_neighbors=1）注入缺陷库加分，
        # 不影响 compute_anomaly_score 内部的 support-sample 查询（n>1）。
        if n_neighbors == 1 and self.defect_bank.dim() == 2 and self.defect_bank.shape[0] > 0:
            d_def = self.euclidean_dist(embedding, self.defect_bank).min(dim=1).values
            patch_scores = apply_defect_boost(patch_scores, d_def, self.fusion_weight)
        return patch_scores, locations

    @property
    def _pinned(self) -> int:
        return int(self.pinned_count.item())

    def add_normal_features(self, feats: torch.Tensor, pinned: bool = False) -> None:
        feats = feats.to(self.memory_bank.device, self.memory_bank.dtype)
        merged, count = merge_pinned(self.memory_bank, self._pinned, feats, pinned)
        self.memory_bank = merged
        self.pinned_count = torch.tensor([count], dtype=torch.long, device=merged.device)
        self.resample_normal_bank()

    def add_defect_features(self, feats: torch.Tensor) -> None:
        feats = feats.to(self.memory_bank.device, self.memory_bank.dtype)
        self.defect_bank = feats if self.defect_bank.numel() == 0 else torch.cat([self.defect_bank, feats])

    def fit_coreset(self) -> None:
        """初始建库：vstack embedding_store → coreset 采样 → memory_bank。"""
        if not self.embedding_store:
            raise DinoError("embedding_store 为空，无法 coreset。请确认 Engine.fit 已遍历训练集。")
        feats = torch.vstack(self.embedding_store)
        self.embedding_store.clear()
        ratio = self.coreset_sampling_ratio
        if ratio < 1.0 and int(len(feats) * ratio) >= 1:
            feats = KCenterGreedy(embedding=feats, sampling_ratio=ratio).sample_coreset()
        self.memory_bank = feats
        self.pinned_count = torch.zeros(1, dtype=torch.long, device=feats.device)
        self.base_bank_size = feats.shape[0]

    def resample_normal_bank(self) -> None:
        """非钉住区超过上限时 coreset 重采样淘汰；钉住区豁免且不计入上限。"""
        if self.base_bank_size is None:
            return
        p = self._pinned
        pinned_part = self.memory_bank[:p]
        unpinned = self.memory_bank[p:]
        cap = coreset_cap(self.base_bank_size, self.bank_cap_ratio, p)
        if len(unpinned) <= cap:
            return
        # +0.5 抵消浮点舍入：cap/len 可能使 int(len * ratio) 少 1（如 cap=1, len=49
        # 时 49*(1/49)=0.9999...→0，KCenterGreedy 抛 ValueError），+0.5 保证取到 cap。
        sampled = KCenterGreedy(embedding=unpinned, sampling_ratio=(cap + 0.5) / len(unpinned)).sample_coreset()
        self.memory_bank = torch.cat([pinned_part, sampled.to(pinned_part.device)])


class DualBankPatchcore(Patchcore):
    def __init__(
        self,
        *,
        backbone: str,
        layers,
        image_size: tuple[int, int] = (224, 224),
        fusion_weight: float = 0.5,
        bank_cap_ratio: float = 1.5,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        pre_trained: bool = True,
        evaluator=True,
    ) -> None:
        pre_processor = Patchcore.configure_pre_processor(image_size=image_size)
        super().__init__(
            backbone=backbone,
            layers=layers,
            pre_trained=False,  # 父类构造的 PatchcoreModel 随即被替换，避免重复加载权重
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
            pre_processor=pre_processor,
            evaluator=evaluator,
        )
        self.model = DualBankPatchcoreModel(
            layers=layers,
            backbone=backbone,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
            fusion_weight=fusion_weight,
            bank_cap_ratio=bank_cap_ratio,
            coreset_sampling_ratio=coreset_sampling_ratio,
        )

    def fit(self) -> None:
        """MemoryBankMixin 钩子在 train epoch 结束时调用；直接用 KCenterGreedy。"""
        self.model.fit_coreset()

    def apply_threshold(self, threshold: float) -> None:
        """把 mean+3σ 阈值注入 PostProcessor（指标/判定/导出共用同一阈值）。

        ManualThreshold 继承纯 torchmetrics.Metric（不接受 fields/strict，
        与 F1AdaptiveThreshold 的 AnomalibMetric 层级不同），但其 update 为
        no-op 且 compute 恒返回 default_value，因此 validation 重算阈值时
        写回的仍是同一阈值，口径不变。

        生效前提：PostProcessor 的判定（pred_label/pred_mask）基于归一化分数
        与归一化阈值比较，依赖 validation 阶段拟合的 normalization min/max，
        未跑过 validation 时归一化阈值为 NaN；本项目应用层判定直接比 raw
        score 与阈值，不经 PostProcessor 归一化路径。
        """
        t = float(threshold)
        pp = self.post_processor
        pp._image_threshold_metric = ManualThreshold(default_value=t)
        pp._pixel_threshold_metric = ManualThreshold(default_value=t)
        pp._image_threshold = torch.tensor(t)
        pp._pixel_threshold = torch.tensor(t)


# ---------------- 特征提取辅助（训练/再训练/反馈共用） ----------------


def extract_embeddings(model: DualBankPatchcoreModel, images: torch.Tensor) -> torch.Tensor:
    """对已预处理图片张量提取 patch embedding，返回 (B*H*W, D)。"""
    with torch.no_grad():
        features = model.feature_extractor(images)
    features = {k: model.feature_pooler(v) for k, v in features.items()}
    embedding = model.generate_embedding(features)
    return model.reshape_embedding(embedding)


def topk_defect_features(model: DualBankPatchcoreModel, image: torch.Tensor, k: int) -> torch.Tensor:
    """取单图异常分数 top-k 的 patch 特征（NG 反馈入缺陷库）。"""
    embedding = extract_embeddings(model, image)
    if model.memory_bank.numel() == 0:
        raise DinoError("正常记忆库为空，无法计算 patch 分数。请先完成基础训练。")
    patch_scores, _ = model.nearest_neighbors(embedding, n_neighbors=1)
    k = min(k, embedding.shape[0])
    idx = torch.topk(patch_scores, k=k).indices
    return embedding[idx]


def save_banks(model: DualBankPatchcoreModel, path: str | Path) -> None:
    """统一存储格式：normal_bank.pt 为字典（含缺陷库与钉住元数据）。"""
    torch.save(
        {
            "memory_bank": model.memory_bank.cpu(),
            "defect_bank": model.defect_bank.cpu(),
            "pinned_count": int(model.pinned_count.item()),
            "base_bank_size": model.base_bank_size,
        },
        path,
    )


def load_banks(model: DualBankPatchcoreModel, bank_dir: str | Path) -> None:
    """从版本目录恢复双库。

    存储约定：normal_bank.pt 为 save_banks 字典格式（主数据源）；
    defect_bank.pt 在无缺陷库时是 registry 写入的 torch.empty(0) 占位张量，
    若是 2D 张量（独立存档的缺陷库）则优先使用。

    请在 model.to(device) 之前调用：本函数加载到 CPU，buffer 随后随 module 迁移。
    """
    bank_dir = Path(bank_dir)
    nb = torch.load(bank_dir / "normal_bank.pt", map_location="cpu", weights_only=True)
    db_path = bank_dir / "defect_bank.pt"
    defect = (
        torch.load(db_path, map_location="cpu", weights_only=True)
        if db_path.exists() else torch.empty(0)
    )
    model.memory_bank = nb["memory_bank"]
    model.defect_bank = defect if defect.dim() == 2 else nb["defect_bank"]
    model.pinned_count = torch.tensor([nb["pinned_count"]], dtype=torch.long)
    model.base_bank_size = nb["base_bank_size"] if nb["base_bank_size"] is not None else nb["memory_bank"].shape[0]
