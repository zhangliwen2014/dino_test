from __future__ import annotations

import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from dino_exp.config import Config
from dino_exp.errors import DinoError

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MVTEC_ROOT = Path("datasets/MVTecAD")  # anomalib 自动下载根目录


@dataclass
class DatasetInfo:
    category: str
    root: Path
    train_good: int
    test_good: int
    defect_types: dict[str, int] = field(default_factory=dict)
    has_masks: bool = False

    @property
    def has_test_good(self) -> bool:
        return self.test_good > 0

    @property
    def degraded(self) -> bool:
        """无 NG 测试图：指标降级（仅阈值校准+逐图分数）。"""
        return sum(self.defect_types.values()) == 0


def _imgs(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)


def dataset_info(category: str, cfg: Config) -> DatasetInfo:
    root = cfg.data_root / category
    train_good = _imgs(root / "train" / "good")
    if not train_good:
        raise DinoError(
            f"类别 '{category}' 缺少 train/good 图片（{root / 'train' / 'good'}）。"
            "请先 `dino dataset download --category ...` 或 `dino dataset import ...` 导入 OK 训练图。"
        )
    test_dir = root / "test"
    defect_types = {
        d.name: len(_imgs(d))
        for d in sorted(test_dir.iterdir())
        if d.is_dir() and d.name != "good" and _imgs(d)
    } if test_dir.is_dir() else {}
    mask_root = root / "mask"
    has_masks = mask_root.is_dir() and any(_imgs(d) for d in mask_root.iterdir() if d.is_dir())
    return DatasetInfo(
        category=category,
        root=root,
        train_good=len(train_good),
        test_good=len(_imgs(test_dir / "good")),
        defect_types=defect_types,
        has_masks=has_masks,
    )


def list_datasets(cfg: Config) -> list[DatasetInfo]:
    if not cfg.data_root.is_dir():
        return []
    return [dataset_info(d.name, cfg) for d in sorted(cfg.data_root.iterdir()) if d.is_dir()]


def import_images(srcs: list[str | Path], category: str, label: str, defect_type: str | None, cfg: Config) -> list[Path]:
    label = label.lower()
    if label not in {"ok", "ng"}:
        raise DinoError(f"label 只能是 ok/ng，得到 '{label}'。")
    if label == "ng" and not defect_type:
        raise DinoError("NG 图片必须指定缺陷类型名（--defect-type），如 scratch。")
    dest_dir = cfg.data_root / category / "test" / ("good" if label == "ok" else defect_type)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for src in srcs:
        src = Path(src)
        if src.suffix.lower() not in IMG_EXTS:
            raise DinoError(f"不支持的图片格式 '{src.suffix}'。支持: {sorted(IMG_EXTS)}")
        dest = dest_dir / src.name
        if dest.exists():
            raise DinoError(f"目标已存在: {dest}。请重命名来源图片或先删除旧文件。")
        shutil.copy2(src, dest)
        out.append(dest)
    return out


def import_mvtec(category: str, cfg: Config) -> Path:
    """调用 anomalib MVTecAD 自动下载，再拷贝/重命名为统一目录规范。"""
    from anomalib.data import MVTecAD

    MVTecAD(root=str(MVTEC_ROOT), category=category).prepare_data()  # 触发下载解压
    src = MVTEC_ROOT / category
    if not src.is_dir():
        raise DinoError(f"MVTec 下载后未找到 {src}。请检查网络后重试 `dino dataset download`。")
    dest = cfg.data_root / category
    if dest.exists():
        raise DinoError(f"目标已存在: {dest}。如需重新导入请先删除该目录。")
    shutil.copytree(src / "train", dest / "train")
    shutil.copytree(src / "test", dest / "test")
    if (src / "ground_truth").is_dir():
        shutil.copytree(src / "ground_truth", dest / "mask")
    return dest


def ok_calibration_images(category: str, cfg: Config, ratio: float = 0.2) -> list[Path]:
    """OK 校准图：优先 test/good；缺失时从 train/good 确定性切 20%（FR-2.3）。"""
    info = dataset_info(category, cfg)
    good_test = _imgs(info.root / "test" / "good")
    if good_test:
        return good_test
    train = _imgs(info.root / "train" / "good")
    n = max(1, int(len(train) * ratio))
    rng = random.Random(42)  # 固定种子保证可复现
    return sorted(rng.sample(train, n))


def test_images_with_labels(category: str, cfg: Config) -> list[tuple[Path, int, str]]:
    """(path, label 0=OK/1=NG, defect_type) 全量 test 清单。"""
    info = dataset_info(category, cfg)
    rows = [(p, 0, "good") for p in _imgs(info.root / "test" / "good")]
    for dt in info.defect_types:
        rows += [(p, 1, dt) for p in _imgs(info.root / "test" / dt)]
    return rows


# 函数名以 test_ 开头，会被 pytest 在被 import 的测试模块中误收集；显式排除
test_images_with_labels.__test__ = False


def build_folder(category: str, cfg: Config):
    """构造 anomalib Folder datamodule。

    多缺陷类型：Folder 的 abnormal_dir/mask_dir 接受 Sequence，直接传缺陷子目录
    路径列表（即设计文档"manifest"的实现形式）。缺 test/good 时 normal_split_ratio=0.2。
    """
    from anomalib.data import Folder

    info = dataset_info(category, cfg)
    defect_dirs = [f"test/{dt}" for dt in info.defect_types]
    mask_dirs = [f"mask/{dt}" for dt in info.defect_types if (info.root / "mask" / dt).is_dir()]
    kwargs: dict = {}
    if defect_dirs:
        kwargs["abnormal_dir"] = defect_dirs  # Sequence[str]：多缺陷类型合并（v2.5.1 已核实支持）
        # 仅当每个缺陷类型都有 mask 目录时才传 mask（保证与 abnormal_dir 对齐）
        kwargs["mask_dir"] = mask_dirs if len(mask_dirs) == len(defect_dirs) else None
    if info.has_test_good:
        kwargs["normal_test_dir"] = "test/good"
    return Folder(
        name=category,
        root=info.root,
        normal_dir="train/good",
        normal_split_ratio=0.0 if info.has_test_good else 0.2,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        **kwargs,
    )
