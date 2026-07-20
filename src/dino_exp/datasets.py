from __future__ import annotations

import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from dino_exp.config import Config
from dino_exp.errors import DinoError

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MVTEC_ROOT = Path("datasets/MVTecAD")  # anomalib 自动下载根目录
SPLIT_SEED = 42  # anomalib 切分种子：保证校准集可复现


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
    if test_dir.is_dir():
        counts = {d.name: len(_imgs(d)) for d in sorted(test_dir.iterdir()) if d.is_dir() and d.name != "good"}
        defect_types = {name: n for name, n in counts.items() if n > 0}
    else:
        defect_types = {}
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
    rows = []
    for d in sorted(cfg.data_root.iterdir()):
        if not d.is_dir():
            continue
        try:
            rows.append(dataset_info(d.name, cfg))
        except DinoError as e:
            warnings.warn(f"跳过不完整类别 '{d.name}': {e}", stacklevel=2)
    return rows


def import_images(srcs: list[str | Path], category: str, label: str, defect_type: str | None, cfg: Config) -> list[Path]:
    label = label.lower()
    if label not in {"ok", "ng"}:
        raise DinoError(f"label 只能是 ok/ng，得到 '{label}'。")
    if label == "ng" and not defect_type:
        raise DinoError("NG 图片必须指定缺陷类型名（--defect-type），如 scratch。")
    dest_dir = cfg.data_root / category / "test" / ("good" if label == "ok" else defect_type)
    # 先全量校验（格式 + 目标重名 + 批次内重名），再统一拷贝：失败零落盘
    pairs = []
    for src in srcs:
        src = Path(src)
        if src.suffix.lower() not in IMG_EXTS:
            raise DinoError(f"不支持的图片格式 '{src.suffix}'。支持: {sorted(IMG_EXTS)}")
        dest = dest_dir / src.name
        if dest.exists():
            raise DinoError(f"目标已存在: {dest}。请重命名来源图片或先删除旧文件。")
        pairs.append((src, dest))
    names = [src.name for src, _ in pairs]
    if len(set(names)) != len(names):
        raise DinoError(f"批次内存在同名图片: {sorted(n for n in names if names.count(n) > 1)}。请重命名后重试。")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for src, dest in pairs:
        shutil.copy2(src, dest)
        out.append(dest)
    return out


def import_mvtec(category: str, cfg: Config) -> Path:
    """调用 anomalib MVTecAD 自动下载，再拷贝/重命名为统一目录规范。

    先拷到临时目录，成功后 rename 为正式目录；任何一步失败都清理临时目录，
    不留下半个数据集。
    """
    from anomalib.data import MVTecAD

    MVTecAD(root=str(MVTEC_ROOT), category=category).prepare_data()  # 触发下载解压
    src = MVTEC_ROOT / category
    if not src.is_dir():
        raise DinoError(f"MVTec 下载后未找到 {src}。请检查网络后重试 `dino dataset download`。")
    dest = cfg.data_root / category
    if dest.exists():
        raise DinoError(f"目标已存在: {dest}。如需重新导入请先删除该目录。")
    tmp = dest.parent / (dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)  # 清理上次失败遗留
    try:
        tmp.mkdir(parents=True)
        shutil.copytree(src / "train", tmp / "train")
        shutil.copytree(src / "test", tmp / "test")
        if (src / "ground_truth").is_dir():
            shutil.copytree(src / "ground_truth", tmp / "mask")
        tmp.rename(dest)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest


def ok_calibration_images(category: str, cfg: Config) -> list[Path]:
    """OK 校准图（FR-2.3）：优先 test/good；缺失时取 anomalib 切分结果中 val 集
    的正常样本（seed=42 从 train/good 切出的 20%），与 build_folder 单一事实源，
    保证校准集不与训练集重叠且可复现。
    """
    info = dataset_info(category, cfg)
    good_test = _imgs(info.root / "test" / "good")
    if good_test:
        return good_test
    dm = build_folder(category, cfg)
    dm.setup("fit")  # 仅构建样本清单与切分，不解码图片
    samples = dm.val_data.samples  # val_split_mode=SAME_AS_TEST → val 含全部 test 样本
    return sorted(Path(p) for p in samples[samples.label_index == 0].image_path)


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
    路径列表（即设计文档"manifest"的实现形式）。缺 test/good 时 anomalib 按
    test_split_ratio=0.2 从 train/good 切出 test 的正常部分（seed=42 可复现）；
    有 test/good 时不切分。seed=42 保证切分可复现；val_split_mode=SAME_AS_TEST
    使 val 集与 test 集一致，ok_calibration_images 据此读取 OK 校准图（单一事实源）。

    注：v2.5.1 的 normal_split_ratio 是死参数（仅存不用），勿依赖。
    """
    from anomalib.data import Folder
    from anomalib.data.utils import ValSplitMode

    info = dataset_info(category, cfg)
    defect_dirs = [f"test/{dt}" for dt in info.defect_types]
    mask_dirs = [f"mask/{dt}" for dt in info.defect_types if (info.root / "mask" / dt).is_dir()]
    kwargs: dict = {}
    if defect_dirs:
        kwargs["abnormal_dir"] = defect_dirs  # Sequence[str]：多缺陷类型合并（v2.5.1 已核实支持）
        if len(mask_dirs) == len(defect_dirs):
            kwargs["mask_dir"] = mask_dirs
        else:
            if mask_dirs:  # 部分缺陷类型有 mask：对齐不了 abnormal_dir，整体弃用并告警
                missing = [dt for dt in info.defect_types if f"mask/{dt}" not in mask_dirs]
                warnings.warn(
                    f"缺陷类型 {'、'.join(missing)} 缺少 mask 目录，mask_dir 已整体弃用，像素级指标不可用",
                    stacklevel=2,
                )
            kwargs["mask_dir"] = None
    if info.has_test_good:
        kwargs["normal_test_dir"] = "test/good"
    return Folder(
        name=category,
        root=info.root,
        normal_dir="train/good",
        test_split_ratio=0.2,  # 无 test/good 时从 train/good 切 20% 作 test 正常样本（真实生效参数）
        val_split_mode=ValSplitMode.SAME_AS_TEST,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        seed=SPLIT_SEED,
        **kwargs,
    )
