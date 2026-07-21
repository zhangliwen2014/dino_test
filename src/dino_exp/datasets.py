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
    error: str | None = None  # 不完整类别的缺漏说明（如缺 train/good），完整类别为 None

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


def _check_category_name(category: str) -> None:
    if not category or category in {".", ".."} or "/" in category or "\\" in category:
        raise DinoError(f"非法类别名 '{category}'。类别名不能含路径分隔符。")


def delete_category(category: str, cfg: Config) -> Path:
    """删除整个类别目录（不可恢复）。仅允许删除 data_root 下的直接子目录。"""
    _check_category_name(category)
    target = cfg.data_root / category
    if not target.is_dir():
        raise DinoError(f"类别 '{category}' 不存在（{target}）。请用 `dino dataset list` 查看现有类别。")
    shutil.rmtree(target)
    return target


def fix_category(category: str, cfg: Config) -> dict:
    """自动修复不完整类别：缺 train/good 时，把 test/good 的图按 8:2 整理（80% 移到
    train/good 作训练集，20% 留在 test/good 作阈值校准集）。返回处理摘要。

    不可修复的情况（抛 DinoError）：类别不存在；test/good 也没有可用图片。
    """
    _check_category_name(category)
    root = cfg.data_root / category
    if not root.is_dir():
        raise DinoError(f"类别 '{category}' 不存在（{root}）。请先导入图片。")
    train_dir = root / "train" / "good"
    if _imgs(train_dir):
        return {"category": category, "moved_to_train": 0, "kept_in_test": 0,
                "note": "train/good 已有图片，类别完整，无需处理。"}
    test_good = _imgs(root / "test" / "good")
    if not test_good:
        raise DinoError(
            f"类别 '{category}' 无法自动修复：train/good 与 test/good 都没有图片。"
            "请先用 `dino dataset import --label ok --split auto` 导入 OK 图，或删除该类别。"
        )
    n_train = max(1, int(len(test_good) * 0.8))
    to_train, kept = test_good[:n_train], test_good[n_train:]
    train_dir.mkdir(parents=True, exist_ok=True)
    for p in to_train:
        p.rename(train_dir / p.name)
    return {"category": category, "moved_to_train": len(to_train), "kept_in_test": len(kept),
            "note": f"已按 8:2 整理：{len(to_train)} 张移入 train/good，{len(kept)} 张留在 test/good。"}


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
    """列出全部类别；不完整类别（如缺 train/good）也返回，error 字段带缺漏说明。"""
    if not cfg.data_root.is_dir():
        return []
    rows = []
    for d in sorted(cfg.data_root.iterdir()):
        if not d.is_dir():
            continue
        try:
            rows.append(dataset_info(d.name, cfg))
        except DinoError as e:
            rows.append(DatasetInfo(category=d.name, root=d, train_good=0, test_good=0, error=str(e)))
    return rows


def import_images(
    srcs: list[str | Path], category: str, label: str, defect_type: str | None, cfg: Config,
    split: str = "test",
) -> list[Path]:
    """导入图片到规范目录。

    split（仅 OK 图有效）："test" → test/good（默认，兼容旧行为）；
    "train" → train/good（训练集）；"auto" → 按文件名确定性 8:2 分入 train/good 与
    test/good（从零建数据集时一步获得训练集与阈值校准集）。NG 图始终进 test/<缺陷类型>。
    """
    label = label.lower()
    if label not in {"ok", "ng"}:
        raise DinoError(f"label 只能是 ok/ng，得到 '{label}'。")
    if label == "ng" and not defect_type:
        raise DinoError("NG 图片必须指定缺陷类型名（--defect-type），如 scratch。")
    if split not in {"train", "test", "auto"}:
        raise DinoError(f"split 只能是 train/test/auto，得到 '{split}'。")

    root = cfg.data_root / category

    def dest_for(src: Path, train: bool) -> Path:
        if label == "ng":
            return root / "test" / defect_type / src.name
        return root / ("train" if train else "test") / "good" / src.name

    # 先全量校验（格式 + 目标重名 + 批次内重名），再统一拷贝：失败零落盘
    sorted_srcs = sorted((Path(s) for s in srcs), key=lambda p: p.name)
    pairs = []
    n = len(sorted_srcs)
    for i, src in enumerate(sorted_srcs):
        if src.suffix.lower() not in IMG_EXTS:
            raise DinoError(f"不支持的图片格式 '{src.suffix}'。支持: {sorted(IMG_EXTS)}")
        if label == "ok" and split == "auto":
            train = i < max(1, int(n * 0.8))  # 前 80% 进 train/good，其余进 test/good
        else:
            train = split == "train"
        dest = dest_for(src, train)
        if dest.exists():
            raise DinoError(f"目标已存在: {dest}。请重命名来源图片或先删除旧文件。")
        pairs.append((src, dest))
    names = [src.name for src, _ in pairs]
    if len(set(names)) != len(names):
        raise DinoError(f"批次内存在同名图片: {sorted(n for n in names if names.count(n) > 1)}。请重命名后重试。")
    out = []
    for src, dest in pairs:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        out.append(dest)
    return out


def import_mvtec(category: str, cfg: Config, force: bool = False) -> Path:
    """调用 anomalib MVTecAD 自动下载，再拷贝/重命名为统一目录规范。

    先拷到临时目录，成功后 rename 为正式目录；任何一步失败都清理临时目录，
    不留下半个数据集。force=True 时先删除已存在的目标目录（用于中断后重下）。
    """
    from anomalib.data import MVTecAD

    dest = cfg.data_root / category
    if dest.exists():
        if not force:
            raise DinoError(
                f"目标已存在: {dest}。若是上次中断留下的半成品，请加 --force 重新下载，"
                "或手动删除该目录后重试。"
            )
        shutil.rmtree(dest)

    MVTecAD(root=str(MVTEC_ROOT), category=category).prepare_data()  # 触发下载解压
    src = MVTEC_ROOT / category
    if not src.is_dir():
        raise DinoError(f"MVTec 下载后未找到 {src}。请检查网络后重试 `dino dataset download`。")
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


def category_images(category: str, cfg: Config) -> list[tuple[str, Path]]:
    """类别下全部图片的 (相对显示路径, 绝对路径) 列表，供 UI 选图（FR-3.2 选图验证）。

    相对路径形如 train/good/t0.png、test/broken/b0.png，按字典序排序。
    """
    info = dataset_info(category, cfg)
    out = []
    for p in sorted(info.root.rglob("*")):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            out.append((p.relative_to(info.root).as_posix(), p))
    return out


def mask_path_for(image_path: Path, defect_type: str, info: DatasetInfo) -> Path | None:
    """NG 测试图对应的 GT mask（MVTec 约定 mask/<defect_type>/<stem>_mask.png）；无则 None。"""
    if defect_type == "good":
        return None
    p = info.root / "mask" / defect_type / f"{image_path.stem}_mask.png"
    return p if p.is_file() else None


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
