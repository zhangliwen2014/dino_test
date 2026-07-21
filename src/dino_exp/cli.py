from __future__ import annotations

import json

import click

from dino_exp.config import load_config
from dino_exp.errors import DinoError


@click.group()
@click.option("--config", "config_path", default=None, help="配置文件路径，默认 config/default.yaml")
@click.pass_context
def main(ctx, config_path):
    """dino — DINO 无监督异常检测试验环境 CLI。"""
    ctx.obj = load_config(config_path)


def _err(fn):
    """统一异常出口：DinoError → 友好报错（含修复建议），退出码 2。"""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except DinoError as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapper


# ---------- dataset ----------

@main.group()
def dataset():
    """数据集管理。"""


@dataset.command("list")
@click.pass_obj
@_err
def dataset_list(cfg):
    from dino_exp.datasets import list_datasets

    rows = list_datasets(cfg)
    if not rows:
        click.echo("无数据集。请先 download 或 import。")
        return
    for info in rows:
        defects = ", ".join(f"{k}:{v}" for k, v in info.defect_types.items()) or "-"
        click.echo(f"{info.category}\ttrain/good={info.train_good}\ttest/good={info.test_good}\t缺陷=[{defects}]"
                   f"{'\t[降级:无NG图]' if info.degraded else ''}")


@dataset.command("download")
@click.option("--category", required=True)
@click.option("--force", is_flag=True, help="目标目录已存在时先删除再重新下载（用于中断后重试）")
@click.pass_obj
@_err
def dataset_download(cfg, category, force):
    from dino_exp.datasets import import_mvtec

    dest = import_mvtec(category, cfg, force=force)
    click.echo(f"已下载并转换: {dest}")


@dataset.command("import")
@click.option("--category", required=True)
@click.option("--label", type=click.Choice(["ok", "ng"]), required=True)
@click.option("--defect-type", default=None)
@click.option("--split", type=click.Choice(["train", "test", "auto"]), default="test", show_default=True,
              help="仅 OK 图有效：train=训练集 train/good；test=测试集 test/good；auto=按文件名 8:2 自动分配")
@click.argument("images", nargs=-1, required=True)
@click.pass_obj
@_err
def dataset_import(cfg, category, label, defect_type, split, images):
    from dino_exp.datasets import import_images

    paths = import_images(list(images), category, label, defect_type, cfg, split=split)
    click.echo(f"已导入 {len(paths)} 张到 {paths[0].parent.parent.name}/{paths[0].parent.name} 等目录")


@dataset.command("preview")
@click.option("--category", required=True)
@click.pass_obj
@_err
def dataset_preview(cfg, category):
    from dino_exp.datasets import dataset_info

    info = dataset_info(category, cfg)
    click.echo(json.dumps({
        "category": info.category, "train_good": info.train_good,
        "test_good": info.test_good, "defect_types": info.defect_types,
        "has_masks": info.has_masks, "degraded": info.degraded,
    }, ensure_ascii=False, indent=2))


# ---------- train / validate / test ----------

@main.command()
@click.option("--category", required=True)
@click.option("--backbone", default=None, help="骨干别名，默认取配置")
@click.option("--coreset", "coreset", type=float, default=None)
@click.option("--image-size", type=int, default=None)
@click.pass_obj
@_err
def train(cfg, category, backbone, coreset, image_size):
    from dino_exp.config import validate_image_size
    from dino_exp.train import train_model

    if backbone:
        cfg.backbone = backbone
        cfg.layers = cfg.backbone_spec.default_layers
    if coreset:
        cfg.coreset_sampling_ratio = coreset
    if image_size:
        validate_image_size(image_size, cfg.backbone_spec.patch_size)
        cfg.image_size = image_size
    result = train_model(category, cfg)
    click.echo(json.dumps(result, ensure_ascii=False, indent=2))


@main.command()
@click.option("--category", required=True)
@click.option("--version", default=None)
@click.option("--full", "full", is_flag=True)
@click.option("--images", "images", multiple=True, help="指定图像路径，可重复；缺省则跑全量验证")
@click.option("--errors-only", is_flag=True)
@click.pass_obj
@_err
def validate(cfg, category, version, full, images, errors_only):
    from dino_exp.validate import filter_errors, validate_full, validate_images

    if images:
        rows = validate_images(category, version, list(images), cfg)
        for r in rows:
            click.echo(f"{r['label']}\tscore={r['score']:.4f}\t{r['heatmap_path']}")
        return
    report = validate_full(category, version, cfg)
    click.echo(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    rows = filter_errors(report["rows"]) if errors_only else report["rows"]
    for r in rows:
        click.echo(f"{r['label_pred']}\tscore={r['score']:.4f}\tgt={r['defect_type']}\t{r['path']}")


@main.command(name="test")
@click.option("--category", required=True)
@click.option("--image", "images", multiple=True, required=True)
@click.option("--version", default=None)
@click.pass_obj
@_err
def test_cmd(cfg, category, images, version):
    from dino_exp.infer import infer_batch

    for r in infer_batch(list(images), version, category=category, cfg=cfg):
        click.echo(f"{r['label']}\tscore={r['score']:.4f}\tthreshold={r['threshold']:.4f}\t{r['heatmap_path']}")


# ---------- feedback / retrain / versions ----------

@main.command()
@click.option("--category", required=True)
@click.option("--image", required=True)
@click.option("--label", type=click.Choice(["ok", "ng"]), required=True)
@click.option("--defect-type", default=None)
@click.option("--score", type=float, default=None, help="预测分数（来自 dino test 输出）")
@click.option("--prediction", default=None)
@click.pass_obj
@_err
def feedback(cfg, category, image, label, defect_type, score, prediction):
    from dino_exp.feedback.store import FeedbackStore
    from dino_exp.models.registry import Registry

    version = Registry(cfg.models_root).current(category)
    rec = FeedbackStore(cfg.feedback_root, category).stage({
        "image_path": image, "model_version": version,
        "prediction": prediction or "unknown", "score": score if score is not None else 0.0,
        "human_label": label, "defect_type": defect_type,
    })
    click.echo(f"已暂存反馈 {rec['id']}（{label}）。")


@main.command()
@click.option("--category", required=True)
@click.option("--yes", is_flag=True, help="跳过确认直接执行")
@click.pass_obj
@_err
def retrain(cfg, category, yes):
    from dino_exp.retrain import preview_retrain
    from dino_exp.retrain import retrain as do_retrain

    pv = preview_retrain(category, cfg)
    click.echo(json.dumps(
        {"current_version": pv["current_version"], "ok": pv["ok"], "ng": pv["ng"],
         "suspicious": len(pv["suspicious"]), "conflicts": pv["conflicts"]},
        ensure_ascii=False, indent=2))
    if pv["suspicious"]:
        click.echo(f"警告: {len(pv['suspicious'])} 条可疑反馈（OK 高分 / NG 低分），请先确认。")
    if pv["conflicts"]:
        click.echo(f"警告: 同图冲突 {len(pv['conflicts'])} 起，将以最新一条为准。")
    if not yes and not click.confirm("确认执行再训练？"):
        click.echo("已取消。")
        return
    result = do_retrain(category, cfg)
    click.echo(json.dumps({k: v for k, v in result.items() if k != "preview"}, ensure_ascii=False, indent=2, default=str))
    if result["warning"]:
        click.echo(f"⚠ {result['warning']}")


@main.command()
@click.option("--category", required=True)
@click.pass_obj
@_err
def versions(cfg, category):
    from dino_exp.models.registry import Registry

    reg = Registry(cfg.models_root)
    cur = reg.current(category)
    for v in reg.list(category):
        click.echo(f"{'*' if v == cur else ' '} {v}")


@main.command()
@click.option("--category", required=True)
@click.argument("version")
@click.pass_obj
@_err
def rollback(cfg, category, version):
    from dino_exp.models.registry import Registry

    Registry(cfg.models_root).rollback(category, version)
    click.echo(f"已回滚到 {version}。")


@main.command()
@click.option("--category", required=True)
@click.option("--version", default=None, help="缺省为当前版本；导出物写入该版本 export/ 目录")
@click.pass_obj
@_err
def export(cfg, category, version):
    """导出当前版本的 OpenVINO 快照（再训练后需对新版本重新导出）。"""
    from dino_exp.infer import export_openvino

    out = export_openvino(category, version, cfg)
    click.echo(f"已导出: {out}")


@main.command()
@click.option("--category", required=True)
@click.argument("feedback_id")
@click.pass_obj
@_err
def unstage(cfg, category, feedback_id):
    """从暂存区单条删除反馈（FR-5.6）。"""
    from dino_exp.feedback.store import FeedbackStore

    if FeedbackStore(cfg.feedback_root, category).remove(feedback_id):
        click.echo(f"已删除反馈 {feedback_id}。")
    else:
        raise DinoError(f"暂存区无反馈 {feedback_id}。可用 `dino retrain --category {category}` 预览现有暂存反馈。")


@main.command()
@click.option("--port", default=7860, show_default=True, help="Web UI 监听端口")
@click.option("--host", default="127.0.0.1", show_default=True,
              help="绑定地址；0.0.0.0 允许局域网访问（无鉴权，仅在可信网络使用）")
@click.pass_obj
@_err
def ui(cfg, port, host):
    from dino_exp.webui.app import launch

    launch(cfg, port=port, host=host)


if __name__ == "__main__":
    main()
