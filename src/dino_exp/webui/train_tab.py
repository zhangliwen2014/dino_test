import dataclasses

import gradio as gr

from dino_exp.config import resolve_backbone, validate_image_size
from dino_exp.train import train_model
from dino_exp.webui.common import category_dropdown
from dino_exp.webui.jobs import JobManager


def build(cfg, jm: JobManager):
    cat = category_dropdown(cfg, label="类别（选择已导入的数据集）")
    backbone = gr.Dropdown(
        ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14",
         "dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"],
        value=cfg.backbone, label="骨干")
    coreset = gr.Slider(0.01, 1.0, value=cfg.coreset_sampling_ratio, label="coreset 采样率")
    image_size = gr.Number(value=cfg.image_size, label="输入尺寸（patch 整数倍）")
    tiles = gr.Dropdown(
        [("off：不切块", "off"), ("auto：固定尺寸切块（推荐）", "auto"),
         ("2x2（旧网格）", "2x2"), ("3x3（旧网格）", "3x3"), ("4x4（旧网格）", "4x4"),
         ("6x6（旧网格）", "6x6"), ("8x8（旧网格）", "8x8")],
        value=cfg.tile_mode, label="切块（大图提升小缺陷分辨率）")
    btn = gr.Button("开始训练", variant="primary")
    status = gr.Textbox(label="状态", interactive=False)
    logs = gr.Textbox(label="日志", interactive=False, lines=12)
    result = gr.JSON(label="验证指标（训练完成自动全量验证）")
    state_jid = gr.State(None)

    def start(c, bb, cs, sz, tm):
        if not c or not c.strip():
            return None, "请先输入类别名（如 bottle）。", ""
        try:
            validate_image_size(int(sz), resolve_backbone(bb).patch_size)
            run_cfg = dataclasses.replace(
                cfg, backbone=bb, coreset_sampling_ratio=float(cs), image_size=int(sz),
                tile_mode=tm)
            run_cfg.layers = list(run_cfg.backbone_spec.default_layers)
            jid = jm.start("train", lambda log: train_model(c, run_cfg, log=log))
            return jid, f"训练已启动: {jid}", ""
        except Exception as exc:
            from dino_exp.webui.common import error_pair

            summary, detail = error_pair(exc)
            return None, summary, detail

    def poll(jid):
        if not jid:
            return "未启动", "", None
        st = jm.status(jid)
        text = "".join(st["logs"])
        if st["state"] == "done":
            return "完成", text, st["result"]
        if st["state"] == "error":
            return f"失败:\n{st['error']}", text, None
        return "运行中...", text, None

    btn.click(start, [cat, backbone, coreset, image_size, tiles], [state_jid, status, logs])
    gr.Timer(1.0).tick(poll, state_jid, [status, logs, result])
