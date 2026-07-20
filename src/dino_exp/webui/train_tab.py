import dataclasses

import gradio as gr

from dino_exp.config import resolve_backbone, validate_image_size
from dino_exp.errors import DinoError
from dino_exp.train import train_model
from dino_exp.webui.jobs import JobManager


def build(cfg, jm: JobManager):
    with gr.Tab("训练"):
        cat = gr.Textbox(label="类别名", placeholder="bottle")
        backbone = gr.Dropdown(
            ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14",
             "dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"],
            value=cfg.backbone, label="骨干")
        coreset = gr.Slider(0.01, 1.0, value=cfg.coreset_sampling_ratio, label="coreset 采样率")
        image_size = gr.Number(value=cfg.image_size, label="输入尺寸（patch 整数倍）")
        btn = gr.Button("开始训练", variant="primary")
        status = gr.Textbox(label="状态", interactive=False)
        logs = gr.Textbox(label="日志", interactive=False, lines=12)
        result = gr.JSON(label="验证指标（训练完成自动全量验证）")
        state_jid = gr.State(None)

        def start(c, bb, cs, sz):
            try:
                validate_image_size(int(sz), resolve_backbone(bb).patch_size)
            except DinoError as exc:
                return None, f"错误: {exc}", ""
            run_cfg = dataclasses.replace(
                cfg, backbone=bb, coreset_sampling_ratio=float(cs), image_size=int(sz))
            run_cfg.layers = list(run_cfg.backbone_spec.default_layers)
            try:
                jid = jm.start("train", lambda log: train_model(c, run_cfg, log=log))
                return jid, f"训练已启动: {jid}", ""
            except DinoError as exc:
                return None, f"错误: {exc}", ""

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

        btn.click(start, [cat, backbone, coreset, image_size], [state_jid, status, logs])
        gr.Timer(1.0).tick(poll, state_jid, [status, logs, result])
