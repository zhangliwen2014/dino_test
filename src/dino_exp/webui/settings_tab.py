import gradio as gr

from dino_exp.config import resolve_device
from dino_exp.webui.common import error_pair


def _env_info(cfg) -> dict:
    import torch

    info = {
        "torch 版本": torch.__version__,
        "CUDA 可用": torch.cuda.is_available(),
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无",
        "配置 device": cfg.device,
        "当前生效设备": resolve_device(cfg),
    }
    return info


def build(cfg):
    env_out = gr.JSON(label="环境信息", value=_env_info(cfg))
    device_sel = gr.Radio(
        [("auto：自动选择性能最高的设备（有 GPU 用 GPU）", "auto"),
         ("cpu：强制 CPU", "cpu"),
         ("cuda：强制 GPU（需已安装 CUDA 版 torch）", "cuda")],
        value=cfg.device if cfg.device in {"auto", "cpu", "cuda"} else "auto",
        label="运行设备（训练/验证/测试共用，立即生效）")
    save_btn = gr.Button("保存为默认（写入 config/default.yaml）")
    msg = gr.Textbox(label="结果", interactive=False)

    def on_device_change(d):
        cfg.device = d
        try:
            return _env_info(cfg), f"已切换为 {d}（立即生效，本次运行有效）"
        except Exception as exc:
            summary, _ = error_pair(exc)
            return _env_info(cfg), summary

    def do_save():
        from pathlib import Path

        import yaml

        p = Path("config/default.yaml")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
        raw["device"] = cfg.device
        p.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
        from dino_exp.logs import get_logger

        get_logger("settings").info("device 已保存到 %s: %s", p, cfg.device)
        return f"已保存 device={cfg.device} 到 {p}（下次启动默认生效）"

    device_sel.change(on_device_change, device_sel, [env_out, msg])
    save_btn.click(do_save, None, msg)
