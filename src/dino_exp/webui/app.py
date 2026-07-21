import gradio as gr

from dino_exp.config import Config
from dino_exp.logs import get_logger, setup_logging
from dino_exp.webui import dataset_tab, settings_tab, test_tab, train_tab, validate_tab
from dino_exp.webui.jobs import JobManager


def launch(cfg: Config, port: int = 7860, host: str = "127.0.0.1") -> None:
    log_file = setup_logging()
    get_logger("webui").info("启动 Web UI: http://%s:%d（日志: %s）", host, port, log_file)
    jm = JobManager()
    with gr.Blocks(title="DINO 异常检测试验环境") as demo:
        gr.Markdown("# DINO 无监督异常检测试验环境")
        dataset_tab.build(cfg)
        train_tab.build(cfg, jm)
        validate_tab.build(cfg)
        test_tab.build(cfg)
        settings_tab.build(cfg)
    demo.queue(default_concurrency_limit=2).launch(server_name=host, server_port=port)


if __name__ == "__main__":
    import sys

    from dino_exp.config import load_config

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    launch(load_config(), port=port, host=host)
