import gradio as gr

from dino_exp.config import Config
from dino_exp.webui import dataset_tab, test_tab, train_tab, validate_tab
from dino_exp.webui.jobs import JobManager


def launch(cfg: Config) -> None:
    jm = JobManager()
    with gr.Blocks(title="DINO 异常检测试验环境") as demo:
        gr.Markdown("# DINO 无监督异常检测试验环境")
        dataset_tab.build(cfg)
        train_tab.build(cfg, jm)
        validate_tab.build(cfg)
        test_tab.build(cfg)
    demo.queue(default_concurrency_limit=2).launch()


if __name__ == "__main__":
    from dino_exp.config import load_config

    launch(load_config())
