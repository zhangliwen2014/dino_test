import gradio as gr

from dino_exp.config import Config
from dino_exp.logs import get_logger, setup_logging
from dino_exp.webui import dataset_tab, settings_tab, test_tab, train_tab, validate_tab
from dino_exp.webui.jobs import JobManager

MENU = ["数据集", "训练", "验证", "测试与反馈", "系统设置"]

_CSS = """
/* CVAT 式左侧导航：去掉 Radio 的圆点，做成菜单按钮样式 */
#side-nav input[type="radio"] { display: none; }
#side-nav label {
    display: block; padding: 10px 14px; margin: 2px 0; border-radius: 6px;
    cursor: pointer; font-size: 15px;
}
#side-nav label:has(input:checked) {
    background: var(--primary-500, #2563eb); color: white; font-weight: 600;
}
#side-nav label:hover { background: var(--neutral-200, #e5e7eb); }
#side-nav label:has(input:checked):hover { background: var(--primary-600, #1d4ed8); }
"""


def launch(cfg: Config, port: int = 7860, host: str = "127.0.0.1") -> None:
    log_file = setup_logging()
    get_logger("webui").info("启动 Web UI: http://%s:%d（日志: %s）", host, port, log_file)
    jm = JobManager()
    with gr.Blocks(title="DINO 异常检测试验环境", css=_CSS) as demo:
        gr.Markdown("# DINO 无监督异常检测试验环境")
        with gr.Row():
            with gr.Column(scale=1, min_width=150):
                menu = gr.Radio(MENU, value=MENU[0], show_label=False,
                                container=False, elem_id="side-nav")
            with gr.Column(scale=6):
                groups = []
                with gr.Group(visible=True) as g:
                    dataset_tab.build(cfg)
                groups.append(g)
                with gr.Group(visible=False) as g:
                    train_tab.build(cfg, jm)
                groups.append(g)
                with gr.Group(visible=False) as g:
                    validate_tab.build(cfg)
                groups.append(g)
                with gr.Group(visible=False) as g:
                    test_tab.build(cfg)
                groups.append(g)
                with gr.Group(visible=False) as g:
                    settings_tab.build(cfg)
                groups.append(g)

        menu.change(
            lambda m: [gr.update(visible=(m == name)) for name in MENU],
            menu, groups,
        )
    demo.queue(default_concurrency_limit=2).launch(server_name=host, server_port=port)


if __name__ == "__main__":
    import sys

    from dino_exp.config import load_config

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    launch(load_config(), port=port, host=host)
