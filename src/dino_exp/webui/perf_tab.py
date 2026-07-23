import gradio as gr

from dino_exp.models.registry import Registry
from dino_exp.perf import format_table, run_perf, save_perf_report
from dino_exp.webui.common import category_dropdown, error_pair
from dino_exp.webui.jobs import JobManager


def build(cfg, jm: JobManager):
    cat = category_dropdown(cfg, label="类别")
    with gr.Row():
        vers = gr.CheckboxGroup(label="版本（默认全选）", choices=[], value=None)
        conc = gr.CheckboxGroup(["1", "4", "8"], value=["1", "4", "8"], label="并发档")
        samples = gr.Number(value=30, label="抽样图片数", precision=0)
    btn = gr.Button("开始性能测试", variant="primary")
    status = gr.Textbox(label="状态", interactive=False)
    table = gr.Textbox(label="对比结果", interactive=False, lines=16)
    state_jid = gr.State(None)

    def _vers(c):
        vs = Registry(cfg.models_root).list(c) if c else []
        return gr.update(choices=vs, value=vs)

    cat.change(_vers, cat, vers)
    gr.Timer(3.0).tick(_vers, cat, vers)

    def start(c, vs, cs, n):
        if not c:
            return None, "请先选择类别", ""
        conc = [int(x) for x in cs] or [1]
        vers = vs or None

        def job(log):
            log(f"性能测试: {c} 版本={vers or '全部'} 并发={conc} 样本={int(n)}")
            report = run_perf(c, vers, conc, int(n), cfg)
            text = format_table(report)
            path = save_perf_report(report)
            log(f"报告已保存: {path}")
            return text

        try:
            jid = jm.start("perf", job)
            return jid, f"性能测试已启动: {jid}", ""
        except Exception as exc:
            summary, detail = error_pair(exc)
            return None, summary, detail

    def poll(jid):
        if not jid:
            return "未启动", ""
        st = jm.status(jid)
        logs = "".join(st["logs"])
        if st["state"] == "done":
            return "完成", (logs + "\n\n" + str(st["result"])) if st["result"] else logs
        if st["state"] == "error":
            return f"失败:\n{st['error']}", logs
        return "运行中...", logs

    btn.click(start, [cat, vers, conc, samples], [state_jid, status, table])
    gr.Timer(1.0).tick(poll, state_jid, [status, table])
