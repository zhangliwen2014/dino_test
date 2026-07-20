import gradio as gr

from dino_exp.models.registry import Registry
from dino_exp.validate import filter_errors, validate_full, validate_images


def build(cfg):
    with gr.Tab("验证"):
        cat = gr.Textbox(label="类别名")
        version = gr.Dropdown(label="版本（留空=当前）", choices=[], value=None)
        errors_only = gr.Checkbox(label="只看误判", value=False)
        btn_full = gr.Button("全量验证", variant="primary")
        metrics = gr.JSON(label="聚合指标")
        rows_out = gr.Dataframe(headers=["判定", "分数", "GT", "路径"], label="逐图结果")

        cat.change(lambda c: gr.update(choices=Registry(cfg.models_root).list(c)),
                   cat, version)

        def do_full(c, v, eo):
            report = validate_full(c, v or None, cfg)
            rows = filter_errors(report["rows"]) if eo else report["rows"]
            table = [[r["label_pred"], round(r["score"], 4), r["defect_type"], r["path"]] for r in rows]
            return report["metrics"], table

        btn_full.click(do_full, [cat, version, errors_only], [metrics, rows_out])

        gr.Markdown("### 选图验证")
        files = gr.File(file_count="multiple", label="上传图片")
        btn_sel = gr.Button("验证所选")
        sel_out = gr.Dataframe(headers=["判定", "分数", "热力图"], label="结果")

        def do_sel(c, v, fs):
            rows = validate_images(c, v or None, [f.name for f in fs], cfg)
            return [[r["label"], round(r["score"], 4), r["heatmap_path"]] for r in rows]

        btn_sel.click(do_sel, [cat, version, files], sel_out)
