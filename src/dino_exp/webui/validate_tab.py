import gradio as gr

from dino_exp.datasets import category_images
from dino_exp.models.registry import Registry
from dino_exp.validate import filter_errors, validate_full, validate_images
from dino_exp.webui.common import error_pair


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

        err_box = gr.Textbox(label="提示", interactive=False)
        with gr.Accordion("错误详情（点击展开堆栈）", open=False):
            err_detail = gr.Textbox(interactive=False, lines=8)

        def do_full(c, v, eo):
            try:
                report = validate_full(c, v or None, cfg)
                rows = filter_errors(report["rows"]) if eo else report["rows"]
                table = [[r["label_pred"], round(r["score"], 4), r["defect_type"], r["path"]] for r in rows]
                return report["metrics"], table, "", ""
            except Exception as exc:
                summary, detail = error_pair(exc)
                return None, [], summary, detail

        btn_full.click(do_full, [cat, version, errors_only],
                       [metrics, rows_out, err_box, err_detail])

        gr.Markdown("### 选图验证")
        with gr.Row():
            srv_img = gr.Dropdown(label="从数据集选图（随类别刷新）", choices=[], value=None, scale=3)
            btn_srv = gr.Button("刷新图片列表", scale=1)
        srv_preview = gr.Image(label="选中图片预览", height=240)
        files = gr.File(file_count="multiple", label="或上传图片")
        btn_sel = gr.Button("验证所选/上传", variant="primary")
        sel_out = gr.Dataframe(headers=["判定", "分数", "热力图"], label="结果")

        def refresh_images(c):
            if not c:
                return gr.update(choices=[])
            try:
                return gr.update(choices=[rel for rel, _ in category_images(c, cfg)])
            except Exception:
                return gr.update(choices=[])

        cat.change(refresh_images, cat, srv_img)
        btn_srv.click(refresh_images, cat, srv_img)

        def preview_selected(c, rel):
            if not rel:
                return None
            return str(dict(category_images(c, cfg)).get(rel, "")) or None

        srv_img.change(preview_selected, [cat, srv_img], srv_preview)

        def do_sel(c, v, rel, fs):
            try:
                paths = [f.name for f in fs] if fs else []
                if rel:
                    abs_map = dict(category_images(c, cfg))
                    if rel in abs_map:
                        paths = [str(abs_map[rel])] + paths
                if not paths:
                    return [], "请先选择服务器图片或上传图片。", ""
                rows = validate_images(c, v or None, paths, cfg)
                return [[r["label"], round(r["score"], 4), r["heatmap_path"]] for r in rows], "", ""
            except Exception as exc:
                summary, detail = error_pair(exc)
                return [], summary, detail

        btn_sel.click(do_sel, [cat, version, srv_img, files],
                      [sel_out, err_box, err_detail])
