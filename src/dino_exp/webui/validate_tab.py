import gradio as gr

from dino_exp.datasets import category_images
from dino_exp.models.registry import Registry
from dino_exp.validate import filter_errors, validate_full, validate_images
from dino_exp.webui.common import category_dropdown, error_pair, verdict_summary_html


def build(cfg):
    with gr.Row():
        cat = category_dropdown(cfg, label="类别（选择已导入的数据集）")
        version = gr.Dropdown(label="版本（留空=当前）", choices=[], value=None)

    def _versions(c):
        return gr.update(choices=Registry(cfg.models_root).list(c) if c else [])

    cat.change(_versions, cat, version)
    gr.Timer(3.0).tick(_versions, cat, version)  # 定时刷新，保证版本列表始终可选

    err_box = gr.Textbox(label="提示", interactive=False)
    with gr.Accordion("错误详情（点击展开堆栈）", open=False):
        err_detail = gr.Textbox(interactive=False, lines=8)

    def _on_heat_select(table, evt: gr.SelectData):
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        try:
            from dino_exp.infer import verdict_frame

            if hasattr(table, "iloc"):  # Gradio 6 传 pandas DataFrame
                label, path = table.iloc[idx, 0], table.iloc[idx, 2]
            else:
                label, path = table[idx][0], table[idx][2]
            return verdict_frame(path, label)  # 热力图 + 判定色外框（绿=OK 红=NG）
        except Exception:
            return None

    with gr.Tabs():
        # ---------------- 全量验证 ----------------
        with gr.Tab("全量验证"):
            errors_only = gr.Checkbox(label="只看误判", value=False)
            btn_full = gr.Button("开始全量验证", variant="primary")
            metrics = gr.JSON(label="聚合指标")
            rows_out = gr.Dataframe(headers=["判定", "分数", "GT", "路径"],
                                    label="逐图结果（点击行查看原图）")
            full_preview = gr.Image(label="原图预览", height=280)

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

            def on_full_select(table, evt: gr.SelectData):
                idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                try:
                    from dino_exp.infer import verdict_frame

                    if hasattr(table, "iloc"):  # Gradio 6 传 pandas DataFrame
                        label, path = table.iloc[idx, 0], table.iloc[idx, 3]
                    else:
                        label, path = table[idx][0], table[idx][3]
                    return verdict_frame(path, label)  # 原图 + 判定色外框
                except Exception:
                    return None

            rows_out.select(on_full_select, rows_out, full_preview)

        # ---------------- 选图验证：从数据集选图 ----------------
        with gr.Tab("从数据集选图"):
            with gr.Row():
                srv_img = gr.Dropdown(label="图片（随类别刷新）", choices=[], value=None, scale=3)
                btn_srv = gr.Button("刷新图片列表", scale=1)
            srv_preview = gr.Image(label="选中图片预览", height=240)
            btn_sel_srv = gr.Button("验证所选", variant="primary")
            sel_verdict = gr.HTML()
            sel_out = gr.Dataframe(headers=["判定", "分数", "热力图"], label="结果（点击行查看热力图）")
            sel_heat = gr.Image(label="热力图预览", height=280)

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

            def do_sel_srv(c, v, rel):
                paths = []
                if rel:
                    abs_map = dict(category_images(c, cfg))
                    if rel in abs_map:
                        paths = [str(abs_map[rel])]
                return _run_sel(c, v, paths)

            btn_sel_srv.click(do_sel_srv, [cat, version, srv_img],
                              [sel_out, err_box, err_detail, sel_heat, sel_verdict])
            sel_out.select(_on_heat_select, sel_out, sel_heat)

        # ---------------- 选图验证：上传图片 ----------------
        with gr.Tab("上传图片"):
            files = gr.File(file_count="multiple", label="选择图片（可多选）")
            btn_sel_up = gr.Button("验证上传图片", variant="primary")
            up_verdict = gr.HTML()
            up_out = gr.Dataframe(headers=["判定", "分数", "热力图"], label="结果（点击行查看热力图）")
            up_heat = gr.Image(label="热力图预览", height=280)

            def do_sel_up(c, v, fs):
                return _run_sel(c, v, [f.name for f in fs] if fs else [])

            btn_sel_up.click(do_sel_up, [cat, version, files],
                             [up_out, err_box, err_detail, up_heat, up_verdict])
            up_out.select(_on_heat_select, up_out, up_heat)

    def _run_sel(c, v, paths):
        try:
            if not paths:
                return [], "请先选择或上传图片。", "", None, ""
            rows = validate_images(c, v or None, paths, cfg)
            table = [[r["label"], round(r["score"], 4), r["heatmap_path"]] for r in rows]
            from dino_exp.infer import verdict_frame

            heat = verdict_frame(rows[0]["heatmap_path"], rows[0]["label"]) if rows else None
            return table, "", "", heat, verdict_summary_html(rows)
        except Exception as exc:
            summary, detail = error_pair(exc)
            return [], summary, detail, None, ""
