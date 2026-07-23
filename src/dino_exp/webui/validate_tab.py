import gradio as gr

from dino_exp.datasets import category_images
from dino_exp.models.registry import Registry
from dino_exp.validate import filter_errors, validate_full, validate_images
from dino_exp.webui.common import category_dropdown, error_pair, verdict_summary_html


def _initial_images(cfg) -> list[str]:
    """首次加载时，默认类别（第一个）的图片列表。"""
    from dino_exp.webui.common import category_choices

    cats = category_choices(cfg)
    if not cats:
        return []
    try:
        return [rel for rel, _ in category_images(cats[0], cfg)]
    except Exception:
        return []


def _label_html(label: str) -> str:
    """判定列彩色单元格（绿 OK / 红 NG），配合 Dataframe datatype=["html", ...]。"""
    color = "#16a34a" if label == "OK" else "#dc2626"
    return f"<span style='color:{color};font-weight:700'>{label}</span>"


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

    def make_result_view():
        """结果查看组件：标记图/热力图双页签 + 上一张/下一张切换。

        返回 (update_fn, outputs, btn_prev, btn_next, results_state, idx_state)。
        update_fn(rows, idx) -> 对应 outputs 的 4 个值。
        """
        results_state = gr.State([])
        idx_state = gr.State(0)
        with gr.Tabs():
            with gr.Tab("标记图（原图+缺陷框）"):
                anno_img = gr.Image(height=300)
            with gr.Tab("异常热力图"):
                heat_img = gr.Image(height=300)
        with gr.Row():
            btn_prev = gr.Button("◀ 上一张")
            pos_label = gr.Markdown("0/0")
            btn_next = gr.Button("下一张 ▶")

        outputs = [anno_img, heat_img, pos_label, idx_state]

        def update_fn(rows, idx):
            if not rows:
                return None, None, "0/0", 0
            idx = idx % len(rows)
            r = rows[idx]
            label = r.get("label") or r.get("label_pred", "")  # 全量=label_pred，选图=label
            return (r.get("annotated_path"), r.get("heatmap_path"),
                    f"第 {idx + 1}/{len(rows)} 张（{label}，分数 {r['score']:.4f}）", idx)

        btn_prev.click(lambda rows, i: update_fn(rows, i - 1),
                       [results_state, idx_state], outputs)
        btn_next.click(lambda rows, i: update_fn(rows, i + 1),
                       [results_state, idx_state], outputs)
        return update_fn, outputs, results_state, idx_state

    def _row_idx(evt: gr.SelectData) -> int:
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        return int(idx or 0)

    with gr.Tabs():
        # ---------------- 全量验证 ----------------
        with gr.Tab("全量验证"):
            with gr.Row():
                errors_only = gr.Checkbox(label="只看误判", value=False)
                thr_override = gr.Number(label="阈值调整（留空=自动，仅影响本次查看）",
                                         value=None, scale=2)
            btn_full = gr.Button("开始全量验证", variant="primary")
            metrics = gr.JSON(label="聚合指标")
            with gr.Row():
                dist_img = gr.Image(label="分数分布（含阈值线）", height=260)
                roc_img = gr.Image(label="ROC 曲线", height=260)
            rows_out = gr.Dataframe(headers=["判定", "分数", "GT", "标记图", "热力图"],
                                    datatype=["html", "number", "str", "str", "str"],
                                    label="逐图结果（点击行查看）")
            full_update, full_outputs, full_rows, full_idx = make_result_view()

            def do_full(c, v, eo, thr):
                try:
                    from dino_exp.validate import aggregate_metrics, relabel_rows

                    report = validate_full(c, v or None, cfg)
                    rows = report["rows"]
                    m = report["metrics"]
                    if thr:  # 手动阈值：重算判定与指标（仅本次查看，不改版本）
                        rows = relabel_rows(rows, float(thr))
                        m = aggregate_metrics(rows, float(thr))
                        m["threshold_manual"] = float(thr)
                    rows = filter_errors(rows) if eo else rows
                    table = [[_label_html(r["label_pred"]), round(r["score"], 4), r["defect_type"],
                              r.get("annotated_path", ""), r.get("heatmap_path", "")] for r in rows]
                    preview = full_update(rows, 0)
                    return (m, report.get("dist_plot"), report.get("roc_plot"),
                            table, "", "", rows, *preview)
                except Exception as exc:
                    summary, detail = error_pair(exc)
                    return None, None, None, [], summary, detail, [], None, None, "0/0", 0

            btn_full.click(do_full, [cat, version, errors_only, thr_override],
                           [metrics, dist_img, roc_img, rows_out, err_box, err_detail,
                            full_rows, *full_outputs])

            def on_full_select(rows, evt: gr.SelectData):
                return full_update(rows, _row_idx(evt))

            rows_out.select(on_full_select, full_rows, full_outputs)

        # ---------------- 选图验证：从数据集选图 ----------------
        with gr.Tab("从数据集选图"):
            with gr.Row():
                srv_img = gr.Dropdown(label="图片", choices=_initial_images(cfg), value=None, scale=3)
                btn_srv = gr.Button("刷新图片列表", scale=1)
            srv_preview = gr.Image(label="选中图片预览", height=240)
            btn_sel_srv = gr.Button("验证所选", variant="primary")
            sel_verdict = gr.HTML()
            sel_out = gr.Dataframe(headers=["判定", "分数", "标记图", "热力图"],
                                   datatype=["html", "number", "str", "str"],
                                   label="结果（点击行查看）")
            sel_update, sel_outputs, sel_rows, sel_idx = make_result_view()

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
                return _run_sel(c, v, paths, sel_update)

            btn_sel_srv.click(do_sel_srv, [cat, version, srv_img],
                              [sel_out, err_box, err_detail, sel_verdict, sel_rows, *sel_outputs])

            def on_sel_select(rows, evt: gr.SelectData):
                return sel_update(rows, _row_idx(evt))

            sel_out.select(on_sel_select, sel_rows, sel_outputs)

        # ---------------- 选图验证：上传图片 ----------------
        with gr.Tab("上传图片"):
            files = gr.File(file_count="multiple", label="选择图片（可多选）")
            btn_sel_up = gr.Button("验证上传图片", variant="primary")
            up_verdict = gr.HTML()
            up_out = gr.Dataframe(headers=["判定", "分数", "标记图", "热力图"],
                                  datatype=["html", "number", "str", "str"],
                                  label="结果（点击行查看）")
            up_update, up_outputs, up_rows, up_idx = make_result_view()

            def do_sel_up(c, v, fs):
                return _run_sel(c, v, [f.name for f in fs] if fs else [], up_update)

            btn_sel_up.click(do_sel_up, [cat, version, files],
                             [up_out, err_box, err_detail, up_verdict, up_rows, *up_outputs])

            def on_up_select(rows, evt: gr.SelectData):
                return up_update(rows, _row_idx(evt))

            up_out.select(on_up_select, up_rows, up_outputs)

    def _run_sel(c, v, paths, update_fn):
        try:
            if not paths:
                return [], "请先选择或上传图片。", "", "", [], None, None, "0/0", 0
            rows = validate_images(c, v or None, paths, cfg)
            table = [[_label_html(r["label"]), round(r["score"], 4),
                      r["annotated_path"], r["heatmap_path"]] for r in rows]
            preview = update_fn(rows, 0)
            return (table, "", "", verdict_summary_html(rows), rows, *preview)
        except Exception as exc:
            summary, detail = error_pair(exc)
            return [], summary, detail, "", [], None, None, "0/0", 0
