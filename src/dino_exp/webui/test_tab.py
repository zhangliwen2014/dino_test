import gradio as gr

from dino_exp.datasets import category_images
from dino_exp.feedback.store import FeedbackStore
from dino_exp.infer import infer_image
from dino_exp.models.registry import Registry
from dino_exp.retrain import preview_retrain, retrain
from dino_exp.webui.common import category_dropdown, error_pair, verdict_html


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


def _verdict_html(label: str, score: float, threshold: float, infer_ms: float | None = None) -> str:
    """OK/NG 彩色判定徽章（绿=OK 红=NG），委托公共实现。"""
    return verdict_html(label, score, threshold, infer_ms)


def build(cfg):
    with gr.Row():
        cat = category_dropdown(cfg, label="类别（选择已导入的数据集）")
        version = gr.Dropdown(label="版本（留空=当前）", choices=[], value=None)
        thr_override = gr.Number(label="阈值（留空=自动）", value=None)

    def _versions(c):
        return gr.update(choices=Registry(cfg.models_root).list(c) if c else [])

    cat.change(_versions, cat, version)
    gr.Timer(3.0).tick(_versions, cat, version)  # 定时刷新，保证版本列表始终可选

    state_score = gr.State(0.0)
    state_pred = gr.State("")
    state_path = gr.State("")

    with gr.Row():
        # 左列：选图/上传（tab 分开，不再堆叠）
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("从数据集选图"):
                    with gr.Row():
                        srv_img = gr.Dropdown(label="图片", choices=_initial_images(cfg), value=None, scale=3)
                        btn_srv = gr.Button("刷新", scale=1)
                    srv_preview = gr.Image(label="选中图片预览", height=260)
                    btn_test_srv = gr.Button("测 试", variant="primary")
                with gr.Tab("上传图片"):
                    img = gr.Image(type="filepath", label="上传图片", height=300)
                    btn_test_up = gr.Button("测 试", variant="primary")

        # 右列：判定结果 + 热力图/缺陷标记
        with gr.Column(scale=1):
            verdict = gr.HTML(label="判定结果")
            with gr.Tabs():
                with gr.Tab("缺陷标记"):
                    anno_out = gr.Image(label="原图+缺陷框", height=300)
                with gr.Tab("异常热力图"):
                    heat_out = gr.Image(label="热力图", height=300)

    err_box = gr.Textbox(label="提示", interactive=False)
    with gr.Accordion("错误详情（点击展开堆栈）", open=False):
        err_detail = gr.Textbox(interactive=False, lines=8)

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

    def _run_test(c, v, path, thr):
        if not path:
            return "", None, None, 0.0, "", "", "请先选择或上传图片。", ""
        try:
            from dino_exp.infer import decide_label, verdict_frame

            r = infer_image(path, v or None, category=c, cfg=cfg)
            label, threshold = r["label"], r["threshold"]
            if thr:  # 手动阈值：重算判定（仅本次查看）
                threshold = float(thr)
                label = decide_label(r["score"], threshold)
            return (_verdict_html(label, r["score"], threshold, r.get("infer_ms")),
                    r["annotated_path"],
                    verdict_frame(r["heatmap_path"], label),
                    r["score"], label, path, "", "")
        except Exception as exc:
            summary, detail = error_pair(exc)
            return "", None, None, 0.0, "", "", summary, detail

    def do_test_srv(c, v, rel, thr):
        path = None
        if rel:
            abs_map = dict(category_images(c, cfg))
            path = str(abs_map.get(rel, "")) or None
        return _run_test(c, v, path, thr)

    def do_test_up(c, v, path, thr):
        return _run_test(c, v, path, thr)

    btn_test_srv.click(do_test_srv, [cat, version, srv_img, thr_override],
                       [verdict, anno_out, heat_out, state_score, state_pred, state_path,
                        err_box, err_detail])
    btn_test_up.click(do_test_up, [cat, version, img, thr_override],
                      [verdict, anno_out, heat_out, state_score, state_pred, state_path,
                       err_box, err_detail])

    gr.Markdown("### 反馈")
    fb_label = gr.Radio(["ok", "ng"], value="ok", label="实际标签")
    fb_dt = gr.Textbox(label="缺陷类型（NG 可填）")
    fb_btn = gr.Button("提交反馈")
    fb_msg = gr.Textbox(label="结果", interactive=False)

    def do_feedback(c, path, label, dt, score, pred):
        if not path:
            return "请先测试一张图片", ""
        try:
            rec = FeedbackStore(cfg.feedback_root, c).stage({
                "image_path": path,
                "model_version": Registry(cfg.models_root).current(c),
                "prediction": pred, "score": float(score),
                "human_label": label, "defect_type": dt or None,
            })
            return f"已暂存 {rec['id']}", ""
        except Exception as exc:
            summary, detail = error_pair(exc)
            return summary, detail

    fb_btn.click(do_feedback,
                 [cat, state_path, fb_label, fb_dt, state_score, state_pred],
                 [fb_msg, err_detail])

    gr.Markdown("### 再训练与版本")
    pv_btn = gr.Button("预览暂存区")
    pv_out = gr.JSON(label="预览")
    rt_btn = gr.Button("执行再训练", variant="primary")
    rt_out = gr.JSON(label="结果")

    def do_preview(c):
        try:
            return preview_retrain(c, cfg)
        except Exception as exc:
            summary, detail = error_pair(exc)
            return {"错误": summary, "详细堆栈": detail}

    def do_retrain(c):
        try:
            return {k: v for k, v in retrain(c, cfg).items() if k != "preview"}
        except Exception as exc:
            summary, detail = error_pair(exc)
            return {"错误": summary, "详细堆栈": detail}

    pv_btn.click(do_preview, cat, pv_out)
    rt_btn.click(do_retrain, cat, rt_out)

    ver_out = gr.Dataframe(headers=["版本", "当前"], label="版本列表")
    with gr.Row():
        rb_ver = gr.Textbox(label="版本号（如 v001）", scale=2)
        rb_btn = gr.Button("回滚到该版本", scale=1)
        del_confirm = gr.Checkbox(label="确认删除", value=False, scale=1)
        del_btn = gr.Button("删除该版本", variant="stop", scale=1)
    rb_msg = gr.Textbox(label="结果", interactive=False)

    def list_versions(c):
        reg = Registry(cfg.models_root)
        cur = reg.current(c)
        return [[v, "✓" if v == cur else ""] for v in reg.list(c)]

    def do_rollback(c, v):
        try:
            Registry(cfg.models_root).rollback(c, v)
            return f"已回滚到 {v}", ""
        except Exception as exc:
            summary, detail = error_pair(exc)
            return summary, detail

    def do_delete(c, v, confirmed):
        if not confirmed:
            return "请先勾选「确认删除」再执行。", ""
        try:
            d = Registry(cfg.models_root).delete(c, v)
            return f"已删除: {d}", ""
        except Exception as exc:
            summary, detail = error_pair(exc)
            return summary, detail

    gr.Timer(5.0).tick(list_versions, cat, ver_out)
    rb_btn.click(do_rollback, [cat, rb_ver], [rb_msg, err_detail])
    del_btn.click(do_delete, [cat, rb_ver, del_confirm], [rb_msg, err_detail])
