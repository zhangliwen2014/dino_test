import gradio as gr

from dino_exp.datasets import category_images
from dino_exp.feedback.store import FeedbackStore
from dino_exp.infer import infer_image
from dino_exp.models.registry import Registry
from dino_exp.retrain import preview_retrain, retrain
from dino_exp.webui.common import error_pair


def build(cfg):
    with gr.Tab("测试与反馈"):
        cat = gr.Textbox(label="类别名")
        version = gr.Dropdown(label="版本（留空=当前）", choices=[], value=None)
        cat.change(lambda c: gr.update(choices=Registry(cfg.models_root).list(c)),
                   cat, version)

        with gr.Row():
            srv_img = gr.Dropdown(label="从数据集选图（随类别刷新）", choices=[], value=None, scale=3)
            btn_srv = gr.Button("刷新图片列表", scale=1)
        srv_preview = gr.Image(label="选中图片预览", height=240)
        img = gr.Image(type="filepath", label="或上传图片（优先于选图）")
        btn = gr.Button("测试", variant="primary")
        label_out = gr.Textbox(label="判定", interactive=False)
        score_out = gr.Number(label="异常分数")
        heat_out = gr.Image(label="热力图")
        state_score = gr.State(0.0)
        state_pred = gr.State("")
        state_path = gr.State("")

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

        def do_test(c, v, rel, path):
            if not path and rel:
                abs_map = dict(category_images(c, cfg))
                path = str(abs_map.get(rel, "")) or None
            if not path:
                return "", 0.0, None, 0.0, "", "", "请先从数据集选图或上传图片。", ""
            try:
                r = infer_image(path, v or None, category=c, cfg=cfg)
                return r["label"], r["score"], r["heatmap_path"], r["score"], r["label"], path, "", ""
            except Exception as exc:
                summary, detail = error_pair(exc)
                return "", 0.0, None, 0.0, "", "", summary, detail

        btn.click(do_test, [cat, version, srv_img, img],
                  [label_out, score_out, heat_out, state_score, state_pred, state_path,
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
        rb_ver = gr.Textbox(label="回滚到版本")
        rb_btn = gr.Button("回滚")
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

        gr.Timer(5.0).tick(list_versions, cat, ver_out)
        rb_btn.click(do_rollback, [cat, rb_ver], [rb_msg, err_detail])
