import gradio as gr

from dino_exp.errors import DinoError
from dino_exp.feedback.store import FeedbackStore
from dino_exp.infer import infer_image
from dino_exp.models.registry import Registry
from dino_exp.retrain import preview_retrain, retrain


def build(cfg):
    with gr.Tab("测试与反馈"):
        cat = gr.Textbox(label="类别名")
        version = gr.Dropdown(label="版本（留空=当前）", choices=[], value=None)
        cat.change(lambda c: gr.update(choices=Registry(cfg.models_root).list(c)),
                   cat, version)
        img = gr.Image(type="filepath", label="上传图片")
        btn = gr.Button("测试", variant="primary")
        label_out = gr.Textbox(label="判定", interactive=False)
        score_out = gr.Number(label="异常分数")
        heat_out = gr.Image(label="热力图")
        state_score = gr.State(0.0)
        state_pred = gr.State("")

        def do_test(c, v, path):
            r = infer_image(path, v or None, category=c, cfg=cfg)
            return r["label"], r["score"], r["heatmap_path"], r["score"], r["label"]

        btn.click(do_test, [cat, version, img],
                  [label_out, score_out, heat_out, state_score, state_pred])

        gr.Markdown("### 反馈")
        fb_label = gr.Radio(["ok", "ng"], value="ok", label="实际标签")
        fb_dt = gr.Textbox(label="缺陷类型（NG 可填）")
        fb_btn = gr.Button("提交反馈")
        fb_msg = gr.Textbox(label="结果", interactive=False)

        def do_feedback(c, path, label, dt, score, pred):
            if not path:
                return "请先测试一张图片"
            try:
                rec = FeedbackStore(cfg.feedback_root, c).stage({
                    "image_path": path,
                    "model_version": Registry(cfg.models_root).current(c),
                    "prediction": pred, "score": float(score),
                    "human_label": label, "defect_type": dt or None,
                })
                return f"已暂存 {rec['id']}"
            except DinoError as exc:
                return f"错误: {exc}"

        fb_btn.click(do_feedback,
                     [cat, img, fb_label, fb_dt, state_score, state_pred], fb_msg)

        gr.Markdown("### 再训练与版本")
        pv_btn = gr.Button("预览暂存区")
        pv_out = gr.JSON(label="预览")
        rt_btn = gr.Button("执行再训练", variant="primary")
        rt_out = gr.JSON(label="结果")
        pv_btn.click(lambda c: preview_retrain(c, cfg), cat, pv_out)
        rt_btn.click(lambda c: {k: v for k, v in retrain(c, cfg).items() if k != "preview"},
                     cat, rt_out)

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
                return f"已回滚到 {v}"
            except DinoError as exc:
                return f"错误: {exc}"

        gr.Timer(5.0).tick(list_versions, cat, ver_out)
        rb_btn.click(do_rollback, [cat, rb_ver], rb_msg)
