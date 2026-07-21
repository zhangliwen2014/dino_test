import gradio as gr

from dino_exp.datasets import category_images, dataset_info, import_images, import_mvtec, list_datasets
from dino_exp.webui.common import error_pair


def build(cfg):
    with gr.Tab("数据集"):
        out = gr.Dataframe(headers=["类别", "train/good", "test/good", "缺陷类型", "状态"],
                           label="数据集列表")

        def refresh():
            rows = []
            for i in list_datasets(cfg):
                if i.error:
                    status = f"不完整: {i.error}"
                else:
                    status = "降级:无NG图" if i.degraded else "正常"
                rows.append([i.category, i.train_good, i.test_good,
                             ", ".join(f"{k}:{v}" for k, v in i.defect_types.items()) or "-",
                             status])
            return rows

        err_detail_box = gr.Accordion("错误详情（点击展开堆栈）", open=False)
        with err_detail_box:
            err_detail = gr.Textbox(interactive=False, lines=8)

        with gr.Row():
            cat_dl = gr.Textbox(label="MVTec 类别名", placeholder="bottle")
            btn_dl = gr.Button("下载 MVTec")
            dl_msg = gr.Textbox(label="结果", interactive=False)

        def do_download(cat):
            try:
                return str(import_mvtec(cat, cfg)), ""
            except Exception as exc:
                summary, detail = error_pair(exc)
                return summary, detail

        btn_dl.click(do_download, cat_dl, [dl_msg, err_detail])

        with gr.Row():
            cat_im = gr.Textbox(label="类别名")
            label_im = gr.Radio(["ok", "ng"], value="ok", label="标签")
            dt_im = gr.Textbox(label="缺陷类型（NG 必填）")
            split_im = gr.Radio(
                [("auto：OK 图 8:2 自动分入训练集/测试集（从零建数据集推荐）", "auto"),
                 ("train：OK 图全部进训练集 train/good", "train"),
                 ("test：OK 图全部进测试集 test/good", "test")],
                value="auto", label="OK 图去向（NG 图始终进 test/<缺陷类型>）")
            files = gr.File(file_count="multiple", label="选择图片")
            btn_im = gr.Button("导入")
            im_msg = gr.Textbox(label="结果", interactive=False)

        def do_import(cat, label, dt, sp, fs):
            try:
                paths = import_images([f.name for f in fs], cat, label, dt or None, cfg, split=sp)
                return f"已导入 {len(paths)} 张", ""
            except Exception as exc:
                summary, detail = error_pair(exc)
                return summary, detail

        btn_im.click(do_import, [cat_im, label_im, dt_im, split_im, files],
                     [im_msg, err_detail])
        cat_info = gr.JSON(label="类别详情")

        def show_info(c):
            if not c:
                return {}
            try:
                return dataset_info(c, cfg).__dict__
            except Exception as exc:
                summary, detail = error_pair(exc)
                return {"错误": summary, "详细堆栈": detail}

        cat_im.change(show_info, cat_im, cat_info)
        gr.Timer(5.0).tick(refresh, outputs=out)

        gr.Markdown("### 图片浏览")
        with gr.Row():
            btn_gallery = gr.Button("查看图片", variant="primary")
            gallery_msg = gr.Textbox(label="说明", interactive=False)
        gallery = gr.Gallery(label="类别图片（点击放大）", columns=6, height=360)

        def show_gallery(c):
            if not c:
                return [], "请先输入类别名"
            try:
                imgs = category_images(c, cfg)
            except Exception as exc:
                summary, _ = error_pair(exc)
                return [], summary
            shown = [str(p) for _, p in imgs[:60]]
            note = f"共 {len(imgs)} 张" + ("，仅显示前 60 张" if len(imgs) > 60 else "")
            return shown, note

        btn_gallery.click(show_gallery, cat_im, [gallery, gallery_msg])
