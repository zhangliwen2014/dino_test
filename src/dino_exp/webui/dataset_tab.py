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
            dir_sel = gr.Dropdown(label="子目录（train/good、test/good、缺陷类型）",
                                  choices=[], value=None, scale=3)
            btn_prev = gr.Button("← 上一页", scale=1)
            btn_next = gr.Button("下一页 →", scale=1)
        page_state = gr.State(0)
        gallery = gr.Gallery(label="图片（点击放大查看）", columns=6, height=380,
                             preview=True, interactive=False)
        gallery_msg = gr.Textbox(label="说明", interactive=False)

        PAGE_SIZE = 24

        def _subdirs(c):
            if not c:
                return []
            try:
                rels = [rel.rsplit("/", 1)[0] for rel, _ in category_images(c, cfg)]
                return sorted(set(rels))
            except Exception:
                return []

        def _page(c, sub, page):
            if not c or not sub:
                return [], 0, "请先输入类别名并选择子目录"
            try:
                imgs = [p for rel, p in category_images(c, cfg)
                        if rel.rsplit("/", 1)[0] == sub]
            except Exception as exc:
                summary, _ = error_pair(exc)
                return [], 0, summary
            total = len(imgs)
            pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
            page = max(0, min(page, pages - 1))
            shown = [str(p) for p in imgs[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]]
            return shown, page, f"{sub}：共 {total} 张，第 {page + 1}/{pages} 页"

        cat_im.change(lambda c: gr.update(choices=_subdirs(c), value=None),
                      cat_im, dir_sel)
        dir_sel.change(lambda c, s: _page(c, s, 0), [cat_im, dir_sel],
                       [gallery, page_state, gallery_msg])
        btn_prev.click(lambda c, s, p: _page(c, s, p - 1), [cat_im, dir_sel, page_state],
                       [gallery, page_state, gallery_msg])
        btn_next.click(lambda c, s, p: _page(c, s, p + 1), [cat_im, dir_sel, page_state],
                       [gallery, page_state, gallery_msg])
