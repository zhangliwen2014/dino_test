import gradio as gr

from dino_exp.datasets import dataset_info, import_images, import_mvtec, list_datasets
from dino_exp.errors import DinoError


def build(cfg):
    with gr.Tab("数据集"):
        out = gr.Dataframe(headers=["类别", "train/good", "test/good", "缺陷类型", "降级"],
                           label="数据集列表")

        def refresh():
            return [[i.category, i.train_good, i.test_good,
                     ", ".join(f"{k}:{v}" for k, v in i.defect_types.items()) or "-",
                     "是" if i.degraded else "否"] for i in list_datasets(cfg)]

        with gr.Row():
            cat_dl = gr.Textbox(label="MVTec 类别名", placeholder="bottle")
            btn_dl = gr.Button("下载 MVTec")
            dl_msg = gr.Textbox(label="结果", interactive=False)

        def do_download(cat):
            try:
                return str(import_mvtec(cat, cfg))
            except DinoError as exc:
                return f"错误: {exc}"

        btn_dl.click(do_download, cat_dl, dl_msg)

        with gr.Row():
            cat_im = gr.Textbox(label="类别名")
            label_im = gr.Radio(["ok", "ng"], value="ok", label="标签")
            dt_im = gr.Textbox(label="缺陷类型（NG 必填）")
            files = gr.File(file_count="multiple", label="选择图片")
            btn_im = gr.Button("导入")
            im_msg = gr.Textbox(label="结果", interactive=False)

        def do_import(cat, label, dt, fs):
            try:
                paths = import_images([f.name for f in fs], cat, label, dt or None, cfg)
                return f"已导入 {len(paths)} 张"
            except DinoError as exc:
                return f"错误: {exc}"

        btn_im.click(do_import, [cat_im, label_im, dt_im, files], im_msg)
        cat_info = gr.JSON(label="类别详情")
        cat_im.change(lambda c: dataset_info(c, cfg).__dict__ if c else {}, cat_im, cat_info)
        gr.Timer(5.0).tick(refresh, outputs=out)
