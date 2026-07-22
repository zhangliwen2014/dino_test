import gradio as gr

from dino_exp.datasets import (
    add_defect_type,
    category_images,
    dataset_info,
    delete_category,
    fix_category,
    import_images,
    import_mvtec,
    list_datasets,
    rename_defect_type,
)
from dino_exp.webui.common import category_choices, category_dropdown, error_pair


def build(cfg):
    err_detail_box = gr.Accordion("错误详情（点击展开堆栈）", open=False)
    with err_detail_box:
        err_detail = gr.Textbox(interactive=False, lines=8)

    with gr.Tabs():
        # ---------------- 概览与浏览（侧栏主从布局：左选类别/子目录，右看图片） ----------------
        with gr.Tab("概览与浏览"):
            page_state = gr.State(0)
            cur_cat = gr.State("")
            shown_state = gr.State([])   # 当前页图片绝对路径列表（供选中映射）
            sel_path = gr.State("")      # 画廊中选中的图片绝对路径

            with gr.Row():
                # 侧栏：类别列表 + 子目录
                with gr.Column(scale=1, min_width=300):
                    out = gr.Dataframe(
                        headers=["类别", "train", "test", "状态"],
                        label="类别（点击行选择）",
                        interactive=False, max_height=330, wrap=True)
                    dir_sel = gr.Radio(label="子目录", choices=[], value=None)
                    with gr.Row():
                        btn_prev = gr.Button("← 上一页")
                        btn_next = gr.Button("下一页 →")

                # 主可视区：面包屑 + 画廊
                with gr.Column(scale=3):
                    crumb = gr.Markdown("未选择类别——点击左侧列表中的一行")
                    gallery = gr.Gallery(label="图片（点击选择/放大）", columns=5,
                                         height=460, preview=True, interactive=False)
                    gallery_msg = gr.Textbox(label="说明", interactive=False)
                    with gr.Accordion("调整选中图片的归类（纠错：OK/NG 放错、缺陷类型归错）", open=False):
                        sel_info = gr.Textbox(label="当前选中", interactive=False)
                        reclass_target = gr.Dropdown(
                            [("OK（测试集 test/good）", "test_good"),
                             ("OK（训练集 train/good）", "train_good"),
                             ("NG（缺陷图）", "ng")],
                            value="test_good", label="移动到")
                        reclass_dt = gr.Textbox(label="缺陷类型（选 NG 时生效，默认 unknown）")
                        reclass_btn = gr.Button("执行移动", variant="primary")
                        reclass_msg = gr.Textbox(label="结果", interactive=False)
                    with gr.Accordion("类别详情", open=False):
                        cat_info = gr.JSON()

            def refresh():
                rows = []
                for i in list_datasets(cfg):
                    if i.error:
                        status = f"不完整: {i.error}"
                    else:
                        status = "降级:无NG图" if i.degraded else "正常"
                    rows.append([i.category, i.train_good, i.test_good, status])
                return rows

            gr.Timer(5.0).tick(refresh, outputs=out)

            PAGE_SIZE = 20

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
                    return [], 0, "请选择类别与子目录"
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
                return shown, page, f"{c} / {sub}：共 {total} 张，第 {page + 1}/{pages} 页"

            def _page4(c, sub, page):
                shown, p, msg = _page(c, sub, page)
                return shown, p, msg, shown  # shown 同时给画廊与 shown_state

            def _info(c):
                try:
                    return dataset_info(c, cfg).__dict__
                except Exception as exc:
                    summary, detail = error_pair(exc)
                    return {"错误": summary, "详细堆栈": detail}

            def on_row_select(evt: gr.SelectData):
                """点击列表行 → 联动：面包屑 + 子目录（自动选第一项）+ 首页画廊 + 详情。"""
                rows = refresh()
                idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                if idx is None or idx >= len(rows):
                    return "未选择类别", gr.update(), [], 0, "", "", {}, []
                c = rows[idx][0]
                subs = _subdirs(c)
                first = subs[0] if subs else None
                shown, page, msg = _page(c, first, 0)
                return (f"**{c}**" + (f" / {first}" if first else ""),
                        gr.update(choices=subs, value=first),
                        shown, page, msg, c, _info(c), shown)

            out.select(on_row_select, None,
                       [crumb, dir_sel, gallery, page_state, gallery_msg, cur_cat, cat_info,
                        shown_state])

            dir_sel.change(lambda c, s: _page4(c, s, 0), [cur_cat, dir_sel],
                           [gallery, page_state, gallery_msg, shown_state])
            btn_prev.click(lambda c, s, p: _page4(c, s, p - 1), [cur_cat, dir_sel, page_state],
                           [gallery, page_state, gallery_msg, shown_state])
            btn_next.click(lambda c, s, p: _page4(c, s, p + 1), [cur_cat, dir_sel, page_state],
                           [gallery, page_state, gallery_msg, shown_state])

            def on_gallery_select(shown, evt: gr.SelectData):
                idx = evt.index
                if idx is None or idx >= len(shown):
                    return "未选中图片", ""
                from pathlib import Path as _P

                abs_p = shown[idx]
                rel = _P(abs_p).relative_to(cfg.data_root / cur_cat.value if False else abs_p)
                return abs_p, abs_p

            def on_gallery_select2(shown, cat, evt: gr.SelectData):
                idx = evt.index
                if idx is None or idx >= len(shown):
                    return "未选中图片", ""
                from pathlib import Path as _P

                abs_p = shown[idx]
                try:
                    rel = str(_P(abs_p).relative_to(cfg.data_root / cat))
                except ValueError:
                    rel = abs_p
                return rel, abs_p

            gallery.select(on_gallery_select2, [shown_state, cur_cat], [sel_info, sel_path])

            def do_reclass(cat, abs_p, target, dt, sub, page):
                if not cat or not abs_p:
                    return "请先在画廊中点击选中一张图片。", "", gr.update(), gr.update(), gr.update()
                from pathlib import Path as _P

                from dino_exp.datasets import move_image

                try:
                    rel = str(_P(abs_p).relative_to(cfg.data_root / cat))
                    dest = move_image(cat, rel, target, dt or None, cfg)
                except Exception as exc:
                    summary, detail = error_pair(exc)
                    return summary, detail, gr.update(), gr.update(), gr.update()
                shown, p, msg, shown2 = _page4(cat, sub, page)
                return f"已移动到: {dest}", "", shown, msg, shown2

            reclass_btn.click(do_reclass,
                              [cur_cat, sel_path, reclass_target, reclass_dt, dir_sel, page_state],
                              [reclass_msg, err_detail, gallery, gallery_msg, shown_state])

        # ---------------- 导入与下载 ----------------
        with gr.Tab("导入与下载"):
            gr.Markdown("### 导入自有图片")
            with gr.Row():
                cat_im = category_dropdown(
                    cfg, allow_custom=True,
                    label="类别名（选择现有类别，或输入新类别名）", refresh=0)
                label_im = gr.Radio(["ok", "ng"], value="ok", label="标签")
                dt_im = gr.Textbox(label="缺陷类型（NG 选填，默认 unknown）")
                split_im = gr.Radio(
                    [("auto：OK 图 8:2 自动分入训练集/测试集（从零建数据集推荐）", "auto"),
                     ("train：OK 图全部进训练集 train/good", "train"),
                     ("test：OK 图全部进测试集 test/good", "test")],
                    value="auto", label="OK 图去向（NG 图始终进 test/<缺陷类型>）")
            files = gr.File(file_count="multiple", label="选择图片")
            btn_im = gr.Button("导入", variant="primary")
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

            gr.Markdown("### 下载 MVTec AD 公开数据集")
            with gr.Row():
                cat_dl = gr.Dropdown(
                    ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
                     "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
                     "transistor", "wood", "zipper"],
                    value="bottle", label="MVTec 类别", scale=2)
                btn_dl = gr.Button("下载", variant="primary")
                btn_dl_force = gr.Button("强制重新下载（删除旧目录）")
            dl_msg = gr.Textbox(label="结果", interactive=False)

            def do_download(cat, force):
                try:
                    return str(import_mvtec(cat, cfg, force=force)), ""
                except Exception as exc:
                    summary, detail = error_pair(exc)
                    return summary, detail

            btn_dl.click(lambda c: do_download(c, False), cat_dl, [dl_msg, err_detail])
            btn_dl_force.click(lambda c: do_download(c, True), cat_dl, [dl_msg, err_detail])

        # ---------------- 类别管理 ----------------
        with gr.Tab("类别管理"):
            gr.Markdown("修复或删除不完整/废弃类别（列表中标注「不完整」的类别在此处理）")
            with gr.Row():
                cat_mgr = gr.Dropdown(label="选择类别", choices=[], scale=2)
                btn_mgr_refresh = gr.Button("刷新", scale=1)
                btn_fix = gr.Button("自动修复（8:2 整理出训练集）", variant="primary", scale=2)
            with gr.Row():
                del_confirm = gr.Checkbox(label="我确认删除该类别全部数据（不可恢复）", value=False)
                btn_del = gr.Button("删除该类别", variant="stop")
            mgr_msg = gr.Textbox(label="结果", interactive=False)

            def _categories():
                return category_choices(cfg)

            btn_mgr_refresh.click(lambda: gr.update(choices=_categories()), None, cat_mgr)
            gr.Timer(10.0).tick(lambda: gr.update(choices=_categories()), None, cat_mgr)

            def do_fix(c):
                if not c:
                    return "请先选择类别", ""
                try:
                    r = fix_category(c, cfg)
                    return r["note"], ""
                except Exception as exc:
                    summary, detail = error_pair(exc)
                    return summary, detail

            def do_delete(c, confirmed):
                if not c:
                    return "请先选择类别", ""
                if not confirmed:
                    return "请先勾选「确认删除」再执行。", ""
                try:
                    target = delete_category(c, cfg)
                    return f"已删除: {target}", ""
                except Exception as exc:
                    summary, detail = error_pair(exc)
                    return summary, detail

            btn_fix.click(do_fix, cat_mgr, [mgr_msg, err_detail])
            btn_del.click(do_delete, [cat_mgr, del_confirm], [mgr_msg, err_detail])

            gr.Markdown("### 缺陷类型维护（新增/改名）")
            with gr.Row():
                dt_cat = gr.Dropdown(label="类别", choices=[], scale=2)
                dt_exist = gr.Dropdown(label="现有缺陷类型", choices=[], scale=2)
            with gr.Row():
                dt_new = gr.Textbox(label="缺陷类型名", placeholder="如：划痕、缺粒")
                btn_dt_add = gr.Button("新增类型")
                btn_dt_rename = gr.Button("改名为输入的名称", variant="primary")
            dt_msg = gr.Textbox(label="结果", interactive=False)

            def _defect_types(c):
                if not c:
                    return gr.update(choices=[])
                try:
                    return gr.update(choices=sorted(dataset_info(c, cfg).defect_types))
                except Exception:
                    return gr.update(choices=[])

            dt_cat.change(_defect_types, dt_cat, dt_exist)
            gr.Timer(10.0).tick(_defect_types, dt_cat, dt_exist)

            def do_dt_add(c, name):
                if not c:
                    return "请先选择类别", ""
                try:
                    return f"已创建: {add_defect_type(c, name, cfg)}", ""
                except Exception as exc:
                    summary, detail = error_pair(exc)
                    return summary, detail

            def do_dt_rename(c, old, new):
                if not c or not old:
                    return "请先选择类别和现有缺陷类型", ""
                try:
                    return f"已改名为: {rename_defect_type(c, old, new, cfg)}", ""
                except Exception as exc:
                    summary, detail = error_pair(exc)
                    return summary, detail

            btn_dt_add.click(do_dt_add, [dt_cat, dt_new], [dt_msg, err_detail])
            btn_dt_rename.click(do_dt_rename, [dt_cat, dt_exist, dt_new], [dt_msg, err_detail])
