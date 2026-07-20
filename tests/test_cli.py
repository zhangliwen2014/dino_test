from click.testing import CliRunner

from dino_exp.cli import main


def test_help_lists_all_commands():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ["dataset", "train", "validate", "test", "feedback", "retrain",
                "versions", "rollback", "export", "unstage", "ui"]:
        assert cmd in result.output


def test_dataset_list_empty(tmp_path, monkeypatch):
    from dino_exp.config import Config

    monkeypatch.setattr("dino_exp.cli.load_config", lambda p=None: Config(data_root=tmp_path / "d"))
    result = CliRunner().invoke(main, ["dataset", "list"])
    assert result.exit_code == 0
    assert "无数据集" in result.output or "category" in result.output.lower()


def test_feedback_requires_label():
    result = CliRunner().invoke(main, ["feedback", "--image", "x.png"])
    assert result.exit_code != 0  # --label 必填


def test_retrain_prints_preview_and_aborts_without_yes(tmp_path, monkeypatch):
    from dino_exp.config import Config

    monkeypatch.setattr("dino_exp.cli.load_config", lambda p=None: Config(
        data_root=tmp_path / "d", models_root=tmp_path / "m", feedback_root=tmp_path / "f"))
    # cli 在命令函数内 `from dino_exp.retrain import preview_retrain`（延迟导入），
    # 因此补丁目标是源模块属性，调用时生效
    monkeypatch.setattr("dino_exp.retrain.preview_retrain", lambda c, cfg: {
        "current_version": "v001", "ok": 1, "ng": 2, "suspicious": [], "conflicts": []})
    result = CliRunner().invoke(main, ["retrain", "--category", "bottle"], input="n\n")
    assert result.exit_code == 0
    assert "v001" in result.output and "取消" in result.output
