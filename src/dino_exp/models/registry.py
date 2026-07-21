from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from dino_exp.errors import DinoError

_VERSION_RE = re.compile(r"^v(\d{3,})$")


class Registry:
    """模型版本库。目录结构见设计文档 §3.4；current 为指针文件（非符号链接）。

    原子性：所有内容先写入 <exp>/.tmp-<v>，成功后 os.rename 一次性替换，
    current 指针文件最后更新（同样走临时文件 + os.replace）。
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _exp_dir(self, experiment: str, create: bool = False) -> Path:
        d = self.root / experiment
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def list(self, experiment: str) -> list[str]:
        d = self._exp_dir(experiment)
        if not d.is_dir():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir() and _VERSION_RE.match(p.name))

    def current(self, experiment: str) -> str | None:
        cur = self._exp_dir(experiment) / "current"
        return cur.read_text(encoding="utf-8").strip() if cur.exists() else None

    def version_dir(self, experiment: str, version: str) -> Path:
        d = self._exp_dir(experiment) / version
        if not d.is_dir():
            raise DinoError(f"版本 {experiment}/{version} 不存在。可用版本: {self.list(experiment)}")
        return d

    def _next_version(self, experiment: str) -> str:
        nums = [int(_VERSION_RE.match(v).group(1)) for v in self.list(experiment)]
        return f"v{(max(nums) + 1) if nums else 1:03d}"

    def create_version(
        self,
        experiment: str,
        *,
        normal_bank: str | Path,
        defect_bank: str | Path | None,
        checkpoint: str | Path | None,
        config: dict,
        metrics: dict,
        meta: dict,
    ) -> str:
        exp = self._exp_dir(experiment, create=True)
        version = self._next_version(experiment)
        tmp = exp / f".tmp-{version}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        try:
            shutil.copy2(normal_bank, tmp / "normal_bank.pt")
            if defect_bank is not None:
                shutil.copy2(defect_bank, tmp / "defect_bank.pt")
            else:
                torch.save(torch.empty(0), tmp / "defect_bank.pt")
            if checkpoint is not None:
                shutil.copy2(checkpoint, tmp / "checkpoint.ckpt")
            (tmp / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            (tmp / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            full_meta = {"created_at": datetime.now(timezone.utc).isoformat(), **meta}
            (tmp / "meta.json").write_text(json.dumps(full_meta, indent=2), encoding="utf-8")
            os.rename(tmp, exp / version)  # 同分区原子替换
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            raise DinoError(f"版本写入失败: {exc}。当前版本未受影响，请检查磁盘空间与路径后重试。") from exc
        self.switch(experiment, version)  # current 指针最后更新
        return version

    def switch(self, experiment: str, version: str) -> None:
        self.version_dir(experiment, version)  # 校验存在
        exp = self._exp_dir(experiment)
        tmp = exp / ".current.tmp"
        tmp.write_text(version + "\n", encoding="utf-8")
        os.replace(tmp, exp / "current")

    def rollback(self, experiment: str, version: str) -> None:
        """回滚 = 切换 current 指针；历史版本目录从不被修改。"""
        self.switch(experiment, version)

    def delete(self, experiment: str, version: str) -> Path:
        """删除指定版本目录（不可恢复）。当前使用中的版本不可删除，需先切换/回滚。"""
        d = self.version_dir(experiment, version)  # 不存在则抛 DinoError
        if self.current(experiment) == version:
            raise DinoError(
                f"版本 {experiment}/{version} 是当前使用中的版本，不能删除。"
                f"请先 `dino rollback --category {experiment} <其他版本>` 切换后再删。"
            )
        shutil.rmtree(d)
        return d
