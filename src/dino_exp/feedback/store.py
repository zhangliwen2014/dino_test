from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dino_exp.errors import DinoError


class FeedbackStore:
    """反馈持久化：staged.jsonl（暂存）+ applied.jsonl（归档）+ images/（图片拷贝）。"""

    def __init__(self, root: str | Path, experiment: str):
        self.dir = Path(root) / experiment
        self.images_dir = self.dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.staged_file = self.dir / "staged.jsonl"
        self.applied_file = self.dir / "applied.jsonl"

    @staticmethod
    def _read(file: Path) -> list[dict]:
        if not file.exists():
            return []
        return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _write(file: Path, rows: list[dict]) -> None:
        file.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def stage(self, record: dict) -> dict:
        src = Path(record["image_path"])
        if not src.exists():
            raise DinoError(f"反馈图片不存在: {src}。请确认路径后重试。")
        fid = uuid.uuid4().hex[:12]
        stored = f"{fid}{src.suffix}"
        shutil.copy2(src, self.images_dir / stored)
        row = {
            "id": fid,
            "stored_image": stored,
            "defect_type": record.get("defect_type"),
            "timestamp": record.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            **{k: record[k] for k in ("image_path", "model_version", "prediction", "score", "human_label")},
        }
        if row["human_label"] not in {"ok", "ng"}:
            raise DinoError(f"human_label 只能是 ok/ng，得到 '{row['human_label']}'。")
        rows = self._read(self.staged_file)
        rows.append(row)
        self._write(self.staged_file, rows)
        return row

    def staged(self) -> list[dict]:
        return self._read(self.staged_file)

    def remove(self, feedback_id: str) -> bool:
        rows = self._read(self.staged_file)
        kept = [r for r in rows if r["id"] != feedback_id]
        if len(kept) == len(rows):
            return False
        self._write(self.staged_file, kept)
        return True

    def apply(self) -> list[dict]:
        from dino_exp.feedback.staging import effective

        rows = self.staged()
        if not rows:
            raise DinoError("暂存区为空，拒绝再训练。请先 `dino feedback ...` 添加反馈。")
        applied = self._read(self.applied_file) + rows
        self._write(self.applied_file, applied)
        self._write(self.staged_file, [])
        return effective(rows)  # 生效集合：同图最新一条为准
