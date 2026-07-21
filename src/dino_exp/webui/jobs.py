"""后台任务管理：worker 线程 + 内存日志队列；UI 用 gr.Timer 轮询 status()。"""

from __future__ import annotations

import threading
import traceback
import uuid
from queue import Queue

from dino_exp.errors import DinoError


class JobManager:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, fn) -> str:
        with self._lock:
            for j in self._jobs.values():
                if j["kind"] == kind and j["state"] == "running":
                    raise DinoError(f"已有 {kind} 任务进行中，请等待完成后再启动。")
            jid = uuid.uuid4().hex[:8]
            queue: Queue = Queue()
            self._jobs[jid] = {"kind": kind, "state": "running", "queue": queue,
                               "logs": [], "result": None, "error": None}

        def run():
            try:
                result = fn(queue.put)
                self._jobs[jid].update(state="done", result=result)
            except DinoError as exc:
                # 应用层错误：信息已含修复建议，直接展示，不刷堆栈
                self._jobs[jid].update(state="error", error=str(exc))
            except Exception:
                # 未知异常：保留堆栈便于排查
                self._jobs[jid].update(state="error", error=traceback.format_exc(limit=5))

        threading.Thread(target=run, daemon=True).start()
        return jid

    def status(self, jid: str) -> dict:
        job = self._jobs[jid]
        while not job["queue"].empty():
            job["logs"].append(str(job["queue"].get()))
        return {k: v for k, v in job.items() if k != "queue"}
