import time

from dino_exp.webui.jobs import JobManager


def test_job_runs_and_collects_logs():
    jm = JobManager()

    def work(log):
        log("step 1")
        log("step 2")
        return {"version": "v001"}

    jid = jm.start("train", work)
    for _ in range(50):
        if jm.status(jid)["state"] == "done":
            break
        time.sleep(0.05)
    st = jm.status(jid)
    assert st["state"] == "done"
    assert st["result"] == {"version": "v001"}
    assert "step 1" in "".join(st["logs"])


def test_job_error_captured():
    jm = JobManager()

    def bad(log):
        raise RuntimeError("boom")

    jid = jm.start("train", bad)
    for _ in range(50):
        if jm.status(jid)["state"] != "running":
            break
        time.sleep(0.05)
    st = jm.status(jid)
    assert st["state"] == "error"
    assert "boom" in st["error"]


def test_one_job_per_kind():
    jm = JobManager()
    jm.start("train", lambda log: time.sleep(2))
    try:
        jm.start("train", lambda log: None)
        raise AssertionError("应当拒绝并发的同类任务")
    except Exception as exc:
        assert "进行中" in str(exc)


def test_job_dinoerror_shows_friendly_message():
    """应用层 DinoError 只展示友好信息，不刷堆栈。"""
    from dino_exp.errors import DinoError

    jm = JobManager()

    def bad(log):
        raise DinoError("类别名不能为空。请输入类别名（如 bottle）后重试。")

    jid = jm.start("train", bad)
    for _ in range(50):
        if jm.status(jid)["state"] != "running":
            break
        time.sleep(0.05)
    st = jm.status(jid)
    assert st["state"] == "error"
    assert st["error"] == "类别名不能为空。请输入类别名（如 bottle）后重试。"
    assert "Traceback" not in st["error"]


def test_job_unknown_error_keeps_traceback():
    jm = JobManager()

    def bad(log):
        raise RuntimeError("boom")

    jid = jm.start("train", bad)
    for _ in range(50):
        if jm.status(jid)["state"] != "running":
            break
        time.sleep(0.05)
    st = jm.status(jid)
    assert st["state"] == "error"
    assert "Traceback" in st["error"] and "boom" in st["error"]
