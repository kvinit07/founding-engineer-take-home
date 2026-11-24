"""Real spawned workers interrupted on both sides of the ledger commit boundary.

Only model responses are scripted. Each scenario copies the runnable application
into a temporary directory so neither shipped data nor API credentials are used.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = r'''
import json
import multiprocessing as mp
from pathlib import Path
import signal
import sqlite3
import sys
import threading
import time

from agent import Agent, run_worker
from storage import Store

ROOT = Path(__file__).resolve().parent


def ledger_count():
    uri = (ROOT / "ledger.db").as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=0.1) as connection:
        return connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]


def main(boundary):
    responses = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "post-process-1", "name": "post_invoice",
             "arguments": {"invoice_id": "invoice-01"}}
        ]},
        {"role": "assistant", "content": "The next request worked.", "tool_calls": []},
    ]
    triggered = threading.Event()
    observer_errors = []
    interrupt_times = []
    notes = []

    def worker(kind, payload, root):
        if kind == "model":
            return responses.pop(0)
        if payload["name"] != "post_approved":
            return run_worker(kind, payload, root)

        done = threading.Event()
        lock_path = ROOT / ".state" / "ledger-post.lock"
        # prepare_post's worker has exited. Recreating its lock file tells the
        # observer that the real posting worker has reached the policy boundary.
        # No ledger code or timing is patched.
        if boundary == "before":
            lock_path.unlink(missing_ok=True)

        def interrupt_at_boundary():
            deadline = time.monotonic() + 8
            while not done.is_set() and time.monotonic() < deadline:
                try:
                    count = ledger_count()
                except sqlite3.Error:
                    count = None
                ready = (lock_path.exists() and count == 1) if boundary == "before" else count == 2
                if ready:
                    interrupt_times.append(time.monotonic())
                    triggered.set()
                    signal.raise_signal(signal.SIGINT)
                    return
                done.wait(0.005)
            if not done.is_set():
                observer_errors.append("Posting did not reach the requested commit boundary.")

        observer = threading.Thread(target=interrupt_at_boundary, daemon=True)
        observer.start()
        try:
            return run_worker(kind, payload, root)
        finally:
            done.set()
            observer.join(timeout=2)
            if observer.is_alive():
                observer_errors.append("Commit-boundary observer did not stop.")

    store = Store(ROOT / ".state" / "conversations.sqlite3")
    try:
        agent = Agent(store, "process-review", root=ROOT, emit=notes.append, worker=worker)
        agent.submit("post invoice-01")
        assert agent.state["pending"]["status"] == "waiting"
        started = time.monotonic()
        agent.approve()
        elapsed = time.monotonic() - started
        assert triggered.is_set(), "A real posting worker was not interrupted"
        stopped_seconds = time.monotonic() - interrupt_times[0]
        assert stopped_seconds < 4, f"Stop took {stopped_seconds:.2f} seconds"
        assert not observer_errors, observer_errors
        assert agent.state["pending"] is None, agent.state["pending"]
        results = [message["content"] for message in agent.state["messages"]
                   if message["role"] == "tool"]
        assert len(results) == 1, results
        saved = store.load("process-review")
        assert saved["pending"] is None
        assert saved["messages"][-1]["content"] == results[0]
        count = ledger_count()
        expected = "cancelled" if boundary == "before" else "posted"
        assert results[0]["status"] == expected, results
        assert count == (1 if boundary == "before" else 2), count
        if boundary == "after":
            assert results[0]["recovered"] is True, results[0]
        agent.submit("Can we continue?")
        assert agent.state["messages"][-1]["content"] == "The next request worked."
        assert agent.state["active"] is False
        print(json.dumps({"boundary": boundary, "count": count,
                          "status": results[0]["status"], "pending": agent.state["pending"],
                          "next_request": agent.state["messages"][-1]["content"],
                          "elapsed_seconds": elapsed, "stopped_seconds": stopped_seconds}))
    finally:
        store.close()


if __name__ == "__main__":
    mp.freeze_support()
    main(sys.argv[1])
'''


class PostingProcessTests(unittest.TestCase):
    def run_scenario(self, boundary):
        with tempfile.TemporaryDirectory(prefix="fourx-process-") as temporary:
            root = Path(temporary)
            # Copy only application source and fixtures, never runtime state.
            for path in ROOT.glob("*.py"):
                shutil.copy2(path, root / path.name)
            shutil.copytree(ROOT / "invoices", root / "invoices")
            shutil.copy2(ROOT / "vendors.json", root / "vendors.json")
            (root / "scenario.py").write_text(textwrap.dedent(SCENARIO), encoding="utf-8")
            process = subprocess.run(
                [sys.executable, "scenario.py", boundary], cwd=root,
                text=True, capture_output=True, timeout=15,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            return json.loads(process.stdout.strip())

    def test_stop_before_real_ledger_commit(self):
        result = self.run_scenario("before")
        self.assertEqual((result["count"], result["status"]), (1, "cancelled"))
        self.assertIsNone(result["pending"])
        self.assertEqual(result["next_request"], "The next request worked.")

    def test_stop_after_confirmed_real_ledger_commit(self):
        result = self.run_scenario("after")
        self.assertEqual((result["count"], result["status"]), (2, "posted"))
        self.assertIsNone(result["pending"])
        self.assertEqual(result["next_request"], "The next request worked.")


if __name__ == "__main__":
    unittest.main()
