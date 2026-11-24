"""Independent edge-case checks for consent, restarts, and interrupted writes."""
import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ledger
from agent import Agent, WorkerError
from storage import Store
from tools import InvoiceTools, ToolError

ROOT = Path(__file__).resolve().parents[1]


def call(call_id, name, **arguments):
    return {"id": call_id, "name": name, "arguments": arguments}


def answer(*calls, text=""):
    return {"role": "assistant", "content": text, "tool_calls": list(calls), "provider": "gemini"}


class HarnessReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / "invoices", self.root / "invoices")
        shutil.copy(ROOT / "vendors.json", self.root / "vendors.json")
        self.patches = [patch.object(ledger, "DB_PATH", self.root / "ledger.db"),
                        patch.object(ledger.time, "sleep", lambda _: None)]
        for item in self.patches:
            item.start()
        self.tools = InvoiceTools(self.root)
        self.store = Store(self.root / "conversation.db")
        self.responses = []
        self.calls = []
        self.post_interrupt = None
        self.agent = Agent(self.store, "review", root=self.root, emit=lambda _: None, worker=self.worker)

    def tearDown(self):
        self.store.close()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def worker(self, kind, payload, root):
        self.calls.append((kind, copy.deepcopy(payload)))
        if kind == "model":
            response = self.responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
        if payload["name"] == "post_approved" and self.post_interrupt == "before":
            raise KeyboardInterrupt
        try:
            result = getattr(self.tools, payload["name"])(**payload.get("arguments", {}))
        except ToolError as exc:
            raise WorkerError(str(exc)) from exc
        if payload["name"] == "post_approved" and self.post_interrupt == "after":
            raise KeyboardInterrupt
        return result

    def request_post(self):
        self.responses = [answer(call("post-1", "post_invoice", invoice_id="invoice-01"))]
        self.agent.submit("post invoice-01")
        self.assertEqual(self.agent.state["pending"]["status"], "waiting")

    def test_restart_after_consent_without_write_requires_fresh_consent(self):
        self.request_post()
        self.agent.state["pending"]["status"] = "approved"
        self.agent.save()
        resumed = Agent(self.store, "review", root=self.root, emit=lambda _: None, worker=self.worker)
        self.assertTrue(resumed.recover())
        self.assertEqual(resumed.state["pending"]["status"], "waiting")
        self.assertFalse(any(p["name"] == "post_approved" for k, p in self.calls if k == "tool"))
        self.assertEqual(len(ledger.entries()), 1)

    def test_restart_after_commit_recovers_original_operation(self):
        self.request_post()
        pending = self.agent.state["pending"]
        pending["status"] = "posting"
        self.agent.save()
        posted = self.tools.post_approved(pending["snapshot"], pending["operation_id"])
        resumed = Agent(self.store, "review", root=self.root, emit=lambda _: None, worker=self.worker)
        self.assertTrue(resumed.recover())
        self.assertIsNone(resumed.state["pending"])
        self.assertEqual(resumed.state["messages"][-1]["content"]["entry_id"], posted["entry_id"])
        self.assertEqual(len(ledger.entries()), 2)

    def test_multiple_posts_are_individually_gated_and_complete_in_order(self):
        self.responses = [answer(call("post-1", "post_invoice", invoice_id="invoice-01"),
                                 call("post-2", "post_invoice", invoice_id="invoice-02")),
                          answer(text="Only invoice-02 was posted.")]
        self.agent.submit("Post both invoices")
        self.agent.decline("Missing PO")
        self.assertEqual(self.agent.state["pending"]["invoice_id"], "invoice-02")
        self.assertEqual(len(ledger.entries()), 1)
        self.agent.approve()
        tool_results = [m["content"] for m in self.agent.state["messages"] if m["role"] == "tool"]
        self.assertEqual([r["status"] for r in tool_results], ["declined", "posted"])
        self.assertEqual(len(ledger.entries()), 2)

    def test_stop_after_commit_reports_success_and_cancels_remaining_calls(self):
        self.responses = [answer(call("post-1", "post_invoice", invoice_id="invoice-01"),
                                 call("read-2", "read_invoice", invoice_id="invoice-02"))]
        self.agent.submit("Post invoice-01 and read invoice-02")
        self.post_interrupt = "after"
        self.agent.approve()
        self.assertIsNone(self.agent.state["pending"])
        results = [m["content"] for m in self.agent.state["messages"] if m["role"] == "tool"]
        self.assertEqual([r["status"] for r in results], ["posted", "cancelled"])
        self.assertEqual(len(ledger.entries()), 2)

    def test_stop_before_commit_does_not_post_or_reuse_consent(self):
        self.request_post()
        self.post_interrupt = "before"
        self.agent.approve()
        self.assertIsNone(self.agent.state["pending"])
        self.assertEqual(self.agent.state["messages"][-1]["content"]["status"], "cancelled")
        self.assertEqual(len(ledger.entries()), 1)

    def test_model_error_allows_provider_switch_and_resume(self):
        self.responses = [WorkerError("GEMINI_API_KEY is missing"), answer(text="Ready on Groq")]
        self.agent.submit("hello")
        self.agent.command("/model groq")
        self.agent.command("/resume")
        self.assertEqual(self.agent.state["messages"][-1]["content"], "Ready on Groq")
        self.assertEqual(self.agent.state["provider"], "groq")
        self.assertFalse(self.agent.state["active"])


if __name__ == "__main__":
    unittest.main()
