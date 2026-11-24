"""Deterministic end-to-end harness checks; never call a paid API or real ledger."""
from __future__ import annotations

from collections import deque
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from agent import Agent, ROOT, WorkerError, run_worker
import ledger
from storage import Store
from tools import InvoiceTools, ToolError


def request(name="post_invoice", **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": uuid4().hex, "name": name, "arguments": arguments}
    ]}


def answer(content="Done."):
    return {"role": "assistant", "content": content, "tool_calls": []}


class ScriptedRunner:
    def __init__(self, tools, responses=()):
        self.tools = tools
        self.responses = deque(responses)
        self.model_inputs = []
        self.fail = None

    def __call__(self, kind, payload, root):
        if self.fail:
            self.fail(kind, payload)
        if kind == "model":
            self.model_inputs.append(copy.deepcopy(payload))
            response = self.responses.popleft()
            if isinstance(response, BaseException):
                raise response
            return response
        try:
            return getattr(self.tools, payload["name"])(**payload["arguments"])
        except ToolError as exc:
            raise WorkerError(str(exc)) from exc


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / "invoices", self.root / "invoices")
        shutil.copyfile(ROOT / "vendors.json", self.root / "vendors.json")
        self.db_patch = patch.object(ledger, "DB_PATH", self.root / "ledger.db")
        self.db_patch.start()
        self.sleep_patch = patch.object(ledger.time, "sleep", return_value=None)
        self.sleep_patch.start()
        self.store = Store(self.root / "state.sqlite3")
        self.output = []
        self.tools = InvoiceTools(self.root)
        self.runner = ScriptedRunner(self.tools)
        self.agent = Agent(self.store, "test", root=self.root,
                           emit=self.output.append, worker=self.runner)

    def tearDown(self):
        self.store.close()
        self.sleep_patch.stop()
        self.db_patch.stop()
        self.temp.cleanup()

    def test_post_requires_approval_and_happens_once(self):
        self.runner.responses.extend([request(invoice_id="invoice-01"), answer("Posted.")])
        self.agent.submit("post invoice-01")
        self.assertEqual(len(ledger.entries()), 1)
        preview = "\n".join(self.output)
        for expected in ("V-001", "SGP/26-27/0412", "2026-10-03", "INR 16862.20"):
            self.assertIn(expected, preview)
        self.agent.approve()
        self.assertEqual(len(ledger.entries()), 2)
        self.assertIsNone(self.agent.state["pending"])
        self.agent.approve()
        self.assertEqual(len(ledger.entries()), 2)

    def test_waiting_approval_survives_restart(self):
        self.runner.responses.extend([request(invoice_id="invoice-02"), answer("Posted.")])
        self.agent.submit("post invoice-02")
        before = copy.deepcopy(self.agent.state["pending"])
        self.store.close()
        self.store = Store(self.root / "state.sqlite3")
        resumed = Agent(self.store, "test", root=self.root, emit=self.output.append, worker=self.runner)
        self.assertEqual(resumed.state["pending"], before)
        resumed.approve()
        entries = ledger.entries(invoice_number="BL-88213")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["amount"], "70800.00")

    def test_decline_reason_reaches_model_without_posting(self):
        self.runner.responses.extend([request(invoice_id="invoice-03"), answer("Not posted: PO not received yet.")])
        self.agent.submit("post invoice-03")
        self.agent.decline("PO not received yet")
        result = self.runner.model_inputs[-1]["messages"][-1]["content"]
        self.assertEqual(result["status"], "declined")
        self.assertEqual(result["reason"], "PO not received yet")
        self.assertEqual(len(ledger.entries()), 1)

    def test_blocked_and_duplicate_rejected_without_approval(self):
        for invoice, expected in (("invoice-04", "blocked"), ("invoice-05", "JE-000001")):
            self.runner.responses.extend([request(invoice_id=invoice), answer("Cannot post.")])
            self.agent.submit(f"post {invoice}")
            self.assertIsNone(self.agent.state["pending"])
            result = self.runner.model_inputs[-1]["messages"][-1]["content"]
            self.assertEqual(result["status"], "error")
            self.assertIn(expected, result["error"])
        self.assertEqual(len(ledger.entries()), 1)

    def test_provider_switch_keeps_exact_tool_history(self):
        self.runner.responses.extend([request("read_invoice", invoice_id="invoice-06"),
                                      answer("One invoice."), answer("INR 21948.00"), answer("One line.")])
        self.agent.submit("what's on invoice-06?")
        previous = copy.deepcopy(self.agent.state["messages"])
        self.agent.command("/model groq")
        self.agent.submit("which vendor and total?")
        self.assertEqual(self.runner.model_inputs[-1]["provider"], "groq")
        self.assertEqual(self.runner.model_inputs[-1]["messages"][:len(previous)], previous)
        self.agent.command("/model gemini")
        self.agent.submit("how many lines?")
        self.assertEqual(self.runner.model_inputs[-1]["provider"], "gemini")
        self.assertEqual(self.runner.model_inputs[-1]["messages"][:len(previous)], previous)

    def test_stop_closes_all_outstanding_calls_and_next_request_works(self):
        first = request("read_invoice", invoice_id="invoice-01")
        second = request("read_invoice", invoice_id="invoice-02")["tool_calls"][0]
        third = request("post_invoice", invoice_id="invoice-03")["tool_calls"][0]
        first["tool_calls"].extend([second, third])
        self.runner.responses.extend([first, answer("Read invoice-01 only.")])

        def interrupt(kind, payload):
            if kind == "tool" and payload["arguments"].get("invoice_id") == "invoice-02":
                raise KeyboardInterrupt

        self.runner.fail = interrupt
        self.agent.submit("read all invoices")
        self.assertFalse(self.agent.state["active"])
        self.assertEqual(self.agent.outstanding(), [])
        results = [m["content"] for m in self.agent.state["messages"] if m["role"] == "tool"]
        self.assertEqual(results[0]["invoice_id"], "invoice-01")
        self.assertEqual([r.get("status") for r in results[1:]], ["cancelled", "cancelled"])
        self.runner.fail = None
        self.agent.submit("which invoices did you get through?")
        self.assertIn("Read invoice-01 only.", self.output[-1])
        self.assertEqual(len(ledger.entries()), 1)

    def test_model_interrupt_leaves_next_request_valid(self):
        self.runner.responses.extend([KeyboardInterrupt(), answer("Works again.")])
        self.agent.submit("hello")
        self.assertFalse(self.agent.state["active"])
        self.agent.submit("hello again")
        self.assertEqual(self.output[-1], "agent> Works again.")

    def test_interrupted_post_after_commit_is_reconciled_without_duplicate(self):
        self.runner.responses.append(request(invoice_id="invoice-01"))
        self.agent.submit("post invoice-01")
        original = self.tools.post_approved

        def commit_then_interrupt(**kwargs):
            original(**kwargs)
            raise KeyboardInterrupt

        with patch.object(self.tools, "post_approved", side_effect=commit_then_interrupt):
            self.agent.approve()
        self.assertEqual(len(ledger.entries()), 2)
        result = self.agent.state["messages"][-1]["content"]
        self.assertEqual(result["status"], "posted")
        self.assertTrue(result["recovered"])
        self.assertIsNone(self.agent.state["pending"])

    def test_crash_after_commit_recovers_saved_posting_intent(self):
        self.runner.responses.append(request(invoice_id="invoice-01"))
        self.agent.submit("post invoice-01")
        pending = self.agent.state["pending"]
        pending["status"] = "posting"
        self.agent.save()
        self.tools.post_approved(pending["snapshot"], pending["operation_id"])
        resumed = Agent(self.store, "test", root=self.root, emit=self.output.append, worker=self.runner)
        self.assertTrue(resumed.recover())
        self.assertIsNone(resumed.state["pending"])
        self.assertEqual(resumed.outstanding(), [])
        self.assertEqual(len(ledger.entries()), 2)

    def test_crash_before_commit_needs_fresh_approval(self):
        self.runner.responses.append(request(invoice_id="invoice-01"))
        self.agent.submit("post invoice-01")
        self.agent.state["pending"]["status"] = "posting"
        self.agent.save()
        resumed = Agent(self.store, "test", root=self.root, emit=self.output.append, worker=self.runner)
        self.assertTrue(resumed.recover())
        self.assertEqual(resumed.state["pending"]["status"], "waiting")
        self.assertEqual(len(ledger.entries()), 1)

    def test_multiple_post_requests_each_need_separate_approval(self):
        response = request(invoice_id="invoice-01")
        response["tool_calls"].extend(request(invoice_id="invoice-02")["tool_calls"])
        self.runner.responses.extend([response, answer("Posted both.")])
        self.agent.submit("post invoice-01 and invoice-02")
        self.agent.approve()
        self.assertEqual(self.agent.state["pending"]["invoice_id"], "invoice-02")
        self.assertEqual(len(ledger.entries()), 2)
        self.agent.approve()
        self.assertEqual(len(ledger.entries()), 3)

    def test_unanswered_tool_after_crash_can_resume(self):
        self.agent.state["messages"] = [{"role": "user", "content": "read invoice-01"},
                                         request("read_invoice", invoice_id="invoice-01")]
        self.agent.state["active"] = True
        self.agent.save()
        self.runner.responses.append(answer("Read."))
        self.agent.command("/resume")
        self.assertEqual(self.agent.outstanding(), [])
        self.assertEqual(self.output[-1], "agent> Read.")

    def test_unknown_and_malformed_tools_return_errors_to_model(self):
        response = request("delete_ledger")
        response["tool_calls"].extend(request("read_invoice", invoice_id="invoice-01", extra="bad")["tool_calls"])
        response["tool_calls"].extend(request("read_invoice", invoice_id="invoice-01")["tool_calls"])
        response["tool_calls"][-1]["arguments"] = "broken JSON"
        self.runner.responses.extend([response, answer("Errors reported.")])
        self.agent.submit("try tools")
        results = [m for m in self.agent.state["messages"] if m["role"] == "tool"]
        self.assertEqual(len(results), 3)
        self.assertTrue(all(m["content"]["status"] == "error" for m in results))

    def test_model_error_can_resume_without_losing_request(self):
        self.runner.responses.extend([WorkerError("Temporary model outage"), answer("Recovered.")])
        self.agent.submit("hello")
        self.assertTrue(self.agent.state["active"])
        self.agent.command("/resume")
        self.assertEqual(self.output[-1], "agent> Recovered.")

    def test_approved_invoice_change_is_rejected(self):
        self.runner.responses.extend([request(invoice_id="invoice-01"), answer("Invoice changed; not posted.")])
        self.agent.submit("post invoice-01")
        path = self.root / "invoices" / "invoice-01.json"
        data = json.loads(path.read_text())
        data["due_date"] = "2026-11-03"
        path.write_text(json.dumps(data))
        self.agent.approve()
        self.assertEqual(len(ledger.entries()), 1)
        self.assertIsNone(self.agent.state["pending"])
        self.assertIn("changed since approval", self.runner.model_inputs[-1]["messages"][-1]["content"]["error"])
        self.assertEqual(self.output[-1], "agent> Invoice changed; not posted.")

    def test_interrupt_during_completion_save_reloads_committed_truth(self):
        self.runner.responses.append(request(invoice_id="invoice-01"))
        self.agent.submit("post invoice-01")
        real_save = self.store.save
        interrupted = False

        def save_then_interrupt(name, state):
            nonlocal interrupted
            real_save(name, state)
            if not interrupted and state["pending"] is None and state["messages"][-1]["role"] == "tool":
                interrupted = True
                raise KeyboardInterrupt

        with patch.object(self.store, "save", side_effect=save_then_interrupt):
            self.agent.approve()
        self.assertTrue(interrupted)
        self.assertEqual(len(ledger.entries()), 2)
        self.assertIsNone(self.agent.state["pending"])
        results = [m for m in self.agent.state["messages"] if m["role"] == "tool"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"]["status"], "posted")


class WorkerCancellationTests(unittest.TestCase):
    def test_real_spawned_tool_stops_within_seconds_and_next_tool_runs(self):
        before = {p.pid for p in mp.active_children()}
        timer = threading.Timer(0.5, signal.raise_signal, args=(signal.SIGINT,))
        started = time.monotonic()
        with patch.dict(os.environ, {"FOURX_READ_DELAY": "8"}):
            timer.start()
            try:
                with self.assertRaises(KeyboardInterrupt):
                    run_worker("tool", {"name": "read_invoice", "arguments": {"invoice_id": "invoice-01"}}, ROOT)
            finally:
                timer.cancel()
                timer.join()
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertEqual({p.pid for p in mp.active_children()}, before)
        result = run_worker("tool", {"name": "read_invoice", "arguments": {"invoice_id": "invoice-06"}}, ROOT)
        self.assertEqual(result["total"], "21948.00")


if __name__ == "__main__":
    unittest.main()
