"""Run the assignment's seven scenarios in order, without a live model API."""
from __future__ import annotations

from collections import deque
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from agent import Agent, ROOT, WorkerError
import ledger
from providers import to_gemini_contents, to_groq_messages
from storage import Store
from tools import InvoiceTools, ToolError


def answer(text):
    return {"role": "assistant", "content": text, "tool_calls": []}


def request(name="post_invoice", **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": uuid4().hex, "name": name, "arguments": arguments}
    ]}


class ScriptedWorker:
    """Replace only model transport; execute real local tools and SQLite writes."""

    def __init__(self, tools):
        self.tools = tools
        self.responses = deque()
        self.inputs = []
        self.interrupt = None

    def __call__(self, kind, payload, root):
        if kind == "model":
            self.inputs.append(copy.deepcopy(payload))
            response = self.responses.popleft()
            return response(payload) if callable(response) else response
        if self.interrupt:
            self.interrupt(payload)
        try:
            return getattr(self.tools, payload["name"])(**payload["arguments"])
        except ToolError as exc:
            raise WorkerError(str(exc)) from exc


class AssignmentAcceptanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shutil.copytree(ROOT / "invoices", self.root / "invoices")
        shutil.copyfile(ROOT / "vendors.json", self.root / "vendors.json")
        database = patch.object(ledger, "DB_PATH", self.root / "ledger.db")
        database.start()
        self.addCleanup(database.stop)
        delay = patch.object(ledger.time, "sleep", return_value=None)
        delay.start()
        self.addCleanup(delay.stop)
        self.store = Store(self.root / "conversations.sqlite3")
        self.addCleanup(lambda: self.store.close())
        self.output = []
        self.worker = ScriptedWorker(InvoiceTools(self.root))
        self.agent = self.make_agent()

    def make_agent(self):
        return Agent(self.store, "assignment-demo", root=self.root,
                     emit=self.output.append, worker=self.worker)

    def test_seven_scenarios_in_one_resumable_conversation(self):
        # 1. Approval displays the exact invoice and performs one write.
        self.worker.responses.extend([request(invoice_id="invoice-01"), answer("Posted invoice-01.")])
        self.agent.submit("post invoice-01")
        for expected in ("V-001", "SGP/26-27/0412", "2026-10-03", "INR 16862.20"):
            self.assertIn(expected, "\n".join(self.output))
        self.assertEqual(len(ledger.entries()), 1)
        self.agent.command("/approve")
        self.assertEqual(len(ledger.entries(invoice_number="SGP/26-27/0412")), 1)

        # 2. Quit, reopen storage and the same conversation, then approve.
        self.worker.responses.extend([request(invoice_id="invoice-02"), answer("Posted invoice-02.")])
        self.agent.submit("post invoice-02")
        pending = copy.deepcopy(self.agent.state["pending"])
        self.assertFalse(self.agent.command("/quit"))
        saved_messages = copy.deepcopy(self.agent.state["messages"])
        self.store.close()
        self.store = Store(self.root / "conversations.sqlite3")
        self.agent = self.make_agent()
        self.assertEqual(self.agent.state["pending"], pending)
        self.assertEqual(self.agent.state["messages"], saved_messages)
        self.assertTrue(self.agent.recover())
        self.agent.show_approval()
        self.assertIn("INR 70800.00", "\n".join(self.output))
        self.agent.command("/approve")
        self.assertEqual([row["amount"] for row in ledger.entries(invoice_number="BL-88213")], ["70800.00"])

        # 3. A refusal and its reason reach the model without posting.
        self.worker.responses.extend([
            request(invoice_id="invoice-03"), answer("Not posted: PO not received yet.")
        ])
        self.agent.submit("post invoice-03")
        self.agent.command("/decline PO not received yet")
        declined = self.worker.inputs[-1]["messages"][-1]["content"]
        self.assertEqual((declined["status"], declined["reason"]), ("declined", "PO not received yet"))
        self.assertIn("PO not received yet", self.output[-1])
        self.assertEqual(ledger.entries(invoice_number="KOS-2026-1187"), [])

        # 4. Both provider switches retain the earlier tool result verbatim.
        self.worker.responses.extend([
            request("read_invoice", invoice_id="invoice-06"), answer("Read invoice-06.")
        ])
        self.agent.submit("what's on invoice-06?")
        before_switch = copy.deepcopy(self.agent.state["messages"])

        def invoice_from_history(payload):
            self.assertEqual(payload["messages"][:len(before_switch)], before_switch)
            return next(message["content"] for message in reversed(payload["messages"])
                        if message["role"] == "tool" and message["name"] == "read_invoice"
                        and message["content"].get("invoice_id") == "invoice-06")

        def vendor_and_total(payload):
            self.assertEqual(payload["provider"], "groq")
            invoice = invoice_from_history(payload)
            return answer(f"{invoice['vendor_name']}: {invoice['currency']} {invoice['total']}.")

        self.agent.command("/model groq")
        self.worker.responses.append(vendor_and_total)
        self.agent.submit("which vendor was that from, and what's the total with tax?")
        self.assertIn("Shree Ganesh Packaging Pvt Ltd: INR 21948.00", self.output[-1])

        def line_count(payload):
            self.assertEqual(payload["provider"], "gemini")
            return answer(f"{len(invoice_from_history(payload)['lines'])} line.")

        self.agent.command("/model gemini")
        self.worker.responses.append(line_count)
        self.agent.submit("and how many lines did it have?")
        self.assertEqual(self.output[-1], "agent> 1 line.")

        # 5. One read completes; stop the second, close every outstanding call.
        summary_start = len(self.agent.state["messages"])
        reads = request("read_invoice", invoice_id="invoice-01")
        for number in range(2, 7):
            reads["tool_calls"].extend(request("read_invoice", invoice_id=f"invoice-{number:02}")["tool_calls"])
        self.worker.responses.append(reads)

        def stop_second_read(payload):
            if payload["name"] == "read_invoice" and payload["arguments"]["invoice_id"] == "invoice-02":
                raise KeyboardInterrupt

        self.worker.interrupt = stop_second_read
        self.agent.submit("read every invoice and summarise each one in a line")
        self.worker.interrupt = None
        self.assertFalse(self.agent.state["active"])
        self.assertEqual(self.agent.outstanding(), [])
        results = [m["content"] for m in self.agent.state["messages"][summary_start:] if m["role"] == "tool"]
        self.assertEqual(results[0]["invoice_id"], "invoice-01")
        self.assertEqual([result.get("status") for result in results[1:]], ["cancelled"] * 5)
        self.assertEqual(len(ledger.entries()), 3)

        def completed_reads(payload):
            finished = [m["content"]["invoice_id"] for m in payload["messages"][summary_start:]
                        if m["role"] == "tool" and m["name"] == "read_invoice"
                        and m["content"].get("status") != "cancelled"]
            self.assertEqual(finished, ["invoice-01"])
            return answer("Read " + ", ".join(finished) + " only in that task.")

        self.worker.responses.append(completed_reads)
        self.agent.submit("which invoices did you get through?")
        self.assertEqual(self.output[-1], "agent> Read invoice-01 only in that task.")

        # 6 and 7. Blocked and opening-duplicate invoices never reach approval.
        for invoice_id, detail in (("invoice-04", "Bank details under verification"),
                                   ("invoice-05", "JE-000001")):
            output_start = len(self.output)
            self.worker.responses.extend([request(invoice_id=invoice_id), answer("Cannot post.")])
            self.agent.submit(f"post {invoice_id}")
            result = self.worker.inputs[-1]["messages"][-1]["content"]
            self.assertEqual(result["status"], "error")
            self.assertIn(detail, result["error"])
            self.assertIsNone(self.agent.state["pending"])
            self.assertNotIn("Approval needed", "\n".join(self.output[output_start:]))

        entries = ledger.entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual({(row["vendor_id"], row["invoice_number"], row["amount"]) for row in entries}, {
            ("V-002", "BL-88190", "25960.00"),
            ("V-001", "SGP/26-27/0412", "16862.20"),
            ("V-002", "BL-88213", "70800.00"),
        })
        self.assertEqual(self.agent.outstanding(), [])
        messages = self.agent.state["messages"]
        requests = [call["id"] for message in messages for call in message.get("tool_calls", [])]
        responses = [message["tool_call_id"] for message in messages if message["role"] == "tool"]
        self.assertCountEqual(requests, responses)
        self.assertEqual(len(responses), len(set(responses)))
        self.assertTrue(to_groq_messages(messages, "test"))
        self.assertTrue(to_gemini_contents(messages))
        self.assertEqual(self.store.load("assignment-demo"), self.agent.state)
        self.assertFalse(self.worker.responses)


if __name__ == "__main__":
    unittest.main()
