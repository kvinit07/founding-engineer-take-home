"""Posting policy tests use copies and a temporary database, never shipped data."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ledger
from tools import InvoiceTools, ToolError
from storage import AppLock

ROOT = Path(__file__).resolve().parents[1]


class InvoiceToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        shutil.copytree(ROOT / "invoices", self.root / "invoices")
        shutil.copy(ROOT / "vendors.json", self.root / "vendors.json")
        self.db_patch = patch.object(ledger, "DB_PATH", self.root / "ledger.db")
        self.sleep_patch = patch.object(ledger.time, "sleep", lambda _: None)
        self.db_patch.start()
        self.sleep_patch.start()
        self.tools = InvoiceTools(self.root)

    def tearDown(self):
        self.sleep_patch.stop()
        self.db_patch.stop()
        self.tmp.cleanup()

    def edit_invoice(self, change):
        path = self.root / "invoices/invoice-01.json"
        invoice = json.loads(path.read_text())
        change(invoice)
        path.write_text(json.dumps(invoice))

    def test_fixtures_and_lookup(self):
        self.assertEqual(len(self.tools.list_invoices()["invoices"]), 6)
        self.assertEqual(self.tools.lookup_vendor("Bluefin")["vendors"][0]["vendor_id"], "V-002")
        invoice = self.tools.read_invoice("invoice-06")
        self.assertEqual((invoice["total"], len(invoice["lines"])), ("21948.00", 1))

    def test_prepare_then_post_once_and_replay_same_operation(self):
        snapshot = self.tools.prepare_post("invoice-01")
        self.assertEqual(snapshot["amount"], "16862.20")
        result = self.tools.post_approved(snapshot, "operation-1")
        self.assertFalse(result["recovered"])
        recovered = self.tools.post_approved(snapshot, "operation-1")
        self.assertTrue(recovered["recovered"])
        self.assertEqual(result["entry_id"], recovered["entry_id"])
        self.assertEqual(len(ledger.entries()), 2)
        with self.assertRaisesRegex(ToolError, "already in the ledger"):
            self.tools.post_approved(snapshot, "different-operation")

    def test_blocked_before_approval(self):
        with self.assertRaisesRegex(ToolError, "V-004.*blocked.*Bank details"):
            self.tools.prepare_post("invoice-04")
        self.assertEqual(len(ledger.entries()), 1)

    def test_opening_duplicate_before_approval(self):
        with self.assertRaisesRegex(ToolError, "JE-000001"):
            self.tools.prepare_post("invoice-05")

    def test_changed_invoice_requires_fresh_approval(self):
        snapshot = self.tools.prepare_post("invoice-01")
        self.edit_invoice(lambda inv: inv.update(due_date="2026-11-03"))
        with self.assertRaisesRegex(ToolError, "changed since approval"):
            self.tools.post_approved(snapshot, "operation-1")
        self.assertEqual(len(ledger.entries()), 1)

    def test_vendor_blocked_after_preview(self):
        snapshot = self.tools.prepare_post("invoice-01")
        path = self.root / "vendors.json"
        vendors = json.loads(path.read_text())
        vendors[0].update(status="blocked", blocked_reason="New hold")
        path.write_text(json.dumps(vendors))
        with self.assertRaisesRegex(ToolError, "blocked.*New hold"):
            self.tools.post_approved(snapshot, "operation-1")
        self.assertEqual(len(ledger.entries()), 1)

    def test_duplicate_appears_after_preview(self):
        snapshot = self.tools.prepare_post("invoice-01")
        ledger.post(vendor_id="V-001", invoice_number=snapshot["invoice_number"],
                    invoice_date=snapshot["invoice_date"], currency="INR", amount="16862.20")
        with self.assertRaisesRegex(ToolError, "already in the ledger"):
            self.tools.post_approved(snapshot, "operation-1")
        self.assertEqual(len(ledger.entries()), 2)

    def test_reconcile_lost_return_value(self):
        snapshot = self.tools.prepare_post("invoice-01")
        entry_id = ledger.post(vendor_id="V-001", invoice_number=snapshot["invoice_number"],
                               invoice_date=snapshot["invoice_date"], currency="INR", amount="16862.20",
                               reference="fourx:operation-1")
        self.assertIsNone(self.tools.reconcile_post(snapshot, "other-operation"))
        self.assertEqual(self.tools.reconcile_post(snapshot, "operation-1")["entry_id"], entry_id)
        self.assertEqual(len(ledger.entries()), 2)

    def test_post_commit_exception_can_be_reconciled_without_retrying(self):
        snapshot = self.tools.prepare_post("invoice-01")
        real_post = ledger.post
        def commit_then_disconnect(**kwargs):
            real_post(**kwargs)
            raise ConnectionError("Connection lost after ledger committed")
        with patch.object(ledger, "post", commit_then_disconnect):
            with self.assertRaises(ConnectionError):
                self.tools.post_approved(snapshot, "operation-1")
        recovered = self.tools.reconcile_post(snapshot, "operation-1")
        self.assertEqual(recovered["status"], "posted")
        self.assertTrue(recovered["recovered"])
        self.assertEqual(len(ledger.entries()), 2)

    def test_invalid_ids_cannot_read_arbitrary_paths(self):
        for invoice_id in ("../vendors", "/etc/passwd", "invoice-01.json", "missing", [], None):
            with self.subTest(invoice_id=invoice_id), self.assertRaises(ToolError):
                self.tools.read_invoice(invoice_id)

    def test_inconsistent_totals_are_rejected(self):
        self.edit_invoice(lambda inv: inv.update(total="999.00"))
        with self.assertRaisesRegex(ToolError, "do not balance"):
            self.tools.prepare_post("invoice-01")

    def test_nonfinite_or_inexact_money_rejected(self):
        for amount in ("NaN", "Infinity", "0.00", "-1.00", "16862.201", 16862.20):
            with self.subTest(amount=amount):
                path = self.root / "invoices/invoice-01.json"
                shutil.copy(ROOT / "invoices/invoice-01.json", path)
                self.edit_invoice(lambda inv: inv.update(total=amount))
                with self.assertRaises(ToolError):
                    self.tools.prepare_post("invoice-01")

    def test_posting_lock_fails_closed(self):
        snapshot = self.tools.prepare_post("invoice-01")
        with AppLock(self.root / ".state" / "ledger-post.lock"):
            for action in (
                lambda: self.tools.prepare_post("invoice-01"),
                lambda: self.tools.post_approved(snapshot, "operation-1"),
                lambda: self.tools.reconcile_post(snapshot, "operation-1"),
            ):
                with self.assertRaisesRegex(ToolError, "Another post is still finishing"):
                    action()
        self.assertEqual(len(ledger.entries()), 1)

    def test_malformed_json_and_unknown_vendor_rejected(self):
        self.edit_invoice(lambda inv: inv.update(vendor_name="Unregistered vendor"))
        with self.assertRaisesRegex(ToolError, "No unique vendor"):
            self.tools.prepare_post("invoice-01")
        (self.root / "invoices/invoice-01.json").write_text("{")
        with self.assertRaisesRegex(ToolError, "Cannot read invoice"):
            self.tools.read_invoice("invoice-01")


if __name__ == "__main__":
    unittest.main()
