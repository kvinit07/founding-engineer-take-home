"""Small, deterministic invoice tools. Models cannot bypass posting policy here."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import date
from decimal import Decimal, DecimalException, InvalidOperation
from pathlib import Path
from typing import Any

from storage import AppLock


class ToolError(Exception):
    """An expected, safe-to-display tool failure."""


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ToolError(f"Cannot read {label}: {exc}") from exc


def _number(value: Any, label: str, *, positive: bool = False) -> Decimal:
    try:
        if isinstance(value, bool):
            raise ValueError
        number = Decimal(str(value))
        if not number.is_finite() or (number <= 0 if positive else number < 0):
            raise ValueError
        return number
    except (InvalidOperation, ValueError, TypeError):
        qualifier = "positive" if positive else "non-negative"
        raise ToolError(f"{label} must be a finite {qualifier} number.") from None


def _money(value: Any, label: str, *, positive: bool = False) -> Decimal:
    if not isinstance(value, str):
        raise ToolError(f"{label} must be an exact money string.")
    number = _number(value, label, positive=positive)
    try:
        if number != number.quantize(Decimal("0.01")):
            raise ToolError(f"{label} must have at most two decimal places.")
    except InvalidOperation:
        raise ToolError(f"{label} is outside the supported money range.") from None
    return number


def _text(obj: dict, key: str, label: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{label} requires a non-empty {key}.")
    return value


class InvoiceTools:
    def __init__(self, root: Path, ledger_api=None):
        self.root = Path(root).resolve()
        if ledger_api is None:
            import ledger
            ledger_api = ledger
        self.ledger = ledger_api

    def _invoice_path(self, invoice_id: str) -> Path:
        folder = (self.root / "invoices").resolve()
        # Membership in the actual fixture set also rejects ../, absolute paths and suffixes.
        paths = {p.stem: p for p in folder.glob("*.json") if p.resolve().parent == folder}
        if not isinstance(invoice_id, str) or invoice_id not in paths:
            raise ToolError("Unknown invoice ID. Use list_invoices to see available IDs.")
        return paths[invoice_id]

    def read_invoice(self, invoice_id: str) -> dict:
        invoice = _read_json(self._invoice_path(invoice_id), invoice_id)
        if not isinstance(invoice, dict):
            raise ToolError(f"{invoice_id} must contain a JSON object.")
        for key in ("invoice_number", "invoice_date", "due_date", "vendor_name", "currency"):
            _text(invoice, key, invoice_id)
        try:
            date.fromisoformat(invoice["invoice_date"])
            date.fromisoformat(invoice["due_date"])
        except ValueError:
            raise ToolError(f"{invoice_id} dates must use YYYY-MM-DD.") from None
        if len(invoice["currency"]) != 3 or not invoice["currency"].isalpha() or not invoice["currency"].isupper():
            raise ToolError(f"{invoice_id} currency must be a three-letter uppercase code.")
        lines, taxes = invoice.get("lines"), invoice.get("taxes")
        if not isinstance(lines, list) or not lines or not isinstance(taxes, list):
            raise ToolError(f"{invoice_id} requires non-empty lines and a taxes list.")
        line_total = Decimal(0)
        for index, line in enumerate(lines, 1):
            label = f"{invoice_id} line {index}"
            if not isinstance(line, dict):
                raise ToolError(f"{label} must be an object.")
            _text(line, "description", label)
            quantity = _number(line.get("quantity"), f"{label} quantity", positive=True)
            price = _money(line.get("unit_price"), f"{label} unit_price")
            amount = _money(line.get("amount"), f"{label} amount")
            try:
                expected = (quantity * price).quantize(Decimal("0.01"))
            except DecimalException:
                raise ToolError(f"{label} amount is outside the supported money range.") from None
            if expected != amount:
                raise ToolError(f"{label} amount does not match quantity × unit_price.")
            line_total += amount
        subtotal = _money(invoice.get("subtotal"), f"{invoice_id} subtotal")
        tax_total = Decimal(0)
        for index, tax in enumerate(taxes, 1):
            label = f"{invoice_id} tax {index}"
            if not isinstance(tax, dict):
                raise ToolError(f"{label} must be an object.")
            _text(tax, "type", label)
            _number(tax.get("rate_percent"), f"{label} rate_percent")
            tax_total += _money(tax.get("amount"), f"{label} amount")
        total = _money(invoice.get("total"), f"{invoice_id} total", positive=True)
        if line_total != subtotal or subtotal + tax_total != total:
            raise ToolError(f"{invoice_id} line amounts, subtotal, taxes and total do not balance.")
        return {**invoice, "invoice_id": invoice_id}

    def list_invoices(self) -> dict:
        folder = (self.root / "invoices").resolve()
        invoices = []
        for path in sorted(folder.glob("*.json")):
            if path.resolve().parent != folder:
                continue
            inv = self.read_invoice(path.stem)
            invoices.append({key: inv[key] for key in (
                "invoice_id", "vendor_name", "invoice_number", "invoice_date", "currency", "total"
            )})
        return {"invoices": invoices}

    def _vendors(self) -> list[dict]:
        vendors = _read_json(self.root / "vendors.json", "vendor list")
        if not isinstance(vendors, list):
            raise ToolError("Vendor list must be a JSON list.")
        ids, names = set(), set()
        for vendor in vendors:
            if not isinstance(vendor, dict):
                raise ToolError("Every vendor must be an object.")
            vendor_id = _text(vendor, "vendor_id", "Vendor")
            name = _text(vendor, "name", "Vendor")
            if vendor.get("status") not in ("active", "blocked"):
                raise ToolError(f"Vendor {vendor_id} has an invalid status.")
            if vendor_id in ids or name in names:
                raise ToolError("Vendor IDs and names must be unique.")
            ids.add(vendor_id)
            names.add(name)
        return vendors

    def lookup_vendor(self, query: str) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ToolError("Vendor query must be a non-empty name or vendor ID.")
        search = query.strip().casefold()
        return {"vendors": [v for v in self._vendors()
                            if search in v["name"].casefold() or search == v["vendor_id"].casefold()]}

    def _snapshot(self, invoice_id: str) -> dict:
        invoice = self.read_invoice(invoice_id)
        vendors = [v for v in self._vendors() if v["name"] == invoice["vendor_name"]]
        if len(vendors) != 1:
            raise ToolError(f"No unique vendor matches {invoice['vendor_name']}.")
        vendor = vendors[0]
        if vendor["status"] != "active":
            reason = vendor.get("blocked_reason", "No reason supplied")
            raise ToolError(f"Vendor {vendor['vendor_id']} ({vendor['name']}) is blocked: {reason}.")
        digest = hashlib.sha256(json.dumps(invoice, sort_keys=True, ensure_ascii=False,
                                           separators=(",", ":")).encode("utf-8")).hexdigest()
        return {"invoice_id": invoice_id, "vendor_id": vendor["vendor_id"],
                "vendor_name": vendor["name"], "vendor_status": vendor["status"],
                "invoice_number": invoice["invoice_number"], "invoice_date": invoice["invoice_date"],
                "currency": invoice["currency"], "amount": invoice["total"], "digest": digest}

    def _existing(self, snapshot: dict) -> list[dict]:
        return self.ledger.entries(vendor_id=snapshot["vendor_id"],
                                   invoice_number=snapshot["invoice_number"])

    @staticmethod
    def _duplicate(entries: list[dict]) -> None:
        if entries:
            raise ToolError(f"Invoice is already in the ledger as {entries[0]['entry_id']}; nothing posted.")

    @contextmanager
    def _posting_lock(self):
        # Survives an orphaned worker and releases automatically when that worker exits.
        lock = AppLock(self.root / ".state" / "ledger-post.lock")
        try:
            lock.__enter__()
        except RuntimeError as exc:
            raise ToolError("Another post is still finishing; try again shortly.") from exc
        try:
            yield
        finally:
            lock.__exit__(None, None, None)

    def prepare_post(self, invoice_id: str) -> dict:
        with self._posting_lock():
            snapshot = self._snapshot(invoice_id)
            self._duplicate(self._existing(snapshot))
            return snapshot

    @staticmethod
    def _reference(operation_id: str) -> str:
        if not isinstance(operation_id, str) or not operation_id.strip() or len(operation_id) > 200:
            raise ToolError("Invalid posting operation ID.")
        return "fourx:" + operation_id

    @staticmethod
    def _result(snapshot: dict, entry_id: str, *, recovered: bool) -> dict:
        return {"status": "posted", "entry_id": entry_id, "invoice_id": snapshot["invoice_id"],
                "vendor_id": snapshot["vendor_id"], "invoice_number": snapshot["invoice_number"],
                "currency": snapshot["currency"], "amount": snapshot["amount"], "recovered": recovered}

    def reconcile_post(self, snapshot: dict, operation_id: str) -> dict | None:
        with self._posting_lock():
            return self._reconcile_post(snapshot, operation_id)

    def _reconcile_post(self, snapshot: dict, operation_id: str) -> dict | None:
        reference = self._reference(operation_id)
        matches = [e for e in self._existing(snapshot) if e.get("reference") == reference]
        if not matches:
            return None
        if len(matches) != 1:
            raise ToolError("Multiple ledger entries share this operation; manual review is required.")
        entry = matches[0]
        if any(entry[key] != snapshot[key] for key in ("invoice_date", "currency")) or (
            _number(entry["amount"], "Ledger amount") != _number(snapshot["amount"], "Approved amount")
        ):
            raise ToolError("Ledger entry differs from the approved details; manual review is required.")
        return self._result(snapshot, entry["entry_id"], recovered=True)

    def post_approved(self, snapshot: dict, operation_id: str) -> dict:
        with self._posting_lock():
            return self._post_approved(snapshot, operation_id)

    def _post_approved(self, snapshot: dict, operation_id: str) -> dict:
        # Replaying the same approved operation is safe after a lost return value.
        recovered = self._reconcile_post(snapshot, operation_id)
        if recovered is not None:
            return recovered
        current = self._snapshot(snapshot["invoice_id"])
        if current != snapshot:
            raise ToolError("Invoice or vendor details changed since approval. Request a new approval.")
        self._duplicate(self._existing(snapshot))
        entry_id = self.ledger.post(vendor_id=snapshot["vendor_id"],
                                    invoice_number=snapshot["invoice_number"],
                                    invoice_date=snapshot["invoice_date"], currency=snapshot["currency"],
                                    amount=snapshot["amount"], reference=self._reference(operation_id))
        return self._result(snapshot, entry_id, recovered=False)
