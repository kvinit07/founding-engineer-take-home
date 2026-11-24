from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from storage import AppLock, Store


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "nested" / "conversations.db"
        self.store = Store(self.path)

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def test_new_conversations_have_independent_defaults(self):
        first = self.store.load("first", "groq")
        self.assertEqual(first, {"provider": "groq", "messages": [], "pending": None, "active": False})
        first["messages"].append({"role": "user", "content": "hello"})
        self.assertEqual(self.store.load("second")["messages"], [])

    def test_round_trip_preserves_pending_and_arbitrary_state(self):
        state = {
            "provider": "groq",
            "messages": [{"role": "user", "content": "₹ 16862.20"}],
            "pending": {"status": "posting", "operation_id": "op-1", "snapshot": {"total": "16862.20"}},
            "active": True,
            "future_field": {"nested": [False, None, 12]},
        }
        self.store.save("demo", state)
        self.store.close()
        self.store = Store(self.path)
        self.assertEqual(self.store.load("demo"), state)

    def test_conversations_are_isolated_and_updates_replace_snapshot(self):
        self.store.save("one", {"revision": 1})
        self.store.save("two", {"revision": 9})
        self.store.save("one", {"revision": 2, "pending": None})
        self.assertEqual(self.store.load("one"), {"revision": 2, "pending": None})
        self.assertEqual(self.store.load("two"), {"revision": 9})

    def test_invalid_json_does_not_replace_saved_state(self):
        self.store.save("demo", {"revision": 1})
        with self.assertRaises(ValueError):
            self.store.save("demo", {"revision": math.nan})
        self.assertEqual(self.store.load("demo"), {"revision": 1})


class AppLockTests(unittest.TestCase):
    def test_rejects_another_process_and_releases_after_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.lock"
            child = (
                "import sys; from pathlib import Path; from storage import AppLock; "
                "lock = AppLock(Path(sys.argv[1])); lock.__enter__()"
            )
            with self.assertRaisesRegex(ValueError, "test exception"):
                with AppLock(path):
                    result = subprocess.run(
                        [sys.executable, "-c", child, str(path)],
                        cwd=Path(__file__).resolve().parents[1],
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Another agent instance", result.stderr)
                    raise ValueError("test exception")
            with AppLock(path):
                pass


if __name__ == "__main__":
    unittest.main()
