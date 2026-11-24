"""No credentials or network: exercise real SDK serialization with mock HTTP."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import providers


SCHEMAS = [{
    "name": "read_invoice", "description": "Read an invoice.",
    "parameters": {"type": "object", "properties": {"invoice_id": {"type": "string"}},
                   "required": ["invoice_id"], "additionalProperties": False},
}]


def foreign_history():
    return [
        {"role": "user", "content": "Read invoice-06."},
        {"role": "assistant", "content": "Checking it.", "provider": "groq", "tool_calls": [
            {"id": "read6", "name": "read_invoice", "arguments": {"invoice_id": "invoice-06"}},
        ]},
        {"role": "tool", "tool_call_id": "read6", "name": "read_invoice", "content": {
            "vendor_name": "Shree Ganesh Packaging Pvt Ltd", "total": "21948.00", "lines": [{}],
        }},
        {"role": "assistant", "provider": "groq", "content": "One line, INR 21948.00.", "tool_calls": []},
        {"role": "user", "content": "Who was the vendor?"},
    ]


class HistoryTests(unittest.TestCase):
    def test_foreign_calls_and_results_preserved_without_fake_signatures(self):
        history = foreign_history()
        original = copy.deepcopy(history)
        contents = providers.to_gemini_contents(history, "gemini-3.8-flash")
        self.assertIn('"invoice_id":"invoice-06"', contents[1]["parts"][0]["text"])
        self.assertIn('"total":"21948.00"', contents[2]["parts"][0]["text"])
        self.assertNotIn("thought_signature", json.dumps(contents))
        self.assertNotIn("function_call", contents[1]["parts"][0])
        self.assertEqual(history, original)

    def test_groq_structured_calls_from_either_provider(self):
        history = foreign_history()
        history[1]["provider"] = "gemini"
        history[1]["native"] = {"some": "provider metadata"}
        messages = providers.to_groq_messages(history, "system text")
        self.assertEqual(messages[0], {"role": "system", "content": "system text"})
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["arguments"], '{"invoice_id":"invoice-06"}')
        self.assertEqual(json.loads(messages[3]["content"])["total"], "21948.00")
        self.assertEqual(messages[3]["tool_call_id"], "read6")
        self.assertNotIn("native", json.dumps(messages))

    def test_model_change_imports_signed_calls_as_history(self):
        message = {"role": "assistant", "content": "", "provider": "gemini", "tool_calls": [
            {"id": "a", "name": "read_invoice", "arguments": {"invoice_id": "invoice-01"}},
        ], "native": {"model": "old-model", "content": {"role": "model", "parts": [{
            "function_call": {"name": "read_invoice", "args": {"invoice_id": "invoice-01"}},
            "thought_signature": "c2lnbmF0dXJl",
        }]}}}
        content = providers.to_gemini_contents([message], "other-model")[0]
        self.assertIn("Historical tool requests", content["parts"][0]["text"])
        self.assertNotIn("thought_signature", json.dumps(content))

    def test_missing_key_is_actionable(self):
        with patch.dict(os.environ, {}, clear=True), patch("providers._load_env"):
            with self.assertRaisesRegex(providers.ProviderError, "GEMINI_API_KEY is missing"):
                providers.generate("gemini", "model", [], SCHEMAS, "prompt")

    @unittest.skipUnless(importlib.util.find_spec("dotenv"), "Install requirements.txt for .env loading test")
    def test_worker_env_file_loads_without_overriding_explicit_environment(self):
        # Patch the module location so this test never reads a real user's .env.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("GEMINI_API_KEY=file-test-key\nGROQ_API_KEY=file-groq-key\n", encoding="utf-8")
            with patch.object(providers, "__file__", str(root / "providers.py")):
                with patch.dict(os.environ, {"GEMINI_API_KEY": "explicit-test-key"}, clear=True):
                    providers._load_env()
                    self.assertEqual(os.environ["GEMINI_API_KEY"], "explicit-test-key")
                    self.assertEqual(os.environ["GROQ_API_KEY"], "file-groq-key")
                (root / ".env").write_text("GEMINI_API_KEY=edited-test-key\n", encoding="utf-8")
                with patch.dict(os.environ, {}, clear=True):
                    providers._load_env()
                    self.assertEqual(os.environ["GEMINI_API_KEY"], "edited-test-key")

    def test_transient_requests_retry_but_auth_failure_does_not(self):
        class Problem(Exception):
            def __init__(self, code):
                self.code = code
        attempts = []
        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise Problem(429)
            return "done"
        with patch("providers.time.sleep") as sleep:
            self.assertEqual(providers._request(flaky, "gemini"), "done")
            self.assertEqual(sleep.call_count, 2)
        def unauthorized():
            raise Problem(401)
        with patch("providers.time.sleep") as sleep:
            with self.assertRaisesRegex(providers.ProviderError, "authentication"):
                providers._request(unauthorized, "groq")
            sleep.assert_not_called()

    def test_api_error_does_not_expose_exception_secret(self):
        def unsafe_error():
            raise ValueError("API key sk-secret-really-private")
        with self.assertRaises(providers.ProviderError) as caught:
            providers._request(unsafe_error, "groq")
        self.assertNotIn("sk-secret", str(caught.exception))


try:
    import httpx
    import httpx2
    from google import genai
    from google.genai import types
    import groq
    HAVE_SDKS = True
except ImportError:
    HAVE_SDKS = False


@unittest.skipUnless(HAVE_SDKS, "Install requirements.txt to test real SDK transport serialization")
class SDKTransportTests(unittest.TestCase):
    def test_gemini_native_parallel_calls_keep_signatures_after_switch_round_trip(self):
        captured = []
        signature = "_wBuYXRpdmUtc2lnbmF0dXJl"
        responses = [
            {"candidates": [{"content": {"role": "model", "parts": [
                {"text": "private thought", "thought": True},
                {"functionCall": {"name": "read_invoice", "args": {"invoice_id": "invoice-01"}}, "thoughtSignature": signature},
                {"functionCall": {"name": "read_invoice", "args": {"invoice_id": "invoice-06"}}},
            ]}, "finishReason": "STOP"}]},
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "Two invoices read."}]}, "finishReason": "STOP"}]},
        ]
        def respond(request):
            captured.append(json.loads(request.content))
            return httpx2.Response(200, json=responses.pop(0))
        actual_client = genai.Client
        def client(**kwargs):
            options = kwargs["http_options"].model_copy(update={
                "httpx_client": httpx2.Client(transport=httpx2.MockTransport(respond)),
            })
            return actual_client(api_key=kwargs["api_key"], http_options=options)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch("providers._load_env"), patch.object(genai, "Client", client):
            history = [{"role": "user", "content": "Read invoices 01 and 06."}]
            first = providers.generate("gemini", providers.DEFAULT_MODELS["gemini"], history, SCHEMAS, "system")
            self.assertEqual(len(first["tool_calls"]), 2)
            self.assertNotIn("private thought", first["content"])
            self.assertEqual(len(captured), 1, "SDK must not execute or loop on tool calls")
            history.append(first)
            for call in first["tool_calls"]:
                history.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": {"total": "21948.00"}})
            # A provider switch only serializes history; saved native payload remains.
            groq_history = providers.to_groq_messages(history, "system")
            self.assertEqual(len(groq_history[2]["tool_calls"]), 2)
            history.extend([
                {"role": "assistant", "content": "Groq remembers both invoices.", "provider": "groq", "tool_calls": []},
                {"role": "user", "content": "Continue on Gemini."},
            ])
            final = providers.generate("gemini", providers.DEFAULT_MODELS["gemini"], history, SCHEMAS, "system")
        self.assertEqual(final["content"], "Two invoices read.")
        signed_part = captured[1]["contents"][1]["parts"][1]
        self.assertEqual(signed_part["thoughtSignature"], signature)
        result_parts = captured[1]["contents"][2]["parts"]
        self.assertEqual(len(result_parts), 2)
        self.assertTrue(all("functionResponse" in part for part in result_parts))
        self.assertEqual(captured[0]["tools"][0]["functionDeclarations"][0]["name"], "read_invoice")

    def test_groq_transport_uses_official_sdk_and_preserves_multiple_calls(self):
        captured = []
        def respond(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "completion-1", "object": "chat.completion", "created": 1,
                "model": "openai/gpt-oss-20b", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "tool_calls": [
                        {"id": "0", "type": "function", "function": {"name": "read_invoice", "arguments": '{"invoice_id":"invoice-01"}'}},
                        {"id": "call-b", "type": "function", "function": {"name": "read_invoice", "arguments": "malformed-json"}},
                    ],
                }}],
            })
        actual_client = groq.Groq
        def client(**kwargs):
            return actual_client(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}), patch("providers._load_env"), patch.object(groq, "Groq", client):
            result = providers.generate("groq", providers.DEFAULT_MODELS["groq"], foreign_history(), SCHEMAS, "system")
            history = foreign_history() + [result]
            for call in result["tool_calls"]:
                history.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": {"ok": True}})
            history.append({"role": "user", "content": "Read it again."})
            repeated = providers.generate("groq", providers.DEFAULT_MODELS["groq"], history, SCHEMAS, "system")
        self.assertEqual(len(captured), 2)
        self.assertNotEqual(result["tool_calls"][0]["id"], repeated["tool_calls"][0]["id"])
        repeated_ids = {call["id"] for call in result["tool_calls"] + repeated["tool_calls"]}
        self.assertEqual(len(repeated_ids), 4)
        sent_calls = captured[1]["messages"][6]["tool_calls"]
        self.assertEqual(sent_calls[0]["id"], captured[1]["messages"][7]["tool_call_id"])
        self.assertEqual(sent_calls[1]["id"], captured[1]["messages"][8]["tool_call_id"])
        self.assertEqual(result["tool_calls"][0]["arguments"], {"invoice_id": "invoice-01"})
        self.assertEqual(result["tool_calls"][1]["arguments"], "malformed-json")
        self.assertEqual(captured[0]["messages"][3]["tool_call_id"], "read6")
        self.assertEqual(captured[0]["tools"][0]["function"]["name"], "read_invoice")


if __name__ == "__main__":
    unittest.main()
