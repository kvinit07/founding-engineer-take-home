# FourX invoice agent

A small Python CLI that lets an accounts-payable clerk inspect invoices and approve ledger posts. The original task is preserved in `ASSIGNMENT.md`; `ledger.py` is unchanged.

## Run

Install **Python 3.11+**. From this folder, run:

```sh
python bootstrap.py --conversation demo
```

Use `python3` if that is your Python command. The bootstrap creates `.venv`, installs the pinned provider SDKs and dotenv, creates `.env` from `.env.example` when needed, and starts the CLI. Subsequent launches reuse installed dependencies unless `requirements.txt` changes. Internet access is needed for installation and provider calls. You can also use `./start.sh` on macOS/Linux, double-click `start.command` on macOS, or double-click `start.bat` on Windows. Keep the folder together when moving it; recreate `.venv` there.

Add your own **`GEMINI_API_KEY` and `GROQ_API_KEY`** to `.env` to use both providers. They are never included in this submission. Defaults are Gemini `gemini-3.8-flash` through `google-genai` and Groq `openai/gpt-oss-20b` through `groq`. Override `GEMINI_MODEL` / `GROQ_MODEL` in `.env`, or pass `--provider gemini|groq --model MODEL_ID`. The chosen model must support tool calling and be available to your account.

Once set up, `.venv/bin/python agent.py --conversation demo` starts directly (Windows: `.venv\Scripts\python.exe agent.py --conversation demo`).

## Use

Type requests such as `post invoice-01` or `which invoices are from Bluefin?`.

- `/approve` posts the displayed invoice; `/decline PO not received yet` returns the refusal to the model.
- `/model gemini` or `/model groq` changes provider while keeping the conversation.
- `/history`, `/help`, and `/quit` inspect history, show commands, and exit.
- Press **Ctrl-C while working** to stop; use `/resume` to continue an interrupted turn, or send a new message.

Restart with the same `--conversation` name to resume, including a waiting approval. `--data-dir PATH` changes conversation storage only; invoices, vendors, and the ledger remain beside the code.

## State and safety

The harness saves provider-independent messages, tool calls/results, selected provider, work status, and pending approval snapshots in `.state/conversations.sqlite3`. Each snapshot is a SQLite transaction using WAL and FULL synchronization. The loop saves between operations and before executing an approved post. Adapters translate history into each official SDK's format. Gemini's native responses retain signatures; historical calls from another provider are preserved as explicit JSON transcripts when importing to Gemini. No agent framework is used.

Posting is a code-controlled tool. Vendor blocking and duplicates `(vendor_id, invoice_number)` are checked before approval and again before execution. The approval binds the vendor, invoice, date, currency, and total shown to the clerk. The original ledger stores entries in `ledger.db`.

The ledger can commit before returning its entry ID. Recovery queries the ledger before retrying an uncertain post, preventing a blind second post. A global OS file lock permits one instance per project folder, including when different conversation storage directories are selected. The OS releases the lock after a crash. Interrupting an in-progress post cannot undo a committed ledger entry; the harness reconciles its outcome.

## Verification and limits

```sh
python bootstrap.py --test
# Once dependencies are installed:
python -m unittest discover -s tests -v
```

Tests run without API keys and use temporary state/ledgers. They cover local safety, persistence, provider conversion, and harness behavior. Local checks used macOS/Python 3.12; the included CI matrix targets macOS, Linux, and Windows with Python 3.11/3.13. Live provider calls were not verified without supplied keys; run the seven scenarios in `ASSIGNMENT.md` before submitting. `python ledger.py` displays project entries. `python ledger.py reset` clears them back to the opening entry, so use it deliberately with a fresh test conversation.

This is a single-user terminal tool. Network outages, rate limits, and model quality still affect responses. Next work would prioritize live acceptance runs, broader crash testing, and dependency updates. Never commit `.env`, `.venv`, `.state`, or ledger databases.
