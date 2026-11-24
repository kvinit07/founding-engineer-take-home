"""Create the local environment, install dependencies, then start the agent."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    if sys.version_info < (3, 11):
        print("Python 3.11 or later is required. Install it, then run this again.", file=sys.stderr)
        return 1
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    try:
        if not python.exists():
            print("Creating .venv...", flush=True)
            subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        requirements = ROOT / "requirements.txt"
        digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
        stamp = environment / ".requirements.sha256"
        if not stamp.exists() or stamp.read_text(encoding="utf-8").strip() != digest:
            print("Installing project dependencies...", flush=True)
            subprocess.run(
                [str(python), "-m", "pip", "install", "-r", str(requirements)],
                cwd=ROOT,
                check=True,
            )
            stamp.write_text(digest + "\n", encoding="utf-8")
        env_file = ROOT / ".env"
        if not env_file.exists():
            shutil.copyfile(ROOT / ".env.example", env_file)
            print("Created .env. Add your provider API keys to enable live model calls.", flush=True)
        arguments = sys.argv[1:]
        if "--test" in arguments:
            if arguments != ["--test"]:
                print("Use --test on its own; agent options are accepted without --test.", file=sys.stderr)
                return 2
            command = [str(python), "-m", "unittest", "discover", "-s", "tests", "-v"]
            return subprocess.run(command, cwd=ROOT).returncode
        command = [str(python), str(ROOT / "agent.py"), *arguments]
        # Replace the launcher so only the agent receives terminal interrupts.
        os.chdir(ROOT)
        os.execv(str(python), command)
        raise AssertionError("execv returned unexpectedly")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        print("Check Python's venv/pip installation and your internet connection, then retry.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
