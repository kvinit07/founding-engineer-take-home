#!/bin/sh
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)" || exit 1
if command -v python3 >/dev/null 2>&1; then
    exec python3 bootstrap.py "$@"
elif command -v python >/dev/null 2>&1; then
    exec python bootstrap.py "$@"
else
    echo "Install Python 3.11 or later, then run this launcher again." >&2
    exit 1
fi
