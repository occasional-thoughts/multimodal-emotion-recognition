#!/bin/bash
# Creates the Python environment. Needs Python 3.11 (TF 2.15 does not support 3.13).
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.11 || echo /opt/homebrew/bin/python3.11)
"$PY" -m venv env
./env/bin/pip install --upgrade pip
./env/bin/pip install -r requirements.txt
echo "Done. Use ./env/bin/python to run everything."
