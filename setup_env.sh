#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

echo "venv is ready"
