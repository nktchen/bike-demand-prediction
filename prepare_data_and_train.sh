#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

.venv/bin/python -m src.train
.venv/bin/python -m src.make_demo_data

echo "done"
echo "now run: .venv/bin/uvicorn src.app:app"
