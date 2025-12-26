#!/usr/bin/env bash
# Helper script to launch Streamlit web interfaces from the project root.
# Usage:
#   ./run_web.sh             # launches the unified app.py
#   ./run_web.sh Lab1/app.py # launches Lab1 Streamlit app
#   ./run_web.sh Lab2/app.py # launches Lab2 Streamlit app

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_APP="${1:-app.py}"
APP_PATH="${PROJECT_ROOT}/${TARGET_APP}"
STREAMLIT_PORT="${STREAMLIT_SERVER_PORT:-8501}"

if [[ ! -f "$APP_PATH" ]]; then
    echo "Error: ${TARGET_APP} not found in project root." >&2
    exit 1
fi

if [[ -x "${PROJECT_ROOT}/venv/bin/streamlit" ]]; then
    STREAMLIT_BIN="${PROJECT_ROOT}/venv/bin/streamlit"
else
    if ! command -v streamlit >/dev/null 2>&1; then
        echo "Error: streamlit executable not found. Activate venv or install dependencies." >&2
        exit 1
    fi
    STREAMLIT_BIN="streamlit"
fi

if command -v lsof >/dev/null 2>&1; then
    if lsof -i TCP:"${STREAMLIT_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Streamlit (or another app) already listens on port ${STREAMLIT_PORT}. Stop it before launching a new instance." >&2
        echo "Hint: export STREAMLIT_SERVER_PORT=<free_port> to override the port." >&2
        exit 1
    fi
fi

cd "$PROJECT_ROOT"
exec "$STREAMLIT_BIN" run "$APP_PATH"
