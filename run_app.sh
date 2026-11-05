#!/usr/bin/env bash
# Starts the Streamlit app on YOUR computer, for checking your work.
# It installs nothing, and never runs git or publishes anything.
# Run it from a terminal (Windows: Git Bash or WSL):   bash run_app.sh
# Stop it with Ctrl+C.

set -euo pipefail

# Move to this script's own folder so it works from anywhere.
cd "$(dirname "${BASH_SOURCE[0]}")"

# Pick the project's Python: the .venv first, then whatever python is active.
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
elif command -v python >/dev/null 2>&1; then
  PY="python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "Error: no Python found. Create the environment first (see SETUP.md, step 1)." >&2
  exit 1
fi

if [ ! -f "streamlit_app.py" ]; then
  echo "Error: streamlit_app.py not found in $(pwd)." >&2
  exit 1
fi

if ! "$PY" -c "import streamlit" >/dev/null 2>&1; then
  echo "Error: Streamlit is not installed for $PY. Create the environment from requirements.txt (see SETUP.md, step 1)." >&2
  exit 1
fi

if [ ! -f "results/readmission.json" ] || [ ! -f "results/diabetes.json" ]; then
  echo "Error: results/ is missing. Run 'python -m analysis' first." >&2
  exit 1
fi

echo "Starting the app. Open the address Streamlit prints below (usually http://localhost:8501)."
echo "Press Ctrl+C to stop."
exec "$PY" -m streamlit run streamlit_app.py
