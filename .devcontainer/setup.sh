#!/usr/bin/env bash
# Codespaces setup: install packages, create .env, build the database from the bundled sample.
# A failure in the sample build is reported but does not fail the codespace.
set -u
echo "== Installing Python packages"
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt || { echo "pip install failed - see above"; exit 1; }
[ -f .env ] || cp .env.example .env
echo "== Building the warehouse from the bundled sample"
if python -m msp.pipeline run --sample; then
  echo "== Ready. Start the dashboard with: python -m streamlit run app/ui.py"
else
  echo "== Sample build failed. Run 'python -m msp.pipeline run --sample' and send the last 20 lines."
fi
exit 0
