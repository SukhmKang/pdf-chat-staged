#!/bin/bash
set -e  # stop on first error

# Create venv only if it doesn't already exist
if [ ! -d "venv39" ]; then
  echo "Creating virtual environment..."
  python3.9 -m venv venv39
fi

# Activate the venv
source venv39/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# process PDFs
python3 pdf_pipeline_orchestrator.py --batch-process --hard-reset

# initialize web_app server
python3 web_app.py

# start the frontend server
uvicorn web_app:app