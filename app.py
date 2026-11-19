"""
Streamlit Cloud Entrypoint for MedStruct.
Redirects execution to app/app.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.app
