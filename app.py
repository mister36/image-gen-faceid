"""
Entry point for uvicorn server.
This module imports the FastAPI app from main.py to enable running with:
uvicorn app:app --host 0.0.0.0 --port 8000
"""

from main import app

# The app variable is imported from main.py and will be used by uvicorn
__all__ = ["app"]
