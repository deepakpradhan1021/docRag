"""
docRag Application Server Entrypoint
====================================
Runs the docRag FastAPI application server and REST endpoints.
For local development:
    python app.py
or:
    uvicorn server:app --reload --port 8000
"""

import os
import uvicorn
from server import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"\nStarting docRag application server at http://{host}:{port}\n")
    uvicorn.run("server:app", host=host, port=port, reload=True)
