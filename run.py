"""Entry point: start the local server and open the one-page app (Stage 5.1).

Runs the FastAPI backend in ``harmonizer.api`` under uvicorn and opens the browser
at the minimal run UI. Everything runs locally under the user's own Earth Engine
account; there is no login and no shared infrastructure (docs/PIPELINE.md,
section 4 and Stage 5.1).

Run:  python run.py   (optionally HOST/PORT env vars, or --no-browser)
"""

from __future__ import annotations

import os
import sys
import threading
import webbrowser


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    open_browser = "--no-browser" not in sys.argv

    url = f"http://{host}:{port}/"
    if open_browser:
        # Open the page shortly after uvicorn starts serving.
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    import uvicorn

    print(f"Legend Harmonizer serving at {url}  (Ctrl+C to stop)")
    uvicorn.run("harmonizer.api:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
