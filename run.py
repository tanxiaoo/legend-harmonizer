"""Entry point: start the local server and open the one-page app (Stage 5.1).

Runs the FastAPI backend in ``harmonizer.api`` under uvicorn and opens the browser
at the minimal run UI. Everything runs locally under the user's own Earth Engine
account; there is no login and no shared infrastructure (docs/PIPELINE.md,
section 4 and Stage 5.1).

Run:  python run.py   (optionally HOST/PORT env vars, or --no-browser)
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import webbrowser


def _check_interpreter() -> None:
    """Refuse to start on an interpreter that cannot actually serve.

    Running ``python run.py`` outside the project virtualenv picks up whatever
    ``python`` is first on PATH. That interpreter usually lacks (or has an
    incompatible) ``rio_tiler``, so the server starts, binds the port, and then
    fails *every* tile request -- which looks like a broken app rather than a
    wrong interpreter. Checked up front, with the fix in the message.

    ``harmonizer`` must be imported FIRST: its package ``__init__`` repairs a
    broken inherited ``PROJ_LIB`` (a machine-wide PostGIS/QGIS install shadowing
    rasterio's own proj.db), and ``rio_tiler`` builds ``CRS.from_epsg(3857)`` at
    module scope. Importing rio_tiler first makes even the correct interpreter
    fail this check -- which it did on the first version of this guard.
    """
    try:
        import harmonizer  # noqa: F401  (repairs PROJ before rasterio loads)
        import rio_tiler  # noqa: F401
    except Exception as exc:
        exe = sys.executable
        sys.exit(
            f"\nThis Python cannot run the app:\n"
            f"  {exe}\n"
            f"  importing rio_tiler failed: {type(exc).__name__}: {exc}\n\n"
            f"Start it with the project virtualenv instead:\n"
            f"  .venv\\Scripts\\python.exe run.py      (Windows)\n"
            f"  .venv/bin/python run.py               (macOS/Linux)\n"
        )


def _check_port_free(host: str, port: int) -> None:
    """Fail early, and usefully, when something already holds the port.

    uvicorn's own message for this is a bare WinError 10048 buried between
    "Application startup complete" and "shutdown complete", which reads as a
    crash rather than "another copy is already running". A second server is not
    harmless either: two of them sampling the same product write the same cache
    file concurrently and can corrupt it (see ``sampling.save_map_sample``).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
            return
        except OSError:
            pass
    sys.exit(
        f"\nPort {port} is already in use on {host} -- another copy of this app "
        f"is probably still running.\n\n"
        f"Find and stop it:\n"
        f"  Windows:  Get-NetTCPConnection -LocalPort {port} -State Listen | "
        f"ForEach-Object {{ Get-Process -Id $_.OwningProcess }}\n"
        f"            Stop-Process -Id <pid> -Force\n"
        f"  macOS/Linux:  lsof -i :{port}    then  kill <pid>\n\n"
        f"Or run this one on a different port:  PORT={port + 1} python run.py\n"
    )


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    open_browser = "--no-browser" not in sys.argv

    _check_interpreter()
    _check_port_free(host, port)

    url = f"http://{host}:{port}/"
    if open_browser:
        # Open the page shortly after uvicorn starts serving.
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    import uvicorn

    print(f"Legend Harmonizer serving at {url}  (Ctrl+C to stop)")
    uvicorn.run("harmonizer.api:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
