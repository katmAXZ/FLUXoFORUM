from __future__ import annotations

import os
import sys
import urllib.request


port = os.getenv("FLUXOFORUM_PORT", "7860")
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)

