"""Launch the VPS backend with an event loop that survives a client hanging up.

WHY THIS FILE EXISTS (2026-08-26). The dev server was dying outright, mid-session, and
every request after it returned ECONNREFUSED:

    OSError: [WinError 64] The specified network name is no longer available
      IocpProactor.accept -> accept_coro -> finish_accept -> ov.getresult()

A client that vanishes between connect and accept raises inside the proactor's accept
coroutine, and that kills the LISTENER, not just the one connection. The server process
stays alive with nothing listening, which is the confusing part: no traceback on the next
request, just a dead port.

Windows asyncio defaults to ProactorEventLoop, and uvicorn has no flag to change it --
`--loop asyncio` still gets the platform default. The policy has to be set in-process
BEFORE uvicorn builds its loop, which is the only thing this file does.

WHY IT SURFACED WHEN IT DID, since the bug is as old as the server: request DURATION
changed. Associate actually runs now instead of silently falling back to Hamming (a
2026-05-30 regression fixed the same day this landed), a 25-step basin walk is 25
sequential slow searches, and a 2-second status poll runs underneath all of it. Long
requests plus an impatient browser is what manufactures abandoned connections. The fix is
here rather than in the search path because the accept race is a bug either way -- making
requests fast again would only hide it.

SELECTOR LOOP CAVEATS, stated so nobody rediscovers them as mysteries: a 512-socket
select() ceiling, and no asyncio subprocess support. Neither matters for this server.
Linux is unaffected by all of the above -- the droplet runs uvicorn directly and should
keep doing so.

Usage (all optional):
    python run_api.py [--host H] [--port N] [--reload]
Defaults match startup.bat: 127.0.0.1:8000, no reload.
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the VPS backend.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    if sys.platform == "win32":
        # ⚠ BEFORE uvicorn.run. Setting it after has no effect: the loop is already built.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        print("[VPS] event loop: WindowsSelector (proactor accept race avoided)")

    import uvicorn
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
