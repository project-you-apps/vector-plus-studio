"""Suite-wide fixtures.

Currently one, and it exists to end a recurring cross-file failure rather than to add
anything.
"""

import asyncio

import pytest


@pytest.fixture(autouse=True)
def preserve_ambient_event_loop():
    """Keep a current event loop installed, because one suite still needs one.

    `test_edit_succession.py` drives its coroutines through the deprecated
    `asyncio.get_event_loop().run_until_complete(...)`, which only works while SOMETHING has
    left a current loop installed. Nothing used to set or clear one, so it worked by
    accident. Any file that calls `asyncio.run()` clears it, and those twelve tests then fail
    with "no current event loop" -- passing alone, failing together, which is the pollution
    shape that cost us 2026-08-10.

    ⚠ THIS IS THE THIRD TIME. test_cart_context.py hit it, test_mount_publishes_a_live_cart.py
    hit it, and each carried its own private copy of this fixture. test_cart_list_access.py
    hit it again on 2026-08-15, which is the point at which a per-file fix is clearly the
    wrong shape: every future file that uses `asyncio.run` would have to know. Hoisted here
    so none of them has to.

    THE REAL FIX IS STILL OPEN: migrate test_edit_succession.py to `asyncio.run` or
    pytest-asyncio on its own terms, then delete this. It is a splint, and it is deliberately
    named like one.
    """
    try:
        previous = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        previous = None
    try:
        yield
    finally:
        if previous is not None and not previous.is_closed():
            asyncio.set_event_loop(previous)
