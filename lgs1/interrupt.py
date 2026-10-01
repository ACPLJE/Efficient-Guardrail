"""Orderly stop for resumable row writers (S5/S6 eval, S3/S6 longbench) at the Slurm time limit.

The sbatch wrapper (`s1ctl.RUN_FORWARD`, jobs submitted with signal="B:TERM@N", requeue=True) forwards Slurm's
time-limit SIGTERM to the python child.  `Stop.install()` turns it into a flag; the row loop calls `Stop.check()`
after every written row, which flushes the writer (the `with RowsWriter` block exits) and ends the process with
rc=3 -- the wrapper then `scontrol requeue`s the SAME job id and the resumed run continues from the ids already on
disk.  Before 9/25 these jobs simply died at the limit (dense YaRN 36 h, LG 8B e2 24 h, GuardReasoner arrays 48 h)
and had to be resubmitted by hand.
"""
from __future__ import annotations

import signal
import sys
import threading
import time
from typing import Optional

RC_REQUEUE = 3


class Stop:
    requested: bool = False
    signum: Optional[int] = None
    at: Optional[float] = None
    _installed: bool = False

    @classmethod
    def install(cls) -> bool:
        """Install SIGTERM / SIGUSR1 handlers (main thread only; idempotent)."""
        if cls._installed:
            return True
        if threading.current_thread() is not threading.main_thread():
            return False

        def handler(signum, frame):  # noqa: ANN001
            cls.requested = True
            cls.signum = signum
            cls.at = time.time()
            print(f"[stop] signal {signum} received -> finishing the current row, flushing, exit {RC_REQUEUE} (requeue)", flush=True)

        for name in ("SIGTERM", "SIGUSR1"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    return False
        cls._installed = True
        return True

    @classmethod
    def reset(cls) -> None:
        cls.requested, cls.signum, cls.at = False, None, None

    @classmethod
    def check(cls, where: str = "") -> None:
        """Raise SystemExit(3) when a stop was requested (call after each written row, inside the writer's `with`)."""
        if cls.requested:
            print(f"[stop] {where or 'row loop'}: stopping for requeue (rc={RC_REQUEUE})", flush=True)
            sys.stdout.flush()
            raise SystemExit(RC_REQUEUE)
