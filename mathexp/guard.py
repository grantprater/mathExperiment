"""Compute guard: idle priority and single-threaded numeric libraries.

Called at CLI start. The machine's CPU is shared with the owner's long jobs, so
everything here runs single-process, sequential, at the lowest priority.
"""

from __future__ import annotations

import os
import sys

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
IDLE_PRIORITY_CLASS = 0x40


def apply() -> str:
    for v in THREAD_VARS:
        os.environ[v] = "1"
    try:
        if sys.platform == "win32":
            import ctypes

            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            ok = k32.SetPriorityClass(ctypes.c_void_p(k32.GetCurrentProcess()), IDLE_PRIORITY_CLASS)
            return "idle" if ok else "priority-unchanged"
        os.nice(19)
        return "nice19"
    except Exception as e:  # never fail the run over priority
        return f"priority-error: {e}"
