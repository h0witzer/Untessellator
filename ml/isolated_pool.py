#!/usr/bin/env python3
"""
isolated_pool.py - run CAD-kernel work in worker processes that are allowed to die.

OCCT and gmsh fail in ways Python cannot catch. Measured on ABC so far:
BRepBuilderAPI_NurbsConvert segfaults (access violation) on some shapes, booleans and
fillets can loop forever, and gmsh can stall inside a surface mesher. A plain
ProcessPoolExecutor cannot cancel a running call, and one segfault breaks the whole pool.

run_isolated() keeps at most `jobs` tasks in flight, so a task's submit time is close to
its start time. When a task overruns `timeout`, the pool is killed and rebuilt, and that
task is reported as a timeout.

When a worker dies, the whole pool breaks, and the pool cannot say which in-flight task
killed it. Retrying every in-flight task a few times and blaming whichever broke the
pool last misattributes crashes. With 8 jobs, the first NURBS validation bake reported
180 "crashes", mostly innocent parts that happened to be running beside a real one. So
every task in flight at a break becomes a SUSPECT, and suspects are re-run one at a time
in a single-worker pool. A crash there is unambiguous; everything else completes
normally.

The worker function must be defined at module level, because Windows spawns workers
rather than forking them. Its tasks must be picklable, and it returns a picklable result.
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures import TimeoutError as FutureTimeout   # not the builtin before 3.11
from concurrent.futures.process import BrokenProcessPool


def _kill(pool):
    for proc in list((getattr(pool, "_processes", None) or {}).values()):
        try:
            proc.kill()
        except Exception:
            pass


def _run_alone(fn, task, timeout, on_fail):
    """One task, one worker: a crash or overrun here is attributable to this task."""
    pool = ProcessPoolExecutor(max_workers=1)
    fut = pool.submit(fn, task)
    try:
        return fut.result(timeout=timeout)
    except BrokenProcessPool:
        return on_fail(task, "crash")
    except FutureTimeout:
        _kill(pool)
        return on_fail(task, "timeout")
    except Exception as exc:
        return on_fail(task, f"error:{type(exc).__name__}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def run_isolated(fn, tasks, jobs: int, timeout: float, on_fail):
    """Yield fn(task) for each task, or on_fail(task, status) where status is
    "timeout", "crash" or "error:<ExceptionName>". Results come in completion order."""
    pending = deque(tasks)
    suspects = deque()
    while pending or suspects:
        while suspects:
            yield _run_alone(fn, suspects.popleft(), timeout, on_fail)
        if not pending:
            break
        pool = ProcessPoolExecutor(max_workers=jobs)
        live, broken = {}, False
        try:
            while (pending or live) and not broken:
                while pending and len(live) < jobs:
                    task = pending.popleft()
                    live[pool.submit(fn, task)] = (task, time.time())
                done, _ = wait(live, timeout=2.0, return_when=FIRST_COMPLETED)
                for fut in done:
                    task, _ = live.pop(fut)
                    try:
                        yield fut.result()
                    except BrokenProcessPool:
                        broken = True
                        suspects.append(task)
                    except Exception as exc:
                        yield on_fail(task, f"error:{type(exc).__name__}")
                now = time.time()
                for fut in [f for f, (_, t0) in live.items() if now - t0 > timeout]:
                    task, _ = live.pop(fut)
                    yield on_fail(task, "timeout")
                    broken = True
            if broken:
                # everything still in flight: a crash may have been any of them; an
                # overrun's neighbours were merely interrupted, so they just go back
                suspects.extend(task for task, _ in live.values())
        finally:
            if broken:
                _kill(pool)
            pool.shutdown(wait=not broken, cancel_futures=True)
