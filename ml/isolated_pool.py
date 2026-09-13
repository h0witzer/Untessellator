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
killed it. So every task in flight at a break becomes a SUSPECT, and each suspect is
re-run ALONE in its own one-worker pool, where a crash is unambiguous.

Two earlier versions got this wrong in opposite directions:
- Retrying in-flight tasks and blaming whichever one broke the pool last misattributed
  crashes. The first NURBS validation bake reported 180, mostly innocent parts that
  happened to be running beside a real one; with exact attribution it was 25.
- Re-running suspects one at a time was exact but throttled. The gmsh training bake,
  where crashes are frequent, fell from about 260 parts per 10 minutes to 1, because
  every break queued up to eight parts behind a single slow worker.

Suspects now run concurrently, up to `jobs` at once, each in its own pool, which keeps
the attribution exact and the throughput intact.

The worker function must be defined at module level, because Windows spawns workers
rather than forking them. Its tasks must be picklable, and it returns a picklable result.
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool


def _kill(pool):
    for proc in list((getattr(pool, "_processes", None) or {}).values()):
        try:
            proc.kill()
        except Exception:
            pass


def _run_suspects(fn, suspects, jobs, timeout, on_fail):
    """Each suspect in its own one-worker pool, up to `jobs` concurrently. A pool that
    breaks here held exactly one task, so the crash belongs to that task."""
    live = {}                                       # future -> (task, pool, start time)
    try:
        while suspects or live:
            while suspects and len(live) < jobs:
                task = suspects.popleft()
                pool = ProcessPoolExecutor(max_workers=1)
                live[pool.submit(fn, task)] = (task, pool, time.time())
            done, _ = wait(live, timeout=2.0, return_when=FIRST_COMPLETED)
            for fut in done:
                task, pool, _ = live.pop(fut)
                try:
                    yield fut.result()
                except BrokenProcessPool:
                    yield on_fail(task, "crash")
                except Exception as exc:
                    yield on_fail(task, f"error:{type(exc).__name__}")
                finally:
                    pool.shutdown(wait=False, cancel_futures=True)
            now = time.time()
            for fut in [f for f, (_, _, t0) in live.items() if now - t0 > timeout]:
                task, pool, _ = live.pop(fut)
                _kill(pool)
                pool.shutdown(wait=False, cancel_futures=True)
                yield on_fail(task, "timeout")
    finally:
        for _, pool, _ in live.values():
            _kill(pool)
            pool.shutdown(wait=False, cancel_futures=True)


def run_isolated(fn, tasks, jobs: int, timeout: float, on_fail):
    """Yield fn(task) for each task, or on_fail(task, status) where status is
    "timeout", "crash" or "error:<ExceptionName>". Results come in completion order."""
    pending = deque(tasks)
    suspects = deque()
    while pending or suspects:
        if suspects:
            yield from _run_suspects(fn, suspects, jobs, timeout, on_fail)
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
                # Everything still in flight: a crash may have been any of them, and an
                # overrun's neighbours were merely interrupted. All are re-run as suspects.
                suspects.extend(task for task, _ in live.values())
        finally:
            if broken:
                _kill(pool)
            pool.shutdown(wait=not broken, cancel_futures=True)
