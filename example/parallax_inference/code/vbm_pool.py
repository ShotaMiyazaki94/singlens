"""Isolate VBM's static scratch state in spawned worker processes."""

import multiprocessing
import os
from importlib import import_module

import numpy as np

from example._cpu import cpu_seconds

_engine = None


def _initialize(tolerance, minannuli, ready):
    global _engine
    try:
        adapter = import_module("example.compare-single-vbml.code.vbm_array")
        _engine = adapter.VBMArray(tolerance=tolerance, minannuli=minannuli)
        ready.put((os.getpid(), cpu_seconds(), _engine.metadata, None))
    except Exception as error:
        ready.put((os.getpid(), cpu_seconds(), None, repr(error)))
        raise


def _evaluate(arguments):
    values = _engine.magnifications(*arguments)
    return values, os.getpid(), cpu_seconds()


class VBMProcessPool:
    """Parallelize batches of walker magnifications, preserving their order."""

    def __init__(self, workers, tolerance, minannuli):
        context = multiprocessing.get_context("spawn")
        ready = context.Queue()
        self._cpu = {}
        self.pool = context.Pool(
            workers, initializer=_initialize, initargs=(tolerance, minannuli, ready)
        )
        try:
            records = [ready.get(timeout=120) for _ in range(workers)]
            if any(record[3] is not None for record in records):
                raise RuntimeError(f"VBM worker setup failed: {[r[3] for r in records]}")
            for pid, seconds, _, _ in records:
                self._cpu[pid] = seconds
            self.workers = workers
            self.metadata = dict(
                workers=workers,
                start_method="spawn",
                interface=records[0][2],
                scope="Independent VBM instances in separate processes; no concurrent calls to its static scratch state. Batches are returned in their input order.",
            )
        except BaseException:
            self.pool.terminate()
            self.pool.join()
            raise
        finally:
            ready.close()
            ready.join_thread()

    @property
    def active_cpu_seconds(self):
        return sum(self._cpu.values())

    def magnifications(self, rho, a1, separation):
        indices = np.array_split(np.arange(len(rho)), min(self.workers, len(rho)))
        results = self.pool.map(
            _evaluate, [(rho[i], a1[i], separation[i]) for i in indices], chunksize=1
        )
        for _, pid, seconds in results:
            self._cpu[pid] = seconds
        return np.concatenate([result[0] for result in results])

    def close(self):
        self.pool.close()
        self.pool.join()
