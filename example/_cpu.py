"""CPU accounting for benchmark processes; no scheduling changes on import."""

import os
import time

try:
    import resource
except ImportError:  # Windows has process CPU timers but no POSIX child accounting.
    resource = None


def cpu_seconds():
    """CPU time of process threads and, where supported, reaped children."""
    if resource is None:
        return time.process_time()
    return sum(
        value.ru_utime + value.ru_stime
        for value in (
            resource.getrusage(resource.RUSAGE_SELF),
            resource.getrusage(resource.RUSAGE_CHILDREN),
        )
    )


def cpu_configuration():
    return dict(
        cpu_accounting=(
            "process threads and reaped child processes"
            if resource is not None
            else "process threads only; child CPU time unavailable on this platform"
        ),
        logical_cpus=os.cpu_count(),
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        environment={
            name: os.environ.get(name)
            for name in (
                "JAX_NUM_CPU_DEVICES",
                "JAX_CPU_ENABLE_ASYNC_DISPATCH",
                "XLA_FLAGS",
                "NPROC",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        scope=(
            "One method at a time on the same CPU host. No CPU affinity is imposed. "
            + (
                "CPU seconds sum user and system time over process threads and completed "
                "worker/compiler processes. "
                if resource is not None
                else "CPU seconds cover process threads only; worker/compiler CPU time is "
                "unavailable and is excluded. "
            )
            + "CPU seconds / wall seconds is average CPU utilization, not a core-count limit."
        ),
    )
