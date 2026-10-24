"""CPU reports distinguish POSIX child accounting from portable process timers."""

from types import SimpleNamespace

from example import _cpu


def test_cpu_seconds_adds_process_and_reaped_child_usage(monkeypatch):
    fake_resource = SimpleNamespace(
        RUSAGE_SELF=0,
        RUSAGE_CHILDREN=1,
        getrusage=lambda who: SimpleNamespace(ru_utime=2.0 + who, ru_stime=0.5),
    )
    monkeypatch.setattr(_cpu, "resource", fake_resource)
    assert _cpu.cpu_seconds() == 6.0
    assert "reaped child processes" in _cpu.cpu_configuration()["cpu_accounting"]


def test_cpu_seconds_fallback_explicitly_excludes_children(monkeypatch):
    monkeypatch.setattr(_cpu, "resource", None)
    monkeypatch.setattr(_cpu.time, "process_time", lambda: 3.25)
    assert _cpu.cpu_seconds() == 3.25
    configuration = _cpu.cpu_configuration()
    assert "process threads only" in configuration["cpu_accounting"]
    assert "worker/compiler CPU time is unavailable and is excluded" in configuration["scope"]
