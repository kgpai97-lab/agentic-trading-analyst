"""forward_run.py --variant wiring (no LLM, ForwardTest stubbed).

Verifies the CLI layer constructs the right ForwardTest(s):
  * prod (default)  → outputs/forward,       no threshold override (0.5 default);
  * full            → outputs/forward_full,  priority_threshold = 1.0;
  * both            → prod THEN full;
  * a variant failure is isolated (the other variant still runs).
"""

from __future__ import annotations

import forward_run


class _StubFT:
    """Captures construction + run calls instead of doing real work."""

    instances: list["_StubFT"] = []

    def __init__(self, forward_dir=None, config=None, debug=False, seed_hold_built=False):
        self.forward_dir = forward_dir
        self.config = dict(config or {})
        self.debug = debug
        self.seed_hold_built = seed_hold_built
        self.ran = None
        _StubFT.instances.append(self)

    def run_all(self, run_date=None):
        self.ran = ("all", run_date)
        return [{"portfolio": "diversified", "weekday": "monday",
                 "date": run_date or "2026-06-15", "target_weights": {"AAA": 1.0}}]

    def run_one(self, only, run_date=None):
        self.ran = ("one", only, run_date)
        return {"portfolio": only, "weekday": "monday",
                "date": run_date or "2026-06-15", "target_weights": {"AAA": 1.0}}


def _patch(monkeypatch):
    _StubFT.instances = []
    monkeypatch.setattr(forward_run, "ForwardTest", _StubFT)


def test_prod_is_default(monkeypatch):
    _patch(monkeypatch)
    rc = forward_run.main(["--date", "2026-06-15"])
    assert rc == 0
    assert len(_StubFT.instances) == 1
    ft = _StubFT.instances[0]
    assert ft.forward_dir == "outputs/forward"
    assert "priority_threshold" not in ft.config   # 0.5 default, no override
    assert ft.seed_hold_built is False             # hold_built is full-run only


def test_full_variant_sets_threshold_one_and_full_dir(monkeypatch):
    _patch(monkeypatch)
    rc = forward_run.main(["--variant", "full", "--date", "2026-06-15"])
    assert rc == 0
    assert len(_StubFT.instances) == 1
    ft = _StubFT.instances[0]
    assert ft.forward_dir == "outputs/forward_full"
    assert ft.config["priority_threshold"] == 1.0
    assert ft.seed_hold_built is True              # hold_built derives from the 1.0 run


def test_both_runs_prod_then_full(monkeypatch):
    _patch(monkeypatch)
    rc = forward_run.main(["--variant", "both", "--date", "2026-06-15"])
    assert rc == 0
    assert len(_StubFT.instances) == 2
    prod, full = _StubFT.instances
    assert prod.forward_dir == "outputs/forward"
    assert "priority_threshold" not in prod.config
    assert full.forward_dir == "outputs/forward_full"
    assert full.config["priority_threshold"] == 1.0


def test_only_passes_through_to_run_one(monkeypatch):
    _patch(monkeypatch)
    forward_run.main(["--variant", "full", "--only", "concentrated"])
    ft = _StubFT.instances[0]
    assert ft.ran[0] == "one" and ft.ran[1] == "concentrated"


def test_variant_failure_is_isolated(monkeypatch):
    """If prod raises, full must still run (and vice-versa)."""
    _patch(monkeypatch)

    real_init = _StubFT.__init__

    def flaky_init(self, forward_dir=None, config=None, debug=False, seed_hold_built=False):
        real_init(self, forward_dir=forward_dir, config=config, debug=debug,
                  seed_hold_built=seed_hold_built)
        if forward_dir == "outputs/forward":          # prod blows up at construction
            raise RuntimeError("boom in prod")

    monkeypatch.setattr(_StubFT, "__init__", flaky_init)
    rc = forward_run.main(["--variant", "both"])
    assert rc == 1                                     # overall failure flagged
    # both were attempted; the full instance still constructed + ran.
    full = [i for i in _StubFT.instances if i.forward_dir == "outputs/forward_full"]
    assert full and full[0].ran is not None
