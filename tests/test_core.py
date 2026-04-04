"""Tests for core types and provenance."""

import json
import tempfile
from pathlib import Path

from earned_understanding_battery.core.types import (
    BatteryResult, DomainConfig, InstrumentResult, SystemClass,
)
from earned_understanding_battery.core.provenance import ProvenanceLog

class TestInstrumentResult:
    def test_basic_creation(self):
        r = InstrumentResult(name="test", passed=True, effect_size=0.5)
        assert r.name == "test"
        assert r.passed is True
        assert r.effect_size == 0.5

    def test_ambiguous_result(self):
        r = InstrumentResult(name="test", passed=None)
        assert r.passed is None

class TestBatteryResult:
    def test_all_pass(self):
        br = BatteryResult(
            system_name="test_system",
            system_class=SystemClass.CLASS_1,
            instrument_results={
                "dev_traj": InstrumentResult("dev_traj", passed=True),
                "integration": InstrumentResult("integration", passed=True),
            },
            provenance_passed=True,
        )
        assert br.compute_overall() is True

    def test_one_fail(self):
        br = BatteryResult(
            system_name="test_system",
            system_class=SystemClass.CLASS_1,
            instrument_results={
                "dev_traj": InstrumentResult("dev_traj", passed=True),
                "integration": InstrumentResult("integration", passed=False),
            },
            provenance_passed=True,
        )
        assert br.compute_overall() is False

    def test_ambiguous(self):
        br = BatteryResult(
            system_name="test_system",
            system_class=SystemClass.CLASS_1,
            instrument_results={
                "dev_traj": InstrumentResult("dev_traj", passed=None),
            },
            provenance_passed=True,
        )
        assert br.compute_overall() is None

    def test_provenance_fail(self):
        br = BatteryResult(
            system_name="test_system",
            system_class=SystemClass.CLASS_1,
            instrument_results={
                "dev_traj": InstrumentResult("dev_traj", passed=True),
            },
            provenance_passed=False,
        )
        assert br.compute_overall() is False

    def test_empty_results(self):
        br = BatteryResult(
            system_name="test", system_class=SystemClass.CLASS_1
        )
        assert br.compute_overall() is None

class TestProvenanceLog:
    def test_log_events(self):
        log = ProvenanceLog()
        log.log_input("hello", step_index=0)
        log.log_state_change(0.0, 0.5, step_index=0)
        log.log_output("world", step_index=0)
        assert log.event_count == 3
        assert log.is_complete()

    def test_incomplete_log(self):
        log = ProvenanceLog()
        log.log_input("hello", step_index=0)
        assert not log.is_complete()

    def test_save_load(self):
        log = ProvenanceLog()
        log.log_input("test", step_index=0)
        log.log_state_change(0.0, 1.0, step_index=0)
        log.log_output("result", step_index=0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        log.save(path)
        loaded = ProvenanceLog.load(path)
        assert loaded.event_count == 3
        assert loaded.is_complete()
        path.unlink()

    def test_measurement_event(self):
        log = ProvenanceLog()
        log.log_measurement("dev_traj", {"passed": True, "effect_size": 0.3})
        assert log.events[0].event_type == "measurement"
        assert log.events[0].data["instrument"] == "dev_traj"

class TestDomainConfig:
    def test_creation(self):
        c = DomainConfig(n_nodes=50, n_communities=4, p_within=0.3, p_between=0.03)
        assert c.n_nodes == 50
        assert c.seed is None

    def test_with_seed(self):
        c = DomainConfig(n_nodes=50, n_communities=4, p_within=0.3, p_between=0.03, seed=42)
        assert c.seed == 42
