import time

import pytest
from prometheus_client import CollectorRegistry

from prime_rl.orchestrator.metrics import OrchestratorPrometheusMetrics
from tests.utils import prom_collect_samples, prom_sample_key, prom_sample_value


def test_orchestrator_metrics_update_and_cleanup():
    registry = CollectorRegistry()
    metrics = OrchestratorPrometheusMetrics(registry)

    update_start = time.time()
    metrics.update(
        {
            "step": 7,
            "progress/ckpt_step": 6,
            "progress/total_tokens": 120,
            "progress/total_samples": 12,
            "time/step": 1.5,
            "time/generate_completions": 0.7,
            "time/wait_for_ckpt": 0.2,
            "time/update_weights": 0.1,
            "scheduler/async_level": 2,
            "scheduler/inflight_rollouts": 4,
            "scheduler/inflight_samples": 8,
            "scheduler/cancelled_rollouts": 1,
            "empty_rollouts/all": 0.25,
            "errored_rollouts/all": 0.125,
            "off_policy_level/all/min": 0.5,
            "off_policy_level/all/max": 3.0,
            "off_policy_level/all/mean": 1.5,
            "error/all/mean": 0.05,
            "event_loop_lag/mean": 0.02,
            "event_loop_lag/max": 0.05,
            "worker/worker_a/pending": 3,
            "worker/worker_b/pending": 1,
            "worker_lag/worker_a/mean": 0.1,
            "worker_lag/worker_b/max": 0.2,
            "pool/easy": 0.2,
            "pool/normal": 0.6,
            "pool/hard": 0.2,
            # These stay in monitor/W&B only and should be ignored here.
            "reward/all/mean": 0.4,
            "reward/env_a/mean": 0.5,
            "batch/env_a": 0.55,
            "solve_none/all": 0.2,
            "stop_condition/env_a/max_turns": 0.25,
            "metrics/env_a/accuracy": 0.8,
        }
    )
    update_end = time.time()

    samples = prom_collect_samples(registry)
    assert prom_sample_value(samples, "orchestrator_step") == pytest.approx(7.0)
    assert prom_sample_value(samples, "orchestrator_ckpt_step") == pytest.approx(6.0)
    assert prom_sample_value(samples, "orchestrator_total_tokens") == pytest.approx(120.0)
    assert prom_sample_value(samples, "orchestrator_total_samples") == pytest.approx(12.0)
    assert prom_sample_value(samples, "orchestrator_step_duration_seconds") == pytest.approx(1.5)
    assert prom_sample_value(samples, "orchestrator_generate_completions_duration_seconds") == pytest.approx(0.7)
    assert prom_sample_value(samples, "orchestrator_wait_for_ckpt_duration_seconds") == pytest.approx(0.2)
    assert prom_sample_value(samples, "orchestrator_update_weights_duration_seconds") == pytest.approx(0.1)
    assert prom_sample_value(samples, "orchestrator_async_level") == pytest.approx(2.0)
    assert prom_sample_value(samples, "orchestrator_inflight_rollouts") == pytest.approx(4.0)
    assert prom_sample_value(samples, "orchestrator_inflight_samples") == pytest.approx(8.0)
    assert prom_sample_value(samples, "orchestrator_cancelled_rollouts") == pytest.approx(1.0)
    assert prom_sample_value(samples, "orchestrator_empty_rollout_rate") == pytest.approx(0.25)
    assert prom_sample_value(samples, "orchestrator_errored_rollout_rate") == pytest.approx(0.125)
    assert prom_sample_value(samples, "orchestrator_off_policy_level_min") == pytest.approx(0.5)
    assert prom_sample_value(samples, "orchestrator_off_policy_level_max") == pytest.approx(3.0)
    assert prom_sample_value(samples, "orchestrator_off_policy_level_mean") == pytest.approx(1.5)
    assert prom_sample_value(samples, "orchestrator_error_rate") == pytest.approx(0.05)
    assert prom_sample_value(samples, "orchestrator_event_loop_lag_seconds", stat="mean") == pytest.approx(0.02)
    assert prom_sample_value(samples, "orchestrator_event_loop_lag_seconds", stat="max") == pytest.approx(0.05)
    assert prom_sample_value(samples, "orchestrator_worker_pending_count", worker="worker_a") == pytest.approx(3.0)
    assert prom_sample_value(samples, "orchestrator_worker_pending_count", worker="worker_b") == pytest.approx(1.0)
    assert prom_sample_value(
        samples, "orchestrator_worker_lag_seconds", worker="worker_a", stat="mean"
    ) == pytest.approx(0.1)
    assert prom_sample_value(
        samples, "orchestrator_worker_lag_seconds", worker="worker_b", stat="max"
    ) == pytest.approx(0.2)
    assert prom_sample_value(samples, "orchestrator_pool_ratio", pool="easy") == pytest.approx(0.2)
    assert prom_sample_value(samples, "orchestrator_pool_ratio", pool="normal") == pytest.approx(0.6)
    assert prom_sample_value(samples, "orchestrator_pool_ratio", pool="hard") == pytest.approx(0.2)
    last_step_ts = prom_sample_value(samples, "orchestrator_last_step_timestamp_seconds")
    assert update_start <= last_step_ts <= update_end

    assert prom_sample_key("orchestrator_reward_mean") not in samples
    assert prom_sample_key("orchestrator_env_reward_mean", env="env_a") not in samples
    assert prom_sample_key("orchestrator_env_batch_ratio", env="env_a") not in samples
    assert prom_sample_key("orchestrator_solve_none") not in samples
    assert prom_sample_key("orchestrator_env_stop_condition", env="env_a", stop_condition="max_turns") not in samples
    assert prom_sample_key("orchestrator_env_metric", env="env_a", metric="accuracy") not in samples
    assert prom_sample_key("orchestrator_throughput_tokens_per_sec") not in samples

    metrics.update(
        {
            "step": 8,
            "worker/worker_a/pending": 2,
            "worker_lag/worker_a/mean": 0.05,
        }
    )

    samples = prom_collect_samples(registry)
    assert prom_sample_value(samples, "orchestrator_step") == pytest.approx(8.0)
    assert prom_sample_value(samples, "orchestrator_worker_pending_count", worker="worker_a") == pytest.approx(2.0)
    assert prom_sample_value(
        samples, "orchestrator_worker_lag_seconds", worker="worker_a", stat="mean"
    ) == pytest.approx(0.05)
    assert prom_sample_key("orchestrator_worker_pending_count", worker="worker_b") not in samples
    assert prom_sample_key("orchestrator_worker_lag_seconds", worker="worker_b", stat="max") not in samples
