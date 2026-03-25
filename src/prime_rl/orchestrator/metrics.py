"""Prometheus metric definitions for the orchestrator."""

import time

from prometheus_client import CollectorRegistry, Gauge


class OrchestratorPrometheusMetrics:
    """Container for orchestrator Prometheus metrics."""

    def __init__(self, registry: CollectorRegistry):
        self.step = Gauge("orchestrator_step", "Current orchestrator step", registry=registry)
        self.ckpt_step = Gauge("orchestrator_ckpt_step", "Current checkpoint step from trainer", registry=registry)
        self.total_tokens = Gauge("orchestrator_total_tokens", "Total tokens processed", registry=registry)
        self.total_samples = Gauge("orchestrator_total_samples", "Total samples processed", registry=registry)
        self.last_step_ts = Gauge(
            "orchestrator_last_step_timestamp_seconds", "Unix timestamp of last step", registry=registry
        )

        self.step_duration = Gauge("orchestrator_step_duration_seconds", "Step duration", registry=registry)
        self.generate_completions_duration = Gauge(
            "orchestrator_generate_completions_duration_seconds", "Generation duration", registry=registry
        )
        self.generation_duration = Gauge(
            "orchestrator_generation_duration_seconds",
            "Per-rollout generation duration",
            ["stat"],
            registry=registry,
        )
        self.wait_for_ckpt_duration = Gauge(
            "orchestrator_wait_for_ckpt_duration_seconds", "Checkpoint wait duration", registry=registry
        )
        self.update_weights_duration = Gauge(
            "orchestrator_update_weights_duration_seconds", "Weight update duration", registry=registry
        )

        self.async_level = Gauge("orchestrator_async_level", "Steps ahead of trainer", registry=registry)
        self.inflight_rollouts = Gauge(
            "orchestrator_inflight_rollouts", "Current number of in-flight rollouts", registry=registry
        )
        self.inflight_samples = Gauge(
            "orchestrator_inflight_samples", "Current number of in-flight samples", registry=registry
        )
        self.cancelled_rollouts = Gauge(
            "orchestrator_cancelled_rollouts", "Cancelled rollouts this step", registry=registry
        )
        self.empty_rollout_rate = Gauge(
            "orchestrator_empty_rollout_rate", "Fraction of empty rollouts", registry=registry
        )
        self.errored_rollout_rate = Gauge(
            "orchestrator_errored_rollout_rate", "Fraction of errored rollouts", registry=registry
        )

        self.off_policy_min = Gauge("orchestrator_off_policy_level_min", "Min off-policy steps", registry=registry)
        self.off_policy_max = Gauge("orchestrator_off_policy_level_max", "Max off-policy steps", registry=registry)
        self.off_policy_mean = Gauge("orchestrator_off_policy_level_mean", "Mean off-policy steps", registry=registry)

        self.error_rate = Gauge("orchestrator_error_rate", "Error rate across rollouts", registry=registry)
        self.event_loop_lag = Gauge(
            "orchestrator_event_loop_lag_seconds", "Event loop lag", ["stat"], registry=registry
        )
        self.worker_pending = Gauge(
            "orchestrator_worker_pending_count", "Pending requests per worker", ["worker"], registry=registry
        )
        self.worker_lag = Gauge(
            "orchestrator_worker_lag_seconds", "Worker event loop lag", ["worker", "stat"], registry=registry
        )
        self.pool_ratio = Gauge("orchestrator_pool_ratio", "Pool distribution", ["pool"], registry=registry)

        self._known_workers: set[str] = set()
        self._known_worker_lag: set[tuple[str, str]] = set()

    def update(self, to_log: dict[str, float]) -> None:
        self.step.set(to_log.get("step", 0))
        self.ckpt_step.set(to_log.get("progress/ckpt_step", 0))
        self.total_tokens.set(to_log.get("progress/total_tokens", 0))
        self.total_samples.set(to_log.get("progress/total_samples", 0))
        self.last_step_ts.set(time.time())

        self.step_duration.set(to_log.get("time/step", 0))
        self.generate_completions_duration.set(to_log.get("time/generate_completions", 0))
        self.wait_for_ckpt_duration.set(to_log.get("time/wait_for_ckpt", 0))
        self.update_weights_duration.set(to_log.get("time/update_weights", 0))

        self.async_level.set(to_log.get("scheduler/async_level", 0))
        self.inflight_rollouts.set(to_log.get("scheduler/inflight_rollouts", 0))
        self.inflight_samples.set(to_log.get("scheduler/inflight_samples", 0))
        self.cancelled_rollouts.set(to_log.get("scheduler/cancelled_rollouts", 0))
        self.empty_rollout_rate.set(to_log.get("empty_rollouts/all", 0))
        self.errored_rollout_rate.set(to_log.get("errored_rollouts/all", 0))

        self.off_policy_min.set(to_log.get("off_policy_level/all/min", 0))
        self.off_policy_max.set(to_log.get("off_policy_level/all/max", 0))
        self.off_policy_mean.set(to_log.get("off_policy_level/all/mean", 0))

        self.error_rate.set(to_log.get("error/all/mean", 0))

        for stat in ["min", "mean", "max"]:
            key = f"generation_ms/all/{stat}"
            if key in to_log:
                self.generation_duration.labels(stat=stat).set(to_log[key] / 1000)

        for stat in ["min", "mean", "med", "p90", "max"]:
            key = f"event_loop_lag/{stat}"
            if key in to_log:
                self.event_loop_lag.labels(stat=stat).set(to_log[key])

        current_workers = set()
        current_worker_lag = set()
        for key, value in to_log.items():
            if key.startswith("worker/") and key.endswith("/pending"):
                worker = key.replace("worker/", "", 1).replace("/pending", "", 1)
                current_workers.add(worker)
                self.worker_pending.labels(worker=worker).set(value)
                continue

            if key.startswith("worker_lag/"):
                _, worker, stat = key.split("/", 2)
                current_worker_lag.add((worker, stat))
                self.worker_lag.labels(worker=worker, stat=stat).set(value)

        removed_workers = self._known_workers - current_workers
        for worker in removed_workers:
            self.worker_pending.remove(worker)
        self._known_workers = current_workers

        removed_worker_lag = self._known_worker_lag - current_worker_lag
        for worker, stat in removed_worker_lag:
            self.worker_lag.remove(worker, stat)
        self._known_worker_lag = current_worker_lag

        for pool in ["easy", "normal", "hard"]:
            key = f"pool/{pool}"
            if key in to_log:
                self.pool_ratio.labels(pool=pool).set(to_log[key])
