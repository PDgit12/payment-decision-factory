"""Tests for Enterprise GPU Factory Infrastructure & Distributed Orchestration Engine."""

import pytest
from payment_decision_factory.orchestration.gpu_factory_orchestrator import (
    ClusterScheduler,
    DeliveryPhase,
    DistributedGPUTrainingCoordinator,
    DistributedTrainingJobConfig,
    GPUEtlBenchmarkReport,
    GPUEtlPipeline,
    GPUInstanceCatalog,
    GPUInstanceSpec,
    GPUInstanceTier,
    GPU_INSTANCE_CATALOG,
    PrecisionMode,
    TrainingRunEstimate,
    Turnkey48HourPipelineScheduler,
    TurnkeyDeliveryPlan,
)


class TestGPUInstanceCatalog:
    """Validate hardware catalog profiles, interconnects, and spot economics."""

    def test_catalog_entries_completeness(self):
        expected_tiers = [
            GPUInstanceTier.AWS_G5_12XLARGE,
            GPUInstanceTier.AWS_G6_12XLARGE,
            GPUInstanceTier.GCP_G2_STANDARD_48,
            GPUInstanceTier.AWS_P4DE_24XLARGE,
            GPUInstanceTier.AWS_P5_48XLARGE,
            GPUInstanceTier.GCP_A3_HIGHGPU_8G,
        ]
        for tier in expected_tiers:
            assert tier in GPU_INSTANCE_CATALOG
            spec = GPU_INSTANCE_CATALOG[tier]
            assert spec.num_gpus >= 4
            assert spec.total_vram_gb >= 96
            assert spec.on_demand_hourly_rate > 0
            assert spec.spot_hourly_rate > 0
            assert spec.spot_discount_pct >= 50.0  # At least 50% discount on spot

    def test_flagship_nvlink_bandwidth(self):
        p4de = GPU_INSTANCE_CATALOG[GPUInstanceTier.AWS_P4DE_24XLARGE]
        p5 = GPU_INSTANCE_CATALOG[GPUInstanceTier.AWS_P5_48XLARGE]
        assert p4de.interconnect_bandwidth_gb_sec == 600.0
        assert p5.interconnect_bandwidth_gb_sec == 900.0
        assert p5.tps_throughput_per_gpu >= 500000


class TestGPUEtlPipeline:
    """Validate GPU ETL benchmarking and speedup against CPU starvation."""

    def test_benchmark_10m_records(self):
        pipeline = GPUEtlPipeline()
        report = pipeline.benchmark_etl_pipeline(
            record_count=10_000_000,
            feature_count=32,
            gpu_tier=GPUInstanceTier.AWS_P4DE_24XLARGE,
        )
        assert isinstance(report, GPUEtlBenchmarkReport)
        assert report.record_count == 10_000_000
        assert report.total_etl_time_sec < 5.0  # Under 5 seconds on 8x A100
        assert report.throughput_rows_per_sec > 2_000_000
        assert report.cpu_speedup_factor > 20.0  # 20x+ speedup over CPU Pandas
        d = report.to_dict()
        assert "throughput_rows_per_sec" in d
        assert "cpu_speedup_factor" in d

    def test_benchmark_50m_records(self):
        pipeline = GPUEtlPipeline()
        report = pipeline.benchmark_etl_pipeline(
            record_count=50_000_000,
            feature_count=32,
            gpu_tier=GPUInstanceTier.AWS_P5_48XLARGE,
        )
        assert report.total_etl_time_sec < 10.0
        assert report.throughput_rows_per_sec > 5_000_000


class TestDistributedGPUTrainingCoordinator:
    """Validate job execution, scaling estimates, and spot savings."""

    def test_job_config_worker_derivation(self):
        cfg = DistributedTrainingJobConfig(
            job_name="client_acme_sub_billion_train",
            instance_tier=GPUInstanceTier.AWS_P4DE_24XLARGE,
            precision=PrecisionMode.BF16,
            scheduler=ClusterScheduler.RAY_TRAIN_KUBERAY,
        )
        assert cfg.num_workers_per_node == 8
        assert cfg.spot_allocation is True
        assert cfg.use_flash_attention is True

    def test_training_run_estimate_mid_tier(self):
        coord = DistributedGPUTrainingCoordinator()
        est = coord.estimate_job_execution(
            dataset_records=15_000_000,
            model_parameters=75_000_000,
            epochs=5,
        )
        assert est.recommended_instance == GPUInstanceTier.AWS_G5_12XLARGE
        assert est.total_training_hours < 8.0
        assert est.spot_cost_usd < 25.0  # Sub-$25 compute cost on spot
        assert est.savings_usd > est.spot_cost_usd

    def test_training_run_estimate_flagship_100m(self):
        coord = DistributedGPUTrainingCoordinator()
        est = coord.estimate_job_execution(
            dataset_records=100_000_000,
            model_parameters=250_000_000,
            epochs=5,
        )
        assert est.recommended_instance == GPUInstanceTier.AWS_P5_48XLARGE
        assert est.total_training_hours < 10.0
        assert est.spot_cost_usd < 250.0  # Training 100M rows on 8x H100 SXM5 for <$250


class TestTurnkey48HourPipelineScheduler:
    """Validate end-to-end 48-hour delivery timeline and phase budgeting."""

    def test_build_48_hour_delivery_plan(self):
        scheduler = Turnkey48HourPipelineScheduler()
        plan = scheduler.build_48_hour_delivery_plan(
            client_name="GlobalFinTech Enterprise",
            dataset_records=50_000_000,
            model_parameters=200_000_000,
            use_spot=True,
        )
        assert isinstance(plan, TurnkeyDeliveryPlan)
        assert plan.client_name == "GlobalFinTech Enterprise"
        assert plan.target_delivery_hours <= 48.0
        assert len(plan.phases) == 5
        
        # Verify chronological phase sequencing
        phases = [p.phase for p in plan.phases]
        assert phases == [
            DeliveryPhase.DAY_0_INGESTION_GPU_ETL,
            DeliveryPhase.DAY_1_DISTRIBUTED_GPU_TRAIN,
            DeliveryPhase.DAY_2_GPU_DIGITAL_TWIN_CERT,
            DeliveryPhase.DAY_2_QUANT_AND_SEALING,
            DeliveryPhase.DAY_3_CLIENT_DEPLOY_CANARY,
        ]
        
        # Compute cost sanity check
        assert plan.total_compute_cost_usd < 150.0  # Under $150 total GPU COGS per client delivery
        
        summary = plan.to_summary_dict()
        assert summary["client_name"] == "GlobalFinTech Enterprise"
        assert "readiness_sla_guarantee" in summary
