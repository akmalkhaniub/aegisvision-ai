"""Automated Verification Suite for AegisVision AI v2.0 (3-Tier CV, ByteTrack OEE, COA Doc-AI & FastAPI)."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.bytetrack_oee import ByteTrackConveyorTracker
from src.core.cv_pipeline import CascadedCVInspector, compute_bbox_iou, compute_polygon_area_px
from src.core.doc_intelligence import (
    SPEC_DOCUMENT_TEMPLATES,
    EngineeringDocIntelligenceParser,
    analyze_custom_uploaded_matrix,
)
from src.core.mlops_benchmark import compute_population_stability_index, get_quantization_benchmarks
from src.core.preprocessing import INSPECTION_SCENARIOS, OpenCVPreprocessor
from src.core.sop_rag_copilot import SOPHybridRAGCopilot


def test_bbox_iou_and_shoelace_polygon_math() -> None:
    """Verify exact geometric IoU and Shoelace polygon area calculations."""
    box_a = [100.0, 100.0, 200.0, 200.0]
    box_b = [100.0, 100.0, 200.0, 200.0]
    assert compute_bbox_iou(box_a, box_b) == 1.0

    disjoint = [300.0, 300.0, 400.0, 400.0]
    assert compute_bbox_iou(box_a, disjoint) == 0.0

    square_poly = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]
    assert compute_polygon_area_px(square_poly) == 5000.0


def test_opencv_clahe_and_homography_calibration() -> None:
    """Verify deterministic frame synthesis, CLAHE contrast gain, and 3x3 homography matrix."""
    pre = OpenCVPreprocessor()
    scenario = INSPECTION_SCENARIOS["pcb-smt-line-01"]
    frame = pre.synthesize_frame_tensor(scenario, illumination_shift_pct=-25.0, perspective_skew_deg=4.0)
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8

    telemetry = pre.apply_clahe_and_homography(frame, scenario, perspective_skew_deg=4.0)
    assert telemetry["calibrated_um_per_px"] == 14.5
    assert len(telemetry["homography_matrix_3x3"]) == 3
    assert telemetry["clahe_contrast_std"] > 20.0


def test_3tier_cascaded_cv_inspector_and_sam2_routing() -> None:
    """Verify Tier-1 YOLO26/Detectron2 detection and Tier-2 SAM 2 zero-shot cascade on uncertain defects."""
    inspector = CascadedCVInspector()
    scenario = INSPECTION_SCENARIOS["pcb-smt-line-01"]

    res_with_sam2 = inspector.run_inspection(
        scenario=scenario,
        conf_threshold=0.25,
        uncertainty_ceiling=0.65,
        enable_zero_shot_sam2=True,
        execution_provider="ONNX_INT8",
    )
    assert res_with_sam2["total_detections"] == 3
    assert res_with_sam2["sam2_cascade_triggers"] >= 1
    assert res_with_sam2["line_verdict"] == "REJECT_CRITICAL_HOLD"

    novel_det = next(d for d in res_with_sam2["detections"] if d["defect_id"] == "DEF-PCB-103")
    assert novel_det["routed_to_sam2"] is True
    assert novel_det["final_confidence"] > novel_det["tier1_raw_confidence"]
    assert novel_det["metrics"]["mask_area_um2"] > 0


def test_bytetrack_2d_kalman_filter_and_oee_analytics() -> None:
    """Verify 2-stage ByteTrack conveyor tracking deduplicates multi-frame observations and recovers glare dips."""
    tracker = ByteTrackConveyorTracker()
    scenario = INSPECTION_SCENARIOS["pcb-smt-line-01"]

    res = tracker.simulate_conveyor_burst(scenario=scenario, num_frames=8, conveyor_speed_px_per_frame=(18.0, 0.0))
    assert res["frames_processed"] == 8
    assert res["unique_tracked_defects"] == 3
    assert res["raw_frame_level_detections"] == 24
    assert res["duplicate_rejects_prevented"] == 21
    assert res["stage2_low_conf_glare_recoveries"] >= 2
    assert res["id_switches"] == 0
    assert 80.0 < res["oee_analytics"]["overall_oee_pct"] < 99.0


def test_document_intelligence_coa_parser_and_custom_image_analyzer() -> None:
    """Verify COA specification extraction, compliance auditing, and custom image matrix Sobel ROI detection."""
    parser = EngineeringDocIntelligenceParser()
    inspector = CascadedCVInspector()
    scenario = INSPECTION_SCENARIOS["pcb-smt-line-01"]
    cv_res = inspector.run_inspection(scenario)

    raw_coa = SPEC_DOCUMENT_TEMPLATES["coa-aerospace-ipc-class3"]["raw_document_text"]
    audit = parser.evaluate_detections_against_spec(raw_coa, cv_res["detections"])
    assert audit["parsed_coa_rules"]["extracted_doc_id"] == "COA-AERO-2026-881"
    assert audit["coa_violations_count"] >= 1
    assert audit["lot_disposition"] == "HOLD_LOT_COA_NONCONFORMANCE"

    # Synthetic 32x32 image matrix with high-contrast anomaly square
    grid = np.full((32, 32), 110.0, dtype=np.float32)
    grid[8:14, 8:14] = 245.0
    custom_res = analyze_custom_uploaded_matrix(grid.tolist(), micrometers_per_pixel=15.0, edge_threshold=20.0)
    assert custom_res["grid_shape"] == [32, 32]
    assert len(custom_res["histogram_8bin_ratios"]) == 8
    assert len(custom_res["detected_custom_rois"]) >= 1


def test_psi_drift_detection_and_onnx_benchmarks() -> None:
    """Verify Population Stability Index (PSI) drift status transitions and quantization benchmarks."""
    pre = OpenCVPreprocessor()
    scenario = INSPECTION_SCENARIOS["ev-battery-weld-02"]
    base_frame = pre.synthesize_frame_tensor(scenario, illumination_shift_pct=0.0)
    same_frame = pre.synthesize_frame_tensor(scenario, illumination_shift_pct=0.0)
    heavy_shift_frame = pre.synthesize_frame_tensor(scenario, illumination_shift_pct=45.0)

    stable_psi = compute_population_stability_index(base_frame, same_frame)
    assert stable_psi["psi_score"] == 0.0
    assert stable_psi["drift_status"] == "STABLE"

    drifted_psi = compute_population_stability_index(base_frame, heavy_shift_frame)
    assert drifted_psi["psi_score"] > 0.10
    assert drifted_psi["drift_status"] in {"MODERATE_DRIFT", "CRITICAL_DRIFT_ALERT"}

    benchmarks = get_quantization_benchmarks()
    assert len(benchmarks) == 4
    trt = next(b for b in benchmarks if b["provider_id"] == "TENSORRT_FP16")
    assert trt["speedup_vs_fp32"] > 4.5


def test_sop_hybrid_rag_and_active_learning_exports() -> None:
    """Verify Hybrid Dense+BM25 RRF retrieval and YOLO26/Detectron2 annotation export."""
    inspector = CascadedCVInspector()
    copilot = SOPHybridRAGCopilot()
    scenario = INSPECTION_SCENARIOS["retail-planogram-04"]

    cv_res = inspector.run_inspection(scenario)
    capa = copilot.retrieve_sop_and_generate_capa(
        detections=cv_res["detections"],
        scenario_title=scenario.title,
        resolution_px=scenario.resolution_px,
    )
    assert "GS1" in capa["cited_standard"] or "HACCP" in capa["cited_standard"]
    assert capa["hybrid_rrf_score"] > 0.70
    assert len(capa["recommended_corrective_actions"]) == 3


def test_fastapi_endpoints_end_to_end() -> None:
    """Verify all FastAPI v2.0 endpoints including conveyor tracking, doc intelligence, and custom image matrix."""
    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["version"] == "2.0.0"

        inspect_resp = client.post(
            "/api/v1/inspect",
            json={
                "scenario_id": "wafer-semicon-05",
                "conf_threshold": 0.25,
                "uncertainty_ceiling": 0.65,
                "enable_zero_shot_sam2": True,
                "execution_provider": "TENSORRT_FP16",
                "illumination_shift_pct": 10.0,
                "perspective_skew_deg": 1.5,
            },
        )
        assert inspect_resp.status_code == 200
        payload = inspect_resp.json()
        assert "conveyor_tracking" in payload
        assert payload["conveyor_tracking"]["unique_tracked_defects"] == 2

        track_resp = client.post(
            "/api/v1/conveyor/track",
            json={"scenario_id": "pcb-smt-line-01", "num_frames": 8, "velocity_x_px": 18.0},
        )
        assert track_resp.status_code == 200
        assert track_resp.json()["duplicate_rejects_prevented"] == 21

        coa_resp = client.post(
            "/api/v1/doc-intelligence/audit",
            json={
                "scenario_id": "pcb-smt-line-01",
                "raw_spec_text": SPEC_DOCUMENT_TEMPLATES["coa-aerospace-ipc-class3"]["raw_document_text"],
            },
        )
        assert coa_resp.status_code == 200
        assert coa_resp.json()["lot_disposition"] == "HOLD_LOT_COA_NONCONFORMANCE"

        grid = [[100.0 if (r < 6 or c < 6) else 230.0 for c in range(16)] for r in range(16)]
        cust_resp = client.post(
            "/api/v1/inspect-custom",
            json={"luminance_grid": grid, "micrometers_per_pixel": 15.0, "edge_threshold": 18.0},
        )
        assert cust_resp.status_code == 200
        assert cust_resp.json()["grid_shape"] == [16, 16]
