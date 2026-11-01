"""FastAPI Backend for AegisVision AI (Industrial CV, ByteTrack OEE, Document Intelligence & SOP RAG Copilot)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from src.core.bytetrack_oee import ByteTrackConveyorTracker
from src.core.cv_pipeline import CascadedCVInspector
from src.core.doc_intelligence import (
    SPEC_DOCUMENT_TEMPLATES,
    EngineeringDocIntelligenceParser,
    analyze_custom_uploaded_matrix,
)
from src.core.mlops_benchmark import (
    RUNTIME_BENCHMARK_PROFILES,
    compute_population_stability_index,
    get_quantization_benchmarks,
)
from src.core.preprocessing import INSPECTION_SCENARIOS, OpenCVPreprocessor
from src.core.sop_rag_copilot import SOPHybridRAGCopilot


class InspectFrameRequest(BaseModel):
    """Strict request contract for running the 3-tier CV + MLOps + SOP RAG pipeline."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    scenario_id: str = Field(
        default="pcb-smt-line-01",
        description="Inspection line scenario ID.",
    )
    conf_threshold: float = Field(
        default=0.25,
        ge=0.05,
        le=0.95,
        description="Minimum confidence threshold for Tier-1 detection.",
    )
    uncertainty_ceiling: float = Field(
        default=0.65,
        ge=0.20,
        le=0.95,
        description="Upper threshold of the uncertainty band that triggers SAM 2 Zero-Shot Tier-2 refinement.",
    )
    enable_zero_shot_sam2: bool = Field(
        default=True,
        description="Whether to trigger Tier-2 SAM 2 zero-shot mask refinement on uncertain/novel anomalies.",
    )
    execution_provider: str = Field(
        default="ONNX_FP16",
        description="Runtime provider: PYTORCH_FP32, ONNX_FP16, ONNX_INT8, or TENSORRT_FP16.",
    )
    illumination_shift_pct: float = Field(
        default=0.0,
        ge=-60.0,
        le=60.0,
        description="Simulated ambient illumination shift percentage for PSI drift & CLAHE validation.",
    )
    perspective_skew_deg: float = Field(
        default=0.0,
        ge=-15.0,
        le=15.0,
        description="Camera planar tilt angle in degrees corrected via 3x3 Homography.",
    )


class ConveyorTrackRequest(BaseModel):
    """Request contract for running multi-frame ByteTrack + 2D Kalman Filter + OEE calculation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    scenario_id: str = Field(default="pcb-smt-line-01")
    num_frames: int = Field(default=8, ge=2, le=30)
    velocity_x_px: float = Field(default=18.0, ge=0.0, le=80.0)


class DocIntelligenceAuditRequest(BaseModel):
    """Request contract for parsing Engineering Spec / COA documents and auditing CV detections."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    scenario_id: str = Field(default="pcb-smt-line-01")
    raw_spec_text: str = Field(
        default=SPEC_DOCUMENT_TEMPLATES["coa-aerospace-ipc-class3"]["raw_document_text"],
        min_length=10,
    )


class CustomMatrixInspectRequest(BaseModel):
    """Request contract for custom uploaded image matrix OpenCV CLAHE + Sobel ROI extraction."""

    model_config = ConfigDict(extra="forbid")

    luminance_grid: List[List[float]] = Field(
        description="2D grayscale luminance array (H x W) in [0, 255] extracted from uploaded image canvas.",
    )
    micrometers_per_pixel: float = Field(default=15.0, ge=0.5, le=1000.0)
    edge_threshold: float = Field(default=26.0, ge=5.0, le=150.0)


class ApprovePseudoLabelRequest(BaseModel):
    """HITL approval payload for promoting zero-shot pseudo-labels into the YOLO/Detectron2 training set."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    scenario_id: str
    reviewer: str = Field(default="Lead Quality Engineer", min_length=2)
    notes: str = Field(default="Verified micrometer mask boundaries against IPC/ISO golden coupon.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize singleton CV Preprocessor, Cascaded Inspector, ByteTrack OEE, Doc Parser & Hybrid RAG Copilot."""
    app.state.preprocessor = OpenCVPreprocessor()
    app.state.inspector = CascadedCVInspector()
    app.state.tracker = ByteTrackConveyorTracker()
    app.state.doc_parser = EngineeringDocIntelligenceParser()
    app.state.rag_copilot = SOPHybridRAGCopilot()
    app.state.distillation_ledger: List[Dict[str, Any]] = []
    yield


app = FastAPI(
    title="AegisVision AI — Industrial & Retail Computer Vision Platform",
    version="2.0.0",
    description=(
        "3-Tier Cascaded Visual Inspection (OpenCV + YOLO26 NMS-Free + Detectron2 Mask R-CNN + SAM 2 Zero-Shot), "
        "ByteTrack Multi-Frame Conveyor Kalman Tracking & OEE, Engineering COA Document Intelligence, "
        "ONNX/TensorRT Quantization Benchmarks, PSI Drift Telemetry & Hybrid RAG CAPA Copilot."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
async def health_check() -> Dict[str, Any]:
    """Readiness probe for Kubernetes / Cloud Run / ECS Fargate."""
    return {
        "status": "healthy",
        "service": "aegisvision-ai",
        "version": "2.0.0",
        "scenarios_loaded": len(INSPECTION_SCENARIOS),
        "runtime_providers": list(RUNTIME_BENCHMARK_PROFILES.keys()),
        "capabilities": [
            "3-Tier Cascaded CV (YOLO26 + Detectron2 + SAM 2)",
            "ByteTrack 2D Kalman Conveyor Tracking & OEE",
            "Engineering COA Document Intelligence",
            "Custom Image CLAHE + Sobel ROI Inspector",
        ],
    }


@app.get("/api/v1/scenarios")
async def list_scenarios() -> Dict[str, Any]:
    """Return all available industrial manufacturing and retail vision scenarios."""
    items = []
    for sc in INSPECTION_SCENARIOS.values():
        items.append(
            {
                "scenario_id": sc.scenario_id,
                "title": sc.title,
                "industry": sc.industry,
                "production_line": sc.production_line,
                "camera_spec": sc.camera_spec,
                "resolution_px": list(sc.resolution_px),
                "micrometers_per_pixel": sc.micrometers_per_pixel,
                "iso_standard": sc.iso_standard,
                "nominal_illumination_lux": sc.nominal_illumination_lux,
                "description": sc.description,
                "defect_count": len(sc.ground_truth_defects),
            }
        )
    return {"scenarios": items, "coa_templates": SPEC_DOCUMENT_TEMPLATES}


@app.get("/api/v1/benchmarks")
async def list_benchmarks() -> Dict[str, Any]:
    """Return PyTorch FP32 vs ONNX FP16 vs ONNX INT8 vs TensorRT optimization benchmarks."""
    return {"benchmarks": get_quantization_benchmarks()}


@app.post("/api/v1/inspect")
async def inspect_frame(payload: InspectFrameRequest) -> Dict[str, Any]:
    """Run end-to-end OpenCV preprocessing, 3-Tier CV detection/segmentation, ByteTrack OEE, PSI drift, and SOP RAG CAPA."""
    if payload.scenario_id not in INSPECTION_SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown scenario_id: {payload.scenario_id}")
    if payload.execution_provider not in RUNTIME_BENCHMARK_PROFILES:
        raise HTTPException(status_code=422, detail=f"Unsupported execution_provider: {payload.execution_provider}")

    scenario = INSPECTION_SCENARIOS[payload.scenario_id]
    preprocessor: OpenCVPreprocessor = app.state.preprocessor
    inspector: CascadedCVInspector = app.state.inspector
    tracker: ByteTrackConveyorTracker = app.state.tracker
    rag_copilot: SOPHybridRAGCopilot = app.state.rag_copilot

    baseline_frame = preprocessor.synthesize_frame_tensor(scenario, illumination_shift_pct=0.0)
    shifted_frame = preprocessor.synthesize_frame_tensor(
        scenario,
        illumination_shift_pct=payload.illumination_shift_pct,
        perspective_skew_deg=payload.perspective_skew_deg,
    )

    opencv_telemetry = preprocessor.apply_clahe_and_homography(
        shifted_frame,
        scenario=scenario,
        perspective_skew_deg=payload.perspective_skew_deg,
    )
    drift_telemetry = compute_population_stability_index(baseline_frame, shifted_frame)

    cv_results = inspector.run_inspection(
        scenario=scenario,
        conf_threshold=payload.conf_threshold,
        uncertainty_ceiling=payload.uncertainty_ceiling,
        enable_zero_shot_sam2=payload.enable_zero_shot_sam2,
        execution_provider=payload.execution_provider,
    )

    conveyor_tracking = tracker.simulate_conveyor_burst(scenario=scenario, num_frames=8)

    capa_report = rag_copilot.retrieve_sop_and_generate_capa(
        detections=cv_results["detections"],
        scenario_title=scenario.title,
        resolution_px=scenario.resolution_px,
    )

    return {
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "title": scenario.title,
            "industry": scenario.industry,
            "production_line": scenario.production_line,
            "camera_spec": scenario.camera_spec,
            "resolution_px": list(scenario.resolution_px),
            "micrometers_per_pixel": scenario.micrometers_per_pixel,
            "iso_standard": scenario.iso_standard,
        },
        "opencv_preprocessing": opencv_telemetry,
        "drift_monitoring": drift_telemetry,
        "cv_inspection": cv_results,
        "conveyor_tracking": conveyor_tracking,
        "capa_rag_report": capa_report,
        "runtime_profile": RUNTIME_BENCHMARK_PROFILES[payload.execution_provider],
    }


@app.post("/api/v1/conveyor/track")
async def track_conveyor_burst(payload: ConveyorTrackRequest) -> Dict[str, Any]:
    """Run multi-frame ByteTrack 2D Kalman Filter tracking and OEE calculation."""
    if payload.scenario_id not in INSPECTION_SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown scenario_id: {payload.scenario_id}")
    scenario = INSPECTION_SCENARIOS[payload.scenario_id]
    tracker: ByteTrackConveyorTracker = app.state.tracker
    return tracker.simulate_conveyor_burst(
        scenario=scenario,
        num_frames=payload.num_frames,
        conveyor_speed_px_per_frame=(payload.velocity_x_px, 0.0),
    )


@app.post("/api/v1/doc-intelligence/audit")
async def audit_against_spec_document(payload: DocIntelligenceAuditRequest) -> Dict[str, Any]:
    """Parse unstructured Engineering Spec / COA text and audit CV detections against extracted limits."""
    if payload.scenario_id not in INSPECTION_SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown scenario_id: {payload.scenario_id}")
    scenario = INSPECTION_SCENARIOS[payload.scenario_id]
    inspector: CascadedCVInspector = app.state.inspector
    doc_parser: EngineeringDocIntelligenceParser = app.state.doc_parser

    parsed_rules = doc_parser.parse_spec_document(payload.raw_spec_text)
    cv_results = inspector.run_inspection(
        scenario=scenario,
        conf_threshold=parsed_rules["confidence_floor_override"],
    )
    return doc_parser.evaluate_detections_against_spec(
        raw_text=payload.raw_spec_text,
        detections=cv_results["detections"],
    )


@app.post("/api/v1/inspect-custom")
async def inspect_custom_uploaded_image(payload: CustomMatrixInspectRequest) -> Dict[str, Any]:
    """Run CLAHE, 8-bin histogram, and Sobel gradient ROI extraction on a custom uploaded image matrix."""
    return analyze_custom_uploaded_matrix(
        luminance_grid=payload.luminance_grid,
        micrometers_per_pixel=payload.micrometers_per_pixel,
        edge_threshold=payload.edge_threshold,
    )


@app.post("/api/v1/distillation/approve")
async def approve_pseudo_labels(payload: ApprovePseudoLabelRequest) -> Dict[str, Any]:
    """Commit verified SAM-2 zero-shot pseudo-labels into the active learning distillation registry."""
    if payload.scenario_id not in INSPECTION_SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown scenario_id: {payload.scenario_id}")

    scenario = INSPECTION_SCENARIOS[payload.scenario_id]
    entry = {
        "commit_id": f"DISTILL-{len(app.state.distillation_ledger) + 101}",
        "scenario_id": scenario.scenario_id,
        "reviewer": payload.reviewer,
        "notes": payload.notes,
        "exported_annotations": len(scenario.ground_truth_defects),
        "target_student_model": "yolo26s-seg.onnx",
        "status": "QUEUED_FOR_NIGHTLY_FINETUNE",
    }
    app.state.distillation_ledger.append(entry)
    return {"status": "committed", "entry": entry, "total_commits": len(app.state.distillation_ledger)}


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_operator_console() -> FileResponse:
    """Serve the Human-Centered UI/UX Operator Console."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Operator console UI not found.")
    return FileResponse(str(index_path))
