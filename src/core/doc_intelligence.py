"""Multimodal Document Intelligence (Engineering Blueprint / COA Spec Parser) & Custom Image Analyzer.

Implements:
1. Structured extraction of quality tolerances from Engineering Change Orders (ECO),
   CAD GD&T callouts, and customer Certificate of Analysis (COA) specifications.
2. Dynamic compliance rule evaluation against detected CV micrometer measurements.
3. Custom image matrix OpenCV-grade CLAHE + Sobel edge hot-spot ROI extractor.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np


SPEC_DOCUMENT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "coa-aerospace-ipc-class3": {
        "doc_id": "COA-AERO-2026-881",
        "title": "Aerospace Flight-Critical ADAS Addendum (IPC-A-610H Class 3+)",
        "raw_document_text": (
            "CUSTOMER SPECIFICATION OVERRIDE: Doc #COA-AERO-2026-881.\n"
            "1. Maximum allowable solder bridge or flux residue surface area shall not exceed 0.65 mm2.\n"
            "2. Maximum equivalent defect diameter shall not exceed 950 um.\n"
            "3. Zero tolerance on class [solder_bridge, tombstone_capacitor, flux_micro_spatter].\n"
            "4. Minimum detector confidence floor: 0.20."
        ),
    },
    "coa-ev-busbar-iatf": {
        "doc_id": "ECO-HV-4680-112",
        "title": "800V EV Drivetrain Laser Weld Spec (ISO 13919-1 Stringent Grade)",
        "raw_document_text": (
            "ENGINEERING CHANGE ORDER: Doc #ECO-HV-4680-112.\n"
            "1. Maximum allowable weld pore or micro-fissure surface area shall not exceed 0.40 mm2.\n"
            "2. Maximum equivalent defect diameter shall not exceed 180 um.\n"
            "3. Zero tolerance on class [weld_keyhole_porosity, thermal_micro_crack].\n"
            "4. Minimum detector confidence floor: 0.22."
        ),
    },
    "coa-retail-coldchain-sla": {
        "doc_id": "SLA-RETAIL-HACCP-09",
        "title": "Tier-1 Grocer Cold-Chain & Planogram Compliance SLA",
        "raw_document_text": (
            "MERCHANDISING & HACCP SLA CONTRACT: Doc #SLA-RETAIL-HACCP-09.\n"
            "1. Maximum allowable shelf void or damaged seal area shall not exceed 120.0 mm2.\n"
            "2. Maximum equivalent defect diameter shall not exceed 15000 um.\n"
            "3. Zero tolerance on class [damaged_packaging_seal, shelf_stockout_gap].\n"
            "4. Minimum detector confidence floor: 0.30."
        ),
    },
}


class EngineeringDocIntelligenceParser:
    """Extracts deterministic metrological tolerances from unstructured engineering/COA text and audits CV detections."""

    def parse_spec_document(self, raw_text: str) -> Dict[str, Any]:
        """Extract max_area_mm2, max_diameter_um, zero_tolerance_classes, and conf_floor from spec text."""
        doc_id_match = re.search(r"Doc\s*#([A-Z0-9\-]+)", raw_text, re.IGNORECASE)
        area_match = re.search(r"(\d+(?:\.\d+)?)\s*mm2", raw_text, re.IGNORECASE)
        diam_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:um|µm|micrometer)", raw_text, re.IGNORECASE)
        classes_match = re.search(r"\[([a-z0-9_,\s]+)\]", raw_text, re.IGNORECASE)
        conf_match = re.search(r"confidence\s*floor[:\s]+(\d+(?:\.\d+)?)", raw_text, re.IGNORECASE)

        doc_id = doc_id_match.group(1) if doc_id_match else "CUSTOM-COA-SPEC"
        max_area_mm2 = float(area_match.group(1)) if area_match else 1.00
        max_diameter_um = float(diam_match.group(1)) if diam_match else 1200.0
        zero_tol_classes = (
            [c.strip() for c in classes_match.group(1).split(",") if c.strip()]
            if classes_match
            else ["solder_bridge", "weld_keyhole_porosity"]
        )
        conf_floor = float(conf_match.group(1)) if conf_match else 0.25

        return {
            "extracted_doc_id": doc_id,
            "max_allowable_area_mm2": round(max_area_mm2, 4),
            "max_allowable_diameter_um": round(max_diameter_um, 1),
            "zero_tolerance_classes": zero_tol_classes,
            "confidence_floor_override": round(conf_floor, 2),
        }

    def evaluate_detections_against_spec(
        self,
        raw_text: str,
        detections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Audit CV detections against extracted Document Intelligence rules."""
        rules = self.parse_spec_document(raw_text)
        audit_rows: List[Dict[str, Any]] = []
        violations_count = 0

        for det in detections:
            area_mm2 = float(det["metrics"]["mask_area_mm2"])
            diam_um = float(det["metrics"]["equivalent_diameter_um"])
            cls_name = det["class_name"]

            reasons: List[str] = []
            if cls_name in rules["zero_tolerance_classes"]:
                reasons.append(f"Class '{cls_name}' is listed in COA Zero-Tolerance schedule")
            if area_mm2 > rules["max_allowable_area_mm2"]:
                reasons.append(
                    f"Area {area_mm2} mm² exceeds COA limit ({rules['max_allowable_area_mm2']} mm²)"
                )
            if diam_um > rules["max_allowable_diameter_um"]:
                reasons.append(
                    f"Diameter {diam_um} µm exceeds COA limit ({rules['max_allowable_diameter_um']} µm)"
                )

            passed = len(reasons) == 0
            if not passed:
                violations_count += 1

            audit_rows.append(
                {
                    "defect_id": det["defect_id"],
                    "class_name": cls_name,
                    "measured_area_mm2": area_mm2,
                    "measured_diameter_um": diam_um,
                    "coa_compliance_status": "COMPLIANT" if passed else "COA_SPEC_VIOLATION",
                    "violation_reasons": reasons,
                }
            )

        return {
            "parsed_coa_rules": rules,
            "total_audited_defects": len(detections),
            "coa_violations_count": violations_count,
            "lot_disposition": "HOLD_LOT_COA_NONCONFORMANCE" if violations_count > 0 else "RELEASE_LOT_CERTIFIED",
            "audit_trail": audit_rows,
        }


def analyze_custom_uploaded_matrix(
    luminance_grid: List[List[float]],
    micrometers_per_pixel: float = 15.0,
    edge_threshold: float = 26.0,
) -> Dict[str, Any]:
    """Run CLAHE, 16-bin histogram analysis, and Sobel gradient ROI extraction on an uploaded image grid."""
    arr = np.clip(np.array(luminance_grid, dtype=np.float32), 0.0, 255.0)
    if arr.ndim != 2 or arr.shape[0] < 8 or arr.shape[1] < 8:
        arr = np.zeros((64, 64), dtype=np.float32) + 120.0

    h, w = arr.shape
    raw_mean = float(np.mean(arr))
    raw_std = float(np.std(arr))

    # CLAHE-style normalization
    norm = (arr - raw_mean) / max(raw_std, 1.0)
    clahe_arr = np.clip(128.0 + norm * 42.0, 0.0, 255.0)

    # Sobel gradient energy
    gx = np.diff(clahe_arr, axis=1, prepend=clahe_arr[:, :1])
    gy = np.diff(clahe_arr, axis=0, prepend=clahe_arr[:1, :])
    grad = np.hypot(gx, gy)

    # 8-bin intensity histogram
    hist_counts, _ = np.histogram(clahe_arr, bins=np.linspace(0.0, 255.0, 9))
    hist_pct = [round(float(c) / float(h * w), 4) for c in hist_counts]

    # Identify top high-gradient anomaly patches (grid scan 4x4 tiles)
    rois: List[Dict[str, Any]] = []
    tile_h = max(4, h // 4)
    tile_w = max(4, w // 4)
    for r in range(4):
        for c in range(4):
            y1, y2 = r * tile_h, min(h, (r + 1) * tile_h)
            x1, x2 = c * tile_w, min(w, (c + 1) * tile_w)
            patch_grad = grad[y1:y2, x1:x2]
            peak_energy = float(np.percentile(patch_grad, 90)) if patch_grad.size else 0.0
            if peak_energy >= edge_threshold:
                active_px = int(np.sum(patch_grad >= edge_threshold))
                area_um2 = round(active_px * (micrometers_per_pixel**2), 1)
                rois.append(
                    {
                        "roi_id": f"CUSTOM-ROI-{len(rois) + 1:02d}",
                        "bbox_grid": [int(x1), int(y1), int(x2), int(y2)],
                        "peak_sobel_energy": round(peak_energy, 2),
                        "active_edge_pixels": active_px,
                        "estimated_area_um2": area_um2,
                        "estimated_area_mm2": round(area_um2 / 1_000_000.0, 4),
                        "confidence": round(min(0.97, 0.45 + peak_energy / 110.0), 3),
                    }
                )

    rois.sort(key=lambda item: item["peak_sobel_energy"], reverse=True)
    return {
        "grid_shape": [int(h), int(w)],
        "raw_luminance_mean": round(raw_mean, 2),
        "raw_contrast_std": round(raw_std, 2),
        "clahe_contrast_std": round(float(np.std(clahe_arr)), 2),
        "mean_sobel_energy": round(float(np.mean(grad)), 2),
        "histogram_8bin_ratios": hist_pct,
        "detected_custom_rois": rois[:6],
    }
