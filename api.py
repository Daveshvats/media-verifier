"""
Media Authenticity Verifier — FastAPI Backend
Serves as the bridge between a custom frontend UI and the forensic analysis pipeline.

Endpoints:
  POST /api/analyze-upload — Upload image + run analysis (multipart form)
  POST /api/analyze-base64 — Send base64 image + run analysis (JSON body)
  GET  /api/health         — Health check
  GET  /api/report/{id}    — Download a generated report image

The API response format is designed to match the frontend UI expectations exactly.
"""

import os
import io
import uuid
import base64
import tempfile
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import core modules
from core.ela import AdaptiveELA
from core.jpeg_analysis import JPEGAnalyzer
from core.noise_analysis import NoiseAnalyzer
from core.copy_move import CopyMoveDetector
from core.metadata import MetadataAnalyzer
from core.scoring import ConfidenceScorer
from models.ai_detector import AIGeneratedDetector
from utils.report_generator import ForensicReportGenerator


# ============================================================
# App Setup
# ============================================================

app = FastAPI(
    title="Media Authenticity Verifier API",
    description="Forensic analysis API for digital evidence authentication",
    version="1.0.0",
)

# CORS — allow any frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Global Pipeline Instance (lazy-loaded)
# ============================================================

class PipelineHolder:
    """Holds pipeline instances so models are loaded only once."""
    ela: Optional[AdaptiveELA] = None
    jpeg: Optional[JPEGAnalyzer] = None
    noise: Optional[NoiseAnalyzer] = None
    copy_move: Optional[CopyMoveDetector] = None
    metadata: Optional[MetadataAnalyzer] = None
    ai: Optional[AIGeneratedDetector] = None
    scorer: Optional[ConfidenceScorer] = None
    reporter: Optional[ForensicReportGenerator] = None
    initialized: bool = False

    @classmethod
    def init(cls):
        if cls.initialized:
            return
        print("Initializing forensic analysis pipeline...")
        cls.ela = AdaptiveELA()
        cls.jpeg = JPEGAnalyzer()
        cls.noise = NoiseAnalyzer()
        cls.copy_move = CopyMoveDetector()
        cls.metadata = MetadataAnalyzer()
        cls.ai = AIGeneratedDetector()
        cls.scorer = ConfidenceScorer()
        cls.reporter = ForensicReportGenerator()
        cls.initialized = True
        print("Pipeline initialization complete.")


# In-memory store for reports (keyed by analysis_id)
report_store: Dict[str, dict] = {}


# ============================================================
# Helper Functions
# ============================================================

def image_to_base64(img: np.ndarray, format: str = '.png') -> str:
    """Convert numpy image to base64 string."""
    if img is None:
        return ""
    success, buf = cv2.imencode(format, img)
    if not success:
        return ""
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def serialize_results(results: Dict[str, Any], analysis_id: str) -> dict:
    """
    Convert analysis results to a JSON-serializable format.
    Replaces numpy arrays with base64-encoded images.
    
    The output format is designed to match the frontend UI expectations:
    - primary_concerns: list of strings (not objects)
    - ai_generated_likelihood: 0-100 float (frontend does NOT multiply by 100)
    - efficientnet_score/clip_score/frequency_score/noise_score: 0-100 range
    - double_compression: float 0-1 (confidence), not boolean
    - clone_confidence: 0-1 float
    """
    serialized = {
        'analysis_id': analysis_id,
    }

    # ELA results
    ela = results.get('ela', {})
    serialized['ela'] = {
        'tampering_likelihood': ela.get('tampering_likelihood', 0),
        'selected_quality': ela.get('selected_quality', 'N/A'),
        'primary_mean_error': float(ela.get('primary_mean_error', 0) or 0),
        'num_anomalous_regions': len(ela.get('region_anomalies', [])),
        'heatmap_base64': image_to_base64(ela.get('heatmap')),
        'primary_ela_base64': image_to_base64(ela.get('primary_ela')),
    }

    # JPEG results
    jpeg = results.get('jpeg', {})
    serialized['jpeg'] = {
        'is_jpeg': bool(jpeg.get('is_jpeg', False)),
        'double_compression': float(jpeg.get('double_compression_confidence', 0) or 0),
        'compression_quality_estimate': jpeg.get('compression_quality_estimate', 0),
        'artifact_score': jpeg.get('artifact_score', 0),
        'quantization_inconsistency': bool(jpeg.get('quantization_tables', {}).get('inconsistency_detected', False)),
        'ghost_map_base64': image_to_base64(jpeg.get('jpeg_ghost_map')),
    }

    # Noise results
    noise = results.get('noise', {})
    noise_map = noise.get('noise_inconsistency_map')
    serialized['noise'] = {
        'overall_noise_score': noise.get('overall_noise_score', 0),
        'noise_inconsistency_score': float(noise.get('noise_inconsistency_score', 0) or 0),
        'frequency_anomaly_score': float(noise.get('frequency_anomaly_score', 0) or 0),
        'noise_map_base64': image_to_base64(noise_map),
    }
    # Pass through any error from noise analysis
    if noise.get('error'):
        serialized['noise']['error'] = noise['error']

    # Copy-move results
    cm = results.get('copy_move', {})
    serialized['copy_move'] = {
        'clone_detected': bool(cm.get('clone_detected', False)),
        'clone_confidence': float(cm.get('clone_confidence', 0) or 0),
        'num_clone_regions': len(cm.get('clone_regions', [])),
        'method_used': cm.get('method_used', []),
        'heatmap_base64': image_to_base64(cm.get('heatmap')),
    }

    # AI detection results — all scores in 0-100 range for frontend
    ai = results.get('ai_detection', {})
    serialized['ai_detection'] = {
        'ai_generated_likelihood': float(ai.get('ai_generated_likelihood', 0) or 0),
        'detection_methods': ai.get('detection_method', []),
        'possible_generator': ai.get('possible_generator'),
        'efficientnet_score': round(float(ai.get('efficientnet_score', 0) or 0) * 100, 1),
        'clip_score': round(float(ai.get('clip_zero_shot_score', 0) or 0) * 100, 1),
        'frequency_score': round(float(ai.get('frequency_artifact_score', 0) or 0) * 100, 1),
        'noise_score': round(float(ai.get('noise_pattern_score', 0) or 0) * 100, 1),
        'heatmap_base64': image_to_base64(ai.get('heatmap')),
        'frequency_spectrum_base64': image_to_base64(ai.get('frequency_spectrum')),
        'noise_map_base64': image_to_base64(ai.get('noise_map')),
    }
    # Pass through any error from AI detection
    if ai.get('error'):
        serialized['ai_detection']['error'] = ai['error']

    # Metadata results
    meta = results.get('metadata', {})
    camera = meta.get('camera_info', {})
    editing = meta.get('editing_traces', {})
    gps = meta.get('gps_data', {})
    timestamps = meta.get('timestamps', {})
    serialized['metadata'] = {
        'exif_present': bool(meta.get('exif_present', False)),
        'anomaly_score': meta.get('anomaly_score', 0),
        'camera_make': camera.get('device_manufacturer', 'Unknown'),
        'camera_model': camera.get('device_model', 'Unknown'),
        'editing_software': editing.get('software_detected', None),
        'is_known_editor': bool(editing.get('is_known_editor', False)),
        'gps_present': bool(gps.get('present', False)),
        'gps_coordinates': f"{gps.get('latitude', 0):.4f}, {gps.get('longitude', 0):.4f}" if gps.get('present') else None,
        'timestamp_inconsistency': bool(timestamps.get('inconsistency_detected', False)),
        'warnings': meta.get('warnings', []),
        'stripping_detected': bool(meta.get('stripping_detected', False)),
    }

    # Scoring results — format for frontend compatibility
    scoring = results.get('scoring', {})
    branch_scores = {}
    for branch, data in scoring.get('branch_scores', {}).items():
        branch_scores[branch] = {
            'score': round(float(data.get('score', 0) or 0), 1),
            'weight': round(float(data.get('weight', 0) or 0), 3),
        }
    
    # Primary concerns: convert from objects to human-readable strings
    # The backend returns objects with 'severity', 'description' keys
    # The frontend expects plain strings with severity prefixes
    primary_concerns_raw = scoring.get('primary_concerns', [])
    primary_concerns = []
    for concern in primary_concerns_raw:
        if isinstance(concern, dict):
            severity = concern.get('severity', 'INFO')
            description = concern.get('description', str(concern))
            primary_concerns.append(f"[{severity}] {description}")
        else:
            primary_concerns.append(str(concern))
    
    serialized['scoring'] = {
        'authenticity_score': round(float(scoring.get('authenticity_score', 0) or 0), 1),
        'suspicion_score': round(float(scoring.get('suspicion_score', 0) or 0), 1),
        'classification': scoring.get('classification', 'N/A'),
        'classification_detail': scoring.get('classification_detail', ''),
        'branch_scores': branch_scores,
        'primary_concerns': primary_concerns,
        'recommendations': scoring.get('recommendations', []),
        'forensic_summary': scoring.get('forensic_summary', ''),
        'analysis_timestamp': scoring.get('analysis_timestamp', ''),
    }

    # Report image
    report_path = results.get('report_path')
    if report_path and os.path.exists(report_path):
        report_img = cv2.imread(report_path)
        serialized['report_image_base64'] = image_to_base64(report_img, '.png')
    else:
        serialized['report_image_base64'] = ""

    return serialized


def run_pipeline(image_path: str,
                 run_ela: bool = True,
                 run_jpeg: bool = True,
                 run_noise: bool = True,
                 run_copy_move: bool = True,
                 run_ai: bool = True,
                 run_metadata: bool = True) -> dict:
    """Run the full forensic pipeline and return serialized results."""
    
    PipelineHolder.init()
    p = PipelineHolder
    results = {}

    # Step 1: ELA
    if run_ela:
        try:
            results['ela'] = p.ela.analyze(image_path)
        except Exception as e:
            results['ela'] = {'error': str(e), 'tampering_likelihood': 0}

    # Step 2: JPEG
    if run_jpeg:
        try:
            results['jpeg'] = p.jpeg.analyze(image_path)
        except Exception as e:
            results['jpeg'] = {'error': str(e), 'artifact_score': 0}

    # Step 3: Noise
    if run_noise:
        try:
            results['noise'] = p.noise.analyze(image_path)
        except Exception as e:
            results['noise'] = {'error': str(e), 'overall_noise_score': 0}

    # Step 4: Copy-Move
    if run_copy_move:
        try:
            results['copy_move'] = p.copy_move.analyze(image_path)
        except Exception as e:
            results['copy_move'] = {'error': str(e), 'clone_confidence': 0}

    # Step 5: AI Detection
    if run_ai:
        try:
            results['ai_detection'] = p.ai.analyze(image_path)
        except Exception as e:
            results['ai_detection'] = {'error': str(e), 'ai_generated_likelihood': 0}

    # Step 6: Metadata
    if run_metadata:
        try:
            results['metadata'] = p.metadata.analyze(image_path)
        except Exception as e:
            results['metadata'] = {'error': str(e), 'anomaly_score': 0}

    # Step 7: Scoring
    scoring_result = p.scorer.compute_score(results)
    results['scoring'] = scoring_result

    # Step 8: Report
    try:
        report_path = p.reporter.generate_report(image_path, results, scoring_result)
        results['report_path'] = report_path
    except Exception as e:
        results['report_error'] = str(e)

    # Serialize
    analysis_id = str(uuid.uuid4())
    serialized = serialize_results(results, analysis_id)

    # Store for later retrieval
    report_store[analysis_id] = serialized

    return serialized


# ============================================================
# API Endpoints
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Pre-load all models on startup so first request isn't slow."""
    PipelineHolder.init()


@app.get("/api/health")
async def health_check():
    """Check if the API is running and models are loaded."""
    return {
        "status": "healthy",
        "service": "Media Authenticity Verifier API",
        "version": "1.0.0",
        "pipeline_loaded": PipelineHolder.initialized,
    }


@app.post("/api/analyze-upload")
async def analyze_upload(
    file: UploadFile = File(...),
    run_ela: bool = Form(True),
    run_jpeg: bool = Form(True),
    run_noise: bool = Form(True),
    run_copy_move: bool = Form(True),
    run_ai: bool = Form(True),
    run_metadata: bool = Form(True),
):
    """
    Upload an image file and run forensic analysis.
    
    Returns JSON with all analysis results including base64-encoded heatmaps.
    The response format matches the frontend UI expectations.
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded file to temp location
    suffix = os.path.splitext(file.filename or 'image.jpg')[1] or '.jpg'
    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(temp_fd)

    try:
        contents = await file.read()
        with open(temp_path, 'wb') as f:
            f.write(contents)

        result = run_pipeline(
            temp_path,
            run_ela=run_ela,
            run_jpeg=run_jpeg,
            run_noise=run_noise,
            run_copy_move=run_copy_move,
            run_ai=run_ai,
            run_metadata=run_metadata,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        try:
            os.unlink(temp_path)
        except OSError:
            pass


@app.post("/api/analyze-base64")
async def analyze_base64(request_body: dict):
    """
    Send a base64-encoded image and run forensic analysis.
    
    Request body:
    {
        "image_base64": "data:image/jpeg;base64,...",
        "run_ela": true,
        "run_jpeg": true,
        "run_noise": true,
        "run_copy_move": true,
        "run_ai": true,
        "run_metadata": true
    }
    """
    image_b64 = request_body.get('image_base64', '')
    if not image_b64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    # Strip data URL prefix if present
    if ',' in image_b64:
        image_b64 = image_b64.split(',', 1)[1]

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
    os.close(temp_fd)

    try:
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)

        result = run_pipeline(
            temp_path,
            run_ela=request_body.get('run_ela', True),
            run_jpeg=request_body.get('run_jpeg', True),
            run_noise=request_body.get('run_noise', True),
            run_copy_move=request_body.get('run_copy_move', True),
            run_ai=request_body.get('run_ai', True),
            run_metadata=request_body.get('run_metadata', True),
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


@app.get("/api/report/{analysis_id}")
async def get_report(analysis_id: str):
    """Download the forensic report image for a given analysis ID."""
    if analysis_id not in report_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    data = report_store[analysis_id]
    report_b64 = data.get('report_image_base64', '')
    
    if not report_b64:
        raise HTTPException(status_code=404, detail="Report image not available")
    
    # Decode base64 and return as PNG
    try:
        img_bytes = base64.b64decode(report_b64)
        return JSONResponse(
            content={"report_image_base64": report_b64},
            media_type="application/json",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Serve Frontend Static Files (production mode)
# When SERVE_STATIC env var is set, serve the built frontend
# from the static/ directory (created by `npm run build`)
# ============================================================

STATIC_DIR = Path(__file__).parent / "static"

if os.environ.get("SERVE_STATIC") and STATIC_DIR.exists():
    # Mount static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """
        Serve the React SPA frontend.
        All non-API routes fall through to index.html (client-side routing).
        """
        # Check if a specific file exists in static/
        file_path = STATIC_DIR / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        # Otherwise serve index.html for SPA routing
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse(
            status_code=404,
            content={"detail": "Frontend not built. Run: cd frontend && npm run build"},
        )


# ============================================================
# Entry Point
# ============================================================

if __name__ == '__main__':
    import uvicorn
    
    print("=" * 60)
    print("  Media Authenticity Verifier — API Server")
    print("  Forensic Analysis Pipeline v1.0")
    print("=" * 60)
    print()
    print("API endpoints:")
    print("  GET  /api/health           — Health check")
    print("  POST /api/analyze-upload   — Upload & analyze image")
    print("  POST /api/analyze-base64   — Send base64 image & analyze")
    print("  GET  /api/report/{id}      — Get report image")
    print()
    print("Starting server at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
