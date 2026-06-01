"""
Media Authenticity Verifier for Digital Evidence
Main Application — Pipeline Orchestrator + Gradio Web Interface

Research-backed multi-branch forensic analysis:
- Adaptive ELA (ELA-TransNet, 2026)
- JPEG Compression Artifacts (double-compression detection, Benford's Law)
- Noise & Frequency Analysis (PRNU, DWT, noise inconsistency)
- Copy-Move Forgery Detection (SIFT + block correlation)
- AI-Generated Image Detection (EfficientNet-B3 + CLIP zero-shot + frequency + noise)
- Metadata & Provenance Analysis (EXIF, C2PA, consistency checks)
- Multi-Branch Confidence Scoring (0-100 authenticity score)

Tech Stack: OpenCV, Pillow, PyWavelets, PyTorch, timm, transformers, Gradio
Note: jpegio is optional (Linux/Mac only) — PIL-based fallback used on Windows
"""

import os
import sys
import time
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

import gradio as gr

# Import core modules
from core.ela import AdaptiveELA
from core.jpeg_analysis import JPEGAnalyzer
from core.noise_analysis import NoiseAnalyzer
from core.copy_move import CopyMoveDetector
from core.metadata import MetadataAnalyzer
from core.scoring import ConfidenceScorer
from models.ai_detector import AIGeneratedDetector
from utils.report_generator import ForensicReportGenerator


class ForensicPipeline:
    """
    Orchestrates all forensic analysis modules and produces
    a comprehensive result with confidence scoring.
    """

    def __init__(self):
        self.ela = AdaptiveELA()
        self.jpeg_analyzer = JPEGAnalyzer()
        self.noise_analyzer = NoiseAnalyzer()
        self.copy_move = CopyMoveDetector()
        self.metadata = MetadataAnalyzer()
        self.ai_detector = AIGeneratedDetector()
        self.scorer = ConfidenceScorer()
        self.report_generator = ForensicReportGenerator()

    def analyze(self, image_path: str, 
                run_ela: bool = True,
                run_jpeg: bool = True,
                run_noise: bool = True,
                run_copy_move: bool = True,
                run_ai: bool = True,
                run_metadata: bool = True,
                progress_callback=None) -> Dict[str, Any]:
        """
        Run full forensic analysis pipeline on an image.
        
        Args:
            image_path: Path to the image file
            run_*: Toggle each analysis branch
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete analysis results with confidence scoring
        """
        results = {}
        total_steps = sum([run_ela, run_jpeg, run_noise, run_copy_move, run_ai, run_metadata])
        step = 0

        def update_progress(msg):
            nonlocal step
            step += 1
            if progress_callback:
                progress_callback(step / (total_steps + 1), msg)

        # Step 1: Adaptive ELA
        if run_ela:
            update_progress("Running Adaptive Error Level Analysis...")
            try:
                results['ela'] = self.ela.analyze(image_path)
            except Exception as e:
                results['ela'] = {'error': str(e), 'tampering_likelihood': 0}

        # Step 2: JPEG Artifact Analysis
        if run_jpeg:
            update_progress("Analyzing JPEG compression artifacts...")
            try:
                results['jpeg'] = self.jpeg_analyzer.analyze(image_path)
            except Exception as e:
                results['jpeg'] = {'error': str(e), 'artifact_score': 0}

        # Step 3: Noise & Frequency Analysis
        if run_noise:
            update_progress("Analyzing noise patterns and frequency domain...")
            try:
                results['noise'] = self.noise_analyzer.analyze(image_path)
            except Exception as e:
                results['noise'] = {'error': str(e), 'overall_noise_score': 0}

        # Step 4: Copy-Move Detection
        if run_copy_move:
            update_progress("Detecting copy-move forgeries...")
            try:
                results['copy_move'] = self.copy_move.analyze(image_path)
            except Exception as e:
                results['copy_move'] = {'error': str(e), 'clone_confidence': 0}

        # Step 5: AI-Generated Image Detection
        if run_ai:
            update_progress("Running AI-generated image detection...")
            try:
                results['ai_detection'] = self.ai_detector.analyze(image_path)
            except Exception as e:
                results['ai_detection'] = {'error': str(e), 'ai_generated_likelihood': 0}

        # Step 6: Metadata Analysis
        if run_metadata:
            update_progress("Analyzing metadata and provenance...")
            try:
                results['metadata'] = self.metadata.analyze(image_path)
            except Exception as e:
                results['metadata'] = {'error': str(e), 'anomaly_score': 0}

        # Step 7: Confidence Scoring
        update_progress("Computing forensic confidence score...")
        scoring_result = self.scorer.compute_score(results)
        results['scoring'] = scoring_result

        # Step 8: Generate Report
        update_progress("Generating forensic report...")
        try:
            report_path = self.report_generator.generate_report(
                image_path, results, scoring_result
            )
            results['report_path'] = report_path
        except Exception as e:
            results['report_error'] = str(e)

        return results


# ============================================================
# Gradio Web Interface
# ============================================================

def format_results_for_display(results: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Format analysis results for Gradio display."""
    scoring = results.get('scoring', {})
    
    # Main result text
    lines = [
        "=" * 60,
        "  MEDIA AUTHENTICITY VERIFIER — FORENSIC REPORT",
        "=" * 60,
        "",
        f"  Classification: {scoring.get('classification', 'N/A')}",
        f"  Authenticity Score: {scoring.get('authenticity_score', 0)}/100",
        f"  Suspicion Score: {scoring.get('suspicion_score', 0)}/100",
        "",
        "-" * 60,
        "  FORENSIC BRANCH RESULTS",
        "-" * 60,
    ]

    for branch, data in scoring.get('branch_scores', {}).items():
        name = branch.replace('_', ' ').title()
        score = data.get('score', 0)
        weight = data.get('weight', 0)
        
        if score <= 30:
            status = "[CLEAR]"
        elif score <= 60:
            status = "[ANOMALY]"
        else:
            status = "[ALERT]"
        
        lines.append(f"  {status} {name}: {score:.1f}/100 (weight: {weight:.0%})")

    # Primary concerns
    concerns = scoring.get('primary_concerns', [])
    if concerns:
        lines.extend([
            "",
            "-" * 60,
            "  PRIMARY CONCERNS",
            "-" * 60,
        ])
        for c in concerns:
            severity = c.get('severity', 'UNKNOWN')
            desc = c.get('description', 'No details')
            lines.append(f"  [{severity}] {desc}")

    # Recommendations
    recommendations = scoring.get('recommendations', [])
    if recommendations:
        lines.extend([
            "",
            "-" * 60,
            "  RECOMMENDATIONS",
            "-" * 60,
        ])
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. {rec}")

    lines.extend([
        "",
        "-" * 60,
        f"  Analysis completed: {scoring.get('analysis_timestamp', 'N/A')}",
        "=" * 60,
    ])

    text = '\n'.join(lines)
    report_path = results.get('report_path')
    
    return text, report_path


def extract_display_images(results: Dict[str, Any]) -> List[np.ndarray]:
    """Extract key visualization images from results."""
    images = []

    # ELA heatmap
    ela = results.get('ela', {})
    if 'heatmap' in ela and ela['heatmap'] is not None:
        images.append(cv2.cvtColor(ela['heatmap'], cv2.COLOR_BGR2RGB))

    # Copy-move heatmap
    cm = results.get('copy_move', {})
    if 'heatmap' in cm and cm['heatmap'] is not None:
        images.append(cv2.cvtColor(cm['heatmap'], cv2.COLOR_BGR2RGB))

    # AI detection heatmap
    ai = results.get('ai_detection', {})
    if 'heatmap' in ai and ai['heatmap'] is not None:
        images.append(cv2.cvtColor(ai['heatmap'], cv2.COLOR_BGR2RGB))

    return images


def run_analysis(image, run_ela, run_jpeg, run_noise, run_copy_move, run_ai, run_metadata,
                 progress=gr.Progress()):
    """
    Main analysis function for Gradio interface.
    """
    if image is None:
        return "Please upload an image to analyze.", None, None, None, None

    # Save uploaded image to temp path (cross-platform)
    temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
    os.close(temp_fd)
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Run pipeline
    pipeline = ForensicPipeline()
    
    def progress_callback(fraction, msg):
        progress(fraction, desc=msg)

    results = pipeline.analyze(
        temp_path,
        run_ela=run_ela,
        run_jpeg=run_jpeg,
        run_noise=run_noise,
        run_copy_move=run_copy_move,
        run_ai=run_ai,
        run_metadata=run_metadata,
        progress_callback=progress_callback,
    )

    # Format results
    text_result, report_path = format_results_for_display(results)
    
    # Extract visualization images
    vis_images = extract_display_images(results)
    
    ela_heatmap = vis_images[0] if len(vis_images) > 0 else None
    cm_heatmap = vis_images[1] if len(vis_images) > 1 else None
    ai_heatmap = vis_images[2] if len(vis_images) > 2 else None

    # Report image
    report_image = None
    if report_path and os.path.exists(report_path):
        report_image = cv2.cvtColor(cv2.imread(report_path), cv2.COLOR_BGR2RGB)

    return text_result, report_image, ela_heatmap, cm_heatmap, ai_heatmap


# Custom CSS and theme (module-level so launch() can access them)
CUSTOM_CSS = """
.gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.main-title {
    text-align: center;
    margin-bottom: 0.5rem;
}
.score-display {
    font-size: 2rem;
    font-weight: bold;
    text-align: center;
    padding: 1rem;
    border-radius: 0.5rem;
}
"""

CUSTOM_THEME = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="amber",
    neutral_hue="slate",
)


def build_gradio_app():
    """Build and return the Gradio application."""

    with gr.Blocks(
        title="Media Authenticity Verifier",
    ) as app:

        # Header
        gr.Markdown(
            """
            # 🔍 Media Authenticity Verifier for Digital Evidence
            ### Multi-Branch Forensic Analysis: ELA · JPEG Artifacts · Noise · Copy-Move · AI Detection · Metadata
            
            Upload an image to perform comprehensive forensic authenticity analysis. The system combines 
            **6 forensic branches** with research-backed scoring to produce a **0–100 authenticity score** 
            and detailed tamper heatmaps.
            """
        )

        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Image")
                input_image = gr.Image(
                    label="Image to Analyze",
                    type="numpy",
                    height=400,
                )
                
                gr.Markdown("### ⚙️ Analysis Options")
                with gr.Row():
                    run_ela = gr.Checkbox(label="Adaptive ELA", value=True)
                    run_jpeg = gr.Checkbox(label="JPEG Artifacts", value=True)
                with gr.Row():
                    run_noise = gr.Checkbox(label="Noise Analysis", value=True)
                    run_copy_move = gr.Checkbox(label="Copy-Move Detection", value=True)
                with gr.Row():
                    run_ai = gr.Checkbox(label="AI Detection", value=True)
                    run_metadata = gr.Checkbox(label="Metadata Analysis", value=True)

                analyze_btn = gr.Button(
                    "🔍 Run Forensic Analysis",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown(
                    """
                    ---
                    **Score Interpretation:**
                    - 🟢 **0–40**: Authentic / Likely Authentic  
                    - 🟡 **41–60**: Suspicious — investigate further  
                    - 🔴 **61–100**: Likely Manipulated / Manipulated
                    
                    **Research Basis:** Multi-branch fusion (Kamble & Uke, 2026) with 
                    88–93% accuracy across forgery types.
                    """
                )

            # Right column: Results
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Analysis Results")
                text_output = gr.Textbox(
                    label="Forensic Report",
                    lines=20,
                    max_lines=40,
                )

                with gr.Tabs():
                    with gr.Tab("📋 Full Report"):
                        report_output = gr.Image(
                            label="Forensic Report Visualization",
                            type="numpy",
                            height=600,
                        )
                    with gr.Tab("🔥 ELA Heatmap"):
                        ela_output = gr.Image(
                            label="Error Level Analysis Heatmap",
                            type="numpy",
                            height=400,
                        )
                    with gr.Tab("🔄 Copy-Move"):
                        cm_output = gr.Image(
                            label="Copy-Move Detection Heatmap",
                            type="numpy",
                            height=400,
                        )
                    with gr.Tab("🤖 AI Detection"):
                        ai_output = gr.Image(
                            label="AI-Generated Detection Heatmap",
                            type="numpy",
                            height=400,
                        )

        # Example images section
        gr.Markdown(
            """
            ---
            ### 📝 About This Tool
            
            **Media Authenticity Verifier** is a research-backed forensic analysis tool that combines 
            6 independent analysis branches into a unified authenticity scoring system:
            
            | Branch | Technique | What It Detects |
            |--------|-----------|----------------|
            | **Adaptive ELA** | Multi-quality Error Level Analysis | Compression inconsistencies, edited regions |
            | **JPEG Artifacts** | DCT analysis, Benford's Law, quantization tables | Double compression, splicing from different sources |
            | **Noise Analysis** | PRNU, DWT frequency, noise inconsistency | Splicing, source camera identification |
            | **Copy-Move** | SIFT + block correlation | Cloned/duplicated regions within image |
            | **AI Detection** | EfficientNet-B3 + CLIP zero-shot + frequency | AI-generated or deepfake content |
            | **Metadata** | EXIF, C2PA, timestamp consistency | Editing traces, provenance verification |
            
            **Developed for:** Delhi IFSO cybercrime evidence screening  
            **Architecture:** Multi-branch fusion with weighted confidence scoring  
            **Target performance:** 88% precision in flagging altered images
            """
        )

        # Wire up the analysis button
        analyze_btn.click(
            fn=run_analysis,
            inputs=[
                input_image,
                run_ela, run_jpeg, run_noise, run_copy_move, run_ai, run_metadata,
            ],
            outputs=[
                text_output, report_output, ela_output, cm_output, ai_output,
            ],
        )

    return app


# ============================================================
# Entry Point
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Media Authenticity Verifier for Digital Evidence")
    print("  Forensic Analysis Pipeline v1.0")
    print("=" * 60)
    print()
    print("Initializing forensic analysis modules...")
    
    app = build_gradio_app()
    
    print("Starting Gradio web interface...")
    print("Access the tool at: http://localhost:7860")
    print()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=CUSTOM_THEME,
        css=CUSTOM_CSS,
    )
