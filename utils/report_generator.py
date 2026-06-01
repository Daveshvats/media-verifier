"""
Forensic Report Generator
Generates investigator-ready PDF reports from analysis results
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
from datetime import datetime
from typing import Dict, Any, List, Optional
import os
import io

# Font setup for matplotlib
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf')
except Exception:
    pass
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class ForensicReportGenerator:
    """
    Generates professional forensic analysis reports with heatmaps,
    scores, and detailed findings for investigators.
    """

    # Color scheme for scores
    SCORE_COLORS = {
        'authentic': '#22c55e',      # Green
        'likely_authentic': '#84cc16', # Light green
        'suspicious': '#f59e0b',       # Amber
        'likely_manipulated': '#f97316', # Orange
        'manipulated': '#ef4444',      # Red
    }

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            # Cross-platform default: ./reports relative to the project directory
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(project_dir, 'reports')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(self, image_path: str, 
                        analysis_results: Dict[str, Any],
                        scoring_result: Dict[str, Any]) -> str:
        """
        Generate a comprehensive forensic report as a multi-panel figure.
        
        Returns:
            Path to the saved report image
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        report_path = os.path.join(self.output_dir, f'report_{image_name}_{timestamp}.png')

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.3)

        # Load original image
        original = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Row 1: Original image + Overall Score
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(original_rgb)
        ax_orig.set_title('Original Image', fontsize=14, fontweight='bold')
        ax_orig.axis('off')

        # Score gauge
        ax_score = fig.add_subplot(gs[0, 1])
        self._draw_score_gauge(ax_score, scoring_result['authenticity_score'],
                               scoring_result['classification'])

        # Classification summary
        ax_summary = fig.add_subplot(gs[0, 2])
        self._draw_summary(ax_summary, scoring_result)

        # Row 2: ELA Results
        ax_ela = fig.add_subplot(gs[1, 0])
        ela_result = analysis_results.get('ela', {})
        if 'primary_ela' in ela_result:
            ela_rgb = cv2.cvtColor(ela_result['primary_ela'], cv2.COLOR_RGB2BGR)
            ax_ela.imshow(ela_rgb)
            ax_ela.set_title(f'Error Level Analysis (Q={ela_result.get("selected_quality", "?")})',
                           fontsize=12, fontweight='bold')
        else:
            ax_ela.text(0.5, 0.5, 'ELA Not Available', ha='center', va='center')
            ax_ela.set_title('Error Level Analysis', fontsize=12, fontweight='bold')
        ax_ela.axis('off')

        # ELA Heatmap
        ax_ela_hm = fig.add_subplot(gs[1, 1])
        if 'heatmap' in ela_result:
            heatmap_rgb = cv2.cvtColor(ela_result['heatmap'], cv2.COLOR_BGR2RGB)
            ax_ela_hm.imshow(heatmap_rgb)
            ax_ela_hm.set_title('ELA Heatmap', fontsize=12, fontweight='bold')
        else:
            ax_ela_hm.text(0.5, 0.5, 'Heatmap Not Available', ha='center', va='center')
            ax_ela_hm.set_title('ELA Heatmap', fontsize=12, fontweight='bold')
        ax_ela_hm.axis('off')

        # Noise Analysis Map
        ax_noise = fig.add_subplot(gs[1, 2])
        noise_result = analysis_results.get('noise', {})
        if 'noise_inconsistency_map' in noise_result and noise_result['noise_inconsistency_map'] is not None:
            ax_noise.imshow(noise_result['noise_inconsistency_map'], cmap='RdBu_r')
            ax_noise.set_title('Noise Inconsistency Map', fontsize=12, fontweight='bold')
        else:
            ax_noise.text(0.5, 0.5, 'Noise Map Not Available', ha='center', va='center')
            ax_noise.set_title('Noise Analysis', fontsize=12, fontweight='bold')
        ax_noise.axis('off')

        # Row 3: Copy-Move + AI Detection Heatmap
        ax_cm = fig.add_subplot(gs[2, 0])
        cm_result = analysis_results.get('copy_move', {})
        if 'heatmap' in cm_result and cm_result['heatmap'] is not None:
            cm_rgb = cv2.cvtColor(cm_result['heatmap'], cv2.COLOR_BGR2RGB)
            ax_cm.imshow(cm_rgb)
            clone_text = "Clone Detected" if cm_result.get('clone_detected') else "No Clone Detected"
            ax_cm.set_title(f'Copy-Move Detection: {clone_text}', fontsize=12, fontweight='bold')
        else:
            ax_cm.text(0.5, 0.5, 'Copy-Move Analysis\nNot Available', ha='center', va='center')
            ax_cm.set_title('Copy-Move Detection', fontsize=12, fontweight='bold')
        ax_cm.axis('off')

        # AI Detection Heatmap
        ax_ai = fig.add_subplot(gs[2, 1])
        ai_result = analysis_results.get('ai_detection', {})
        if 'heatmap' in ai_result and ai_result['heatmap'] is not None:
            ai_rgb = cv2.cvtColor(ai_result['heatmap'], cv2.COLOR_BGR2RGB)
            ax_ai.imshow(ai_rgb)
            ai_pct = ai_result.get('ai_generated_likelihood', 0)
            ax_ai.set_title(f'AI Detection Heatmap ({ai_pct}% AI likelihood)',
                          fontsize=12, fontweight='bold')
        else:
            ax_ai.text(0.5, 0.5, 'AI Detection\nNot Available', ha='center', va='center')
            ax_ai.set_title('AI-Generated Detection', fontsize=12, fontweight='bold')
        ax_ai.axis('off')

        # JPEG Ghost Map
        ax_ghost = fig.add_subplot(gs[2, 2])
        jpeg_result = analysis_results.get('jpeg', {})
        if 'jpeg_ghost_map' in jpeg_result and jpeg_result['jpeg_ghost_map'] is not None:
            ax_ghost.imshow(jpeg_result['jpeg_ghost_map'], cmap='hot')
            ax_ghost.set_title('JPEG Ghost Map', fontsize=12, fontweight='bold')
        else:
            ax_ghost.text(0.5, 0.5, 'JPEG Analysis\nNot Available', ha='center', va='center')
            ax_ghost.set_title('JPEG Artifact Analysis', fontsize=12, fontweight='bold')
        ax_ghost.axis('off')

        # Row 4: Branch Score Breakdown Bar Chart
        ax_bar = fig.add_subplot(gs[3, :2])
        self._draw_branch_breakdown(ax_bar, scoring_result.get('branch_scores', {}))

        # Metadata Summary
        ax_meta = fig.add_subplot(gs[3, 2])
        self._draw_metadata_summary(ax_meta, analysis_results.get('metadata', {}))

        # Row 5: Forensic Summary and Recommendations
        ax_forensic = fig.add_subplot(gs[4, :])
        self._draw_forensic_summary(ax_forensic, scoring_result)

        # Add overall title
        fig.suptitle(
            'Media Authenticity Verifier — Forensic Analysis Report',
            fontsize=18, fontweight='bold', y=0.98
        )

        # Save
        plt.savefig(report_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return report_path

    def _draw_score_gauge(self, ax, score: float, classification: str):
        """Draw a semi-circular gauge showing the authenticity score."""
        # Determine color
        if score >= 80:
            color = self.SCORE_COLORS['authentic']
        elif score >= 60:
            color = self.SCORE_COLORS['likely_authentic']
        elif score >= 40:
            color = self.SCORE_COLORS['suspicious']
        elif score >= 20:
            color = self.SCORE_COLORS['likely_manipulated']
        else:
            color = self.SCORE_COLORS['manipulated']

        # Create gauge
        theta = np.linspace(np.pi, 0, 100)
        # Background arc
        for i in range(99):
            t = theta[i:i+2]
            # Color gradient from green to red
            frac = i / 99
            if frac < 0.2:
                c = self.SCORE_COLORS['manipulated']
            elif frac < 0.4:
                c = self.SCORE_COLORS['likely_manipulated']
            elif frac < 0.6:
                c = self.SCORE_COLORS['suspicious']
            elif frac < 0.8:
                c = self.SCORE_COLORS['likely_authentic']
            else:
                c = self.SCORE_COLORS['authentic']
            ax.plot(np.cos(t), np.sin(t), color=c, linewidth=20, solid_capstyle='butt')

        # Needle
        needle_angle = np.pi * (1 - score / 100)
        ax.plot([0, 0.7 * np.cos(needle_angle)], [0, 0.7 * np.sin(needle_angle)],
                color='#1f2937', linewidth=3, zorder=5)
        ax.plot(0, 0, 'o', color='#1f2937', markersize=8, zorder=6)

        # Score text
        ax.text(0, -0.15, f'{score}', fontsize=32, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(0, -0.35, classification, fontsize=11, fontweight='bold',
                ha='center', va='center', color=color)

        # Scale labels
        ax.text(-1.05, -0.1, '0', fontsize=9, ha='center', va='top')
        ax.text(1.05, -0.1, '100', fontsize=9, ha='center', va='top')

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.5, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Authenticity Score', fontsize=14, fontweight='bold')

    def _draw_summary(self, ax, scoring: Dict[str, Any]):
        """Draw text summary panel."""
        ax.axis('off')
        lines = [
            f"Analysis: {scoring.get('classification', 'N/A')}",
            f"Score: {scoring.get('authenticity_score', 0)}/100",
            f"Suspicion: {scoring.get('suspicion_score', 0)}/100",
            "",
            "Branch Results:",
        ]

        for branch, data in scoring.get('branch_scores', {}).items():
            name = branch.replace('_', ' ').title()
            score = data.get('score', 0)
            symbol = '✓' if score <= 30 else ('⚠' if score <= 60 else '✗')
            lines.append(f"  {symbol} {name}: {score:.0f}")

        concerns = scoring.get('primary_concerns', [])
        if concerns:
            lines.append("")
            lines.append("Primary Concerns:")
            for c in concerns[:3]:
                lines.append(f"  • {c.get('description', c.get('branch', 'Unknown'))}")

        text = '\n'.join(lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#e2e8f0'))
        ax.set_title('Analysis Summary', fontsize=14, fontweight='bold')

    def _draw_branch_breakdown(self, ax, branch_scores: Dict[str, Any]):
        """Draw horizontal bar chart of branch scores."""
        if not branch_scores:
            ax.text(0.5, 0.5, 'No branch scores available', ha='center', va='center')
            ax.axis('off')
            return

        branches = []
        scores = []
        colors = []
        weights = []

        for branch, data in branch_scores.items():
            branches.append(branch.replace('_', ' ').title())
            scores.append(data.get('score', 0))
            weights.append(data.get('weight', 0))
            s = data.get('score', 0)
            if s <= 30:
                colors.append('#22c55e')
            elif s <= 60:
                colors.append('#f59e0b')
            else:
                colors.append('#ef4444')

        y_pos = np.arange(len(branches))
        bars = ax.barh(y_pos, scores, color=colors, height=0.6, edgecolor='white')

        # Add score labels
        for i, (score, weight) in enumerate(zip(scores, weights)):
            ax.text(score + 1, i, f'{score:.0f} (w={weight:.0%})',
                   va='center', fontsize=10)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(branches, fontsize=11)
        ax.set_xlabel('Anomaly Score (0-100)', fontsize=11)
        ax.set_xlim(0, 110)
        ax.set_title('Forensic Branch Analysis Breakdown', fontsize=14, fontweight='bold')
        ax.axvline(x=40, color='#f59e0b', linestyle='--', alpha=0.5, label='Suspicious threshold')
        ax.axvline(x=60, color='#ef4444', linestyle='--', alpha=0.5, label='Manipulated threshold')
        ax.legend(loc='best', fontsize=9)
        ax.invert_yaxis()

    def _draw_metadata_summary(self, ax, metadata: Dict[str, Any]):
        """Draw metadata analysis summary."""
        ax.axis('off')

        lines = ["Metadata Analysis"]

        camera = metadata.get('camera_info', {})
        if camera.get('device_description', 'Unknown') != 'Unknown':
            lines.append(f"Device: {camera['device_description']}")
        else:
            lines.append("Device: Not identified")

        if metadata.get('exif_present'):
            lines.append(f"EXIF: Present")
            editing = metadata.get('editing_traces', {})
            if editing.get('is_known_editor'):
                lines.append(f"Editor: {editing.get('software_detected', 'Unknown')}")
            else:
                lines.append("Editor: None detected")
        else:
            lines.append("EXIF: Missing/Stripped")

        gps = metadata.get('gps_data', {})
        if gps.get('present'):
            lines.append(f"GPS: {gps['latitude']:.4f}, {gps['longitude']:.4f}")
        else:
            lines.append("GPS: Not available")

        ts = metadata.get('timestamps', {})
        if ts.get('inconsistency_detected'):
            lines.append("Timestamps: INCONSISTENT")
        else:
            lines.append("Timestamps: Consistent")

        lines.append(f"Anomaly Score: {metadata.get('anomaly_score', 0)}/100")

        # Warnings
        warnings = metadata.get('warnings', [])
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in warnings[:2]:
                lines.append(f"  • {w[:60]}...")

        text = '\n'.join(lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#e2e8f0'))
        ax.set_title('Metadata', fontsize=14, fontweight='bold')

    def _draw_forensic_summary(self, ax, scoring: Dict[str, Any]):
        """Draw the forensic summary and recommendations panel."""
        ax.axis('off')

        # Forensic summary
        summary = scoring.get('forensic_summary', 'No summary available.')
        recommendations = scoring.get('recommendations', [])

        text_parts = [
            "FORENSIC SUMMARY",
            "=" * 80,
            summary,
            "",
            "RECOMMENDATIONS",
            "=" * 80,
        ]
        for i, rec in enumerate(recommendations, 1):
            text_parts.append(f"{i}. {rec}")

        text_parts.append("")
        text_parts.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        text_parts.append("Tool: Media Authenticity Verifier v1.0")

        text = '\n'.join(text_parts)
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fffbeb', edgecolor='#f59e0b'))
        ax.set_title('Forensic Report', fontsize=14, fontweight='bold')
