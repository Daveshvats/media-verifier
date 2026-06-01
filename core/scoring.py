"""
Forensic Confidence Scoring Engine
Research basis: 
- Unified Multi-Branch Framework (Kamble & Uke, 2026) — 88-93% accuracy
- AI-Based Legal Evidence Authenticity Scoring (Vijesh P et al., 2026) — 0-100 score
Key insight: Multi-branch fusion with weighted signals produces robust, legally defensible scores
"""

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime


class ConfidenceScorer:
    """
    Multi-branch forensic confidence scoring engine.
    
    Combines signals from all forensic analysis branches into a single
    0-100 authenticity score with detailed breakdown. The scoring follows
    the research-validated multi-branch fusion approach where each branch
    contributes a weighted signal to the final score.
    
    Score interpretation:
    - 0-20: Authentic (high confidence in integrity)
    - 21-40: Likely authentic (minor anomalies, could be compression artifacts)
    - 41-60: Suspicious (significant anomalies detected, further investigation needed)
    - 61-80: Likely manipulated (strong forensic evidence of tampering)
    - 81-100: Manipulated (very strong forensic evidence of tampering)
    """

    # Weights for each forensic branch (based on research reliability)
    # These weights reflect the established reliability of each technique:
    # - ELA: 20% — useful but has known limitations (compression-dependent)
    # - JPEG Artifacts: 15% — deterministic, court-friendly, but JPEG-only
    # - Noise Analysis: 15% — strong for splicing detection
    # - Copy-Move: 15% — definitive when detected, but narrow scope
    # - AI Detection: 20% — critical for modern threats, but evolving
    # - Metadata: 15% — deterministic, but can be stripped without tampering
    DEFAULT_WEIGHTS = {
        'ela': 0.20,
        'jpeg_artifacts': 0.15,
        'noise_analysis': 0.15,
        'copy_move': 0.15,
        'ai_detection': 0.20,
        'metadata': 0.15,
    }

    # Confidence classification thresholds
    CLASSIFICATIONS = [
        (0, 20, 'AUTHENTIC', 'No significant forensic anomalies detected. Image appears to be an unmodified capture from a camera device.'),
        (21, 40, 'LIKELY AUTHENTIC', 'Minor anomalies detected that could be explained by normal processing (compression, sharing). No strong evidence of tampering.'),
        (41, 60, 'SUSPICIOUS', 'Significant forensic anomalies detected. The image may have been modified. Further investigation is recommended.'),
        (61, 80, 'LIKELY MANIPULATED', 'Strong forensic evidence of image manipulation detected. Multiple analysis branches indicate tampering.'),
        (81, 100, 'MANIPULATED', 'Very strong forensic evidence of deliberate image manipulation. Results are consistent across multiple analysis methods.'),
    ]

    def __init__(self, weights: Dict[str, float] = None):
        """
        Args:
            weights: Custom weights for each forensic branch. 
                    Must sum to 1.0. Uses DEFAULT_WEIGHTS if not specified.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()

    def _validate_weights(self):
        """Ensure weights sum to approximately 1.0."""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            # Normalize
            for key in self.weights:
                self.weights[key] /= total

    def compute_score(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the forensic confidence score from all analysis results.
        
        Args:
            analysis_results: Dictionary containing results from all forensic
                            analysis branches (ELA, JPEG, noise, copy-move,
                            AI detection, metadata)
        
        Returns:
            Comprehensive scoring result with authenticity score, classification,
            branch breakdown, and forensic summary
        """
        # Extract branch scores
        branch_scores = self._extract_branch_scores(analysis_results)

        # Compute weighted final score
        # Note: branch scores are 0-100 where higher = more suspicious
        # We convert to authenticity: 100 - weighted_suspicion
        weighted_suspicion = 0.0
        for branch, weight in self.weights.items():
            if branch in branch_scores:
                weighted_suspicion += branch_scores[branch]['score'] * weight

        # Authenticity score (inverse of suspicion)
        authenticity_score = round(100 - weighted_suspicion, 1)

        # Determine classification
        classification = self._classify(authenticity_score)

        # Generate detailed breakdown
        breakdown = self._generate_breakdown(branch_scores, authenticity_score)

        # Generate forensic summary
        forensic_summary = self._generate_forensic_summary(
            branch_scores, authenticity_score, classification
        )

        # Identify primary concerns
        primary_concerns = self._identify_primary_concerns(branch_scores)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            authenticity_score, branch_scores, primary_concerns
        )

        result = {
            'authenticity_score': authenticity_score,
            'suspicion_score': round(weighted_suspicion, 1),
            'classification': classification['label'],
            'classification_detail': classification['description'],
            'branch_scores': branch_scores,
            'breakdown': breakdown,
            'primary_concerns': primary_concerns,
            'recommendations': recommendations,
            'forensic_summary': forensic_summary,
            'analysis_timestamp': datetime.now().isoformat(),
            'weights_used': self.weights,
        }

        return result

    def _extract_branch_scores(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize scores from each analysis branch."""
        branch_scores = {}

        # ELA score
        if 'ela' in results:
            ela_result = results['ela']
            branch_scores['ela'] = {
                'score': ela_result.get('tampering_likelihood', 0),
                'detail': {
                    'mean_error': ela_result.get('primary_mean_error', 0),
                    'anomalous_regions': len(ela_result.get('region_anomalies', [])),
                    'quality_used': ela_result.get('selected_quality', 'N/A'),
                },
                'weight': self.weights.get('ela', 0),
            }

        # JPEG artifact score
        if 'jpeg' in results:
            jpeg_result = results['jpeg']
            branch_scores['jpeg_artifacts'] = {
                'score': jpeg_result.get('artifact_score', 0),
                'detail': {
                    'is_jpeg': jpeg_result.get('is_jpeg', False),
                    'double_compression': jpeg_result.get('double_compression', False),
                    'quality_estimate': jpeg_result.get('compression_quality_estimate', 0),
                },
                'weight': self.weights.get('jpeg_artifacts', 0),
            }

        # Noise analysis score
        if 'noise' in results:
            noise_result = results['noise']
            branch_scores['noise_analysis'] = {
                'score': noise_result.get('overall_noise_score', 0),
                'detail': {
                    'inconsistency_score': noise_result.get('noise_inconsistency_score', 0),
                    'frequency_anomaly': noise_result.get('frequency_anomaly_score', 0),
                },
                'weight': self.weights.get('noise_analysis', 0),
            }

        # Copy-move score
        if 'copy_move' in results:
            cm_result = results['copy_move']
            cm_score = cm_result.get('clone_confidence', 0) * 100
            if cm_result.get('clone_detected', False):
                cm_score = max(cm_score, 50)  # Minimum 50 if clone detected
            branch_scores['copy_move'] = {
                'score': round(cm_score, 1),
                'detail': {
                    'clone_detected': cm_result.get('clone_detected', False),
                    'methods_used': cm_result.get('method_used', []),
                    'num_clone_regions': len(cm_result.get('clone_regions', [])),
                },
                'weight': self.weights.get('copy_move', 0),
            }

        # AI detection score
        if 'ai_detection' in results:
            ai_result = results['ai_detection']
            branch_scores['ai_detection'] = {
                'score': ai_result.get('ai_generated_likelihood', 0),
                'detail': {
                    'methods': ai_result.get('detection_method', []),
                    'possible_generator': ai_result.get('possible_generator'),
                    'efficientnet_score': ai_result.get('efficientnet_score', 0),
                    'clip_score': ai_result.get('clip_zero_shot_score', 0),
                    'frequency_score': ai_result.get('frequency_artifact_score', 0),
                    'noise_score': ai_result.get('noise_pattern_score', 0),
                },
                'weight': self.weights.get('ai_detection', 0),
            }

        # Metadata score
        if 'metadata' in results:
            meta_result = results['metadata']
            branch_scores['metadata'] = {
                'score': meta_result.get('anomaly_score', 0),
                'detail': {
                    'exif_present': meta_result.get('exif_present', False),
                    'editing_detected': meta_result.get('editing_traces', {}).get('is_known_editor', False),
                    'stripping_detected': meta_result.get('stripping_detected', False),
                    'timestamp_inconsistency': meta_result.get('timestamps', {}).get('inconsistency_detected', False),
                },
                'weight': self.weights.get('metadata', 0),
            }

        return branch_scores

    def _classify(self, authenticity_score: float) -> Dict[str, str]:
        """Classify the authenticity score into a category."""
        for low, high, label, description in self.CLASSIFICATIONS:
            if low <= authenticity_score <= high:
                return {'label': label, 'description': description}
        
        # Edge case
        if authenticity_score > 100:
            return {'label': 'AUTHENTIC', 'description': self.CLASSIFICATIONS[0][3]}
        return {'label': 'MANIPULATED', 'description': self.CLASSIFICATIONS[-1][3]}

    def _generate_breakdown(self, branch_scores: Dict[str, Any],
                             authenticity_score: float) -> List[Dict[str, Any]]:
        """Generate a detailed breakdown of each branch's contribution."""
        breakdown = []

        for branch, data in branch_scores.items():
            contribution = round(data['score'] * data['weight'], 1)
            breakdown.append({
                'branch': branch.replace('_', ' ').title(),
                'raw_score': data['score'],
                'weight': data['weight'],
                'weighted_contribution': contribution,
                'impact': 'HIGH' if data['score'] > 60 else ('MEDIUM' if data['score'] > 30 else 'LOW'),
            })

        # Sort by impact
        breakdown.sort(key=lambda x: x['weighted_contribution'], reverse=True)

        return breakdown

    def _identify_primary_concerns(self, branch_scores: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify the most concerning forensic findings."""
        concerns = []

        for branch, data in branch_scores.items():
            if data['score'] > 40:
                severity = 'CRITICAL' if data['score'] > 70 else 'HIGH' if data['score'] > 50 else 'MODERATE'
                concern = {
                    'branch': branch.replace('_', ' ').title(),
                    'severity': severity,
                    'score': data['score'],
                }

                # Add specific concern descriptions
                detail = data.get('detail', {})
                if branch == 'ela' and detail.get('anomalous_regions', 0) > 3:
                    concern['description'] = f"{detail['anomalous_regions']} anomalous regions detected in error level analysis"
                elif branch == 'jpeg_artifacts' and detail.get('double_compression'):
                    concern['description'] = "Double JPEG compression detected — image was saved/edited at least twice"
                elif branch == 'noise_analysis' and detail.get('inconsistency_score', 0) > 30:
                    concern['description'] = "Noise inconsistency suggests possible splicing from different sources"
                elif branch == 'copy_move' and detail.get('clone_detected'):
                    concern['description'] = f"Copy-move forgery detected: {detail.get('num_clone_regions', 0)} cloned region(s)"
                elif branch == 'ai_detection' and data['score'] > 50:
                    gen = detail.get('possible_generator')
                    concern['description'] = f"AI-generated content detected{': ' + gen if gen else ''}"
                elif branch == 'metadata' and detail.get('editing_detected'):
                    concern['description'] = "Editing software traces found in metadata"
                elif branch == 'metadata' and detail.get('stripping_detected'):
                    concern['description'] = "Metadata appears to have been stripped or is missing"
                else:
                    concern['description'] = f"Elevated forensic anomaly score: {data['score']:.1f}"

                concerns.append(concern)

        concerns.sort(key=lambda x: x['score'], reverse=True)
        return concerns

    def _generate_recommendations(self, authenticity_score: float,
                                   branch_scores: Dict[str, Any],
                                   concerns: List[Dict]) -> List[str]:
        """Generate actionable recommendations for investigators."""
        recommendations = []

        if authenticity_score >= 80:
            recommendations.append(
                "No further forensic analysis needed. Image passes all authenticity checks."
            )
        elif authenticity_score >= 60:
            recommendations.append(
                "Minor anomalies detected. These may be caused by normal image processing "
                "(compression, resizing). Consider the source and context of the image."
            )
        elif authenticity_score >= 40:
            recommendations.append(
                "SUSPICIOUS: Significant anomalies detected. Request the original file "
                "from the source device for comparison. Check if a verified original exists."
            )
            recommendations.append(
                "Cross-reference with other evidence sources. Do not rely solely on this "
                "image as evidence without additional verification."
            )
        elif authenticity_score >= 20:
            recommendations.append(
                "LIKELY MANIPULATED: Strong forensic evidence of tampering. This image "
                "should NOT be used as primary evidence without further investigation."
            )
            recommendations.append(
                "Request the original device/camera for PRNU comparison. "
                "If possible, obtain the original file directly from the source device."
            )
            recommendations.append(
                "Document all findings for potential legal proceedings. "
                "Consider engaging a professional forensic image analyst for court-ready analysis."
            )
        else:
            recommendations.append(
                "MANIPULATED: Very strong evidence of deliberate image manipulation. "
                "This image should be treated as unreliable evidence."
            )
            recommendations.append(
                "CRITICAL: This image shows multiple forensic indicators consistent with "
                "deliberate tampering. Professional forensic analysis is strongly recommended."
            )

        # Specific recommendations based on branch findings
        for concern in concerns:
            if 'copy-move' in concern['branch'].lower():
                recommendations.append(
                    "Copy-move forgery detected: Examine the cloned regions in the heatmap. "
                    "The duplicated areas may be used to conceal or duplicate objects."
                )
            if 'ai' in concern['branch'].lower():
                recommendations.append(
                    "AI-generated content detected: This image may have been entirely or "
                    "partially generated by AI. It should not be treated as a photograph "
                    "of a real event without additional verification."
                )
            if 'metadata' in concern['branch'].lower():
                recommendations.append(
                    "Metadata anomalies: Check the original source device for comparison. "
                    "Missing or inconsistent metadata does not prove tampering but reduces "
                    "the ability to verify the image's provenance."
                )

        return recommendations

    def _generate_forensic_summary(self, branch_scores: Dict[str, Any],
                                    authenticity_score: float,
                                    classification: Dict[str, str]) -> str:
        """Generate a human-readable forensic summary paragraph."""
        lines = []

        # Overall assessment
        lines.append(
            f"Forensic Analysis Result: {classification['label']} "
            f"(Authenticity Score: {authenticity_score}/100)"
        )

        # Key findings
        n_branches = len(branch_scores)
        n_concerning = sum(1 for b in branch_scores.values() if b['score'] > 40)
        n_clear = sum(1 for b in branch_scores.values() if b['score'] <= 20)

        lines.append(
            f"Analysis completed across {n_branches} forensic branches: "
            f"{n_clear} clear, {n_branches - n_clear - n_concerning} with minor anomalies, "
            f"{n_concerning} with significant findings."
        )

        # Specific findings
        for branch, data in branch_scores.items():
            if data['score'] > 40:
                branch_name = branch.replace('_', ' ').title()
                lines.append(
                    f"- {branch_name}: Score {data['score']:.1f}/100 "
                    f"(contributes {data['score'] * data['weight']:.1f} to overall suspicion)"
                )

        return ' '.join(lines)
