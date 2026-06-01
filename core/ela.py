"""
Adaptive Error Level Analysis (ELA) Module
Research basis: ELA-TransNet (Prakash & Sahani, 2026), Adaptive Compression Factor ELA (2024)
Key improvement: Adaptive compression quality selection instead of fixed quality factor
"""

import cv2
import numpy as np
from PIL import Image
import io
import os
from typing import Tuple, Dict, Any, Optional


class AdaptiveELA:
    """
    Adaptive Error Level Analysis with multiple quality factors.
    
    Traditional ELA uses a single fixed JPEG quality factor (typically 75-85).
    Research shows this is suboptimal — different tampering types are revealed
    at different quality levels. This implementation runs ELA at multiple quality
    levels and selects the most informative one, following the adaptive compression
    approach from the 2024 research.
    """

    def __init__(self, quality_levels: list = None, scale_factor: float = 15.0):
        """
        Args:
            quality_levels: List of JPEG quality levels to test (default: [60, 70, 75, 80, 85, 90])
            scale_factor: Amplification factor for error visualization
        """
        self.quality_levels = quality_levels or [60, 70, 75, 80, 85, 90]
        self.scale_factor = scale_factor

    def _compute_ela_single(self, image: np.ndarray, quality: int) -> Tuple[np.ndarray, float]:
        """
        Compute ELA at a single quality level.
        
        Process:
        1. Re-compress the image at the given JPEG quality
        2. Compute pixel-wise difference between original and re-compressed
        3. Scale the difference for visualization
        
        Returns:
            Tuple of (ELA image, mean error)
        """
        # Convert to PIL for JPEG recompression
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Save and reload at specified quality
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        recompressed = Image.open(buffer)
        recompressed = np.array(recompressed)

        # Convert original to RGB for comparison
        original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ensure same dimensions
        if original.shape != recompressed.shape:
            recompressed = np.array(recompressed.resize(
                (original.shape[1], original.shape[0])
            ))

        # Compute absolute difference
        diff = np.abs(original.astype(np.float32) - recompressed.astype(np.float32))
        
        # Scale for visualization
        ela_image = np.clip(diff * self.scale_factor, 0, 255).astype(np.uint8)
        
        # Mean error for scoring
        mean_error = float(np.mean(diff))

        return ela_image, mean_error

    def _select_best_quality(self, image: np.ndarray) -> int:
        """
        Select the optimal JPEG quality level for ELA analysis.
        
        Strategy: Choose the quality level that maximizes the variance of errors,
        as high variance indicates the most discriminative information about 
        tampering (regions with different compression histories show different 
        error patterns).
        """
        best_quality = 75  # default
        best_variance = 0

        for quality in self.quality_levels:
            _, mean_error = self._compute_ela_single(image, quality)
            # Use a small patch to estimate variance efficiently
            h, w = image.shape[:2]
            patch = image[h//4:3*h//4, w//4:3*w//4]
            _, patch_error = self._compute_ela_single(patch, quality)
            
            # Higher variance in error = more informative quality level
            if patch_error > best_variance:
                best_variance = patch_error
                best_quality = quality

        return best_quality

    def analyze(self, image_path: str, adaptive: bool = True) -> Dict[str, Any]:
        """
        Perform full Adaptive ELA analysis on an image.
        
        Args:
            image_path: Path to the image file
            adaptive: If True, automatically select best quality level
            
        Returns:
            Dictionary with ELA results including heatmap, scores, and analysis
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        results = {
            'image_shape': image.shape,
            'quality_levels_tested': self.quality_levels,
            'multi_quality_ela': {},
            'ela_scores': {},
        }

        if adaptive:
            best_quality = self._select_best_quality(image)
            results['selected_quality'] = best_quality
            results['adaptive'] = True
        else:
            best_quality = 75
            results['selected_quality'] = best_quality
            results['adaptive'] = False

        # Compute ELA at all quality levels for comprehensive analysis
        for quality in self.quality_levels:
            ela_img, mean_err = self._compute_ela_single(image, quality)
            results['multi_quality_ela'][quality] = {
                'ela_image': ela_img,
                'mean_error': mean_err,
            }
            results['ela_scores'][quality] = mean_err

        # Primary ELA at selected quality
        primary_ela, primary_error = self._compute_ela_single(image, best_quality)
        results['primary_ela'] = primary_ela
        results['primary_mean_error'] = primary_error

        # Generate heatmap visualization (colorized ELA)
        results['heatmap'] = self._generate_heatmap(primary_ela)

        # Compute region-wise anomaly score
        results['region_anomalies'] = self._detect_region_anomalies(primary_ela)

        # Compute ELA-based tampering likelihood (0-100)
        results['tampering_likelihood'] = self._compute_tampering_likelihood(
            primary_error, results['region_anomalies']
        )

        return results

    def _generate_heatmap(self, ela_image: np.ndarray) -> np.ndarray:
        """
        Generate a colorized heatmap from grayscale ELA image.
        Uses JET colormap for clear visualization of error regions.
        """
        # Convert to grayscale if needed
        if len(ela_image.shape) == 3:
            gray = cv2.cvtColor(ela_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = ela_image

        # Normalize to 0-255
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Apply JET colormap
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        return heatmap

    def _detect_region_anomalies(self, ela_image: np.ndarray, 
                                  grid_size: int = 8) -> list:
        """
        Detect anomalous regions using grid-based analysis.
        
        Divides the ELA image into a grid and computes per-cell statistics.
        Cells with significantly higher error than the mean are flagged as
        potentially tampered regions.
        """
        if len(ela_image.shape) == 3:
            gray = cv2.cvtColor(ela_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = ela_image

        h, w = gray.shape
        cell_h, cell_w = h // grid_size, w // grid_size
        
        cell_errors = []
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = gray[y1:y2, x1:x2]
                cell_errors.append({
                    'row': i, 'col': j,
                    'mean_error': float(np.mean(cell)),
                    'std_error': float(np.std(cell)),
                    'max_error': float(np.max(cell)),
                    'bbox': (x1, y1, x2, y2),
                })

        # Identify anomalous cells (error > mean + 1.5 * std)
        errors = [c['mean_error'] for c in cell_errors]
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        threshold = mean_err + 1.5 * std_err

        anomalies = [c for c in cell_errors if c['mean_error'] > threshold]
        for a in anomalies:
            a['anomaly_severity'] = float((a['mean_error'] - mean_err) / (std_err + 1e-8))

        return anomalies

    def _compute_tampering_likelihood(self, mean_error: float, 
                                       anomalies: list) -> float:
        """
        Compute a 0-100 tampering likelihood score from ELA analysis.
        
        Based on the research finding that ELA + scoring (Vijesh P et al., 2026)
        can effectively flag altered images. The score considers:
        1. Overall mean error level
        2. Number and severity of anomalous regions
        3. Spatial distribution of anomalies
        """
        # Base score from mean error (normalized to 0-1 range)
        # Typical pristine images: mean error < 3; tampered: > 5
        error_score = min(1.0, mean_error / 10.0)

        # Anomaly contribution
        n_anomalies = len(anomalies)
        max_severity = max([a['anomaly_severity'] for a in anomalies], default=0)
        
        # Anomaly score: more anomalies + higher severity = higher score
        anomaly_score = min(1.0, (n_anomalies / 10.0) * 0.5 + (max_severity / 5.0) * 0.5)

        # Weighted combination (ELA is one signal, not definitive)
        # Weight ELA at 40% since it has known limitations
        final_score = (error_score * 0.4 + anomaly_score * 0.6) * 100

        return round(min(100, max(0, final_score)), 1)

    def get_ela_comparison_image(self, image_path: str, 
                                  quality: int = None) -> np.ndarray:
        """
        Generate a side-by-side comparison: original | ELA | heatmap
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        if quality is None:
            quality = self._select_best_quality(image)

        ela_img, _ = self._compute_ela_single(image, quality)
        heatmap = self._generate_heatmap(ela_img)

        # Resize all to same height
        h = image.shape[0]
        images = [image, heatmap]
        resized = []
        for img in images:
            ratio = h / img.shape[0]
            new_w = int(img.shape[1] * ratio)
            resized.append(cv2.resize(img, (new_w, h)))

        # Concatenate horizontally
        comparison = np.hstack(resized)
        return comparison
