"""
Noise & Frequency Domain Analysis Module
Research basis: PRNU camera fingerprint, DWT-based manipulation detection
Key insight: Camera sensors leave unique noise patterns (PRNU); inconsistencies reveal splicing
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
import pywt


class NoiseAnalyzer:
    """
    Noise pattern and frequency domain analysis for forensic examination.
    
    Analyzes:
    1. PRNU (Photo Response Non-Uniformity) - camera sensor fingerprint
    2. Noise inconsistency map - different noise levels indicate splicing
    3. DWT (Discrete Wavelet Transform) frequency analysis
    4. Local Noise Level estimation for region-level anomalies
    """

    def __init__(self, wavelet: str = 'db8', noise_threshold: float = 2.0):
        """
        Args:
            wavelet: Wavelet type for DWT decomposition
            noise_threshold: Standard deviations above mean to flag as anomaly
        """
        self.wavelet = wavelet
        self.noise_threshold = noise_threshold

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive noise and frequency analysis.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        results = {
            'noise_inconsistency_map': None,
            'noise_inconsistency_score': 0.0,
            'dwt_analysis': {},
            'local_noise_map': None,
            'prnu_likelihood': 0.0,
            'frequency_anomaly_score': 0.0,
            'overall_noise_score': 0.0,
        }

        # 1. Extract noise residual (approximation of PRNU pattern)
        noise_residual = self._extract_noise_residual(image)
        results['noise_residual'] = noise_residual

        # 2. Compute noise inconsistency map
        inconsistency_map, inconsistency_score = self._compute_noise_inconsistency(image)
        results['noise_inconsistency_map'] = inconsistency_map
        results['noise_inconsistency_score'] = inconsistency_score

        # 3. Local noise level estimation
        local_noise_map = self._estimate_local_noise(image)
        results['local_noise_map'] = local_noise_map

        # 4. DWT frequency analysis
        dwt_results = self._dwt_analysis(image)
        results['dwt_analysis'] = dwt_results

        # 5. Frequency anomaly detection
        results['frequency_anomaly_score'] = self._compute_frequency_anomaly(dwt_results)

        # 6. Overall noise-based tampering score
        results['overall_noise_score'] = self._compute_overall_score(results)

        return results

    def _extract_noise_residual(self, image: np.ndarray) -> np.ndarray:
        """
        Extract noise residual using denoising filter.
        
        The noise residual is the difference between the original image
        and a denoised version. This captures the camera's PRNU pattern
        plus any other noise sources.
        
        PRNU (Photo Response Non-Uniformity) is a unique fingerprint
        left by each camera sensor. Different noise patterns in different
        regions of an image indicate the regions came from different cameras
        (i.e., splicing).
        """
        # Convert to float
        img_float = image.astype(np.float32)

        # Use non-local means denoising (preserves edges better)
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None, h=10, hForColorComponents=10,
                templateWindowSize=7, searchWindowSize=21
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                image, None, h=10,
                templateWindowSize=7, searchWindowSize=21
            )

        denoised_float = denoised.astype(np.float32)
        
        # Noise residual = original - denoised
        noise_residual = img_float - denoised_float

        return noise_residual

    def _compute_noise_inconsistency(self, image: np.ndarray, 
                                      grid_size: int = 8) -> Tuple[np.ndarray, float]:
        """
        Compute noise inconsistency map using grid-based analysis.
        
        Divides the image into blocks and estimates local noise variance.
        Blocks with significantly different noise levels from the majority
        likely came from a different source (splicing indicator).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        cell_h, cell_w = h // grid_size, w // grid_size

        noise_residual = self._extract_noise_residual(image)
        if len(noise_residual.shape) == 3:
            noise_gray = np.mean(noise_residual, axis=2)
        else:
            noise_gray = noise_residual

        # Compute per-cell noise variance
        variances = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = noise_gray[y1:y2, x1:x2]
                variances[i, j] = np.var(cell)

        # Compute inconsistency: how much each cell deviates from the median
        median_var = np.median(variances)
        mad = np.median(np.abs(variances - median_var))  # Median Absolute Deviation
        
        # Normalized deviation (robust z-score)
        if mad > 0:
            z_scores = (variances - median_var) / (mad * 1.4826)  # 1.4826 for consistency with std
        else:
            z_scores = np.zeros_like(variances)

        # Create inconsistency map
        inconsistency_map = cv2.resize(
            z_scores.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC
        )

        # Score: fraction of cells with high inconsistency
        anomaly_cells = np.sum(np.abs(z_scores) > self.noise_threshold)
        total_cells = grid_size * grid_size
        score = float(anomaly_cells / total_cells * 100)

        return inconsistency_map, score

    def _estimate_local_noise(self, image: np.ndarray, 
                               block_size: int = 64) -> np.ndarray:
        """
        Estimate local noise level using the MAD (Median Absolute Deviation)
        estimator in the wavelet domain. This is the state-of-the-art
        noise estimation approach.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape

        noise_map = np.zeros_like(gray, dtype=np.float32)
        count_map = np.zeros_like(gray, dtype=np.float32)

        for y in range(0, h - block_size + 1, block_size // 2):
            for x in range(0, w - block_size + 1, block_size // 2):
                block = gray[y:y+block_size, x:x+block_size].astype(np.float32)

                # Wavelet decomposition
                coeffs = pywt.dwt2(block, self.wavelet)
                if coeffs and len(coeffs) == 2:
                    cH, cV, cD = coeffs[1]  # Detail coefficients
                    
                    # MAD-based noise estimation (robust)
                    sigma = np.median(np.abs(cD)) / 0.6745

                    noise_map[y:y+block_size, x:x+block_size] += sigma
                    count_map[y:y+block_size, x:x+block_size] += 1

        # Average overlapping estimates
        count_map[count_map == 0] = 1
        noise_map = noise_map / count_map

        return noise_map

    def _dwt_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Discrete Wavelet Transform analysis for manipulation detection.
        
        Tampered regions often exhibit:
        1. Different frequency characteristics in high-frequency subbands
        2. Inconsistent wavelet coefficient distributions
        3. Artifacts at block boundaries (for JPEG-based manipulations)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray_float = gray.astype(np.float32)

        # Multi-level DWT decomposition
        results = {'levels': {}}
        
        try:
            coeffs = pywt.wavedec2(gray_float, self.wavelet, level=3)
            
            for level_idx, (cH, cV, cD) in enumerate(coeffs[1:], 1):
                level_key = f'level_{level_idx}'
                results['levels'][level_key] = {
                    'horizontal_energy': float(np.mean(cH**2)),
                    'vertical_energy': float(np.mean(cV**2)),
                    'diagonal_energy': float(np.mean(cD**2)),
                    'total_energy': float(np.mean(cH**2 + cV**2 + cD**2)),
                    'coeff_kurtosis': float(self._kurtosis(cD.flatten())),
                    'entropy': float(self._entropy(cD)),
                }

            # Check for frequency anomalies
            energies = [results['levels'][f'level_{i}']['total_energy'] 
                       for i in range(1, min(4, len(coeffs)))]
            if len(energies) >= 2:
                # Normal: energy decreases with level. Reversal = anomaly
                results['energy_monotonic'] = all(
                    energies[i] >= energies[i+1] for i in range(len(energies)-1)
                )

        except Exception as e:
            results['error'] = str(e)

        return results

    def _compute_frequency_anomaly(self, dwt_results: Dict[str, Any]) -> float:
        """
        Compute frequency anomaly score from DWT analysis.
        """
        if 'error' in dwt_results:
            return 0.0

        score = 0.0
        levels = dwt_results.get('levels', {})

        if not levels:
            return 0.0

        # Check energy monotonicity violation
        if not dwt_results.get('energy_monotonic', True):
            score += 20  # Non-monotonic energy is suspicious

        # Check for abnormally high kurtosis (indicates periodic artifacts)
        for level_key, level_data in levels.items():
            kurt = level_data.get('coeff_kurtosis', 0)
            if abs(kurt) > 10:  # Normal images have kurtosis ~3-8
                score += 10

        # Check entropy anomalies
        entropies = [level_data.get('entropy', 0) for level_data in levels.values()]
        if entropies:
            mean_entropy = np.mean(entropies)
            for e in entropies:
                if abs(e - mean_entropy) > 2:  # Large deviation
                    score += 5

        return min(100, score)

    def _compute_overall_score(self, results: Dict[str, Any]) -> float:
        """
        Compute overall noise-based tampering score (0-100).
        """
        score = 0.0

        # Noise inconsistency is a strong indicator
        score += results['noise_inconsistency_score'] * 0.5

        # Frequency anomalies add weight
        score += results['frequency_anomaly_score'] * 0.3

        # DWT energy monotonicity violation
        dwt = results.get('dwt_analysis', {})
        if not dwt.get('energy_monotonic', True):
            score += 10

        return round(min(100, score), 1)

    @staticmethod
    def _kurtosis(data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        n = len(data)
        if n < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3)

    @staticmethod
    def _entropy(data: np.ndarray) -> float:
        """Compute Shannon entropy of data."""
        hist, _ = np.histogram(data.flatten(), bins=256, density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))
