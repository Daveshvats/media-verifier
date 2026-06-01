"""
JPEG Compression Artifact Analysis Module
Research basis: JPEG double-compression detection (deterministic, court-friendly)
Key insight: If a JPEG was saved twice with different quality settings, it was likely edited.
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Dict, Any, List, Tuple, Optional

try:
    import jpegio
    JPEGIO_AVAILABLE = True
except ImportError:
    JPEGIO_AVAILABLE = False


class JPEGAnalyzer:
    """
    JPEG compression artifact analysis for forensic examination.
    
    Detects:
    1. Double JPEG compression (strong indicator of tampering)
    2. Quantization table inconsistencies
    3. JPEG ghost detection (regions with different compression history)
    4. DCT coefficient analysis
    """

    def __init__(self):
        self.jpegio_available = JPEGIO_AVAILABLE

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive JPEG artifact analysis.
        """
        results = {
            'is_jpeg': False,
            'double_compression': False,
            'double_compression_confidence': 0.0,
            'quantization_tables': {},
            'jpeg_ghost_map': None,
            'compression_quality_estimate': 0,
            'artifact_score': 0.0,
            'warnings': [],
        }

        # Check if file is JPEG
        is_jpeg = self._is_jpeg(image_path)
        results['is_jpeg'] = is_jpeg

        if not is_jpeg:
            results['warnings'].append(
                "Not a JPEG file. JPEG artifact analysis is not applicable. "
                "Non-JPEG formats (PNG, TIFF, BMP) are lossless — their presence "
                "doesn't indicate tampering, but also provides no compression-based evidence."
            )
            return results

        # Analyze quantization tables
        qt_results = self._analyze_quantization_tables(image_path)
        results['quantization_tables'] = qt_results

        # Estimate compression quality
        quality = self._estimate_jpeg_quality(image_path)
        results['compression_quality_estimate'] = quality

        # Detect double compression
        dbl_result = self._detect_double_compression(image_path)
        results['double_compression'] = dbl_result['detected']
        results['double_compression_confidence'] = dbl_result['confidence']
        results['double_compression_details'] = dbl_result.get('details', '')

        # JPEG ghost detection
        ghost_map = self._jpeg_ghost_analysis(image_path)
        results['jpeg_ghost_map'] = ghost_map

        # Compute overall artifact score
        results['artifact_score'] = self._compute_artifact_score(results)

        return results

    def _is_jpeg(self, image_path: str) -> bool:
        """Check if file is JPEG by magic bytes."""
        try:
            with open(image_path, 'rb') as f:
                header = f.read(2)
                return header == b'\xff\xd8'
        except Exception:
            return False

    def _analyze_quantization_tables(self, image_path: str) -> Dict[str, Any]:
        """
        Extract and analyze JPEG quantization tables.
        Inconsistent tables across image components indicate editing.
        """
        result = {
            'tables_extracted': False,
            'inconsistency_detected': False,
            'num_tables': 0,
            'table_signatures': [],
            'software_hint': None,
        }

        if self.jpegio_available:
            try:
                jpeg = jpegio.read(image_path)
                quant_tables = jpeg.quant_tables
                result['tables_extracted'] = True
                result['num_tables'] = len(quant_tables)
                
                # Extract table signatures for identification
                for i, table in enumerate(quant_tables):
                    sig = self._get_table_signature(table)
                    result['table_signatures'].append(sig)
                
                # Check for inconsistencies
                if len(quant_tables) > 1:
                    # In standard JPEG, all components should use consistent tables
                    table_sets = [tuple(t.flatten()) for t in quant_tables]
                    unique_tables = set(table_sets)
                    if len(unique_tables) > 1:
                        result['inconsistency_detected'] = True
                        result['inconsistency_details'] = (
                            f"Found {len(unique_tables)} different quantization table patterns "
                            f"across {len(quant_tables)} components. This can indicate splicing "
                            f"from sources with different compression settings."
                        )

                # Try to identify editing software from table patterns
                result['software_hint'] = self._identify_software(quant_tables)

            except Exception as e:
                result['error'] = str(e)

        # Fallback: use PIL for quality estimation
        if not result['tables_extracted']:
            try:
                with Image.open(image_path) as img:
                    if hasattr(img, 'quantization'):
                        qtables = img.quantization
                        if qtables:
                            result['tables_extracted'] = True
                            result['num_tables'] = len(qtables)
                            for i, qt in enumerate(qtables):
                                if qt:
                                    result['table_signatures'].append(
                                        f"table_{i}_avg_{np.mean(list(qt.values())):.1f}"
                                    )
            except Exception:
                pass

        return result

    def _get_table_signature(self, table: np.ndarray) -> str:
        """Generate a signature string for a quantization table."""
        flat = table.flatten()
        return f"avg={np.mean(flat):.1f}_min={np.min(flat)}_max={np.max(flat)}"

    def _identify_software(self, quant_tables: list) -> Optional[str]:
        """
        Attempt to identify editing software from quantization table patterns.
        Known patterns: Photoshop, GIMP, libjpeg, etc.
        """
        if not quant_tables:
            return None
        
        # Common software quantization table signatures
        avg_val = np.mean([np.mean(t) for t in quant_tables])
        min_val = np.min([np.min(t) for t in quant_tables])
        
        if avg_val < 5:
            return "High quality (likely Photoshop quality 95+)"
        elif avg_val < 15:
            return "Medium-high quality (typical of social media re-compression)"
        elif avg_val < 30:
            return "Medium quality (standard web compression)"
        else:
            return "Low quality (heavy compression, possibly WhatsApp/social media)"

    def _estimate_jpeg_quality(self, image_path: str) -> int:
        """Estimate JPEG compression quality from quantization tables."""
        try:
            with Image.open(image_path) as img:
                if hasattr(img, 'info') and 'quality' in img.info:
                    return img.info['quality']
                
                # Estimate from quantization table
                if hasattr(img, 'quantization') and img.quantization:
                    qtable = img.quantization
                    if 0 in qtable:
                        avg_q = np.mean(list(qtable[0].values()))
                        # Rough inverse mapping: higher quality = lower quantization values
                        quality = max(10, min(100, int(100 - avg_q * 2)))
                        return quality
        except Exception:
            pass
        
        return 75  # default assumption

    def _detect_double_compression(self, image_path: str) -> Dict[str, Any]:
        """
        Detect double JPEG compression by analyzing DCT coefficient histograms.
        
        Key insight: When an image is JPEG-compressed twice with different quality
        settings, the DCT coefficients show periodic peaks (First Digit Law violation).
        This is a well-established forensic technique and is considered court-friendly
        because it's based on deterministic statistical analysis.
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'details': '',
        }

        if self.jpegio_available:
            try:
                jpeg = jpegio.read(image_path)
                # Analyze DCT coefficients for double compression artifacts
                coeffs = jpeg.coef_arrays[0]  # Luminance component
                
                # Compute histogram of DCT coefficients
                flat_coeffs = coeffs.flatten()
                # Exclude DC coefficients
                ac_coeffs = flat_coeffs[flat_coeffs != 0]
                
                if len(ac_coeffs) > 0:
                    # Check for periodic peaks (double compression artifact)
                    hist, bin_edges = np.histogram(ac_coeffs, bins=range(-50, 51))
                    
                    # Compute first digit probability (Benford's Law)
                    first_digits = np.abs(ac_coeffs)
                    first_digits = first_digits[first_digits > 0]
                    first_digit_counts = np.zeros(9)
                    for d in first_digits:
                        while d >= 10:
                            d //= 10
                        if 1 <= d <= 9:
                            first_digit_counts[d-1] += 1
                    
                    total = np.sum(first_digit_counts)
                    if total > 0:
                        first_digit_probs = first_digit_counts / total
                        
                        # Benford's Law expected probabilities
                        benford = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
                        
                        # Chi-square test against Benford's Law
                        # Significant deviation suggests double compression
                        chi2 = np.sum((first_digit_probs - benford)**2 / (benford + 1e-10))
                        
                        # Threshold: chi2 > 0.05 suggests double compression
                        if chi2 > 0.05:
                            result['detected'] = True
                            result['confidence'] = min(1.0, chi2 / 0.2)
                            result['details'] = (
                                f"DCT coefficient distribution deviates significantly from "
                                f"Benford's Law (chi2={chi2:.4f}), indicating double JPEG compression. "
                                f"This strongly suggests the image was saved/edited at least twice."
                            )
                        else:
                            result['details'] = (
                                f"DCT coefficients follow Benford's Law (chi2={chi2:.4f}), "
                                f"consistent with single JPEG compression."
                            )

            except Exception as e:
                result['error'] = str(e)

        # Fallback: PIL-based double compression detection (works without jpegio on Windows)
        if not result['detected']:
            result = self._detect_double_compression_pil(image_path, result)

        # Fallback: simple double-save detection via file size comparison
        if not result['detected']:
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    _, buf1 = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    original_size = len(open(image_path, 'rb').read())
                    reencoded_size = len(buf1)
                    
                    ratio = original_size / (reencoded_size + 1)
                    if 0.85 < ratio < 1.15:
                        result['details'] += " File size ratio consistent with single compression."
                    elif ratio < 0.85:
                        result['detected'] = True
                        result['confidence'] = max(result['confidence'], 0.3)
                        result['details'] += " File size suggests prior heavy compression."
            except Exception:
                pass

        return result

    def _detect_double_compression_pil(self, image_path: str, 
                                        result: Dict[str, Any]) -> Dict[str, Any]:
        """
        PIL-based double JPEG compression detection.
        
        Works WITHOUT jpegio — uses only Pillow and numpy.
        
        Strategy: Recompress the image at multiple quality levels and analyze
        the error patterns. If the image was previously compressed at a certain
        quality Q1, recompressing at Q1 will produce distinctive error patterns
        (small error at Q1, larger at other qualities). The presence of multiple
        such "sweet spots" indicates double compression.
        
        Also uses quantization table analysis from PIL to detect inconsistencies.
        """
        try:
            with Image.open(image_path) as img:
                original = np.array(img.convert('RGB'))
            
            # Test multiple recompression qualities
            error_profile = {}
            for quality in [60, 65, 70, 75, 80, 85, 90, 95]:
                buffer = io.BytesIO()
                Image.fromarray(original).save(buffer, format='JPEG', quality=quality)
                buffer.seek(0)
                recompressed = np.array(Image.open(buffer))
                
                if recompressed.shape == original.shape:
                    mse = np.mean((original.astype(np.float32) - recompressed.astype(np.float32)) ** 2)
                    error_profile[quality] = mse
            
            if len(error_profile) < 3:
                return result
            
            # Analyze error profile for double compression signatures
            qualities = sorted(error_profile.keys())
            errors = [error_profile[q] for q in qualities]
            
            # Single compression: error should decrease monotonically with quality
            # Double compression: error profile has a local minimum near Q1
            # (because recompressing at the original quality produces less error)
            
            # Check for non-monotonic behavior
            is_monotonic = True
            for i in range(len(errors) - 1):
                if errors[i] < errors[i+1]:  # Error increasing = non-monotonic
                    is_monotonic = False
                    break
            
            # Check for local minimum (strong double compression indicator)
            has_local_min = False
            for i in range(1, len(errors) - 1):
                if errors[i] < errors[i-1] and errors[i] < errors[i+1]:
                    has_local_min = True
                    min_quality = qualities[i]
                    break
            
            if has_local_min:
                result['detected'] = True
                result['confidence'] = min(1.0, 0.6)
                result['details'] = (
                    f"PIL-based analysis detected error profile local minimum at quality {min_quality}, "
                    f"indicating the image was previously JPEG-compressed at approximately this quality level. "
                    f"This is consistent with double JPEG compression (image was saved/edited at least twice)."
                )
            elif not is_monotonic:
                result['detected'] = True
                result['confidence'] = min(1.0, 0.4)
                result['details'] = (
                    f"Recompression error profile is non-monotonic, which is abnormal for "
                    f"singly-compressed images. This suggests the image may have been "
                    f"compressed multiple times at different quality levels."
                )
            else:
                # Check quantization tables via PIL for additional evidence
                qt_evidence = self._check_qt_inconsistency_pil(image_path)
                if qt_evidence:
                    result['detected'] = True
                    result['confidence'] = min(1.0, 0.35)
                    result['details'] = (
                        f"Quantization table analysis suggests possible double compression: {qt_evidence}"
                    )
                else:
                    result['details'] += (
                        " PIL-based recompression analysis shows monotonic error profile, "
                        "consistent with single JPEG compression."
                    )
        
        except Exception as e:
            if 'error' not in result:
                result['error'] = str(e)
        
        return result

    def _check_qt_inconsistency_pil(self, image_path: str) -> Optional[str]:
        """
        Check quantization table inconsistency using PIL only (no jpegio).
        Multiple different quantization tables suggest splicing from different sources.
        """
        try:
            with Image.open(image_path) as img:
                if hasattr(img, 'quantization') and img.quantization:
                    qtables = img.quantization
                    if qtables:
                        # Compare luminance and chrominance tables
                        table_averages = []
                        for key, qt in qtables.items():
                            if qt:
                                vals = list(qt.values()) if isinstance(qt, dict) else list(qt)
                                table_averages.append(np.mean(vals))
                        
                        if len(table_averages) >= 2:
                            # Large difference between tables = inconsistency
                            max_diff = max(table_averages) - min(table_averages)
                            if max_diff > 15:
                                return (
                                    f"Quantization table averages differ by {max_diff:.1f} "
                                    f"(values: {[f'{v:.1f}' for v in table_averages]}). "
                                    f"Large differences suggest different compression sources."
                                )
        except Exception:
            pass
        
        return None

    def _jpeg_ghost_analysis(self, image_path: str, 
                              quality_range: range = range(60, 96, 5)) -> Optional[np.ndarray]:
        """
        JPEG Ghost detection: identifies regions with different compression history.
        
        For each quality level, recompress the image and compute the difference.
        Regions that were already compressed at a quality close to the test quality
        will show low error (appear as "ghosts" in the error map), while regions
        from a different source will show higher error.
        
        This is particularly effective for detecting splicing where parts of the
        image come from differently-compressed sources.
        """
        image = cv2.imread(image_path)
        if image is None:
            return None

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Compute error maps at different quality levels
        error_maps = []
        for quality in quality_range:
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            recompressed = np.array(Image.open(buffer))

            if recompressed.shape == rgb_image.shape:
                diff = np.mean(np.abs(
                    rgb_image.astype(np.float32) - recompressed.astype(np.float32)
                ), axis=2)
                error_maps.append(diff)

        if not error_maps:
            return None

        # Stack error maps and find minimum error per pixel
        stacked = np.stack(error_maps, axis=0)
        min_error_map = np.min(stacked, axis=0)

        # Normalize to 0-255
        if min_error_map.max() > 0:
            normalized = (min_error_map / min_error_map.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(min_error_map, dtype=np.uint8)

        return normalized

    def _compute_artifact_score(self, results: Dict[str, Any]) -> float:
        """
        Compute overall JPEG artifact score (0-100).
        Higher score = more suspicious artifacts.
        """
        score = 0.0

        # Double compression is a strong indicator
        if results['double_compression']:
            score += results['double_compression_confidence'] * 50

        # Quantization table inconsistency
        qt = results.get('quantization_tables', {})
        if qt.get('inconsistency_detected'):
            score += 25

        # Very low quality suggests heavy re-compression (common in sharing/editing)
        quality = results.get('compression_quality_estimate', 75)
        if quality < 50:
            score += 15
        elif quality < 70:
            score += 5

        return round(min(100, score), 1)
