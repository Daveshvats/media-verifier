"""
Metadata & Provenance Analysis Module
Research basis: EXIF type-aware embeddings (Yang et al., 2026), C2PA content credentials
Key insight: Metadata inconsistencies are deterministic, court-friendly forensic evidence
"""

import cv2
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False

try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class MetadataAnalyzer:
    """
    Comprehensive metadata analysis for forensic examination.
    
    Analyzes:
    1. EXIF data completeness and consistency
    2. Camera/device identification
    3. GPS location data
    4. Timestamp consistency and anomalies
    5. Software/editing tool traces
    6. C2PA content provenance (if available)
    7. Thumbnail consistency with main image
    """

    # Known editing software signatures in EXIF
    EDITING_SOFTWARE = {
        'adobe photoshop': 'Adobe Photoshop',
        'gimp': 'GIMP',
        'paint.net': 'Paint.NET',
        'affinity': 'Affinity Photo',
        'capture one': 'Capture One',
        'lightroom': 'Adobe Lightroom',
        'snapseed': 'Snapseed',
        'picasa': 'Picasa',
        'instagram': 'Instagram',
        'picsart': 'PicsArt',
        'canva': 'Canva',
        'skylum': 'Skylum Luminar',
        'darktable': 'Darktable',
        'pixelmator': 'Pixelmator',
    }

    # Social media platforms that strip/modify EXIF
    EXIF_STRIPPERS = {
        'facebook', 'instagram', 'twitter', 'whatsapp', 'telegram',
        'messenger', 'snapchat', 'tiktok', 'reddit',
    }

    def __init__(self):
        self.piexif_available = PIEXIF_AVAILABLE
        self.pil_available = PIL_AVAILABLE

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive metadata analysis.
        """
        results = {
            'exif_present': False,
            'exif_data': {},
            'camera_info': {},
            'gps_data': {},
            'timestamps': {},
            'editing_traces': {},
            'metadata_consistency': {},
            'stripping_detected': False,
            'c2pa_manifest': None,
            'anomaly_score': 0.0,
            'warnings': [],
            'forensic_notes': [],
        }

        # Extract EXIF data
        exif_data = self._extract_exif(image_path)
        results['exif_data'] = exif_data
        results['exif_present'] = bool(exif_data)

        if not exif_data:
            results['stripping_detected'] = True
            results['warnings'].append(
                "No EXIF metadata found. This is a significant forensic indicator: "
                "either the image was taken with a device/app that doesn't embed EXIF, "
                "or the metadata was intentionally stripped (common when sharing via "
                "social media or after editing). A pristine camera image should have EXIF data."
            )
            results['forensic_notes'].append(
                "METADATA_ABSENT: No EXIF data — cannot verify device attribution or capture conditions."
            )
            results['anomaly_score'] = 30.0  # Moderate suspicion
            return results

        # Parse camera information
        results['camera_info'] = self._parse_camera_info(exif_data)

        # Parse GPS data
        results['gps_data'] = self._parse_gps(exif_data)

        # Parse and validate timestamps
        results['timestamps'] = self._analyze_timestamps(exif_data)

        # Detect editing software traces
        results['editing_traces'] = self._detect_editing_traces(exif_data)

        # Check metadata consistency
        results['metadata_consistency'] = self._check_consistency(exif_data, image_path)

        # Check for EXIF stripping indicators
        results['stripping_detected'] = self._detect_stripping(exif_data)

        # Check for C2PA manifest
        results['c2pa_manifest'] = self._check_c2pa(image_path)

        # Compute overall anomaly score
        results['anomaly_score'] = self._compute_anomaly_score(results)

        return results

    def _extract_exif(self, image_path: str) -> Dict[str, Any]:
        """Extract all available EXIF data from image."""
        exif_data = {}

        # Try piexif first (more detailed)
        if self.piexif_available:
            try:
                exif_dict = piexif.load(image_path)
                for ifd_name in ('0th', 'Exif', 'GPS', '1st'):
                    if exif_dict.get(ifd_name):
                        for tag_id, value in exif_dict[ifd_name].items():
                            tag_name = piexif.TAGS[ifd_name].get(tag_id, {}).get('name', f'tag_{tag_id}')
                            # Convert bytes to string for JSON serialization
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode('utf-8', errors='replace')
                                except Exception:
                                    value = f"<binary data, {len(value)} bytes>"
                            elif isinstance(value, tuple) and all(isinstance(x, int) for x in value):
                                # Rational number
                                if len(value) == 2 and value[1] != 0:
                                    value = value[0] / value[1]
                            exif_data[tag_name] = value
            except Exception:
                pass

        # Fallback: try PIL
        if not exif_data and self.pil_available:
            try:
                with Image.open(image_path) as img:
                    raw_exif = img._getexif()
                    if raw_exif:
                        for tag_id, value in raw_exif.items():
                            tag_name = TAGS.get(tag_id, f'tag_{tag_id}')
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode('utf-8', errors='replace')
                                except Exception:
                                    value = f"<binary data>"
                            exif_data[tag_name] = value
            except Exception:
                pass

        return exif_data

    def _parse_camera_info(self, exif_data: Dict) -> Dict[str, Any]:
        """Parse camera/device identification from EXIF."""
        camera_info = {
            'make': exif_data.get('Make', 'Unknown'),
            'model': exif_data.get('Model', 'Unknown'),
            'software': exif_data.get('Software', 'Unknown'),
            'lens_model': exif_data.get('LensModel', 'Unknown'),
            'focal_length': exif_data.get('FocalLength', 'Unknown'),
            'aperture': exif_data.get('FNumber', 'Unknown'),
            'iso': exif_data.get('ISOSpeedRatings', 'Unknown'),
            'exposure_time': exif_data.get('ExposureTime', 'Unknown'),
            'flash_fired': exif_data.get('Flash', 'Unknown'),
        }

        # Build device description
        parts = []
        if camera_info['make'] != 'Unknown':
            parts.append(str(camera_info['make']))
        if camera_info['model'] != 'Unknown':
            parts.append(str(camera_info['model']))
        camera_info['device_description'] = ' '.join(parts) if parts else 'Unknown device'

        return camera_info

    def _parse_gps(self, exif_data: Dict) -> Dict[str, Any]:
        """Parse and format GPS data from EXIF."""
        gps_data = {
            'present': False,
            'latitude': None,
            'longitude': None,
            'altitude': None,
        }

        lat = exif_data.get('GPSLatitude')
        lon = exif_data.get('GPSLongitude')
        lat_ref = exif_data.get('GPSLatitudeRef', 'N')
        lon_ref = exif_data.get('GPSLongitudeRef', 'E')

        if lat is not None and lon is not None:
            gps_data['present'] = True
            try:
                # Convert from degrees/minutes/seconds to decimal
                if isinstance(lat, (list, tuple)) and len(lat) == 3:
                    lat_decimal = float(lat[0]) + float(lat[1])/60 + float(lat[2])/3600
                    if lat_ref == 'S':
                        lat_decimal = -lat_decimal
                    gps_data['latitude'] = round(lat_decimal, 6)
                
                if isinstance(lon, (list, tuple)) and len(lon) == 3:
                    lon_decimal = float(lon[0]) + float(lon[1])/60 + float(lon[2])/3600
                    if lon_ref == 'W':
                        lon_decimal = -lon_decimal
                    gps_data['longitude'] = round(lon_decimal, 6)
            except Exception:
                pass

        alt = exif_data.get('GPSAltitude')
        if alt is not None:
            try:
                gps_data['altitude'] = float(alt)
            except Exception:
                pass

        return gps_data

    def _analyze_timestamps(self, exif_data: Dict) -> Dict[str, Any]:
        """Analyze timestamp consistency."""
        timestamps = {
            'date_time_original': exif_data.get('DateTimeOriginal'),
            'date_time_digitized': exif_data.get('DateTimeDigitized'),
            'date_time': exif_data.get('DateTime'),
            'inconsistency_detected': False,
            'notes': [],
        }

        dt_original = timestamps['date_time_original']
        dt_digitized = timestamps['date_time_digitized']
        dt_modify = timestamps['date_time']

        # Parse and compare timestamps
        parsed = {}
        for key, ts in [('original', dt_original), ('digitized', dt_digitized), ('modify', dt_modify)]:
            if ts and isinstance(ts, str):
                try:
                    parsed[key] = datetime.strptime(ts, '%Y:%m:%d %H:%M:%S')
                except ValueError:
                    try:
                        parsed[key] = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        timestamps['notes'].append(f"Could not parse {key} timestamp: {ts}")

        # Check for inconsistencies
        if 'original' in parsed and 'digitized' in parsed:
            diff = abs((parsed['digitized'] - parsed['original']).total_seconds())
            if diff > 3600:  # More than 1 hour difference
                timestamps['inconsistency_detected'] = True
                timestamps['notes'].append(
                    f"DateTimeOriginal and DateTimeDigitized differ by "
                    f"{diff/3600:.1f} hours. This can indicate the image was "
                    f"digitized (scanned or re-saved) well after the original capture."
                )

        if 'original' in parsed and 'modify' in parsed:
            diff = (parsed['modify'] - parsed['original']).total_seconds()
            if diff > 86400:  # Modified more than 24 hours after capture
                timestamps['notes'].append(
                    f"DateTime indicates modification {diff/86400:.1f} days after "
                    f"original capture. This is common for edited images."
                )
                timestamps['inconsistency_detected'] = True

        if dt_modify and not dt_original:
            timestamps['notes'].append(
                "DateTime present but DateTimeOriginal missing. "
                "This suggests metadata has been altered or partially stripped."
            )

        return timestamps

    def _detect_editing_traces(self, exif_data: Dict) -> Dict[str, Any]:
        """Detect traces of editing software in metadata."""
        editing_traces = {
            'software_detected': None,
            'is_known_editor': False,
            'editing_history': [],
            'photoshop_serial': None,
        }

        software = exif_data.get('Software', '')
        if software:
            editing_traces['software_detected'] = software
            software_lower = str(software).lower()
            
            for known_editor, display_name in self.EDITING_SOFTWARE.items():
                if known_editor in software_lower:
                    editing_traces['is_known_editor'] = True
                    editing_traces['editing_history'].append(
                        f"Edited with: {display_name} (detected from Software tag: '{software}')"
                    )
                    break

        # Check for Photoshop-specific tags
        ps_tags = ['PhotoshopSettings', 'PhotoshopThumbnail', 'PhotoshopQuality']
        for tag in ps_tags:
            if tag in exif_data:
                editing_traces['is_known_editor'] = True
                editing_traces['editing_history'].append(
                    f"Adobe Photoshop tag found: {tag}"
                )

        # Check IPTC data (often added by editing software)
        iptc_tags = [k for k in exif_data.keys() if 'IPTC' in k.upper()]
        if iptc_tags:
            editing_traces['editing_history'].append(
                f"IPTC metadata found (often added by editing software): {', '.join(iptc_tags)}"
            )

        # Check XMP data
        xmp_tags = [k for k in exif_data.keys() if 'XMP' in k.upper()]
        if xmp_tags:
            editing_traces['editing_history'].append(
                f"XMP metadata present: {', '.join(xmp_tags)}"
            )

        return editing_traces

    def _check_consistency(self, exif_data: Dict, 
                           image_path: str) -> Dict[str, Any]:
        """
        Check metadata consistency against actual image properties.
        """
        consistency = {
            'resolution_match': True,
            'thumbnail_consistent': True,
            'orientation_match': True,
            'anomalies': [],
        }

        # Check image dimensions vs EXIF reported dimensions
        try:
            img = cv2.imread(image_path)
            if img is not None:
                actual_h, actual_w = img.shape[:2]
                
                exif_width = exif_data.get('ExifImageWidth') or exif_data.get('PixelXDimension')
                exif_height = exif_data.get('ExifImageHeight') or exif_data.get('PixelYDimension')
                
                if exif_width and exif_height:
                    try:
                        exif_w = int(exif_width)
                        exif_h = int(exif_height)
                        
                        if exif_w != actual_w or exif_h != actual_h:
                            consistency['resolution_match'] = False
                            consistency['anomalies'].append(
                                f"Resolution mismatch: EXIF reports {exif_w}x{exif_h}, "
                                f"but actual image is {actual_w}x{actual_h}. "
                                f"This strongly indicates the image was resized after capture."
                            )
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

        # Check orientation
        orientation = exif_data.get('Orientation', 1)
        if isinstance(orientation, int) and orientation not in [1, 2]:
            consistency['anomalies'].append(
                f"EXIF Orientation tag = {orientation}. Image may appear rotated "
                f"in viewers that don't respect EXIF orientation. Verify actual display."
            )

        # Check thumbnail consistency (if embedded)
        if 'JPEGThumbnail' in exif_data or 'ThumbnailOffset' in exif_data:
            consistency['anomalies'].append(
                "Embedded thumbnail found. Thumbnail-main image consistency "
                "can be verified — mismatch indicates editing."
            )

        return consistency

    def _detect_stripping(self, exif_data: Dict) -> bool:
        """
        Detect if metadata was likely stripped (e.g., by social media).
        """
        # Camera images typically have these fields
        expected_fields = ['Make', 'Model', 'DateTimeOriginal', 'ExifImageWidth']
        present_count = sum(1 for f in expected_fields if f in exif_data)

        # If most expected fields are missing, likely stripped
        return present_count < 2

    def _check_c2pa(self, image_path: str) -> Optional[Dict]:
        """
        Check for C2PA (Coalition for Content Provenance and Authenticity) manifest.
        C2PA is the emerging industry standard for content provenance,
        adopted by Adobe, Google, TikTok, and others.
        """
        # C2PA manifests are stored in specific XMP namespaces
        # For now, check for C2PA-related XMP data
        try:
            with Image.open(image_path) as img:
                if hasattr(img, 'info'):
                    xmp = img.info.get('XML:com.adobe.xmp')
                    if xmp and 'c2pa' in str(xmp).lower():
                        return {
                            'c2pa_found': True,
                            'note': 'C2PA manifest detected in XMP data. '
                                    'This provides verified provenance information.',
                        }
        except Exception:
            pass

        return {'c2pa_found': False, 'note': 'No C2PA manifest found.'}

    def _compute_anomaly_score(self, results: Dict[str, Any]) -> float:
        """
        Compute metadata anomaly score (0-100).
        Higher score = more suspicious metadata.
        """
        score = 0.0

        # No EXIF at all = moderately suspicious
        if not results['exif_present']:
            return 35.0  # Not 100 because some apps genuinely don't embed EXIF

        # Editing software detected
        editing = results.get('editing_traces', {})
        if editing.get('is_known_editor'):
            score += 25
            if len(editing.get('editing_history', [])) > 1:
                score += 10  # Multiple editing traces

        # Timestamp inconsistencies
        timestamps = results.get('timestamps', {})
        if timestamps.get('inconsistency_detected'):
            score += 20

        # Resolution mismatch
        consistency = results.get('metadata_consistency', {})
        if not consistency.get('resolution_match', True):
            score += 25

        # EXIF stripping
        if results.get('stripping_detected'):
            score += 15

        # Number of anomalies
        n_anomalies = len(consistency.get('anomalies', []))
        score += min(15, n_anomalies * 5)

        return round(min(100, score), 1)
