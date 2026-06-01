"""
Copy-Move Forgery Detection Module
Research basis: SIFT/SURF keypoint matching + block-based overlap analysis
Implements both keypoint-based and block-based approaches for robustness
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class CopyMoveDetector:
    """
    Copy-Move (Clone) Forgery Detection.
    
    Detects when a region of an image has been copied and pasted elsewhere
    within the same image. This is one of the most common manipulation types
    (used to duplicate or hide objects).
    
    Uses a dual approach:
    1. Keypoint-based: SIFT feature matching for robust detection against
       rotation, scaling, and JPEG compression
    2. Block-based: Overlapping block analysis for detecting smooth region cloning
    """

    def __init__(self, 
                 min_match_count: int = 10,
                 ransac_threshold: float = 5.0,
                 block_size: int = 16,
                 overlap: int = 8):
        """
        Args:
            min_match_count: Minimum SIFT matches to consider as clone detection
            ransac_threshold: RANSAC reprojection threshold for filtering matches
            block_size: Block size for block-based analysis
            overlap: Overlap between adjacent blocks
        """
        self.min_match_count = min_match_count
        self.ransac_threshold = ransac_threshold
        self.block_size = block_size
        self.overlap = overlap

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Perform copy-move forgery detection.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        results = {
            'clone_detected': False,
            'clone_confidence': 0.0,
            'clone_regions': [],
            'keypoint_matches': 0,
            'heatmap': None,
            'method_used': [],
        }

        # Method 1: SIFT-based keypoint matching
        sift_results = self._sift_detection(image)
        results['method_used'].append('SIFT')
        results['keypoint_matches'] = sift_results['match_count']

        if sift_results['detected']:
            results['clone_detected'] = True
            results['clone_confidence'] = max(
                results['clone_confidence'], sift_results['confidence']
            )
            results['clone_regions'].extend(sift_results['regions'])

        # Method 2: Block-based correlation analysis
        block_results = self._block_based_detection(image)
        results['method_used'].append('Block-Correlation')

        if block_results['detected']:
            results['clone_detected'] = True
            results['clone_confidence'] = max(
                results['clone_confidence'], block_results['confidence']
            )
            results['clone_regions'].extend(block_results['regions'])

        # Generate heatmap highlighting cloned regions
        results['heatmap'] = self._generate_heatmap(image, results['clone_regions'])

        return results

    def _sift_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """
        SIFT-based copy-move detection.
        
        Process:
        1. Extract SIFT keypoints and descriptors
        2. Match features using FLANN matcher
        3. Filter matches using geometric consistency (RANSAC)
        4. Cluster matched regions to identify clone pairs
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'match_count': 0,
            'regions': [],
        }

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initialize SIFT
        try:
            sift = cv2.SIFT_create()
        except AttributeError:
            try:
                sift = cv2.xfeatures2d.SIFT_create()
            except AttributeError:
                return result

        # Detect keypoints and compute descriptors
        kp, des = sift.detectAndCompute(gray, None)

        if des is None or len(kp) < self.min_match_count:
            return result

        # Match features using FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        try:
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des, des, k=2)
        except Exception:
            return result

        # Apply Lowe's ratio test and filter self-matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # Lowe's ratio test
                if m.distance < 0.7 * n.distance:
                    # Filter self-matches (same keypoint)
                    pt1 = kp[m.queryIdx].pt
                    pt2 = kp[m.trainIdx].pt
                    dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                    if dist > 20:  # Minimum displacement to consider as copy-move
                        good_matches.append((m, n))

        result['match_count'] = len(good_matches)

        if len(good_matches) < self.min_match_count:
            return result

        # Extract matched points for RANSAC
        src_pts = np.float32([kp[m.queryIdx].pt for m, n in good_matches])
        dst_pts = np.float32([kp[m.trainIdx].pt for m, n in good_matches])

        # RANSAC to find geometric transformation
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 
                                          self.ransac_threshold)
            inliers = int(np.sum(mask)) if mask is not None else 0

            if inliers >= self.min_match_count:
                result['detected'] = True
                result['confidence'] = min(1.0, inliers / (self.min_match_count * 3))

                # Identify clone regions from inlier matches
                inlier_mask = mask.flatten().astype(bool)
                inlier_src = src_pts[inlier_mask]
                inlier_dst = dst_pts[inlier_mask]

                # Cluster into regions
                src_region = self._cluster_points(inlier_src, image.shape)
                dst_region = self._cluster_points(inlier_dst, image.shape)

                if src_region and dst_region:
                    result['regions'].append({
                        'type': 'copy_move_sift',
                        'source_region': src_region,
                        'destination_region': dst_region,
                        'inlier_count': inliers,
                    })

        except Exception:
            pass

        return result

    def _block_based_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Block-based copy-move detection using correlation analysis.
        
        Divides the image into overlapping blocks, computes features for each,
        and finds block pairs with suspiciously high correlation (indicating cloning).
        This catches smooth-region cloning that SIFT misses.
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'regions': [],
        }

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        bs = self.block_size
        step = bs - self.overlap

        if h < bs * 2 or w < bs * 2:
            return result

        # Extract block features (DCT coefficients as compact representation)
        blocks = []
        positions = []
        for y in range(0, h - bs + 1, step):
            for x in range(0, w - bs + 1, step):
                block = gray[y:y+bs, x:x+bs].astype(np.float32)
                # DCT-based feature (low-frequency coefficients)
                dct_block = cv2.dct(block)
                # Use zigzag-scan of top-left 4x4 as feature
                feature = dct_block[:4, :4].flatten()
                # Normalize
                feature = feature / (np.linalg.norm(feature) + 1e-8)
                blocks.append(feature)
                positions.append((x, y))

        if len(blocks) < 2:
            return result

        blocks = np.array(blocks)
        
        # Find similar block pairs
        # For efficiency, use a simplified approach with batched computation
        n_blocks = len(blocks)
        suspicious_pairs = []
        
        # Sample-based approach for large images
        max_compare = min(n_blocks, 2000)
        indices = np.random.choice(n_blocks, max_compare, replace=False) if n_blocks > max_compare else np.arange(n_blocks)
        
        sampled_blocks = blocks[indices]
        sampled_positions = [positions[i] for i in indices]
        
        # Compute pairwise cosine similarity
        similarity_matrix = np.dot(sampled_blocks, sampled_blocks.T)
        
        # Find high-similarity pairs that are spatially distant
        threshold = 0.95  # Very high correlation threshold
        for i in range(len(sampled_blocks)):
            for j in range(i + 1, len(sampled_blocks)):
                if similarity_matrix[i, j] > threshold:
                    pos_i = sampled_positions[i]
                    pos_j = sampled_positions[j]
                    # Must be spatially separated
                    dist = np.sqrt(
                        (pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2
                    )
                    if dist > bs * 2:  # Minimum spatial separation
                        suspicious_pairs.append((pos_i, pos_j, similarity_matrix[i, j]))

        if len(suspicious_pairs) > 5:  # Threshold for significance
            result['detected'] = True
            result['confidence'] = min(1.0, len(suspicious_pairs) / 50)

            # Cluster suspicious pairs into regions
            source_pts = [(p[0][0], p[0][1]) for p in suspicious_pairs]
            dest_pts = [(p[1][0], p[1][1]) for p in suspicious_pairs]

            src_region = self._cluster_points(
                np.array(source_pts, dtype=np.float32), image.shape
            )
            dst_region = self._cluster_points(
                np.array(dest_pts, dtype=np.float32), image.shape
            )

            if src_region and dst_region:
                result['regions'].append({
                    'type': 'copy_move_block',
                    'source_region': src_region,
                    'destination_region': dst_region,
                    'pair_count': len(suspicious_pairs),
                })

        return result

    def _cluster_points(self, points: np.ndarray, 
                        image_shape: Tuple[int, ...]) -> Optional[Dict]:
        """
        Cluster matched points into a bounding region.
        """
        if len(points) < 2:
            return None

        x_min = int(np.min(points[:, 0]))
        x_max = int(np.max(points[:, 0]))
        y_min = int(np.min(points[:, 1]))
        y_max = int(np.max(points[:, 1]))

        # Add padding
        pad = 20
        h, w = image_shape[:2]
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)

        return {
            'bbox': (x_min, y_min, x_max, y_max),
            'num_points': len(points),
            'center': (int(np.mean(points[:, 0])), int(np.mean(points[:, 1]))),
        }

    def _generate_heatmap(self, image: np.ndarray, 
                          regions: List[Dict]) -> np.ndarray:
        """
        Generate heatmap highlighting detected clone regions.
        """
        heatmap = np.zeros(image.shape[:2], dtype=np.float32)

        for region in regions:
            for key in ['source_region', 'destination_region']:
                if key in region and region[key]:
                    bbox = region[key]['bbox']
                    x1, y1, x2, y2 = bbox
                    # Draw filled region on heatmap
                    heatmap[y1:y2, x1:x2] = 1.0

        # Apply Gaussian blur for smooth visualization
        if heatmap.max() > 0:
            heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)

        # Colorize
        colorized = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay on original image
        overlay = cv2.addWeighted(image, 0.6, colorized, 0.4, 0)

        return overlay
