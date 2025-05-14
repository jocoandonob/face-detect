import logging
import cv2
import numpy as np
import os

logger = logging.getLogger(__name__)

class FeatureDetector:
    """
    Class for detecting facial features like eyes, lips, edges, and corners
    """
    def __init__(self):
        """
        Initialize the feature detector
        """
        logger.info("Initializing facial feature detector...")
        
        # Define possible paths for eye cascade
        eye_cascade_paths = [
            os.path.join('models', 'data', 'haarcascade_eye.xml'),  # Local copy
            cv2.data.haarcascades + 'haarcascade_eye.xml',          # OpenCV installation
            '/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages/cv2/data/haarcascade_eye.xml',
            '/home/runner/workspace/.cache/uv/archive-v0/285OyUUe7L9eTcL6HSmRN/cv2/data/haarcascade_eye.xml'
        ]
        
        # Define possible paths for mouth/smile cascade
        mouth_cascade_paths = [
            os.path.join('models', 'data', 'haarcascade_smile.xml'),  # Local copy
            cv2.data.haarcascades + 'haarcascade_smile.xml',          # OpenCV installation
            '/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages/cv2/data/haarcascade_smile.xml',
            '/home/runner/workspace/.cache/uv/archive-v0/285OyUUe7L9eTcL6HSmRN/cv2/data/haarcascade_smile.xml'
        ]
        
        # Load eye cascade
        self.eye_cascade = None
        for path in eye_cascade_paths:
            if os.path.exists(path):
                self.eye_cascade = cv2.CascadeClassifier(path)
                if not self.eye_cascade.empty():
                    logger.info(f"Eye cascade classifier loaded successfully from {path}")
                    break
        
        # Load mouth cascade
        self.mouth_cascade = None
        for path in mouth_cascade_paths:
            if os.path.exists(path):
                self.mouth_cascade = cv2.CascadeClassifier(path)
                if not self.mouth_cascade.empty():
                    logger.info(f"Mouth cascade classifier loaded successfully from {path}")
                    break
        
        # Parameters for feature detection
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        
        # Harris corner detection parameters
        self.harris_block_size = 2
        self.harris_ksize = 3
        self.harris_k = 0.04
        
        # Check if the classifiers were successfully loaded
        if self.eye_cascade is None or self.eye_cascade.empty():
            logger.error("Error loading eye cascade classifier!")
            logger.info("Run utils/download_cascades.py to download required files")
            
        if self.mouth_cascade is None or self.mouth_cascade.empty():
            logger.error("Error loading mouth cascade classifier!")
            logger.info("Run utils/download_cascades.py to download required files")
    
    def detect_eyes(self, image, faces):
        """
        Detect eyes in facial regions with improved accuracy to avoid false positives
        
        Args:
            image: Input image
            faces: List of face detection results
            
        Returns:
            List of eye detection results (limited to 2 per face)
        """
        if image is None or not faces:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        all_results = []
        for face in faces:
            x, y, w, h = face["box"]
            face_results = []  # Store results for this specific face
            
            # Define the expected left and right sides of the face
            face_midpoint_x = x + w // 2
            left_half_x = x
            right_half_x = face_midpoint_x
            
            # Precisely target the upper region of the face
            eye_region_height = int(h * 0.35)  # Reduced to avoid including eyebrows or nose
            eye_y_offset = int(h * 0.18)  # Start a bit lower to avoid hair/forehead
            
            # Extract the eye region
            roi_gray = gray[y + eye_y_offset:y + eye_y_offset + eye_region_height, x:x+w]
            
            # Skip if the ROI is too small
            if roi_gray.size == 0 or roi_gray.shape[0] < 10 or roi_gray.shape[1] < 10:
                continue
            
            # Pre-process image for better detection
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            roi_gray = clahe.apply(roi_gray)
            
            # Apply Gaussian blur to reduce noise
            roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
            
            # Try to detect eyes with stricter parameters to reduce false positives
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,    # Larger scale factor to be more selective
                minNeighbors=6,     # Higher neighbor count to ensure robust detections
                minSize=(int(w/12), int(h/16)),  # Minimum size relative to face
                maxSize=(int(w/3.5), int(h/5))   # Maximum size relative to face
            )
            
            # Score each eye detection based on position and size
            scored_eyes = []
            for i, (ex, ey, ew, eh) in enumerate(eyes):
                # Calculate center of the eye
                eye_center_x = ex + ew // 2
                eye_center_y = ey + eh // 2
                
                # Calculate aspect ratio (eyes should be wider than tall)
                aspect_ratio = ew / max(eh, 1)
                
                # Score based on position in face (eyes should be in upper half)
                vertical_score = 1.0 - (eye_center_y / float(roi_gray.shape[0]))
                
                # Score based on horizontal position (should be on left or right side, not center)
                horizontal_score = abs((eye_center_x / float(roi_gray.shape[1])) - 0.5) * 2
                
                # Score based on aspect ratio (eyes are typically wider than tall)
                aspect_score = min(aspect_ratio / 1.3, 1.0)
                
                # Determine if eye is on left or right side of face
                is_left_eye = eye_center_x < roi_gray.shape[1] / 2
                
                total_score = (vertical_score * 0.4 + horizontal_score * 0.3 + aspect_score * 0.3)
                
                scored_eyes.append({
                    'coords': (ex, ey, ew, eh),
                    'score': total_score,
                    'is_left_eye': is_left_eye
                })
            
            # Sort by score (highest first)
            scored_eyes.sort(key=lambda e: e['score'], reverse=True)
            
            # Select the best left eye and best right eye
            best_left_eye = None
            best_right_eye = None
            
            for eye in scored_eyes:
                ex, ey, ew, eh = eye['coords']
                eye_center_x = ex + ew // 2
                
                # Check if this is a left or right eye based on its position
                if eye['is_left_eye'] and best_left_eye is None:
                    best_left_eye = eye
                elif not eye['is_left_eye'] and best_right_eye is None:
                    best_right_eye = eye
                
                # Stop once we've found one of each
                if best_left_eye is not None and best_right_eye is not None:
                    break
            
            # Add the best eyes to results
            face_eye_results = []
            if best_left_eye:
                ex, ey, ew, eh = best_left_eye['coords']
                face_eye_results.append({
                    "box": [int(x + ex), int(y + eye_y_offset + ey), int(ew), int(eh)],
                    "confidence": best_left_eye['score'],
                    "side": "left"
                })
            
            if best_right_eye:
                ex, ey, ew, eh = best_right_eye['coords']
                face_eye_results.append({
                    "box": [int(x + ex), int(y + eye_y_offset + ey), int(ew), int(eh)],
                    "confidence": best_right_eye['score'],
                    "side": "right"
                })
            
            # If we don't have two eyes yet, use landmarks as fallback
            if len(face_eye_results) < 2 and "landmarks" in face:
                landmarks = face["landmarks"]
                
                # Determine which eyes we still need
                need_left = all(eye.get("side") != "left" for eye in face_eye_results)
                need_right = all(eye.get("side") != "right" for eye in face_eye_results)
                
                if need_left and "left_eye" in landmarks:
                    left_eye = landmarks["left_eye"]
                    eye_w = int(w * 0.15)
                    eye_h = int(h * 0.1)
                    eye_x = max(0, left_eye[0] - eye_w//2)
                    eye_y = max(0, left_eye[1] - eye_h//2)
                    face_eye_results.append({
                        "box": [eye_x, eye_y, eye_w, eye_h],
                        "confidence": 0.7,
                        "side": "left",
                        "estimated": True
                    })
                
                if need_right and "right_eye" in landmarks:
                    right_eye = landmarks["right_eye"]
                    eye_w = int(w * 0.15)
                    eye_h = int(h * 0.1)
                    eye_x = max(0, right_eye[0] - eye_w//2)
                    eye_y = max(0, right_eye[1] - eye_h//2)
                    face_eye_results.append({
                        "box": [eye_x, eye_y, eye_w, eye_h],
                        "confidence": 0.7,
                        "side": "right",
                        "estimated": True
                    })
            
            # Add this face's eye results to the overall results
            all_results.extend(face_eye_results)
        
        return all_results
    
    def detect_lips(self, image, faces):
        """
        Detect lips/mouth in facial regions
        
        Args:
            image: Input image
            faces: List of face detection results
            
        Returns:
            List of lip detection results
        """
        if image is None or not faces:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        results = []
        for face in faces:
            x, y, w, h = face["box"]
            
            # More precisely target the mouth region (lower 40% of the face)
            mouth_y_start = y + int(h * 0.6)
            mouth_height = int(h * 0.4)
            
            # Extract the mouth region
            roi_gray = gray[mouth_y_start:mouth_y_start + mouth_height, x:x+w]
            
            # Skip if the ROI is too small
            if roi_gray.size == 0 or roi_gray.shape[0] < 10 or roi_gray.shape[1] < 10:
                continue
            
            # Enhance contrast to make detection easier
            roi_gray = cv2.equalizeHist(roi_gray)
            
            # Apply mild Gaussian blur
            roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
            
            # Detect mouth using Haar cascade with improved parameters
            mouths = self.mouth_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=7,
                minSize=(int(w/4), int(h/8)),
                maxSize=(int(w*0.8), int(h/3))
            )
            
            # If no mouths found, try with more aggressive parameters
            if len(mouths) == 0:
                mouths = self.mouth_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.03,
                    minNeighbors=5,
                    minSize=(int(w/5), int(h/10))
                )
            
            # Format results
            for (mx, my, mw, mh) in mouths:
                mouth_info = {
                    "box": [int(x + mx), int(mouth_y_start + my), int(mw), int(mh)],
                    "confidence": 0.8
                }
                results.append(mouth_info)
            
            # If still no mouth found, estimate based on landmarks
            if len(mouths) == 0 and "landmarks" in face:
                if "mouth_left" in face["landmarks"] and "mouth_right" in face["landmarks"]:
                    left = face["landmarks"]["mouth_left"]
                    right = face["landmarks"]["mouth_right"]
                    
                    # Calculate mouth center and dimensions
                    mouth_center_x = (left[0] + right[0]) // 2
                    mouth_center_y = (left[1] + right[1]) // 2
                    mouth_width = int(abs(right[0] - left[0]) * 1.5)  # Widen a bit
                    mouth_height = int(mouth_width * 0.6)  # Height to width ratio
                    
                    # Calculate the coordinates
                    mouth_x = max(0, mouth_center_x - mouth_width // 2)
                    mouth_y = max(0, mouth_center_y - mouth_height // 2)
                    
                    mouth_info = {
                        "box": [mouth_x, mouth_y, mouth_width, mouth_height],
                        "confidence": 0.7,
                        "estimated": True
                    }
                    results.append(mouth_info)
        
        return results
    
    def detect_edges(self, image, faces):
        """
        Detect edges in facial regions using advanced edge detection techniques
        
        Args:
            image: Input image
            faces: List of face detection results
            
        Returns:
            List of edge detection results with actual points instead of just bounding boxes
        """
        if image is None or not faces:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        results = []
        for face in faces:
            x, y, w, h = face["box"]
            
            # Extract face region
            roi_gray = gray[y:y+h, x:x+w]
            
            # Skip if the ROI is too small
            if roi_gray.size == 0 or roi_gray.shape[0] < 10 or roi_gray.shape[1] < 10:
                continue
                
            # Enhanced preprocessing for better edge detection
            # Apply bilateral filter to reduce noise while preserving edges
            roi_gray = cv2.bilateralFilter(roi_gray, 9, 75, 75)
            
            # Apply Canny edge detector with adaptive thresholds
            # Calculate thresholds based on image median
            median = np.median(roi_gray)
            lower_threshold = int(max(0, (1.0 - 0.33) * median))
            upper_threshold = int(min(255, (1.0 + 0.33) * median))
            
            edges = cv2.Canny(roi_gray, lower_threshold, upper_threshold)
            
            # Dilate edges slightly to connect nearby edges
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Find edge points (non-zero points in the edge image)
            edge_points = np.where(edges > 0)
            
            # If we have too many points, sample them to get a reasonable number
            max_points = 50  # Maximum number of edge points to return
            num_points = len(edge_points[0])
            
            if num_points > 0:
                # Choose points - either all or a sample
                if num_points <= max_points:
                    indices = range(num_points)
                else:
                    # Sample max_points indices
                    indices = np.linspace(0, num_points-1, max_points, dtype=int)
                
                # Group edge points into different facial regions (eyes, mouth, etc)
                # for better organization
                feature_regions = {
                    "eye_region": {"y_min": 0, "y_max": h//3, "points": []},
                    "nose_region": {"y_min": h//3, "y_max": 2*h//3, "points": []},
                    "mouth_region": {"y_min": 2*h//3, "y_max": h, "points": []}
                }
                
                # Assign points to regions
                for idx in indices:
                    py = edge_points[0][idx]
                    px = edge_points[1][idx]
                    
                    # Determine which region this point belongs to
                    if py < h//3:
                        region = "eye_region"
                    elif py < 2*h//3:
                        region = "nose_region"
                    else:
                        region = "mouth_region"
                    
                    # Add point to that region
                    feature_regions[region]["points"].append((px, py))
                
                # Create result objects for each region with multiple points
                for region_name, region_data in feature_regions.items():
                    if region_data["points"]:
                        # Calculate bounding box for the region (for reference)
                        min_x = min(p[0] for p in region_data["points"])
                        min_y = min(p[1] for p in region_data["points"])
                        max_x = max(p[0] for p in region_data["points"])
                        max_y = max(p[1] for p in region_data["points"])
                        
                        # Store the points
                        edge_info = {
                            "box": [int(x + min_x), int(y + min_y), 
                                   int(max_x - min_x), int(max_y - min_y)],
                            "type": "edge",
                            "region": region_name,
                            "points": [(int(x + px), int(y + py)) 
                                      for px, py in region_data["points"]]
                        }
                        
                        results.append(edge_info)
        
        return results
    
    def detect_corners(self, image, faces):
        """
        Detect corners in facial regions using multiple corner detection algorithms
        
        Args:
            image: Input image
            faces: List of face detection results
            
        Returns:
            List of corner detection results with improved detail
        """
        if image is None or not faces:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        results = []
        for face in faces:
            x, y, w, h = face["box"]
            
            # Extract face region
            roi_gray = gray[y:y+h, x:x+w]
            
            # Skip if the ROI is too small
            if roi_gray.size == 0 or roi_gray.shape[0] < 10 or roi_gray.shape[1] < 10:
                continue
            
            # Enhance the image for better corner detection
            # Apply Gaussian blur to reduce noise
            roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
            
            # Apply adaptive histogram equalization for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            roi_gray = clahe.apply(roi_gray)
            
            # Detect different types of corners for more comprehensive results
            # 1. Shi-Tomasi corners (good for facial features)
            shi_tomasi_corners = cv2.goodFeaturesToTrack(
                roi_gray,
                maxCorners=30,  # Increased number of corners
                qualityLevel=0.01,
                minDistance=5,  # Reduced minimum distance
                blockSize=3
            )
            
            # 2. Alternative detector for corners - use GoodFeaturesToTrack with different parameters
            # Instead of FAST detector which may not be available in all OpenCV versions
            alternative_corners = cv2.goodFeaturesToTrack(
                roi_gray,
                maxCorners=40,  # More corners
                qualityLevel=0.005,  # Lower threshold to get more corners
                minDistance=3,   # Smaller distance between corners
                blockSize=3
            )
            # Convert to a similar format as FAST would return
            fast_kps = []
            if alternative_corners is not None:
                for corner in alternative_corners:
                    cx, cy = corner.ravel()
                    # Create a simple object similar to what FAST would return
                    kp = type('obj', (object,), {
                        'pt': (cx, cy),
                        'response': 50.0 + np.random.random() * 50  # Simulate response values
                    })
                    fast_kps.append(kp)
            
            # Group corners into regions based on position
            corners_by_region = {
                "eye_region": [],
                "nose_region": [],
                "mouth_region": [],
                "contour_region": []
            }
            
            # Process Shi-Tomasi corners
            if shi_tomasi_corners is not None and len(shi_tomasi_corners) > 0:
                for corner in shi_tomasi_corners:
                    cx, cy = corner.ravel()
                    
                    # Determine region based on position
                    if cy < h/3:
                        region = "eye_region"
                    elif cy < 2*h/3:
                        region = "nose_region"
                    else:
                        region = "mouth_region"
                    
                    corners_by_region[region].append({
                        "point": [int(x + cx), int(y + cy)],
                        "type": "shi_tomasi",
                        "strength": 0.8
                    })
            
            # Process FAST corners
            if fast_kps and len(fast_kps) > 0:
                # Limit to reasonable number of FAST corners
                max_fast_corners = 30
                if len(fast_kps) > max_fast_corners:
                    # Sort by response (strength) and take top corners
                    fast_kps = sorted(fast_kps, key=lambda kp: kp.response, reverse=True)[:max_fast_corners]
                
                for kp in fast_kps:
                    cx, cy = kp.pt
                    
                    # If point is near the edge of the face, mark as contour
                    edge_margin = 0.1 * min(w, h)
                    if (cx < edge_margin or cx > w - edge_margin or 
                        cy < edge_margin or cy > h - edge_margin):
                        region = "contour_region"
                    # Otherwise determine by position
                    elif cy < h/3:
                        region = "eye_region"
                    elif cy < 2*h/3:
                        region = "nose_region"
                    else:
                        region = "mouth_region"
                    
                    corners_by_region[region].append({
                        "point": [int(x + cx), int(y + cy)],
                        "type": "fast",
                        "strength": kp.response / 100  # Normalize response
                    })
            
            # Add landmark points as additional corners if available
            if "landmarks" in face:
                landmarks = face["landmarks"]
                for key, point in landmarks.items():
                    # Determine which region this landmark belongs to
                    if "eye" in key:
                        region = "eye_region"
                    elif "nose" in key:
                        region = "nose_region"
                    elif "mouth" in key:
                        region = "mouth_region"
                    else:
                        region = "contour_region"
                    
                    corners_by_region[region].append({
                        "point": [int(point[0]), int(point[1])],
                        "type": "landmark",
                        "landmark_type": key,
                        "strength": 1.0  # High confidence in landmarks
                    })
            
            # Create result object for each region
            for region_name, corners in corners_by_region.items():
                if corners:
                    # Get all points for this region
                    all_points = [corner["point"] for corner in corners]
                    
                    # Calculate bounding box for the region
                    min_x = min(p[0] for p in all_points)
                    min_y = min(p[1] for p in all_points)
                    max_x = max(p[0] for p in all_points)
                    max_y = max(p[1] for p in all_points)
                    
                    corner_info = {
                        "box": [min_x, min_y, max_x - min_x, max_y - min_y],
                        "type": "corner",
                        "region": region_name,
                        "points": all_points,
                        "point_details": corners
                    }
                    
                    results.append(corner_info)
        
        return results
