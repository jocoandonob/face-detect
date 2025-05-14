import os
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Class for face detection using OpenCV's Haar Cascade
    """
    def __init__(self):
        """
        Initialize the face detector with a pre-trained model
        """
        logger.info("Initializing face detector...")
        
        # Using OpenCV's pre-trained face detector (Haar Cascade)
        # Try different paths for the cascade file
        cascade_paths = [
            os.path.join('models', 'data', 'haarcascade_frontalface_default.xml'),  # Local copy
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',          # OpenCV installation
            '/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml',
            '/home/runner/workspace/.cache/uv/archive-v0/285OyUUe7L9eTcL6HSmRN/cv2/data/haarcascade_frontalface_default.xml'
        ]
        
        self.face_cascade = None
        for path in cascade_paths:
            if os.path.exists(path):
                self.face_cascade = cv2.CascadeClassifier(path)
                if not self.face_cascade.empty():
                    logger.info(f"Face cascade classifier loaded successfully from {path}")
                    break
        
        # Check if the classifier was successfully loaded
        if self.face_cascade is None or self.face_cascade.empty():
            logger.error("Error loading face cascade classifier!")
            logger.info("Run utils/download_cascades.py to download required files")
    
    def detect_faces(self, image):
        """
        Detect faces in the input image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            List of dictionaries with face information (coordinates, confidence)
        """
        if image is None:
            logger.error("Input image is None")
            return []
        
        if self.face_cascade is None or self.face_cascade.empty():
            logger.error("Face cascade classifier not loaded")
            return []
        
        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Haar cascade to detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Format results
        results = []
        for (x, y, w, h) in faces:
            face_info = {
                "box": [int(x), int(y), int(w), int(h)],
                "confidence": 0.9,  # Haar cascade doesn't provide confidence scores
                "landmarks": {
                    "nose": [int(x + w/2), int(y + h/2)],
                    "mouth_left": [int(x + w/4), int(y + 3*h/4)],
                    "mouth_right": [int(x + 3*w/4), int(y + 3*h/4)],
                    "left_eye": [int(x + w/4), int(y + h/3)],
                    "right_eye": [int(x + 3*w/4), int(y + h/3)]
                }
            }
            results.append(face_info)
        
        return results
