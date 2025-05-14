import os
import logging
import numpy as np
import io
from flask import Flask, request, jsonify, render_template, abort
import cv2
from models.face_detector import FaceDetector
from models.feature_detector import FeatureDetector
from utils.image_processing import preprocess_image, encode_image_to_base64

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")

# Initialize detectors
face_detector = FaceDetector()
feature_detector = FeatureDetector()

@app.route('/', methods=['GET'])
def read_root():
    """
    Serves the main HTML page for the face detection application.
    """
    return render_template("index.html")

@app.route('/api/detect', methods=['POST'])
def detect_face():
    """
    API endpoint to detect faces and facial features in an uploaded image.
    
    Parameters:
    - file: Image file to analyze (JPG, PNG)
    
    Returns:
    - JSON object with detection results
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    
    # Check content type if available, or validate by filename
    content_type = getattr(file, 'content_type', '')
    filename = file.filename or ''
    
    if not (content_type and content_type.startswith("image/")) and not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
        return jsonify({"error": "File must be an image (JPG, PNG, or BMP)"}), 400
    
    try:
        # Read and process image
        content = file.read()
        image = preprocess_image(content)
        
        if image is None or image.size == 0:
            return jsonify({"error": "Invalid image provided"}), 400
        
        # Run detections
        face_results = face_detector.detect_faces(image)
        
        if not face_results:
            return jsonify({
                "faces": [],
                "eyes": [],
                "lips": [],
                "edges": [],
                "corners": []
            })
        
        # Process facial features
        eye_results = feature_detector.detect_eyes(image, face_results)
        lip_results = feature_detector.detect_lips(image, face_results)
        edge_results = feature_detector.detect_edges(image, face_results)
        corner_results = feature_detector.detect_corners(image, face_results)
        
        # Format response
        response = {
            "faces": face_results,
            "eyes": eye_results,
            "lips": lip_results,
            "edges": edge_results,
            "corners": corner_results
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

@app.route('/api/process_image', methods=['POST'])
def process_image():
    """
    Process an image and return the processed image with detections
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    
    # Check content type if available, or validate by filename
    content_type = getattr(file, 'content_type', '')
    filename = file.filename or ''
    
    if not (content_type and content_type.startswith("image/")) and not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
        return jsonify({"error": "File must be an image (JPG, PNG, or BMP)"}), 400
    
    try:
        # Read and process image
        content = file.read()
        image = preprocess_image(content)
        
        if image is None or image.size == 0:
            return jsonify({"error": "Invalid image provided"}), 400
        
        # Detect faces
        faces = face_detector.detect_faces(image)
        
        # Draw face rectangles
        result_image = image.copy()
        for face in faces:
            x, y, w, h = face["box"]
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Get facial features for this face
            eyes = feature_detector.detect_eyes(image, [face])
            lips = feature_detector.detect_lips(image, [face])
            
            # Draw eyes with different colors based on side
            for eye in eyes:
                ex, ey, ew, eh = eye["box"]
                # Use different colors for left and right eyes
                if eye.get("side") == "left":
                    color = (255, 165, 0)  # Orange for left eye
                elif eye.get("side") == "right":
                    color = (0, 165, 255)  # Blue for right eye
                else:
                    color = (255, 0, 0)    # Red for unclassified
                
                # Draw rectangle
                cv2.rectangle(result_image, (ex, ey), (ex+ew, ey+eh), color, 2)
                
                # Add a dot at the center for better visualization
                center_x = ex + ew//2
                center_y = ey + eh//2
                cv2.circle(result_image, (center_x, center_y), 2, color, -1)
                
            # Draw lips
            for lip in lips:
                lx, ly, lw, lh = lip["box"]
                cv2.rectangle(result_image, (lx, ly), (lx+lw, ly+lh), (0, 0, 255), 2)
        
        # Encode the processed image to base64
        encoded_image = encode_image_to_base64(result_image)
        
        return jsonify({"processed_image": encoded_image})
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

@app.errorhandler(Exception)
def global_exception_handler(exc):
    """
    Global exception handler for the application
    """
    logger.error(f"Global exception: {str(exc)}")
    return jsonify({"error": f"An unexpected error occurred: {str(exc)}"}), 500