import cv2
import numpy as np
import base64

def preprocess_image(content):
    """
    Process raw image content into a format usable by OpenCV
    
    Args:
        content: Raw image content (bytes)
        
    Returns:
        Processed image as numpy array
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(content, np.uint8)
    
    # Decode the image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Ensure the image was properly decoded
    if image is None:
        return None
    
    # Resize if too large (to improve performance)
    max_dimension = 800
    height, width = image.shape[:2]
    
    if height > max_dimension or width > max_dimension:
        # Calculate the ratio
        if height > width:
            ratio = max_dimension / height
            new_height = max_dimension
            new_width = int(width * ratio)
        else:
            ratio = max_dimension / width
            new_width = max_dimension
            new_height = int(height * ratio)
        
        # Resize the image
        image = cv2.resize(image, (new_width, new_height))
    
    return image

def encode_image_to_base64(image):
    """
    Encode an OpenCV image to base64 string
    
    Args:
        image: OpenCV image (numpy array)
        
    Returns:
        Base64 encoded string of the image
    """
    # Encode the image to JPG format
    _, buffer = cv2.imencode('.jpg', image)
    
    # Convert to base64
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    return encoded_image

def draw_detections(image, detections):
    """
    Draw detection results on an image with improved visualization
    
    Args:
        image: Original image
        detections: Dictionary of detection results
        
    Returns:
        Image with detections drawn
    """
    result = image.copy()
    
    # Draw face detections
    for face in detections.get('faces', []):
        x, y, w, h = face['box']
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add face label
        confidence = face.get('confidence', 0) * 100
        label = f"Face: {confidence:.1f}%"
        cv2.putText(result, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw landmarks if available
        landmarks = face.get('landmarks', {})
        for point_name, (px, py) in landmarks.items():
            color = (0, 0, 255)  # Default red
            if 'eye' in point_name:
                color = (255, 0, 0)  # Blue for eyes
            elif 'mouth' in point_name:
                color = (0, 0, 255)  # Red for mouth
            elif 'nose' in point_name:
                color = (0, 255, 255)  # Yellow for nose
                
            cv2.circle(result, (px, py), 3, color, -1)
            # Add small labels for landmark points
            cv2.putText(result, point_name, (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Draw eye detections with different colors for left and right eyes
    for eye in detections.get('eyes', []):
        x, y, w, h = eye['box']
        
        # Use different colors for left and right eyes
        if eye.get('side') == 'left':
            color = (255, 165, 0)  # Orange for left eye
            label = "Left Eye"
        elif eye.get('side') == 'right':
            color = (0, 165, 255)  # Blue for right eye
            label = "Right Eye"
        else:
            color = (255, 0, 0)    # Red for unclassified eye
            label = "Eye"
            
        # Draw rectangle for eye
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        
        # Add center dot
        center_x = x + w//2
        center_y = y + h//2
        cv2.circle(result, (center_x, center_y), 2, color, -1)
        
        # Add label with confidence if available
        confidence = eye.get('confidence', 0)
        if isinstance(confidence, float):
            confidence *= 100  # Convert to percentage
            label = f"{label}: {confidence:.1f}%"
        
        cv2.putText(result, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw lip detections
    for lip in detections.get('lips', []):
        x, y, w, h = lip['box']
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Add label
        confidence = lip.get('confidence', 0)
        if isinstance(confidence, float):
            confidence *= 100  # Convert to percentage
            label = f"Mouth: {confidence:.1f}%"
            cv2.putText(result, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Draw edge detections with multiple points instead of just bounding boxes
    for edge in detections.get('edges', []):
        # Get region information
        region = edge.get('region', 'unknown')
        
        # Different color for each region
        if 'eye' in region:
            color = (255, 0, 0)  # Blue for eye regions
        elif 'nose' in region:
            color = (0, 255, 255)  # Yellow for nose regions
        elif 'mouth' in region:
            color = (0, 0, 255)  # Red for mouth regions
        else:
            color = (0, 165, 255)  # Orange for other regions
        
        # Draw a bounding box for reference (with reduced opacity)
        x, y, w, h = edge['box']
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 1)
        
        # Label the region
        cv2.putText(result, f"Edges: {region}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw individual edge points
        if 'points' in edge:
            for point in edge['points']:
                px, py = point
                # Draw a small dot for each edge point
                cv2.circle(result, (px, py), 1, color, -1)
    
    # Draw corner points with improved visualization
    for corner in detections.get('corners', []):
        # Get region information
        region = corner.get('region', 'unknown')
        
        # Different base color for each region
        if 'eye' in region:
            base_color = (255, 165, 0)  # Orange for eye regions
        elif 'nose' in region:
            base_color = (0, 255, 0)  # Green for nose regions
        elif 'mouth' in region:
            base_color = (255, 0, 255)  # Magenta for mouth regions
        elif 'contour' in region:
            base_color = (255, 255, 255)  # White for contour regions
        else:
            base_color = (0, 255, 255)  # Yellow for other regions
        
        # Draw a light bounding box for reference
        if 'box' in corner:
            x, y, w, h = corner['box']
            # Use a transparent/thin box 
            cv2.rectangle(result, (x, y), (x+w, y+h), base_color, 1)
            
            # Label the region
            cv2.putText(result, f"Corners: {region}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, base_color, 1)
        
        # Draw individual corner points with type-specific styling
        if 'points' in corner:
            for i, point in enumerate(corner['points']):
                px, py = point
                
                # Get point details if available
                point_detail = corner.get('point_details', [])[i] if i < len(corner.get('point_details', [])) else None
                point_type = point_detail.get('type', 'generic') if point_detail else 'generic'
                point_strength = point_detail.get('strength', 0.5) if point_detail else 0.5
                
                # Adjust color based on point type
                if point_type == 'landmark':
                    # Special color for landmarks
                    color = (255, 255, 0)  # Yellow
                    size = 3  # Larger size for landmarks
                    label = point_detail.get('landmark_type', '') if point_detail else ''
                elif point_type == 'shi_tomasi':
                    color = (0, 255, 0)  # Green 
                    size = 2
                    label = ''
                elif point_type == 'fast':
                    color = (0, 165, 255)  # Orange
                    size = 2
                    label = ''
                else:
                    color = base_color
                    size = 2
                    label = ''
                
                # Draw the corner point with size based on strength
                radius = int(1 + point_strength * 3)  # Scale radius with strength
                cv2.circle(result, (px, py), radius, color, -1)
                
                # For important points, add tiny labels
                if label:
                    cv2.putText(result, label, (px+4, py+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                    
        # For backwards compatibility, also handle the old format
        elif 'point' in corner:
            px, py = corner['point']
            # Different colors based on corner type
            if corner.get('type') == 'landmark':
                # For landmark corners, use cyan
                color = (255, 255, 0)
                label = corner.get('landmark_type', 'landmark')
            else:
                # For detected corners, use yellow
                color = (0, 255, 255)
                label = 'corner'
                
            cv2.circle(result, (px, py), 3, color, -1)
            
            # Add tiny label
            cv2.putText(result, label, (px+4, py+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    return result
