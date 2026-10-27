import cv2
import numpy as np
import os
import base64
from flask import current_app
from app.services.embedding_service import embedding_service

class FaceService:
    def __init__(self):
        # OpenCV DNN face detector (more accurate than Haar cascade)
        self._detector = None
    
    def _get_detector(self):
        if self._detector is None:
            prototxt = os.path.join(os.path.dirname(__file__), '..', 'static', 'models', 'deploy.prototxt')
            caffemodel = os.path.join(os.path.dirname(__file__), '..', 'static', 'models', 'res10_300x300_ssd_iter_140000.caffemodel')
            
            if os.path.exists(prototxt) and os.path.exists(caffemodel):
                self._detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            else:
                # Fallback to Haar cascade
                self._detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return self._detector
    
    def detect_faces_dnn(self, image_bgr):
        """Detect faces using InsightFace (returns bboxes)."""
        try:
            from app.services.embedding_service import embedding_service
            model = embedding_service._get_model()
            if model:
                faces = model.get(image_bgr)
                results = []
                for f in faces:
                    x1, y1, x2, y2 = [int(v) for v in f.bbox]
                    results.append({
                        'bbox': (x1, y1, x2-x1, y2-y1),
                        'confidence': float(f.det_score)
                    })
                return results
        except Exception:
            pass
        
        # Fallback: OpenCV Haar
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return [{'bbox': tuple(f), 'confidence': 0.9} for f in faces]

    
    def decode_base64_image(self, b64_string):
        """Decode a base64 image string to a numpy BGR image."""
        try:
            # Strip any data URI prefix (handles all variants)
            if ',' in b64_string:
                b64_string = b64_string.split(',', 1)[1]

            # Fix padding if missing
            b64_string = b64_string.strip()
            missing_padding = len(b64_string) % 4
            if missing_padding:
                b64_string += '=' * (4 - missing_padding)

            img_bytes = base64.b64decode(b64_string)
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img  # Returns None if decode fails, caller handles it
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"decode_base64_image failed: {e}")
            return None
    
    def crop_face(self, image_bgr, bbox, padding=0.2):
        """Crop face with padding from image."""
        x, y, w, h = bbox
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image_bgr.shape[1], x + w + pad_x)
        y2 = min(image_bgr.shape[0], y + h + pad_y)
        return image_bgr[y1:y2, x1:x2]
    
    def save_face_photo(self, image_bgr, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image_bgr)
    
    def assess_quality(self, image_bgr):
        """Assess face image quality: sharpness, brightness."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        
        quality = {
            'sharpness': float(laplacian_var),
            'brightness': float(brightness),
            'sharp_ok': laplacian_var > 100,
            'bright_ok': 40 < brightness < 220,
            'overall': 'good' if laplacian_var > 100 and 40 < brightness < 220 else 'poor'
        }
        return quality
    
    def draw_face_box(self, image_bgr, faces, label=""):
        """Draw bounding boxes on detected faces."""
        img = image_bgr.copy()
        for face in faces:
            x, y, w, h = face['bbox']
            conf = face.get('confidence', 0)
            color = (0, 220, 100)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            if label:
                cv2.putText(img, f"{label} {conf:.0%}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img
    
    def image_to_base64(self, image_bgr):
        _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

face_service = FaceService()
