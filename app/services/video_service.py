import cv2
import os
import numpy as np
from flask import current_app
from app.services.face_service import face_service
from app.services.embedding_service import embedding_service

class VideoService:
    def process_video(self, video_path, embeddings_folder, sample_rate=30):
        """
        Process a video, extract frames, run face recognition.
        Returns list of recognition results.
        """
        if not os.path.exists(video_path):
            return {'error': f'Video not found: {video_path}'}
        
        all_embeddings = embedding_service.load_all_embeddings(embeddings_folder)
        if not all_embeddings:
            return {'error': 'No enrolled drivers found. Please enroll drivers first.'}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Cannot open video file.'}
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results = []
        frame_num = 0
        
        thumbnails_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'thumbnails')
        os.makedirs(thumbnails_dir, exist_ok=True)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % sample_rate == 0:
                faces = face_service.detect_faces_dnn(frame)
                
                for face_info in faces:
                    emb, bbox = embedding_service.extract_embedding(frame)
                    if emb is not None:
                        driver_id, score = embedding_service.identify_face(
                            emb, all_embeddings,
                            threshold=current_app.config['FACE_RECOGNITION_THRESHOLD']
                        )
                        
                        # Save thumbnail
                        x, y, w, h = face_info['bbox']
                        thumb = frame[max(0,y-20):y+h+20, max(0,x-20):x+w+20]
                        thumb_name = f"thumb_{os.path.basename(video_path)}_{frame_num}.jpg"
                        thumb_path = os.path.join(thumbnails_dir, thumb_name)
                        cv2.imwrite(thumb_path, thumb)
                        
                        results.append({
                            'frame': frame_num,
                            'timestamp': frame_num / fps,
                            'driver_id': driver_id,
                            'confidence': score,
                            'status': 'identified' if driver_id else 'unknown',
                            'thumbnail': f"uploads/thumbnails/{thumb_name}"
                        })
            
            frame_num += 1
        
        cap.release()
        return {
            'total_frames': total_frames,
            'processed_frames': frame_num,
            'fps': fps,
            'results': results,
            'summary': self._summarize_results(results)
        }
    
    def _summarize_results(self, results):
        """Aggregate frame-level results into a driver summary."""
        driver_counts = {}
        for r in results:
            did = r['driver_id'] or 'unknown'
            if did not in driver_counts:
                driver_counts[did] = {'count': 0, 'confidences': [], 'first_frame': r['frame']}
            driver_counts[did]['count'] += 1
            driver_counts[did]['confidences'].append(r['confidence'])
        
        summary = []
        for did, data in driver_counts.items():
            summary.append({
                'driver_id': did,
                'appearances': data['count'],
                'avg_confidence': float(np.mean(data['confidences'])),
                'first_seen_frame': data['first_frame']
            })
        summary.sort(key=lambda x: x['appearances'], reverse=True)
        return summary
    
    def extract_frame(self, video_path, frame_number):
        """Extract a specific frame from video."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    
    def list_videos(self, videos_folder):
        """List all video files in the videos folder."""
        if not os.path.exists(videos_folder):
            return []
        
        videos = []
        for fname in os.listdir(videos_folder):
            ext = fname.rsplit('.', 1)[-1].lower()
            if ext in {'mp4', 'avi', 'mov', 'mkv'}:
                path = os.path.join(videos_folder, fname)
                size = os.path.getsize(path)
                videos.append({
                    'filename': fname,
                    'size': size,
                    'size_mb': round(size / 1024 / 1024, 1),
                    'path': path
                })
        return videos

video_service = VideoService()
