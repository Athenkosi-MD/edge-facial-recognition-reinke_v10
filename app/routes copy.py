from flask import Blueprint, render_template, request, jsonify, current_app, redirect, url_for
from werkzeug.utils import secure_filename
from app import db
from app.models import Driver, RecognitionEvent
from app.services.face_service import face_service
from app.services.embedding_service import embedding_service
from app.services.video_service import video_service
import os
import cv2
import numpy as np
import base64
import json

# ── Custom JSON encoder so stray numpy types never crash the app ──────────────
from flask.json.provider import DefaultJSONProvider

class _NumpySafeProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

main = Blueprint('main', __name__)

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

# ─── Dashboard ───────────────────────────────────────────────────────────────

@main.route('/')
def index():
    return redirect(url_for('main.dashboard'))

@main.route('/dashboard')
def dashboard():
    total_drivers = Driver.query.filter_by(status='active').count()
    total_events = RecognitionEvent.query.count()
    recent_events = RecognitionEvent.query.order_by(RecognitionEvent.timestamp.desc()).limit(5).all()
    recent_drivers = Driver.query.order_by(Driver.enrollment_date.desc()).limit(4).all()
    return render_template('dashboard.html',
        total_drivers=total_drivers,
        total_events=total_events,
        recent_events=recent_events,
        recent_drivers=recent_drivers
    )

# ─── Drivers List ────────────────────────────────────────────────────────────

@main.route('/drivers')
def drivers():
    all_drivers = Driver.query.order_by(Driver.enrollment_date.desc()).all()
    return render_template('drivers.html', drivers=all_drivers)

@main.route('/api/drivers/<int:driver_id>', methods=['DELETE'])
def delete_driver(driver_id):
    driver = Driver.query.get_or_404(driver_id)
    # Remove embedding file
    if driver.embedding_path and os.path.exists(driver.embedding_path):
        os.remove(driver.embedding_path)
    db.session.delete(driver)
    db.session.commit()
    return jsonify({'success': True})

@main.route('/api/drivers/<int:driver_id>/toggle', methods=['POST'])
def toggle_driver(driver_id):
    driver = Driver.query.get_or_404(driver_id)
    driver.status = 'inactive' if driver.status == 'active' else 'active'
    db.session.commit()
    return jsonify({'success': True, 'status': driver.status})

# ─── Enrollment ──────────────────────────────────────────────────────────────

@main.route('/enroll')
def enroll():
    return render_template('enroll.html')

@main.route('/api/detect-face', methods=['POST'])
def detect_face():
    """Called from webcam stream to detect and validate face."""
    data = request.json
    b64 = data.get('image')
    if not b64:
        return jsonify({'error': 'No image provided'}), 400
    
    image = face_service.decode_base64_image(b64)
    if image is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    faces = face_service.detect_faces_dnn(image)
    
    if not faces:
        return jsonify({'detected': False, 'message': 'No face detected'})
    if len(faces) > 1:
        return jsonify({'detected': False, 'message': 'Multiple faces detected - please ensure only one person is in frame'})
    
    face = faces[0]
    quality = face_service.assess_quality(image)

    # Draw box and return annotated image
    annotated = face_service.draw_face_box(image, faces)
    annotated_b64 = face_service.image_to_base64(annotated)

# Explicitly cast everything to JSON-safe Python types
    # (OpenCV / numpy can return int32, float32, bool_ etc.)
    # Handle quality values that may be strings (like 'poor', 'good') or numbers
    quality_safe = {}
    for k, v in quality.items():
        if isinstance(v, (bool, np.bool_)):
            quality_safe[k] = bool(v)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            quality_safe[k] = float(v)
        else:
            quality_safe[k] = str(v)  # Keep strings as strings
    
    return jsonify({
        'detected':       True,
        'confidence':     float(face['confidence']),
        'quality':        quality_safe,
        'annotated_image': 'data:image/jpeg;base64,' + annotated_b64,
        'bbox':           [int(v) for v in face['bbox']]
    })

@main.route('/api/enroll', methods=['POST'])
def enroll_driver():
    """Complete enrollment - save driver + compute embedding."""
    import traceback

    try:
        # ── Parse form data ───────────────────────────────────────────────────
        first_name     = request.form.get('first_name', '').strip()
        last_name      = request.form.get('last_name', '').strip()
        email          = request.form.get('email', '').strip()
        phone          = request.form.get('phone', '').strip()
        license_number = request.form.get('license_number', '').strip()
        images_json    = request.form.get('captured_images', '[]')

        if not first_name or not last_name:
            return jsonify({'error': 'First name and last name are required'}), 400

        try:
            captured_images = json.loads(images_json)
        except Exception:
            captured_images = []

        uploaded_files = request.files.getlist('photos')

        # ── Decode all images ─────────────────────────────────────────────────
        all_images = []

        for idx, b64_str in enumerate(captured_images):
            try:
                img = face_service.decode_base64_image(b64_str)
                if img is not None:
                    all_images.append(img)
            except Exception as e:
                current_app.logger.warning(f"[ENROLL] Skipping webcam image {idx}: {e}")

        for f in uploaded_files:
            if f and allowed_file(f.filename, current_app.config['ALLOWED_EXTENSIONS']):
                try:
                    img_bytes = f.read()
                    arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        all_images.append(img)
                except Exception as e:
                    current_app.logger.warning(f"[ENROLL] Skipping uploaded file '{f.filename}': {e}")

        current_app.logger.info(f"[ENROLL] Total images collected: {len(all_images)}")

        min_photos = current_app.config['ENROLLMENT_MIN_PHOTOS']
        if len(all_images) < min_photos:
            return jsonify({'error': f'At least {min_photos} face photos are required. Got {len(all_images)}.'}), 400

        # ── Ensure InsightFace model loads before we proceed ──────────────────
        current_app.logger.info("[ENROLL] Loading InsightFace model...")
        try:
            from app.services.embedding_service import embedding_service as emb_svc
            model = emb_svc._get_model()
            if model is None:
                return jsonify({'error': 'Face recognition model failed to load. Check server logs.'}), 500
            current_app.logger.info("[ENROLL] InsightFace model ready.")
        except Exception as e:
            current_app.logger.error(f"[ENROLL] Model load error: {e}\n{traceback.format_exc()}")
            return jsonify({'error': f'Face recognition model error: {str(e)}'}), 500

        # ── Generate driver ID ────────────────────────────────────────────────
        driver_id_str = Driver.generate_driver_id()
        current_app.logger.info(f"[ENROLL] Generated driver ID: {driver_id_str}")

        # ── Create photos directory ───────────────────────────────────────────
        driver_photos_dir = os.path.join(current_app.config['DRIVER_PHOTOS_FOLDER'], driver_id_str)
        os.makedirs(driver_photos_dir, exist_ok=True)

        # ── Extract embeddings ────────────────────────────────────────────────
        embeddings = []
        profile_photo_rel = None
        max_photos = current_app.config['ENROLLMENT_MAX_PHOTOS']

        for i, img in enumerate(all_images[:max_photos]):
            try:
                current_app.logger.info(f"[ENROLL] Processing image {i+1}/{min(len(all_images), max_photos)}")
                faces = face_service.detect_faces_dnn(img)
                if not faces:
                    current_app.logger.warning(f"[ENROLL] No face detected in image {i}")
                    continue

                emb, bbox = embedding_service.extract_embedding(img)
                if emb is None:
                    current_app.logger.warning(f"[ENROLL] No embedding extracted from image {i}")
                    continue

                embeddings.append(emb)
                photo_name = f"photo_{i:03d}.jpg"
                photo_path = os.path.join(driver_photos_dir, photo_name)
                cv2.imwrite(photo_path, img)

                if i == 0:
                    profile_photo_rel = f"../data/drivers/{driver_id_str}/{photo_name}"

                current_app.logger.info(f"[ENROLL] ✓ Embedding {len(embeddings)} extracted")

            except Exception as e:
                current_app.logger.error(f"[ENROLL] Error processing image {i}: {e}\n{traceback.format_exc()}")
                continue  # Skip bad images, don't crash

        current_app.logger.info(f"[ENROLL] Total valid embeddings: {len(embeddings)}")

        if len(embeddings) < 3:
            return jsonify({
                'error': (
                    f'Could not extract enough valid face embeddings ({len(embeddings)}/3 succeeded). '
                    'Please retake photos with better lighting and a clear, front-facing view.'
                )
            }), 400

        # ── Save embeddings ───────────────────────────────────────────────────
        embedding_path = os.path.join(current_app.config['EMBEDDINGS_FOLDER'], f"{driver_id_str}.pkl")
        try:
            embedding_service.save_embeddings(driver_id_str, embeddings, embedding_path)
            current_app.logger.info(f"[ENROLL] Embeddings saved to {embedding_path}")
        except Exception as e:
            current_app.logger.error(f"[ENROLL] Failed to save embeddings: {e}\n{traceback.format_exc()}")
            return jsonify({'error': f'Failed to save face data: {str(e)}'}), 500

        # ── Save profile photo ────────────────────────────────────────────────
        static_photo = None
        if all_images:
            try:
                static_photo_name = f"{driver_id_str}_profile.jpg"
                static_photo_path = os.path.join(current_app.config['UPLOAD_FOLDER'], static_photo_name)
                os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
                cv2.imwrite(static_photo_path, all_images[0])
                static_photo = f"uploads/{static_photo_name}"
                current_app.logger.info(f"[ENROLL] Profile photo saved: {static_photo_path}")
            except Exception as e:
                current_app.logger.warning(f"[ENROLL] Could not save profile photo: {e}")
                # Non-fatal — continue without profile photo

        # ── Create DB record ──────────────────────────────────────────────────
        try:
            driver = Driver(
                driver_id=driver_id_str,
                first_name=first_name,
                last_name=last_name,
                email=email or None,
                phone=phone or None,
                license_number=license_number or None,
                profile_photo=static_photo,
                embedding_path=embedding_path,
                photo_count=len(embeddings)
            )
            db.session.add(driver)
            db.session.commit()
            current_app.logger.info(f"[ENROLL] ✓ Driver '{driver_id_str}' saved to DB.")
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"[ENROLL] DB error: {e}\n{traceback.format_exc()}")
            return jsonify({'error': f'Database error: {str(e)}'}), 500

        return jsonify({
            'success': True,
            'driver': driver.to_dict(),
            'message': f'Driver {first_name} {last_name} enrolled successfully with {len(embeddings)} face samples'
        })

    except Exception as e:
        current_app.logger.error(f"[ENROLL] Unexpected error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Unexpected server error: {str(e)}'}), 500

# ─── Recognition ─────────────────────────────────────────────────────────────

@main.route('/recognition')
def recognition():
    videos = video_service.list_videos(current_app.config['VIDEOS_FOLDER'])
    return render_template('recognition.html', videos=videos)

@main.route('/api/recognize-video', methods=['POST'])
def recognize_video():
    """Run recognition on a selected or uploaded video."""
    video_filename = request.form.get('video_filename')
    
    if 'video_file' in request.files and request.files['video_file'].filename:
        vf = request.files['video_file']
        if allowed_file(vf.filename, current_app.config['ALLOWED_VIDEO_EXTENSIONS']):
            filename = secure_filename(vf.filename)
            video_path = os.path.join(current_app.config['VIDEOS_FOLDER'], filename)
            vf.save(video_path)
            video_filename = filename
    
    if not video_filename:
        return jsonify({'error': 'No video provided'}), 400
    
    video_path = os.path.join(current_app.config['VIDEOS_FOLDER'], video_filename)
    sample_rate = int(request.form.get('sample_rate', 30))
    
    result = video_service.process_video(
        video_path,
        current_app.config['EMBEDDINGS_FOLDER'],
        sample_rate=sample_rate
    )
    
    if 'error' in result:
        return jsonify(result), 400
    
    # Save events to DB
    for r in result.get('results', []):
        if r['driver_id']:
            driver = Driver.query.filter_by(driver_id=r['driver_id']).first()
            if driver:
                event = RecognitionEvent(
                    driver_id=driver.id,
                    video_filename=video_filename,
                    confidence=r['confidence'],
                    frame_number=r['frame'],
                    status=r['status'],
                    thumbnail_path=r.get('thumbnail')
                )
                db.session.add(event)
    
    db.session.commit()
    
    # Enrich summary with driver info
    for s in result.get('summary', []):
        if s['driver_id'] != 'unknown':
            d = Driver.query.filter_by(driver_id=s['driver_id']).first()
            s['driver_name'] = f"{d.first_name} {d.last_name}" if d else s['driver_id']
            s['profile_photo'] = d.profile_photo if d else None
        else:
            s['driver_name'] = 'Unknown Person'
            s['profile_photo'] = None
    
    return jsonify(result)

@main.route('/api/recognize-image', methods=['POST'])
def recognize_image():
    """Recognize driver from a single image (webcam or upload)."""
    b64 = request.json.get('image') if request.is_json else None
    
    if not b64:
        return jsonify({'error': 'No image provided'}), 400
    
    image = face_service.decode_base64_image(b64)
    if image is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    emb, bbox = embedding_service.extract_embedding(image)
    if emb is None:
        return jsonify({'detected': False, 'message': 'No face detected in image'})
    
    all_embeddings = embedding_service.load_all_embeddings(current_app.config['EMBEDDINGS_FOLDER'])
    driver_id, score = embedding_service.identify_face(
        emb, all_embeddings,
        threshold=current_app.config['FACE_RECOGNITION_THRESHOLD']
    )
    
    result = {'detected': True, 'confidence': score}
    
    if driver_id:
        driver = Driver.query.filter_by(driver_id=driver_id).first()
        if driver:
            result['identified'] = True
            result['driver'] = driver.to_dict()
        else:
            result['identified'] = False
    else:
        result['identified'] = False
        result['message'] = 'Face detected but not recognized'
    
    return jsonify(result)
