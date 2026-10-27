from flask import Blueprint, render_template, request, jsonify, current_app, redirect, url_for, session
from werkzeug.utils import secure_filename
from app import db
from app.models import (Driver, RecognitionEvent, User, AuditLog, CustomField, DriverCustomFieldValue, Trip,
                        PipelineRun, DriverStateHistory)
from app.services import lifecycle_service as lifecycle
from app.services.face_service import face_service
from app.services.embedding_service import embedding_service
from app.services.video_service import video_service
from app.services.storage_service import (
    upload_driver_photo, save_embeddings as r2_save_embeddings,
    load_all_embeddings as r2_load_all_embeddings, upload_profile_photo
)
import os
import cv2
import numpy as np
import base64
import json
from datetime import datetime, timedelta
from sqlalchemy import func, case, or_

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

# ── Auth guard ────────────────────────────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated

def get_current_user():
    if 'user_id' in session:
        return User.query.filter_by(user_id=session['user_id']).first()
    return None


def permission_required(permission):
    """Require a User permission property (e.g. 'can_view_drivers').

    API routes (/api/...) get a JSON 403; pages flash a message and go back
    to the dashboard.  Use below @login_required."""
    from functools import wraps
    from flask import flash

    def wrapper(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            current = get_current_user()
            if not current or not current.is_active:
                session.clear()
                if request.path.startswith('/api/'):
                    return jsonify({'error': 'Not logged in'}), 401
                return redirect(url_for('auth.login'))
            if not getattr(current, permission, False):
                if request.path.startswith('/api/'):
                    return jsonify({'error': 'Insufficient permissions'}), 403
                flash('You do not have permission to access that page.', 'error')
                return redirect(url_for('main.dashboard'))
            return f(*args, **kwargs)
        return decorated
    return wrapper


def log_action(action, target_type=None, target_id=None, details=None):
    try:
        from app.models import AuditLog as AL
        entry = AL(
            user_id=session.get('user_id'),
            username=session.get('username', 'unknown'),
            action=action,
            target_type=target_type,
            target_id=str(target_id) if target_id else None,
            details=details,
            ip_address=request.remote_addr,
        )
        db.session.add(entry)
        db.session.commit()
    except Exception as e:
        current_app.logger.warning(f'[AUDIT] {e}')


def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

# ── Webcam image pre-processor (mirrors image_preparation.py logic) ───────────
_WEBCAM_MAX_LONG_SIDE = 480   # matches DEFAULT_MAX_LONG_SIDE in image_preparation.py
_WEBCAM_MAX_ASPECT    = 2.0   # matches DEFAULT_MAX_ASPECT
_WEBCAM_JPEG_QUALITY  = 85    # matches DEFAULT_JPEG_QUALITY

def _prepare_webcam_image(image_bgr):
    """
    Resize and normalise a raw webcam BGR frame before enrollment.
    Mirrors the transformations in image_preparation.py so webcam photos
    are treated identically to manually uploaded enrollment photos.

    Steps:
      1. Fix extreme aspect ratios (crop excess height/width)
      2. Downscale so the longest side ≤ _WEBCAM_MAX_LONG_SIDE
      3. Re-encode as JPEG at _WEBCAM_JPEG_QUALITY to keep payload small
    """
    if image_bgr is None:
        return None

    h, w = image_bgr.shape[:2]

    # Step 1 — aspect-ratio crop (same logic as fix_aspect_ratio in image_preparation.py)
    ratio = (h / w) if h > w else (w / h)
    if ratio > _WEBCAM_MAX_ASPECT:
        if h > w:                          # too tall — keep top (face is usually there)
            new_h = int(w * _WEBCAM_MAX_ASPECT)
            image_bgr = image_bgr[:new_h, :]
        else:                              # too wide — keep centred strip
            new_w = int(h * _WEBCAM_MAX_ASPECT)
            left = (w - new_w) // 2
            image_bgr = image_bgr[:, left:left + new_w]
        h, w = image_bgr.shape[:2]

    # Step 2 — downscale (same logic as resize_image in image_preparation.py)
    long_side = max(h, w)
    if long_side > _WEBCAM_MAX_LONG_SIDE:
        scale = _WEBCAM_MAX_LONG_SIDE / long_side
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Step 3 — re-encode at reduced JPEG quality to shrink payload further
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, _WEBCAM_JPEG_QUALITY]
    ok, buf = cv2.imencode('.jpg', image_bgr, encode_params)
    if not ok:
        return image_bgr  # fall back to uncompressed if encode fails
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# ─── Dashboard ───────────────────────────────────────────────────────────────

@main.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return redirect(url_for('main.dashboard'))

@main.route('/dashboard')
@login_required
def dashboard():
    from app.services.storage_service import get_profile_photo_url
    import os
    total_drivers = Driver.query.filter_by(status='active').filter(
        Driver.enrollment_state.in_(['AUTO_ENROLLED', 'FULLY_ENROLLED'])).count()
    total_events = RecognitionEvent.query.count()
    recent_events = RecognitionEvent.query.order_by(RecognitionEvent.timestamp.desc()).limit(5).all()
    recent_drivers = Driver.query.order_by(Driver.enrollment_date.desc()).limit(4).all()
    for d in recent_drivers:
        if d.profile_photo:
            signed = get_profile_photo_url(d.driver_id)
            local_path = os.path.join(current_app.root_path, 'static', d.profile_photo)
            if signed:
                d.photo_url = signed
            elif os.path.exists(local_path):
                d.photo_url = f"/static/{d.profile_photo}"
            else:
                d.photo_url = None
        else:
            d.photo_url = None
    return render_template('dashboard.html',
        total_drivers=total_drivers,
        total_events=total_events,
        recent_events=recent_events,
        recent_drivers=recent_drivers
    )

# ─── Drivers List ────────────────────────────────────────────────────────────

@main.route('/drivers')
@login_required
@permission_required('can_view_drivers')
def drivers():
    from app.services.storage_service import get_profile_photo_url
    all_drivers = Driver.query.order_by(Driver.enrollment_date.desc()).all()
    # Attach a signed URL (or local static path fallback) to each driver object
    for d in all_drivers:
        if d.profile_photo:
            # If stored as an R2 key like "uploads/DRV-xxx_profile.jpg"
            if d.profile_photo.startswith('uploads/') or not d.profile_photo.startswith('/'):
                driver_id_str = d.driver_id
                signed = get_profile_photo_url(driver_id_str)
                # Fallback to local static file if R2 isn't reachable
                import os
                local_path = os.path.join(current_app.root_path, 'static', d.profile_photo)
                if signed:
                    d.photo_url = signed
                elif os.path.exists(local_path):
                    d.photo_url = f"/static/{d.profile_photo}"
                else:
                    d.photo_url = None
            else:
                d.photo_url = d.profile_photo
        else:
            d.photo_url = None
    state_counts, status_counts = {}, {}
    for d in all_drivers:
        state_counts[d.state] = state_counts.get(d.state, 0) + 1
        st_ = 'deleted' if d.is_deleted else 'active'
        status_counts[st_] = status_counts.get(st_, 0) + 1
    return render_template('drivers.html', drivers=all_drivers, state_counts=state_counts,
                           status_counts=status_counts, initial_state=request.args.get('state', ''))

@main.route('/api/drivers/<int:driver_id>', methods=['DELETE'])
@login_required
@permission_required('can_delete_drivers')
def delete_driver(driver_id):
    from app.services.storage_service import delete_driver_all
    driver = Driver.query.get_or_404(driver_id)
    driver_id_str = driver.driver_id
    # Remove local embedding file if it exists
    if driver.embedding_path and os.path.exists(driver.embedding_path):
        try:
            os.remove(driver.embedding_path)
        except Exception:
            pass
    # Remove all R2 photos and embeddings
    try:
        delete_driver_all(driver_id_str)
    except Exception as e:
        current_app.logger.warning(f'[DELETE] R2 cleanup partial error for {driver_id_str}: {e}')
    # Delete custom field values
    try:
        from app.models import DriverCustomFieldValue
        DriverCustomFieldValue.query.filter_by(driver_id=driver.id).delete()
    except Exception:
        pass
    # The recognizer's templates (embeddings_buffalo_l/, provisional/) are not
    # the web app's to delete - queue it for the scripts.
    lifecycle.enqueue(driver_id_str, 'DELETE')
    db.session.delete(driver)
    db.session.commit()
    log_action('DELETE_DRIVER', target_type='Driver', target_id=driver_id_str,
               details=f'Driver {driver_id_str} permanently deleted')
    return jsonify({'success': True})

@main.route('/api/drivers/<int:driver_id>/toggle', methods=['POST'])
@login_required
@permission_required('can_change_driver_status')
def toggle_driver(driver_id):
    driver = Driver.query.get_or_404(driver_id)
    driver.status = 'deleted' if driver.status == 'active' else 'active'
    db.session.commit()
    log_action('TOGGLE_DRIVER', target_type='Driver', target_id=driver.driver_id,
               details=f'Status set to {driver.status}')
    return jsonify({'success': True, 'status': driver.status})


# ─── Driver List Export (CSV, no images) ──────────────────────────────────────

@main.route('/api/drivers/export-list')
@login_required
@permission_required('can_export_drivers')
def export_driver_list():
    """Export all drivers as CSV (no images) — available to admin, support, client."""
    import csv
    import io as _io
    from flask import send_file, Response

    current = get_current_user()
    if not current or not current.can_export_drivers:
        return jsonify({'error': 'Insufficient permissions'}), 403

    all_drivers = Driver.query.order_by(Driver.enrollment_date.desc()).all()

    output = _io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'Driver ID', 'First Name', 'Last Name', 'Email', 'Phone',
        'Role', 'Status', 'Enrollment State', 'Enrollment Date', 'Photo Count'
    ])
    for d in all_drivers:
        writer.writerow([
            d.driver_id,
            d.first_name,
            d.last_name,
            d.email or '',
            d.phone or '',
            d.role or '',
            d.status_label,
            d.state_label,
            d.enrollment_date.strftime('%Y-%m-%d %H:%M') if d.enrollment_date else '',
            d.photo_count or 0,
        ])

    output.seek(0)
    log_action('EXPORT_DRIVER_LIST', details=f'Exported {len(all_drivers)} drivers to CSV')
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=driver_list.csv'}
    )

# ─── Enrollment ──────────────────────────────────────────────────────────────

@main.route('/enroll')
@login_required
@permission_required('can_enroll_drivers')
def enroll():
    promote = None
    promote_id = (request.args.get('promote') or '').strip()
    if promote_id:
        promote = Driver.query.filter_by(driver_id=promote_id).first()
        if promote and promote.state not in ('UNIDENTIFIED', 'AUTO_ENROLLED'):
            promote = None
    return render_template('enroll.html', promote=promote)

@main.route('/api/detect-face', methods=['POST'])
@login_required
@permission_required('can_enroll_drivers')
def detect_face():
    """Called from webcam stream to detect and validate face."""
    data = request.json
    b64 = data.get('image')
    if not b64:
        return jsonify({'error': 'No image provided'}), 400
    
    image = face_service.decode_base64_image(b64)
    if image is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    image = _prepare_webcam_image(image)   # resize before detection
    if image is None:
        return jsonify({'error': 'Image processing failed'}), 400
    
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
@login_required
@permission_required('can_enroll_drivers')
def enroll_driver():
    """Complete enrollment - save driver + compute embedding."""
    import traceback

    try:
        # ── Parse form data ───────────────────────────────────────────────────
        first_name     = request.form.get('first_name', '').strip()
        last_name      = request.form.get('last_name', '').strip()
        email          = request.form.get('email', '').strip()
        phone          = request.form.get('phone', '').strip()
        role           = request.form.get('role', '').strip()
        images_json    = request.form.get('captured_images', '[]')

        if not first_name or not last_name:
            return jsonify({'error': 'First name and last name are required'}), 400

        # Validate role against allowed options
        ALLOWED_ROLES = {'Driver', 'Mechanic', 'Operations'}
        if role and role not in ALLOWED_ROLES:
            return jsonify({'error': f'Invalid role. Must be one of: {", ".join(sorted(ALLOWED_ROLES))}'}), 400
        if not role:
            role = 'Driver'

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
                    img = _prepare_webcam_image(img)   # resize + compress like image_preparation.py
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
                        img = _prepare_webcam_image(img)   # normalise size/quality
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

        # ── Driver ID: new, or the existing one when promoting an ─────────────
        # UNIDENTIFIED / AUTO_ENROLLED driver to FULLY_ENROLLED. Same DRV id
        # for life, so trips already allocated in Trust Track stay correct.
        promote_id = (request.form.get('promote_driver_id') or '').strip()
        promote = None
        if promote_id:
            promote = Driver.query.filter_by(driver_id=promote_id).first()
            if not promote:
                return jsonify({'error': f'Driver {promote_id} not found'}), 404
            if promote.state not in ('UNIDENTIFIED', 'AUTO_ENROLLED'):
                return jsonify({'error': f'{promote_id} is already {promote.state_label}'}), 400
            driver_id_str = promote.driver_id
            current_app.logger.info(f"[ENROLL] Promoting {driver_id_str} "
                                    f"({promote.enrollment_state} -> FULLY_ENROLLED)")
        else:
            driver_id_str = Driver.generate_driver_id()
            current_app.logger.info(f"[ENROLL] Generated driver ID: {driver_id_str}")

        # ── Photos go to R2 — no local directory needed ───────────────────────

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
                photo_bytes = cv2.imencode('.jpg', img)[1].tobytes()
                upload_driver_photo(driver_id_str, photo_name, photo_bytes)

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

        # ── Save embeddings to R2 ─────────────────────────────────────────────
        try:
            embedding_path = r2_save_embeddings(driver_id_str, embeddings)
            current_app.logger.info(f"[ENROLL] Embeddings saved to R2: {embedding_path}")
        except Exception as e:
            current_app.logger.error(f"[ENROLL] Failed to save embeddings: {e}\n{traceback.format_exc()}")
            return jsonify({'error': f'Failed to save face data: {str(e)}'}), 500

        # ── Save profile photo ────────────────────────────────────────────────
        static_photo = None
        if all_images:
            try:
                photo_bytes = cv2.imencode('.jpg', all_images[0])[1].tobytes()
                upload_profile_photo(driver_id_str, photo_bytes)
                static_photo = f"uploads/{driver_id_str}_profile.jpg"
                current_app.logger.info(f"[ENROLL] Profile photo saved to R2: {static_photo}")
            except Exception as e:
                current_app.logger.warning(f"[ENROLL] Could not save profile photo: {e}")
                # Non-fatal — continue without profile photo

        # ── Create / promote DB record ────────────────────────────────────────
        try:
            if promote:
                driver = promote
                driver.first_name, driver.last_name = first_name, last_name
                driver.email = email or None
                driver.phone = phone or None
                driver.role = role
                driver.status = 'active'
                if static_photo:
                    driver.profile_photo = static_photo
                driver.embedding_path = embedding_path
                driver.photo_count = len(embeddings)
                lifecycle.mark_fully_enrolled(driver)
            else:
                driver = Driver(
                    driver_id=driver_id_str,
                    first_name=first_name,
                    last_name=last_name,
                    email=email or None,
                    phone=phone or None,
                    role=role,
                    profile_photo=static_photo,
                    embedding_path=embedding_path,
                    photo_count=len(embeddings),
                    enrollment_state='FULLY_ENROLLED',
                    confirmed_by=session.get('username'),
                    confirmed_at=datetime.utcnow(),
                )
                db.session.add(driver)
                db.session.add(DriverStateHistory(
                    driver_id=driver_id_str, from_state=None, to_state='FULLY_ENROLLED',
                    changed_by=session.get('username'), note='Enrolled in the web app'))
            db.session.commit()
            current_app.logger.info(f"[ENROLL] ✓ Driver '{driver_id_str}' saved to DB.")
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"[ENROLL] DB error: {e}\n{traceback.format_exc()}")
            return jsonify({'error': f'Database error: {str(e)}'}), 500

        # ── Hand over to the recognizer scripts ──────────────────────────────
        # They build the buffalo_l templates from these photos (keeping any
        # cab templates) and create / rename the Trust Track driver.
        lifecycle.enqueue(driver_id_str, 'PROMOTE' if promote else 'ENROLL')
        db.session.commit()

        log_action('PROMOTE_DRIVER' if promote else 'ENROLL_DRIVER', target_type='Driver',
                   target_id=driver_id_str,
                   details=f'{first_name} {last_name} fully enrolled ({len(embeddings)} samples)')
        return jsonify({
            'success': True,
            'driver': driver.to_dict(),
            'message': (f'Driver {first_name} {last_name} '
                        f'{"promoted to fully enrolled" if promote else "enrolled successfully"} '
                        f'with {len(embeddings)} face samples. Recognition templates and Trust Track '
                        f'are updated by the recognition service within a few minutes.')
        })

    except Exception as e:
        current_app.logger.error(f"[ENROLL] Unexpected error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Unexpected server error: {str(e)}'}), 500

# ─── Recognition ─────────────────────────────────────────────────────────────

@main.route('/recognition')
@login_required
def recognition():
    videos = video_service.list_videos(current_app.config['VIDEOS_FOLDER'])
    return render_template('recognition.html', videos=videos)

# ─── Driver Detail Page ───────────────────────────────────────────────────────

@main.route('/drivers/<driver_id>')
@login_required
@permission_required('can_view_drivers')
def driver_detail(driver_id):
    driver = Driver.query.filter_by(driver_id=driver_id).first_or_404()
    events = RecognitionEvent.query.filter_by(driver_id=driver.id).order_by(RecognitionEvent.timestamp.desc()).limit(50).all()

    # Load all active custom field definitions
    custom_fields = CustomField.query.filter_by(is_active=True).order_by(CustomField.sort_order, CustomField.id).all()

    # Build a dict of field_id -> current value for this driver
    existing_values = {
        cv.custom_field_id: cv
        for cv in DriverCustomFieldValue.query.filter_by(driver_id=driver.id).all()
    }

    # Attach .current_value and parsed .options_list to each field
    import json as _json
    for cf in custom_fields:
        val_obj = existing_values.get(cf.id)
        cf.current_value = val_obj.value if val_obj else ''
        # Parse options from JSON string to Python list so Jinja2 can iterate
        if cf.options and isinstance(cf.options, str):
            try:
                cf.options_list = _json.loads(cf.options)
            except Exception:
                cf.options_list = []
        elif isinstance(cf.options, list):
            cf.options_list = cf.options
        else:
            cf.options_list = []

    state_history = (DriverStateHistory.query.filter_by(driver_id=driver.driver_id)
                     .order_by(DriverStateHistory.changed_at.desc()).limit(20).all())
    return render_template('driver_detail.html', driver=driver, events=events,
                           custom_fields=custom_fields, state_history=state_history,
                           pending_actions=lifecycle.pending_actions(driver.driver_id))

# ─── Driver Edit ──────────────────────────────────────────────────────────────

@main.route('/api/drivers/<int:driver_id>/edit', methods=['POST'])
@login_required
@permission_required('can_modify_driver')
def edit_driver(driver_id):
    # Fetch driver — works with all Flask-SQLAlchemy versions
    driver = Driver.query.filter_by(id=driver_id).first()
    if not driver:
        return jsonify({'error': 'Driver not found'}), 404

    data = request.get_json(silent=True) or {}

    # Safe strip helper — avoids AttributeError if value is None
    def _s(val, fallback=''):
        return (val or fallback).strip()

    first_name = _s(data.get('first_name'), driver.first_name or '')
    last_name  = _s(data.get('last_name'),  driver.last_name  or '')

    if not first_name or not last_name:
        return jsonify({'error': 'First name and last name are required'}), 400

    if driver.name_pending and (first_name, last_name) != (driver.first_name, driver.last_name):
        driver.name_pending = False      # the auto-enrolled driver now has a real name
    driver.first_name = first_name
    driver.last_name  = last_name
    driver.email      = (data.get('email') or '').strip() or None
    driver.phone      = (data.get('phone') or '').strip() or None

    new_role = data.get('role')
    if new_role:
        ALLOWED_ROLES = {'Driver', 'Mechanic', 'Operations'}
        if new_role not in ALLOWED_ROLES:
            return jsonify({'error': f'Invalid role. Must be one of: {", ".join(sorted(ALLOWED_ROLES))}'}), 400
        driver.role = new_role

    new_status = data.get('status')
    if new_status and new_status != driver.status:
        if new_status not in Driver.STATUSES:
            return jsonify({'error': 'Invalid status. Must be Active or Deleted'}), 400
        if not get_current_user().can_change_driver_status:
            return jsonify({'error': 'You do not have permission to change a driver\'s status'}), 403
        driver.status = new_status

    try:
        db.session.commit()
        if driver.trusttrack_id or driver.enrollment_state in ('AUTO_ENROLLED', 'FULLY_ENROLLED'):
            lifecycle.enqueue(driver.driver_id, 'SYNC_TT')   # scripts push the new name
            db.session.commit()
        log_action('EDIT_DRIVER', target_type='Driver', target_id=driver.driver_id,
                   details=f'Updated: {driver.first_name} {driver.last_name}')
        return jsonify({'success': True, 'driver': driver.to_dict()})
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f'[EDIT_DRIVER] DB error: {e}')
        return jsonify({'error': str(e)}), 500

# ─── Driver Custom Field Values ───────────────────────────────────────────────

@main.route('/api/drivers/<driver_id>/custom-fields', methods=['GET'])
@login_required
@permission_required('can_view_drivers')
def get_driver_custom_fields(driver_id):
    """Return all active custom fields with the driver's current values."""
    driver = Driver.query.filter_by(driver_id=driver_id).first_or_404()
    fields = CustomField.query.filter_by(is_active=True).order_by(CustomField.sort_order, CustomField.id).all()
    existing = {
        cv.custom_field_id: cv.value
        for cv in DriverCustomFieldValue.query.filter_by(driver_id=driver.id).all()
    }
    result = []
    for cf in fields:
        d = cf.to_dict()
        d['value'] = existing.get(cf.id, '')
        result.append(d)
    return jsonify({'fields': result})


@main.route('/api/drivers/<driver_id>/custom-fields', methods=['POST'])
@login_required
@permission_required('can_modify_driver')
def save_driver_custom_fields(driver_id):
    """Upsert custom field values for a driver.
    Expects JSON: { values: { field_id: value, ... } }
    """
    driver = Driver.query.filter_by(driver_id=driver_id).first_or_404()
    data = request.get_json() or {}
    values = data.get('values', {})

    # Build lookup of existing records
    existing = {
        cv.custom_field_id: cv
        for cv in DriverCustomFieldValue.query.filter_by(driver_id=driver.id).all()
    }

    for field_id_str, value in values.items():
        try:
            field_id = int(field_id_str)
        except (ValueError, TypeError):
            continue
        cf = CustomField.query.get(field_id)
        if not cf or not cf.is_active:
            continue
        if field_id in existing:
            existing[field_id].value = value
        else:
            new_val = DriverCustomFieldValue(
                driver_id=driver.id,
                custom_field_id=field_id,
                value=value,
            )
            db.session.add(new_val)

    try:
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500




@main.route('/api/drivers/<driver_id>/photos', methods=['GET'])
@login_required
@permission_required('can_view_drivers')
def list_driver_photos(driver_id):
    from app.services.storage_service import list_driver_photos as r2_list, get_presigned_url
    driver = Driver.query.filter_by(driver_id=driver_id).first_or_404()
    try:
        keys = r2_list(driver_id)
    except Exception as e:
        current_app.logger.error(f'[PHOTOS] list_driver_photos error for {driver_id}: {e}')
        return jsonify({'photos': [], 'error': 'Could not reach storage'}), 200
    photos = []
    profile_key = driver.profile_photo  # stored as R2 key
    for key in keys:
        url = get_presigned_url(key, expires_in=3600) or ''
        photos.append({'key': key, 'url': url, 'is_profile': (key == profile_key)})
    return jsonify({'photos': photos})

@main.route('/api/drivers/<driver_id>/photos', methods=['POST'])
@login_required
@permission_required('can_modify_driver')
def add_driver_photos(driver_id):
    import uuid as _uuid
    driver = Driver.query.filter_by(driver_id=driver_id).first_or_404()
    from app.services.storage_service import upload_driver_photo, list_driver_photos as r2_list
    files = request.files.getlist('photos')
    added = 0
    for f in files:
        if not f:
            continue
        # Accept files by content even when filename lacks extension
        fname = f.filename or ''
        ext   = fname.rsplit('.', 1)[-1].lower() if '.' in fname else 'jpg'
        if ext not in current_app.config.get('ALLOWED_EXTENSIONS', {'png', 'jpg', 'jpeg'}):
            continue
        try:
            img_bytes = f.read()
            if not img_bytes:
                continue
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                current_app.logger.warning(f'[ADD_PHOTO] Could not decode image from {fname}')
                continue
            img          = _prepare_webcam_image(img)
            photo_bytes  = cv2.imencode('.jpg', img)[1].tobytes()
            # Use a unique name so concurrent uploads never collide
            unique_name  = f'photo_{_uuid.uuid4().hex[:8]}.jpg'
            upload_driver_photo(driver_id, unique_name, photo_bytes)
            added += 1
        except Exception as e:
            current_app.logger.warning(f'[ADD_PHOTO] Error processing {fname}: {e}')
    try:
        driver.photo_count = len(r2_list(driver_id))
        db.session.commit()
    except Exception as e:
        current_app.logger.warning(f'[ADD_PHOTO] Count update error: {e}')
    if added:
        log_action('ADD_PHOTOS', target_type='Driver', target_id=driver_id,
                   details=f'Added {added} photo(s)')
    return jsonify({'added': added, 'total': driver.photo_count})

@main.route('/api/drivers/<driver_id>/photos', methods=['DELETE'])
@login_required
@permission_required('can_modify_driver')
def remove_driver_photo(driver_id):
    from app.services.storage_service import _get_client, _bucket, list_driver_photos as r2_list
    current = get_current_user()
    if not current or not current.can_modify_driver:
        return jsonify({'error': 'Insufficient permissions'}), 403
    driver = Driver.query.filter_by(driver_id=driver_id).first_or_404()
    key = request.get_json().get('key')
    if not key or not key.startswith(f'drivers/{driver_id}/'):
        return jsonify({'error': 'Invalid key'}), 400
    remaining = r2_list(driver_id)
    if len(remaining) <= 1:
        return jsonify({'error': 'Cannot remove the last photo — driver must have at least 1'}), 400
    try:
        _get_client().delete_object(Bucket=_bucket(), Key=key)
        if driver.profile_photo == key:
            others = [k for k in remaining if k != key]
            driver.profile_photo = others[0] if others else None
        driver.photo_count = max(0, driver.photo_count - 1)
        db.session.commit()
        log_action('REMOVE_PHOTO', target_type='Driver', target_id=driver_id,
                   details=f'Removed photo: {key}')
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/api/drivers/<driver_id>/export')
@login_required
@permission_required('can_export_drivers')
def export_driver_full(driver_id):
    """Export driver info as CSV + all images inside a ZIP."""
    import zipfile, io as _io, csv
    from app.services.storage_service import (
        list_driver_photos as r2_list, download_driver_photo
    )
    from flask import send_file

    current = get_current_user()
    if not current or not current.can_export_drivers:
        return jsonify({'error': 'Insufficient permissions'}), 403

    driver = Driver.query.filter_by(driver_id=driver_id).first_or_404()

    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. CSV with driver info + custom field values
        csv_buf = _io.StringIO()
        writer  = csv.writer(csv_buf)
        headers = ['Driver ID','First Name','Last Name','Email','Phone',
                   'Role','Status','Enrollment State','Enrollment Date','Photo Count']
        values  = [
            driver.driver_id, driver.first_name, driver.last_name,
            driver.email or '', driver.phone or '', driver.role or '',
            driver.status_label, driver.state_label,
            driver.enrollment_date.strftime('%Y-%m-%d %H:%M') if driver.enrollment_date else '',
            driver.photo_count or 0,
        ]
        try:
            from app.models import CustomField, DriverCustomFieldValue
            fields = CustomField.query.filter_by(is_active=True).order_by(
                CustomField.sort_order, CustomField.id).all()
            cf_vals = {cv.custom_field_id: cv.value
                       for cv in DriverCustomFieldValue.query.filter_by(driver_id=driver.id).all()}
            for cf in fields:
                headers.append(cf.name)
                values.append(cf_vals.get(cf.id, ''))
        except Exception:
            pass
        writer.writerow(headers)
        writer.writerow(values)
        zf.writestr(f'{driver_id}/driver_info.csv', csv_buf.getvalue())

        # 2. Profile photo — use the actual stored key from DB
        if driver.profile_photo:
            try:
                profile_bytes = download_driver_photo(driver.profile_photo)
                zf.writestr(f'{driver_id}/images/profile.jpg', profile_bytes)
            except Exception as e:
                current_app.logger.warning(f'[EXPORT] Profile photo error: {e}')

        # 3. All training/face photos
        try:
            keys = r2_list(driver_id)
            for key in keys:
                try:
                    data = download_driver_photo(key)
                    filename = key.split('/')[-1]
                    zf.writestr(f'{driver_id}/images/{filename}', data)
                except Exception:
                    pass
        except Exception:
            pass

    buf.seek(0)
    log_action('EXPORT_DRIVER', target_type='Driver', target_id=driver_id,
               details='Full package export (.zip)')
    return send_file(buf, mimetype='application/zip',
                     as_attachment=True,
                     download_name=f'{driver_id}_export.zip')


@main.route('/api/drivers/<driver_id>/photos/download')
@login_required
@permission_required('can_export_drivers')
def download_driver_photos(driver_id):
    import zipfile, io as _io
    from app.services.storage_service import list_driver_photos as r2_list, download_driver_photo
    from flask import send_file
    driver = Driver.query.filter_by(driver_id=driver_id).first_or_404()
    keys = r2_list(driver_id)
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for key in keys:
            try:
                data = download_driver_photo(key)
                zf.writestr(key.split('/')[-1], data)
            except Exception:
                pass
    buf.seek(0)
    return send_file(buf, mimetype='application/zip',
                     as_attachment=True,
                     download_name=f'{driver_id}_photos.zip')

# ─── Profile Photo ────────────────────────────────────────────────────────────

@main.route('/api/drivers/<driver_id>/profile-photo', methods=['GET'])
@login_required
@permission_required('can_view_drivers')
def get_profile_photo(driver_id):
    from app.services.storage_service import get_profile_photo_url
    from flask import redirect as _redirect
    url = get_profile_photo_url(driver_id)
    if url:
        return _redirect(url)
    return ('', 404)

@main.route('/api/drivers/<driver_id>/profile-photo', methods=['POST'])
@login_required
@permission_required('can_modify_driver')
def set_profile_photo(driver_id):
    from app.services.storage_service import _get_client, _bucket
    driver = Driver.query.filter_by(driver_id=driver_id).first_or_404()
    key = request.get_json().get('key')
    if not key:
        return jsonify({'error': 'No key provided'}), 400
    driver.profile_photo = key
    db.session.commit()
    try:
        url = _get_client().generate_presigned_url('get_object',
            Params={'Bucket': _bucket(), 'Key': key}, ExpiresIn=3600)
    except Exception:
        url = ''
    log_action('SET_PROFILE_PHOTO', target_type='Driver', target_id=driver_id,
               details=f'Profile photo set to: {key}')
    return jsonify({'success': True, 'url': url})

@main.route('/api/drivers/<driver_id>/profile-photo-upload', methods=['POST'])
@login_required
@permission_required('can_modify_driver')
def upload_profile_photo_route(driver_id):
    from app.services.storage_service import upload_profile_photo, get_profile_photo_url
    driver = Driver.query.filter_by(driver_id=driver_id).first_or_404()
    f = request.files.get('photo')
    if not f:
        return jsonify({'error': 'No file'}), 400
    try:
        img_bytes = f.read()
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        photo_bytes = cv2.imencode('.jpg', img)[1].tobytes()
        key = upload_profile_photo(driver_id, photo_bytes)
        driver.profile_photo = key
        db.session.commit()
        url = get_profile_photo_url(driver_id) or ''
        return jsonify({'success': True, 'url': url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─── Recompute Embeddings ─────────────────────────────────────────────────────

@main.route('/api/drivers/<driver_id>/recompute', methods=['POST'])
@login_required
@permission_required('can_modify_driver')
def recompute_embeddings(driver_id):
    import traceback
    from app.services.storage_service import list_driver_photos as r2_list, download_driver_photo, save_embeddings as r2_save
    driver = Driver.query.filter_by(driver_id=driver_id).first_or_404()

    # List photos from R2
    try:
        keys = r2_list(driver_id)
    except Exception as e:
        current_app.logger.error(f'[RECOMPUTE] Could not list photos for {driver_id}: {e}')
        return jsonify({'error': f'Could not reach storage: {str(e)}'}), 500

    if not keys:
        return jsonify({'error': 'No photos found in storage for this driver'}), 400

    # Ensure embedding model is loaded
    try:
        model = embedding_service._get_model()
        if model is None:
            return jsonify({'error': 'Face recognition model not available — check server logs'}), 500
    except Exception as e:
        current_app.logger.error(f'[RECOMPUTE] Model load error: {e}')
        return jsonify({'error': f'Model load error: {str(e)}'}), 500

    embeddings = []
    for key in keys:
        try:
            img_bytes = download_driver_photo(key)
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                current_app.logger.warning(f'[RECOMPUTE] Cannot decode {key}')
                continue
            emb, _ = embedding_service.extract_embedding(img)
            if emb is not None:
                embeddings.append(emb)
            else:
                current_app.logger.warning(f'[RECOMPUTE] No face found in {key}')
        except Exception as e:
            current_app.logger.warning(f'[RECOMPUTE] Skipping {key}: {e}')

    if len(embeddings) < 1:
        return jsonify({'error': f'No valid faces found in {len(keys)} photo(s). Ensure photos contain clear front-facing faces.'}), 400

    try:
        r2_save(driver_id, embeddings)
    except Exception as e:
        current_app.logger.error(f'[RECOMPUTE] Save embeddings error: {e}')
        return jsonify({'error': f'Could not save face data: {str(e)}'}), 500

    try:
        driver.photo_count = len(keys)
        db.session.commit()
    except Exception as e:
        current_app.logger.warning(f'[RECOMPUTE] DB update error: {e}')

    log_action('RECOMPUTE_EMBEDDINGS', target_type='Driver', target_id=driver_id,
               details=f'Recomputed {len(embeddings)} embeddings from {len(keys)} photos')
    return jsonify({'success': True, 'embeddings': len(embeddings), 'photos': len(keys)})

@main.route('/api/recognize-video', methods=['POST'])
@login_required
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
@login_required
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
    
    all_embeddings = r2_load_all_embeddings()
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


# ─── Reports Dashboard ───────────────────────────────────────────────────────
# Reads exclusively from the `trips_reinke_logistics_v2` table, which is loaded by
# the standalone pipeline/load_trips.py script. This blueprint never talks to
# the FM-Track API directly.

def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, '%Y-%m-%d')
    except ValueError:
        return None


def _apply_trip_filters(query, args):
    """Applies vehicle / driver / assignment-status / search / date-range filters.
    `args` behaves like a dict with .get() — works with both request.args and a plain dict."""
    vehicle = (args.get('vehicle') or '').strip()
    driver = (args.get('driver') or '').strip()
    status = (args.get('status') or 'all').strip().lower()
    include_excluded = (args.get('include_excluded') or '0') in ('1', 'true', 'True')
    search = (args.get('search') or '').strip()
    date_from = _parse_date(args.get('date_from'))
    date_to = _parse_date(args.get('date_to'))

    if not include_excluded:
        query = query.filter(Trip.is_excluded.is_(False))
    if vehicle:
        query = query.filter(Trip.vehicle == vehicle)
    if driver:
        query = query.filter(Trip.driver_name == driver)
    if status == 'assigned':
        query = query.filter(Trip.has_driver.is_(True))
    elif status == 'unassigned':
        query = query.filter(Trip.has_driver.is_(False))
    if search:
        like = f"%{search}%"
        query = query.filter(or_(
            Trip.vehicle.ilike(like),
            Trip.driver_name.ilike(like),
            Trip.start_location.ilike(like),
            Trip.start_zone_name.ilike(like),
            Trip.end_location.ilike(like),
        ))
    if date_from:
        query = query.filter(Trip.start_time >= date_from)
    if date_to:
        query = query.filter(Trip.start_time < date_to + timedelta(days=1))
    return query


def _fmt_duration(minutes):
    if minutes is None:
        return "—"
    h, m = divmod(int(round(minutes)), 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


@main.route('/reports')
@login_required
def reports():
    vehicles = [v[0] for v in db.session.query(Trip.vehicle).distinct().order_by(Trip.vehicle).all()]
    drivers = [d[0] for d in db.session.query(Trip.driver_name)
               .filter(Trip.driver_name.isnot(None)).distinct().order_by(Trip.driver_name).all()]
    last_run = PipelineRun.query.order_by(PipelineRun.id.desc()).first()
    return render_template('reports.html',
        vehicles=vehicles, drivers=drivers,
        last_run=last_run.to_dict() if last_run else None)


@main.route('/api/reports/meta')
@login_required
def reports_meta():
    vehicles = [v[0] for v in db.session.query(Trip.vehicle).distinct().order_by(Trip.vehicle).all()]
    drivers = [d[0] for d in db.session.query(Trip.driver_name)
               .filter(Trip.driver_name.isnot(None)).distinct().order_by(Trip.driver_name).all()]
    last_run = PipelineRun.query.order_by(PipelineRun.id.desc()).first()
    return jsonify({
        'vehicles': vehicles,
        'drivers': drivers,
        'last_run': last_run.to_dict() if last_run else None,
    })


@main.route('/api/reports/summary')
@login_required
def reports_summary():
    base = _apply_trip_filters(Trip.query, request.args)
    assigned_q = base.filter(Trip.has_driver.is_(True))
    unassigned_q = base.filter(Trip.has_driver.is_(False))

    assigned_count = assigned_q.count()
    unassigned_count = unassigned_q.count()
    total = assigned_count + unassigned_count

    assigned_km = assigned_q.with_entities(func.coalesce(func.sum(Trip.km), 0.0)).scalar() or 0.0
    unassigned_km = unassigned_q.with_entities(func.coalesce(func.sum(Trip.km), 0.0)).scalar() or 0.0
    assigned_dur = assigned_q.with_entities(func.coalesce(func.sum(Trip.duration_min), 0.0)).scalar() or 0.0
    unassigned_dur = unassigned_q.with_entities(func.coalesce(func.sum(Trip.duration_min), 0.0)).scalar() or 0.0

    # Excluded trailers are reported separately regardless of the include_excluded toggle,
    # honouring only the date range / search so the panel stays meaningful.
    excl_args = dict(request.args)
    excl_args.pop('status', None)
    excl_args.pop('vehicle', None)
    excl_args.pop('driver', None)
    excl_args['include_excluded'] = '1'
    excl_q = _apply_trip_filters(Trip.query, excl_args).filter(Trip.is_excluded.is_(True))
    excluded_count = excl_q.count()
    excluded_vehicles = excl_q.with_entities(Trip.vehicle).distinct().count()

    return jsonify({
        'total_trips': total,
        'assigned_count': assigned_count,
        'unassigned_count': unassigned_count,
        'assigned_pct': (assigned_count / total) if total else 0,
        'unassigned_pct': (unassigned_count / total) if total else 0,
        'assigned_km': round(assigned_km, 1),
        'unassigned_km': round(unassigned_km, 1),
        'assigned_duration': _fmt_duration(assigned_dur),
        'unassigned_duration': _fmt_duration(unassigned_dur),
        'excluded_count': excluded_count,
        'excluded_vehicles': excluded_vehicles,
    })


@main.route('/api/reports/vehicles')
@login_required
def reports_vehicles():
    """Vehicles ranked by unmatched trips — honours every filter except 'vehicle' itself."""
    args = request.args.to_dict()
    args.pop('vehicle', None)
    q = _apply_trip_filters(Trip.query, args)

    rows = q.with_entities(
        Trip.vehicle,
        func.count(Trip.id).label('total'),
        func.sum(case((Trip.has_driver.is_(True), 1), else_=0)).label('assigned'),
        func.sum(case((Trip.has_driver.is_(False), 1), else_=0)).label('unassigned'),
        func.coalesce(func.sum(case((Trip.has_driver.is_(False), Trip.km), else_=0.0)), 0.0).label('unassigned_km'),
    ).group_by(Trip.vehicle).all()

    result = []
    for vehicle, total, assigned, unassigned, unassigned_km in rows:
        result.append({
            'vehicle': vehicle,
            'total_trips': int(total or 0),
            'assigned': int(assigned or 0),
            'unassigned': int(unassigned or 0),
            'pct_unassigned': (unassigned / total) if total else 0,
            'unassigned_km': round(unassigned_km or 0.0, 1),
        })
    result.sort(key=lambda r: (-r['unassigned'], -r['pct_unassigned']))
    return jsonify({'vehicles': result})


@main.route('/api/reports/trips')
@login_required
def reports_trips():
    q = _apply_trip_filters(Trip.query, request.args)

    sort_map = {
        'start_time': Trip.start_time, 'vehicle': Trip.vehicle,
        'driver_name': Trip.driver_name, 'km': Trip.km, 'duration_min': Trip.duration_min,
    }
    sort_col = sort_map.get(request.args.get('sort', 'start_time'), Trip.start_time)
    order = request.args.get('order', 'desc').lower()
    q = q.order_by(sort_col.desc() if order == 'desc' else sort_col.asc())

    try:
        page = max(int(request.args.get('page', 1)), 1)
    except ValueError:
        page = 1
    try:
        per_page = min(max(int(request.args.get('per_page', 25)), 1), 200)
    except ValueError:
        per_page = 25

    total = q.count()
    rows = q.offset((page - 1) * per_page).limit(per_page).all()

    return jsonify({
        'rows': [r.to_dict() for r in rows],
        'total': total,
        'page': page,
        'per_page': per_page,
        'pages': max(1, (total + per_page - 1) // per_page),
    })


def _export_csv(trips):
    import csv
    import io as _io
    from flask import Response

    output = _io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Vehicle', 'Date', 'Start Time', 'End Time', 'Duration (min)', 'KM',
                      'Driver', 'Driver Identified', 'Start Location', 'End Location'])
    for t in trips:
        writer.writerow([
            t.vehicle,
            t.start_time.strftime('%Y-%m-%d') if t.start_time else '',
            t.start_time.strftime('%H:%M') if t.start_time else '',
            t.end_time.strftime('%H:%M') if t.end_time else '',
            round(t.duration_min, 1) if t.duration_min is not None else '',
            round(t.km, 2) if t.km is not None else '',
            (t.driver_name or '') + (' (facial recognition)' if getattr(t, 'driver_source', None) == 'recognition' else ''),
            'Yes' if t.has_driver else 'No',
            t.start_zone_name or t.start_location or '',
            t.end_location or '',
        ])
    output.seek(0)
    filename = f"trip_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
    return Response(output.getvalue(), mimetype='text/csv',
                     headers={'Content-Disposition': f'attachment; filename={filename}'})


def _export_xlsx(trips):
    import io as _io
    from flask import send_file
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    NAVY = "2E4057"
    hdr_font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    hdr_fill = PatternFill("solid", start_color=NAVY)
    hdr_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    body_font = Font(name="Arial", size=10)
    bold_font = Font(name="Arial", size=10, bold=True)
    thin = Border(left=Side(style="thin"), right=Side(style="thin"),
                   top=Side(style="thin"), bottom=Side(style="thin"))
    fill_even = PatternFill("solid", start_color="EAF0FB")
    fill_odd = PatternFill("solid", start_color="FFFFFF")
    flag_fill = PatternFill("solid", start_color="FDEDEC")

    total = len(trips)
    assigned = [t for t in trips if t.has_driver]
    unassigned = [t for t in trips if not t.has_driver]
    assigned_km = sum(t.km or 0 for t in assigned)
    unassigned_km = sum(t.km or 0 for t in unassigned)
    unassigned_dur = sum(t.duration_min or 0 for t in unassigned)

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Driver Assignment Report"
    ws["A1"].font = Font(name="Arial", bold=True, size=15, color=NAVY)
    ws["A2"] = (f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC   |   "
                f"Filtered export from the Reinke Logistics Reports dashboard")
    ws["A2"].font = Font(name="Arial", italic=True, size=9, color="666666")

    kpis = [
        ("Total trips", total),
        ("Driver identified", f"{len(assigned)} ({(len(assigned)/total if total else 0):.0%})"),
        ("Driver NOT identified", f"{len(unassigned)} ({(len(unassigned)/total if total else 0):.0%})"),
        ("KM — driver identified", f"{assigned_km:,.1f} km"),
        ("KM — no driver identified", f"{unassigned_km:,.1f} km"),
        ("Driving time — no driver identified", _fmt_duration(unassigned_dur)),
    ]
    r = 4
    for label, val in kpis:
        ws.cell(row=r, column=1, value=label).font = bold_font
        ws.cell(row=r, column=2, value=val).font = body_font
        r += 1
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 34

    ws2 = wb.create_sheet("Trip Detail")
    headers = ["Vehicle", "Date", "Start Time", "End Time", "Duration", "KM",
               "Driver", "Driver Identified", "Start Location", "End Location"]
    for c, h in enumerate(headers, 1):
        cell = ws2.cell(row=1, column=c, value=h)
        cell.font, cell.fill, cell.alignment, cell.border = hdr_font, hdr_fill, hdr_align, thin
    widths = [22, 12, 11, 11, 11, 10, 24, 16, 40, 40]
    for i, w in enumerate(widths, 1):
        ws2.column_dimensions[get_column_letter(i)].width = w

    for i, t in enumerate(trips):
        row = i + 2
        fill = flag_fill if not t.has_driver else (fill_even if row % 2 == 0 else fill_odd)
        values = [
            t.vehicle,
            t.start_time.strftime('%Y-%m-%d') if t.start_time else '',
            t.start_time.strftime('%H:%M') if t.start_time else '',
            t.end_time.strftime('%H:%M') if t.end_time else '',
            _fmt_duration(t.duration_min),
            round(t.km, 2) if t.km is not None else '',
            (t.driver_name or '') + (' (facial recognition)' if getattr(t, 'driver_source', None) == 'recognition' else ''),
            'Yes' if t.has_driver else 'No',
            t.start_zone_name or t.start_location or '',
            t.end_location or '',
        ]
        for c, v in enumerate(values, 1):
            cell = ws2.cell(row=row, column=c, value=v)
            cell.font, cell.fill, cell.border = body_font, fill, thin
            cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws2.freeze_panes = "A2"

    buf = _io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    filename = f"trip_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.xlsx"
    return send_file(buf, as_attachment=True, download_name=filename,
                      mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@main.route('/api/reports/export')
@login_required
def reports_export():
    fmt = request.args.get('format', 'xlsx').lower()
    q = _apply_trip_filters(Trip.query, request.args).order_by(Trip.vehicle, Trip.start_time)
    trips = q.all()
    log_action('EXPORT_REPORT', details=f'Exported {len(trips)} trip rows as {fmt}')
    if fmt == 'csv':
        return _export_csv(trips)
    return _export_xlsx(trips)

# ─── Unidentified drivers: OCC review queue ──────────────────────────────────
# The recognizer scripts (separate from this app) create an UNIDENTIFIED
# driver the first time they see a clear face that matches nobody, allocate
# the trip to it, and keep alerting until OCC acts here. This app only writes
# the decision to Postgres and queues the side effects (R2 templates, Trust
# Track) in driver_lifecycle_actions_reinke_logistics for the scripts:
#   confirm  -> AUTO_ENROLLED (alerts stop, Trust Track renamed)
#   merge    -> into an existing driver (it was someone we know); the
#               provisional record is set to status 'deleted'
#   reject   -> AUTO_ENROLLED back to UNIDENTIFIED (alerts resume)
#   enroll   -> FULLY_ENROLLED via /enroll?promote=<id> (photos + consent)
# Enrollment states: Fully Enrolled, Auto-enrolled, Unidentified.
# Statuses: Active, Deleted.

def _driver_photo_url(d):
    from app.services.storage_service import get_profile_photo_url
    return get_profile_photo_url(d.driver_id) if d.profile_photo else None


def _require_modify():
    current = get_current_user()
    if not current or not current.can_review_unidentified:
        return jsonify({'error': 'Insufficient permissions'}), 403
    return None


@main.route('/unidentified')
@login_required
@permission_required('can_review_unidentified')
def unidentified():
    state = request.args.get('state', 'UNIDENTIFIED').upper()
    if state not in ('UNIDENTIFIED', 'AUTO_ENROLLED'):
        state = 'UNIDENTIFIED'
    rows = (Driver.query.filter_by(enrollment_state=state).filter(Driver.status == 'active')
            .order_by(Driver.last_seen_at.desc().nullslast(), Driver.id.desc()).all())
    for d in rows:
        d.photo_url = _driver_photo_url(d)
    counts = dict(db.session.query(Driver.enrollment_state, func.count(Driver.id))
                  .filter(Driver.enrollment_state.in_(['UNIDENTIFIED', 'AUTO_ENROLLED']))
                  .filter(Driver.status == 'active')
                  .group_by(Driver.enrollment_state).all())
    return render_template('unidentified.html', drivers=rows, state=state, counts=counts)


@main.route('/api/unidentified/<driver_id>/details')
@login_required
@permission_required('can_review_unidentified')
def unidentified_details(driver_id):
    from app.services.storage_service import list_provisional_photos, get_presigned_url
    d = Driver.query.filter_by(driver_id=driver_id).first_or_404()
    try:
        photos = [{'key': k, 'url': get_presigned_url(k) or ''} for k in list_provisional_photos(driver_id)]
    except Exception as e:
        current_app.logger.warning(f'[UNIDENTIFIED] photos for {driver_id}: {e}')
        photos = []
    # captured_at (snapshot time, SAST) is added by the recognizer; fall back
    # to the processing timestamp (UTC) if that column isn't there.
    from sqlalchemy import text as _text
    try:
        rows = db.session.execute(_text(
            "SELECT video_filename, status, confidence, COALESCE(captured_at, timestamp) "
            "FROM recognition_events_reinke_logistics WHERE driver_id = :pk "
            "ORDER BY COALESCE(captured_at, timestamp) DESC LIMIT 25"), {'pk': d.id}).fetchall()
    except Exception:
        db.session.rollback()
        rows = [(e.video_filename, e.status, e.confidence, e.timestamp) for e in
                RecognitionEvent.query.filter_by(driver_id=d.id)
                .order_by(RecognitionEvent.timestamp.desc()).limit(25).all()]
    sightings = [{'vehicle': v, 'status': stt, 'confidence': round((c or 0) * 100, 1),
                  'time': t.strftime('%Y-%m-%d %H:%M') if t else None} for v, stt, c, t in rows]
    history = [h.to_dict() for h in DriverStateHistory.query.filter_by(driver_id=driver_id)
               .order_by(DriverStateHistory.changed_at.desc()).limit(10).all()]
    pending = [{'action': a.action, 'label': a.label, 'status': a.status, 'error': a.error}
               for a in lifecycle.pending_actions(driver_id)]
    return jsonify({'driver': d.to_dict(), 'photos': photos,
                    'sightings': sightings, 'history': history, 'pending': pending})


@main.route('/api/drivers/<driver_id>/confirm', methods=['POST'])
@login_required
@permission_required('can_review_unidentified')
def confirm_driver(driver_id):
    denied = _require_modify()
    if denied:
        return denied
    d = Driver.query.filter_by(driver_id=driver_id).first_or_404()
    data = request.get_json(silent=True) or {}
    client_name = (data.get('confirmed_by_client') or '').strip()
    note = (data.get('note') or '').strip()
    first = (data.get('first_name') or '').strip()
    last = (data.get('last_name') or '').strip()
    role = (data.get('role') or 'Driver').strip()
    if role not in {'Driver', 'Mechanic', 'Operations'}:
        return jsonify({'error': 'Invalid role'}), 400
    try:
        lifecycle.confirm(d, client_name, note, first_name=first, last_name=last,
                          phone=(data.get('phone') or '').strip(),
                          email=(data.get('email') or '').strip(), role=role)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f'[CONFIRM] {driver_id}: {e}')
        return jsonify({'error': str(e)}), 500
    log_action('CONFIRM_DRIVER', target_type='Driver', target_id=driver_id,
               details=f'Auto-enrolled ({d.first_name} {d.last_name}); authorized by client '
                       f'contact {client_name}' + (f'; note: {note}' if note else ''))
    return jsonify({'success': True, 'driver': d.to_dict()})


@main.route('/api/drivers/<driver_id>/reject', methods=['POST'])
@login_required
@permission_required('can_review_unidentified')
def reject_driver(driver_id):
    denied = _require_modify()
    if denied:
        return denied
    d = Driver.query.filter_by(driver_id=driver_id).first_or_404()
    note = ((request.get_json(silent=True) or {}).get('note') or '').strip() or None
    try:
        lifecycle.reject(d, note)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    log_action('MARK_UNIDENTIFIED', target_type='Driver', target_id=driver_id, details=note or 'Marked unidentified')
    return jsonify({'success': True, 'driver': d.to_dict()})


@main.route('/api/drivers/<driver_id>/merge', methods=['POST'])
@login_required
@permission_required('can_review_unidentified')
def merge_driver(driver_id):
    denied = _require_modify()
    if denied:
        return denied
    d = Driver.query.filter_by(driver_id=driver_id).first_or_404()
    data = request.get_json(silent=True) or {}
    target = Driver.query.filter_by(driver_id=(data.get('target_driver_id') or '').strip()).first()
    if not target:
        return jsonify({'error': 'Target driver not found'}), 404
    try:
        lifecycle.merge(d, target, note=(data.get('note') or '').strip() or None)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f'[MERGE] {driver_id} -> {target.driver_id}: {e}')
        return jsonify({'error': str(e)}), 500
    log_action('MERGE_DRIVER', target_type='Driver', target_id=driver_id,
               details=f'Merged into {target.driver_id}')
    return jsonify({'success': True, 'target': target.to_dict()})


@main.route('/api/drivers/<driver_id>/trusttrack-sync', methods=['POST'])
@login_required
@permission_required('can_review_unidentified')
def trusttrack_sync_driver(driver_id):
    denied = _require_modify()
    if denied:
        return denied
    d = Driver.query.filter_by(driver_id=driver_id).first_or_404()
    lifecycle.enqueue(d.driver_id, 'SYNC_TT')
    db.session.commit()
    return jsonify({'success': True, 'queued': True, 'driver': d.to_dict()})


@main.route('/api/drivers/search')
@login_required
@permission_required('can_review_unidentified')
def search_drivers():
    """Merge-target picker: active enrolled drivers matching a name / id fragment."""
    q = (request.args.get('q') or '').strip()
    query = (Driver.query.filter(Driver.enrollment_state.in_(['AUTO_ENROLLED', 'FULLY_ENROLLED']))
             .filter(Driver.status == 'active'))
    if q:
        like = f'%{q}%'
        query = query.filter(or_(Driver.driver_id.ilike(like), Driver.first_name.ilike(like),
                                 Driver.last_name.ilike(like)))
    rows = query.order_by(Driver.first_name, Driver.last_name).limit(20).all()
    return jsonify({'drivers': [{'driver_id': r.driver_id, 'name': f'{r.first_name} {r.last_name}',
                                 'state': r.state_label} for r in rows]})
