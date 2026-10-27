from app import db
from datetime import datetime
import uuid
from werkzeug.security import generate_password_hash, check_password_hash


class User(db.Model):
    __tablename__ = 'users_facial_recognition_reinke_logistics'

    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.String(36), unique=True, nullable=False,
                              default=lambda: str(uuid.uuid4()))
    username      = db.Column(db.String(80),  unique=True, nullable=False)
    email         = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role          = db.Column(db.String(20),  default='client')   # admin | support | client
    is_active     = db.Column(db.Boolean,     default=True)
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)
    last_login    = db.Column(db.DateTime,    nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id':         self.id,
            'user_id':    self.user_id,
            'username':   self.username,
            'email':      self.email,
            'role':       self.role,
            'is_active':  self.is_active,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else None,
            'last_login': self.last_login.strftime('%Y-%m-%d %H:%M') if self.last_login else None,
        }

    # ── Permission helpers ────────────────────────────────────────────────────
    # Each property checks the RolePermission table first (DB override), then
    # falls back to the hard-coded default.  This means the Permissions UI
    # in /users can change behaviour in real time without a code deploy.
    #
    # Admins always get True, and LOCKED_GRANTS below always win over the DB,
    # so a stale override row can never take those permissions away.  Other
    # locked cells are enforced by NOT exposing them as editable in the UI.

    # Permissions a role always has, whatever the Permissions tab says.
    # Clients must always be able to enroll, view and edit drivers and export
    # the driver list (mirrored in auth.save_permissions and users.html).
    LOCKED_GRANTS = {
        'client': {'can_enroll_drivers', 'can_view_drivers',
                   'can_modify_driver', 'can_export_drivers'},
    }

    def _perm(self, name: str, default: bool) -> bool:
        if self.role == 'admin':
            return True        # admins are never locked out of anything
        if name in self.LOCKED_GRANTS.get(self.role, ()):
            return True
        try:
            from app.models import RolePermission
            return RolePermission.get(self.role, name, default)
        except Exception:
            return default

    @property
    def can_create_client_user(self):
        return self._perm('can_create_client_user', self.role in ('admin', 'support'))

    @property
    def can_create_support_user(self):
        return self._perm('can_create_support_user', self.role == 'admin')

    @property
    def can_enroll_drivers(self):
        return self._perm('can_enroll_drivers', self.role in ('admin', 'support', 'client'))

    @property
    def can_modify_driver(self):
        """Edit driver details, custom-field values and face photos."""
        return self._perm('can_modify_driver', self.role in ('admin', 'support', 'client'))

    @property
    def can_change_driver_status(self):
        """Set a driver's status: Active / Deleted."""
        return self._perm('can_change_driver_status', self.role in ('admin', 'support'))

    @property
    def can_delete_drivers(self):
        return self._perm('can_delete_drivers', self.role == 'admin')

    @property
    def can_review_unidentified(self):
        """Confirm (auto-enroll) / mark unidentified / merge drivers, Trust Track sync."""
        return self._perm('can_review_unidentified', self.role in ('admin', 'support', 'client'))

    @property
    def can_export_drivers(self):
        return self._perm('can_export_drivers', self.role in ('admin', 'support', 'client'))

    @property
    def can_view_drivers(self):
        return self._perm('can_view_drivers', self.role in ('admin', 'support', 'client'))

    @property
    def can_manage_custom_fields(self):
        return self._perm('can_manage_custom_fields', self.role in ('admin', 'support'))

    @property
    def can_create_delete_custom_fields(self):
        return self._perm('can_create_delete_custom_fields', self.role == 'admin')

    @property
    def generate_email_alerts(self):
        return self._perm('generate_email_alerts', self.role in ('admin', 'support', 'client'))

    DRIVER_PERMISSIONS = ('can_view_drivers', 'can_enroll_drivers', 'can_modify_driver',
                          'can_change_driver_status', 'can_delete_drivers',
                          'can_export_drivers', 'can_review_unidentified')

    def driver_permissions(self) -> dict:
        """{permission: bool} for templates (exposed as `perms`)."""
        return {p: bool(getattr(self, p)) for p in self.DRIVER_PERMISSIONS}


class AuditLog(db.Model):
    """Records all significant actions performed by users."""
    __tablename__ = 'audit_logs_reinke_logistics'

    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.String(36), nullable=True)
    username    = db.Column(db.String(80),  nullable=True)
    action      = db.Column(db.String(100), nullable=False)
    target_type = db.Column(db.String(50),  nullable=True)
    target_id   = db.Column(db.String(80),  nullable=True)
    details     = db.Column(db.Text,        nullable=True)
    ip_address  = db.Column(db.String(45),  nullable=True)
    timestamp   = db.Column(db.DateTime,    default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':          self.id,
            'user_id':     self.user_id,
            'username':    self.username,
            'action':      self.action,
            'target_type': self.target_type,
            'target_id':   self.target_id,
            'details':     self.details,
            'ip_address':  self.ip_address,
            'timestamp':   self.timestamp.strftime('%Y-%m-%d %H:%M:%S') if self.timestamp else None,
        }


class CustomField(db.Model):
    __tablename__ = 'custom_fields_reinke_logistics'

    id           = db.Column(db.Integer, primary_key=True)
    name         = db.Column(db.String(100), nullable=False, unique=True)
    field_type   = db.Column(db.String(30),  nullable=False, default='text')
    is_required  = db.Column(db.Boolean, default=False)
    is_active    = db.Column(db.Boolean, default=True)
    options      = db.Column(db.Text, nullable=True)  # JSON for select types
    sort_order   = db.Column(db.Integer, default=0)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at   = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        import json as _json
        opts = []
        if self.options:
            try:
                opts = _json.loads(self.options)
            except Exception:
                pass
        return {
            'id':          self.id,
            'name':        self.name,
            'field_type':  self.field_type,
            'is_required': self.is_required,
            'is_active':   self.is_active,
            'options':     opts,
            'sort_order':  self.sort_order,
            'created_at':  self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else None,
        }


class DriverCustomFieldValue(db.Model):
    __tablename__ = 'driver_custom_field_values_reinke_logistics'

    id              = db.Column(db.Integer, primary_key=True)
    driver_id       = db.Column(db.Integer, db.ForeignKey('drivers_reinke_logistics.id'), nullable=False)
    custom_field_id = db.Column(db.Integer, db.ForeignKey('custom_fields_reinke_logistics.id'), nullable=False)
    value           = db.Column(db.Text, nullable=True)

    driver       = db.relationship('Driver', backref='custom_values')
    custom_field = db.relationship('CustomField')

    def to_dict(self):
        return {
            'id':              self.id,
            'driver_id':       self.driver_id,
            'custom_field_id': self.custom_field_id,
            'field_name':      self.custom_field.name if self.custom_field else '',
            'field_type':      self.custom_field.field_type if self.custom_field else 'text',
            'value':           self.value,
        }


class Driver(db.Model):
    __tablename__ = 'drivers_reinke_logistics'
    
    id = db.Column(db.Integer, primary_key=True)
    driver_id = db.Column(db.String(20), unique=True, nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(200), unique=True)
    phone = db.Column(db.String(20))
    role = db.Column(db.String(20), default='Driver')
    enrollment_date = db.Column(db.DateTime, default=datetime.utcnow)
    profile_photo = db.Column(db.String(200))
    status = db.Column(db.String(20), default='active')   # active | deleted  (see STATUSES)
    embedding_path = db.Column(db.String(200))
    photo_count = db.Column(db.Integer, default=0)

    # ── Enrollment lifecycle ──────────────────────────────────────────────────
    # Only three enrollment states exist:
    # UNIDENTIFIED   unknown / not authorized cab face (created by the recognizer,
    #                or reverted by OCC); alerts until OCC sorts it out
    # AUTO_ENROLLED  OCC confirmed a legitimate driver; no alerts
    # FULLY_ENROLLED enrolled with photos in this web app (or promoted)
    # A merged record (same person as `merged_into`) or an expired one keeps its
    # enrollment state and gets status 'deleted'.
    enrollment_state = db.Column(db.String(20), nullable=False, default='FULLY_ENROLLED', index=True)
    trusttrack_id = db.Column(db.String(64))
    trusttrack_sync_pending = db.Column(db.Boolean, nullable=False, default=False)
    first_seen_at = db.Column(db.DateTime)
    first_seen_vehicle = db.Column(db.String(100))
    last_seen_at = db.Column(db.DateTime)
    last_seen_vehicle = db.Column(db.String(100))
    sighting_count = db.Column(db.Integer, nullable=False, default=0)
    confirmed_by = db.Column(db.String(100))          # web-app user (OCC operator) who recorded it
    confirmed_at = db.Column(db.DateTime)
    confirmed_by_client = db.Column(db.String(150))   # client contact who vouched the driver is legit
    name_pending = db.Column(db.Boolean, nullable=False, default=False, server_default=db.false())
    merged_into = db.Column(db.String(30))
    state_note = db.Column(db.Text)
    # JSON [{driver_id, similarity}] - nearest confirmed drivers, written by
    # the recognizer (the web app never reads the face templates itself)
    lookalikes = db.Column(db.Text)

    recognitions = db.relationship('RecognitionEvent', backref='driver', lazy=True)

    STATE_LABELS = {
        'FULLY_ENROLLED': 'Fully Enrolled',
        'AUTO_ENROLLED': 'Auto-enrolled',
        'UNIDENTIFIED': 'Unidentified',
    }
    # Legacy values that may still come from older rows / the recognizer.
    LEGACY_STATES = {'REJECTED': 'UNIDENTIFIED', 'EXPIRED': 'UNIDENTIFIED', 'MERGED': 'UNIDENTIFIED'}

    STATUS_LABELS = {'active': 'Active', 'deleted': 'Deleted'}
    STATUSES = tuple(STATUS_LABELS)

    # First name given to a driver confirmed without a name (last name keeps the
    # recognizer's "<vehicle> <date>" label) until a real name is added.
    AUTO_ENROLLED_PLACEHOLDER = 'Auto Enrolled'

    @property
    def state(self):
        st = self.enrollment_state or 'FULLY_ENROLLED'
        return self.LEGACY_STATES.get(st, st)

    @property
    def state_label(self):
        return self.STATE_LABELS.get(self.state, self.state)

    @property
    def status_label(self):
        return self.STATUS_LABELS.get(self.status or 'active', 'Deleted')

    @property
    def is_deleted(self):
        return (self.status or 'active') != 'active'

    @property
    def is_provisional(self):
        return self.state == 'UNIDENTIFIED'

    def to_dict(self):
        return {
            'id': self.id,
            'driver_id': self.driver_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': f"{self.first_name} {self.last_name}",
            'email': self.email,
            'phone': self.phone,
            'role': self.role,
            'enrollment_date': self.enrollment_date.strftime('%Y-%m-%d %H:%M') if self.enrollment_date else None,
            'profile_photo': self.profile_photo,
            'status': self.status,
            'status_label': self.status_label,
            'photo_count': self.photo_count,
            'enrollment_state': self.state,
            'state_label': self.state_label,
            'trusttrack_id': self.trusttrack_id,
            'trusttrack_sync_pending': bool(self.trusttrack_sync_pending),
            'first_seen_at': self.first_seen_at.strftime('%Y-%m-%d %H:%M') if self.first_seen_at else None,
            'first_seen_vehicle': self.first_seen_vehicle,
            'last_seen_at': self.last_seen_at.strftime('%Y-%m-%d %H:%M') if self.last_seen_at else None,
            'last_seen_vehicle': self.last_seen_vehicle,
            'sighting_count': self.sighting_count or 0,
            'confirmed_by': self.confirmed_by,
            'confirmed_at': self.confirmed_at.strftime('%Y-%m-%d %H:%M') if self.confirmed_at else None,
            'confirmed_by_client': self.confirmed_by_client,
            'name_pending': bool(self.name_pending),
            'merged_into': self.merged_into,
        }

    @staticmethod
    def generate_driver_id():
        """DRV-xxxxxx, checked for uniqueness (the recognizer creates ids in
        the same space, so a blind random pick can collide)."""
        for _ in range(50):
            candidate = f"DRV-{str(uuid.uuid4().int)[:6]}"
            if not Driver.query.filter_by(driver_id=candidate).first():
                return candidate
        return f"DRV-{uuid.uuid4().hex[:8].upper()}"


class DriverStateHistory(db.Model):
    """Audit trail of enrollment-state changes (who confirmed whom, and when)."""
    __tablename__ = 'driver_state_history_reinke_logistics'

    id = db.Column(db.Integer, primary_key=True)
    driver_id = db.Column(db.String(30), nullable=False, index=True)
    from_state = db.Column(db.String(20))
    to_state = db.Column(db.String(20), nullable=False)
    changed_by = db.Column(db.String(100))
    note = db.Column(db.Text)
    changed_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False,
                           server_default=db.func.current_timestamp())

    def to_dict(self):
        return {
            'driver_id': self.driver_id,
            'from_state': self.from_state,
            'to_state': self.to_state,
            'changed_by': self.changed_by,
            'note': self.note,
            'changed_at': self.changed_at.strftime('%Y-%m-%d %H:%M') if self.changed_at else None,
        }


class DriverLifecycleAction(db.Model):
    """Work queue from the web app to the recognizer scripts.

    The web app records OCC's decision in the DB and queues the side effects
    here; the scripts (which own the face templates in R2 and the Trust Track
    integration) process PENDING rows and mark them DONE / FAILED."""
    __tablename__ = 'driver_lifecycle_actions_reinke_logistics'

    id = db.Column(db.Integer, primary_key=True)
    driver_id = db.Column(db.String(30), nullable=False, index=True)
    action = db.Column(db.String(20), nullable=False)   # CONFIRM|REJECT|MERGE|ENROLL|PROMOTE|SYNC_TT|DELETE
    payload = db.Column(db.Text)                        # JSON, e.g. {"target_driver_id": "DRV-..."}
    requested_by = db.Column(db.String(100))
    requested_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False,
                             server_default=db.func.current_timestamp())
    status = db.Column(db.String(12), nullable=False, default='PENDING', server_default='PENDING',
                       index=True)  # PENDING|RUNNING|DONE|FAILED
    attempts = db.Column(db.Integer, nullable=False, default=0, server_default='0')
    processed_at = db.Column(db.DateTime)
    error = db.Column(db.Text)

    LABELS = {'CONFIRM': 'Trust Track rename', 'REJECT': 'Trust Track rename', 'SYNC_TT': 'Trust Track sync',
              'MERGE': 'Move face templates + Trust Track rename', 'ENROLL': 'Build recognition templates',
              'PROMOTE': 'Build recognition templates', 'DELETE': 'Purge recognition data'}

    @property
    def label(self):
        return self.LABELS.get(self.action, self.action)


class DriverAlertThrottle(db.Model):
    """Written by the recognizer; one row per (driver, vehicle, alert kind)."""
    __tablename__ = 'driver_alert_throttle_reinke_logistics'

    driver_id = db.Column(db.String(30), primary_key=True)
    vehicle = db.Column(db.String(100), primary_key=True)
    kind = db.Column(db.String(20), primary_key=True, default='unidentified',
                     server_default='unidentified')   # unidentified
    last_alert_at = db.Column(db.DateTime, nullable=False)
    alert_count = db.Column(db.Integer, nullable=False, default=1, server_default='1')


class RecognitionEvent(db.Model):
    __tablename__ = 'recognition_events_reinke_logistics'
    
    id = db.Column(db.Integer, primary_key=True)
    driver_id = db.Column(db.Integer, db.ForeignKey('drivers_reinke_logistics.id'), nullable=True)
    driver_id_string = db.Column(db.String(20))
    video_filename = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    confidence = db.Column(db.Float)
    frame_number = db.Column(db.Integer)
    status = db.Column(db.String(20))
    thumbnail_path = db.Column(db.String(200))
    
    def to_dict(self):
        return {
            'id': self.id,
            'driver_id': self.driver_id,
            'driver_id_string': self.driver_id_string,
            'driver_name': f"{self.driver.first_name} {self.driver.last_name}" if self.driver else "Unknown",
            'video_filename': self.video_filename,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M') if self.timestamp else None,
            'confidence': round(self.confidence * 100, 1) if self.confidence else 0,
            'status': self.status,
            'thumbnail_path': self.thumbnail_path
        }


class RecognitionEventLatest(db.Model):
    __tablename__ = 'recognition_events_latest_reinke_logistics'
    
    id = db.Column(db.Integer, primary_key=True)
    driver_id = db.Column(db.Integer, nullable=True)
    driver_id_string = db.Column(db.String(50))
    video_filename = db.Column(db.String(255), unique=True)
    frame_number = db.Column(db.Integer)
    confidence = db.Column(db.Float)
    status = db.Column(db.String(50))
    thumbnail_path = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'driver_id': self.driver_id,
            'driver_id_string': self.driver_id_string,
            'video_filename': self.video_filename,
            'frame_number': self.frame_number,
            'confidence': self.confidence,
            'status': self.status,
            'thumbnail_path': self.thumbnail_path,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M') if self.updated_at else None,
        }


class RolePermission(db.Model):
    """Stores per-role permission overrides. Rows only exist when the default is overridden."""
    __tablename__ = 'role_permissions_reinke_logistics'

    id         = db.Column(db.Integer, primary_key=True)
    role       = db.Column(db.String(20), nullable=False)      # admin | support | client
    permission = db.Column(db.String(80), nullable=False)      # e.g. 'can_modify_driver'
    granted    = db.Column(db.Boolean, nullable=False, default=True)
    updated_at = db.Column(db.DateTime, default=__import__('datetime').datetime.utcnow,
                           onupdate=__import__('datetime').datetime.utcnow)

    __table_args__ = (db.UniqueConstraint('role', 'permission', name='uq_role_perm'),)

    @staticmethod
    def get(role: str, permission: str, default: bool) -> bool:
        row = RolePermission.query.filter_by(role=role, permission=permission).first()
        return row.granted if row else default

    def to_dict(self):
        return {
            'role':       self.role,
            'permission': self.permission,
            'granted':    self.granted,
        }


class Trip(db.Model):
    """
    One vehicle trip pulled from FM-Track, enriched with driver-assignment info.
    Loaded exclusively by the standalone pipeline/load_trips.py script — the web
    app only ever reads this table (never writes trip data itself).

    All timestamps are naive datetimes representing SAST (UTC+2) wall-clock time,
    matching what the FM-Track API / extractor scripts have always shown.
    """
    __tablename__ = 'trips_reinke_logistics_v2'

    id = db.Column(db.Integer, primary_key=True)

    # Stable de-duplication key so re-running the pipeline safely upserts
    # instead of creating duplicate rows: vehicle + start + end time.
    external_id = db.Column(db.String(64), unique=True, nullable=False, index=True)

    vehicle = db.Column(db.String(100), nullable=False, index=True)

    start_time = db.Column(db.DateTime, nullable=True, index=True)
    end_time   = db.Column(db.DateTime, nullable=True)
    duration_min = db.Column(db.Float, nullable=True)

    start_location = db.Column(db.Text, nullable=True)
    end_location   = db.Column(db.Text, nullable=True)
    start_zone_name = db.Column(db.String(150), nullable=True, index=True)
    start_lat = db.Column(db.Float, nullable=True)
    start_lng = db.Column(db.Float, nullable=True)
    end_lat   = db.Column(db.Float, nullable=True)
    end_lng   = db.Column(db.Float, nullable=True)

    km = db.Column(db.Float, nullable=True)
    trip_type = db.Column(db.String(40), nullable=True)

    driver_name = db.Column(db.String(150), nullable=True, index=True)   # raw name from FM-Track / facial recognition
    has_driver  = db.Column(db.Boolean, nullable=False, default=False, index=True)
    # Where driver_name came from: 'trusttrack' (on the trip in Trust Track), 'recognition'
    # (no driver in Trust Track - the driver facial recognition saw in the vehicle during the
    # trip) or 'state' (only a state profile such as "no face" was on the vehicle).
    driver_source = db.Column(db.String(20), nullable=True)
    # The web app's DRV id of that driver (names are refreshed from the drivers table every load)
    driver_code = db.Column(db.String(30), nullable=True, index=True)

    # Trailers / towed units never carry a driver of their own — flagged so the
    # dashboard can exclude them from assignment-rate calculations by default.
    is_excluded = db.Column(db.Boolean, nullable=False, default=False, index=True)

    source     = db.Column(db.String(30), default='fmtrack')
    loaded_at  = db.Column(db.DateTime, default=datetime.utcnow)   # last pipeline run that touched this row
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.Index('ix_trips_v2_vehicle_start', 'vehicle', 'start_time'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'vehicle': self.vehicle,
            'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else None,
            'end_time': self.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.end_time else None,
            'duration_min': round(self.duration_min, 1) if self.duration_min is not None else None,
            'start_location': self.start_location,
            'end_location': self.end_location,
            'start_zone_name': self.start_zone_name,
            'km': round(self.km, 2) if self.km is not None else None,
            'trip_type': self.trip_type,
            'driver_name': self.driver_name or None,
            'has_driver': self.has_driver,
            'driver_source': self.driver_source,
            'driver_code': self.driver_code,
            'is_excluded': self.is_excluded,
        }


class PipelineRun(db.Model):
    """Audit trail for the standalone data-load pipeline (pipeline/load_trips.py)."""
    __tablename__ = 'pipeline_runs_reinke_logistics'

    id = db.Column(db.Integer, primary_key=True)
    started_at  = db.Column(db.DateTime, default=datetime.utcnow)
    finished_at = db.Column(db.DateTime, nullable=True)
    status      = db.Column(db.String(20), default='running')   # running | success | failed
    range_from  = db.Column(db.DateTime, nullable=True)
    range_to    = db.Column(db.DateTime, nullable=True)
    trips_fetched  = db.Column(db.Integer, default=0)
    trips_inserted = db.Column(db.Integer, default=0)
    trips_updated  = db.Column(db.Integer, default=0)
    vehicles_seen  = db.Column(db.Integer, default=0)
    drivers_seen   = db.Column(db.Integer, default=0)
    message = db.Column(db.Text, nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'started_at': self.started_at.strftime('%Y-%m-%d %H:%M:%S') if self.started_at else None,
            'finished_at': self.finished_at.strftime('%Y-%m-%d %H:%M:%S') if self.finished_at else None,
            'status': self.status,
            'range_from': self.range_from.strftime('%Y-%m-%d %H:%M:%S') if self.range_from else None,
            'range_to': self.range_to.strftime('%Y-%m-%d %H:%M:%S') if self.range_to else None,
            'trips_fetched': self.trips_fetched,
            'trips_inserted': self.trips_inserted,
            'trips_updated': self.trips_updated,
            'vehicles_seen': self.vehicles_seen,
            'drivers_seen': self.drivers_seen,
            'message': self.message,
        }


class GeoZone(db.Model):
    """
    A named geofence zone (circle or polygon) pulled from FM-Track, used to
    tag/lookup trip start & end locations. Loaded exclusively by the
    standalone pipeline/load_trips.py script — the web app only ever reads
    this table.
    """
    __tablename__ = 'geozones_reinke_logistics'

    id = db.Column(db.Integer, primary_key=True)

    # FM-Track's own zone id, used as the de-duplication / upsert key.
    external_id = db.Column(db.String(64), unique=True, nullable=False, index=True)

    name  = db.Column(db.String(150), nullable=True)
    notes = db.Column(db.Text, nullable=True)

    zone_type = db.Column(db.String(20), nullable=True)   # CIRCLE | POLYGON

    # Bounding box (used for fast containment pre-filtering).
    min_lat = db.Column(db.Float, nullable=True)
    max_lat = db.Column(db.Float, nullable=True)
    min_lng = db.Column(db.Float, nullable=True)
    max_lng = db.Column(db.Float, nullable=True)

    # Circle-specific fields.
    center_lat = db.Column(db.Float, nullable=True)
    center_lng = db.Column(db.Float, nullable=True)
    radius_m   = db.Column(db.Float, nullable=True)

    # Full GeoJSON geometry (for polygons, or for precise point-in-polygon checks).
    geometry_json = db.Column(db.Text, nullable=True)

    loaded_at  = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'external_id': self.external_id,
            'name': self.name,
            'notes': self.notes,
            'zone_type': self.zone_type,
            'min_lat': self.min_lat,
            'max_lat': self.max_lat,
            'min_lng': self.min_lng,
            'max_lng': self.max_lng,
            'center_lat': self.center_lat,
            'center_lng': self.center_lng,
            'radius_m': self.radius_m,
        }
