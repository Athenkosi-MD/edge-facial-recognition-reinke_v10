"""
REPLACE your existing app/__init__.py with this file.
Changes made:
  - Registers the auth Blueprint
  - Adds SECRET_KEY check
  - Sets session lifetime
"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os
from datetime import timedelta
from dotenv import load_dotenv

# Load .env early so credentials are in os.environ before config.py reads them
load_dotenv(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '.env'))

db = SQLAlchemy()


def _migrate_states_and_statuses(engine, tables):
    """Fold legacy enrollment states / statuses into the current set.

    Enrollment states: FULLY_ENROLLED, AUTO_ENROLLED, UNIDENTIFIED.
    Statuses: active, deleted. Safe to run on every start-up."""
    import logging
    from sqlalchemy import text
    log = logging.getLogger(__name__)
    D = 'drivers_reinke_logistics'
    H = 'driver_state_history_reinke_logistics'
    Q = 'driver_lifecycle_actions_reinke_logistics'
    with engine.begin() as conn:
        # MERGED: back to the state it had before the merge (history), else UNIDENTIFIED
        merged = conn.execute(text(f"SELECT driver_id FROM {D} WHERE enrollment_state = 'MERGED'")).fetchall()
        for (drv,) in merged:
            prev = None
            if H in tables:
                prev = conn.execute(text(
                    f"SELECT from_state FROM {H} WHERE driver_id = :d AND to_state = 'MERGED' "
                    f"ORDER BY id DESC LIMIT 1"), {'d': drv}).scalar()
            if prev not in ('AUTO_ENROLLED', 'FULLY_ENROLLED'):
                prev = 'UNIDENTIFIED'
            conn.execute(text(f"UPDATE {D} SET enrollment_state = :s, status = 'deleted' "
                              f"WHERE driver_id = :d"), {'s': prev, 'd': drv})
        n_exp = conn.execute(text(
            f"UPDATE {D} SET enrollment_state = 'UNIDENTIFIED', status = 'deleted' "
            f"WHERE enrollment_state = 'EXPIRED'")).rowcount
        # REJECTED ("Unauthorised") is now simply Unidentified
        conn.execute(text(
            f"UPDATE {D} SET first_name = 'Unidentified' WHERE enrollment_state = 'REJECTED' "
            f"AND first_name IN ('Unauthorised', 'Unauthorized')"))
        n_rej = conn.execute(text(
            f"UPDATE {D} SET enrollment_state = 'UNIDENTIFIED' WHERE enrollment_state = 'REJECTED'")).rowcount
        n_any = conn.execute(text(
            f"UPDATE {D} SET enrollment_state = 'UNIDENTIFIED' WHERE enrollment_state IS NULL "
            f"OR enrollment_state NOT IN ('FULLY_ENROLLED', 'AUTO_ENROLLED', 'UNIDENTIFIED')")).rowcount
        # Statuses: active | deleted
        n_st = conn.execute(text(
            f"UPDATE {D} SET status = 'deleted' WHERE status IS NOT NULL "
            f"AND status NOT IN ('active', 'deleted')")).rowcount
        conn.execute(text(f"UPDATE {D} SET status = 'active' WHERE status IS NULL"))
        # Auto-enrolled without a name: "Confirmed <vehicle> <date>" -> "Auto Enrolled ..."
        renamed = conn.execute(text(
            f"SELECT driver_id FROM {D} WHERE enrollment_state = 'AUTO_ENROLLED' "
            f"AND first_name = 'Confirmed'")).fetchall()
        if renamed:
            conn.execute(text(
                f"UPDATE {D} SET first_name = 'Auto Enrolled' WHERE enrollment_state = 'AUTO_ENROLLED' "
                f"AND first_name = 'Confirmed'"))
            if Q in tables:   # let the recognizer push the new name to Trust Track
                for (drv,) in renamed:
                    conn.execute(text(
                        f"INSERT INTO {Q} (driver_id, action, requested_by, requested_at, status, attempts) "
                        f"VALUES (:d, 'SYNC_TT', 'system', CURRENT_TIMESTAMP, 'PENDING', 0)"), {'d': drv})
        if merged or n_exp or n_rej or n_any or n_st or renamed:
            log.info(f"Migration: states/statuses - {len(merged)} merged, {n_exp} expired, "
                     f"{n_rej} unauthorised->unidentified, {n_any} other state, {n_st} status -> deleted, "
                     f"{len(renamed)} 'Confirmed' -> 'Auto Enrolled'")


def create_app(config=None):
    app = Flask(__name__)

    from config import Config
    app.config.from_object(Config)

    if config:
        app.config.update(config)

    # Session settings
    app.config.setdefault('PERMANENT_SESSION_LIFETIME', timedelta(hours=8))

    # Only videos are stored locally; all other files go to R2
    os.makedirs(app.config['VIDEOS_FOLDER'], exist_ok=True)
    # Local embeddings cache used by video_service for offline recognition
    os.makedirs(app.config.get('EMBEDDINGS_FOLDER',
                               os.path.join(app.root_path, '..', 'data', 'embeddings')),
                exist_ok=True)

    db.init_app(app)

    # Register numpy-safe JSON encoder (prevents int32/float32 serialisation errors)
    from app.routes import _NumpySafeProvider
    app.json_provider_class = _NumpySafeProvider
    app.json = _NumpySafeProvider(app)

    with app.app_context():
        from app import models
        db.create_all()   # creates RolePermission table automatically

        # ── Lightweight migration: add `role` column to drivers if missing ────
        # Safe to run every startup. Required when an existing drivers.db was
        # created before the License -> Role change.
        try:
            from sqlalchemy import text, inspect
            inspector = inspect(db.engine)
            if 'drivers_reinke_logistics' in inspector.get_table_names():
                cols = {c['name'] for c in inspector.get_columns('drivers_reinke_logistics')}
                if 'role' not in cols:
                    with db.engine.begin() as conn:
                        conn.execute(text(
                            "ALTER TABLE drivers_reinke_logistics ADD COLUMN role VARCHAR(20) DEFAULT 'Driver'"
                        ))
                    import logging
                    logging.getLogger(__name__).info(
                        "Migration: added 'role' column to drivers table."
                    )

            # ── Enrollment-state columns (UNIDENTIFIED / AUTO / FULLY ...) ──
            # Column-by-column so it works on Postgres and SQLite alike.
            # Existing drivers get the default FULLY_ENROLLED.
            if 'drivers_reinke_logistics' in inspector.get_table_names():
                cols = {c['name'] for c in inspector.get_columns('drivers_reinke_logistics')}
                new_cols = [
                    ("enrollment_state", "VARCHAR(20) NOT NULL DEFAULT 'FULLY_ENROLLED'"),
                    ("trusttrack_id", "VARCHAR(64)"),
                    ("trusttrack_sync_pending", "BOOLEAN NOT NULL DEFAULT FALSE"),
                    ("first_seen_at", "TIMESTAMP"),
                    ("first_seen_vehicle", "VARCHAR(100)"),
                    ("last_seen_at", "TIMESTAMP"),
                    ("last_seen_vehicle", "VARCHAR(100)"),
                    ("sighting_count", "INTEGER NOT NULL DEFAULT 0"),
                    ("confirmed_by", "VARCHAR(100)"),
                    ("confirmed_at", "TIMESTAMP"),
                    ("merged_into", "VARCHAR(30)"),
                    ("state_note", "TEXT"),
                    ("lookalikes", "TEXT"),
                    ("confirmed_by_client", "VARCHAR(150)"),
                    ("name_pending", "BOOLEAN NOT NULL DEFAULT FALSE"),
                ]
                with db.engine.begin() as conn:
                    for name, ddl in new_cols:
                        if name not in cols:
                            conn.execute(text(
                                f"ALTER TABLE drivers_reinke_logistics ADD COLUMN {name} {ddl}"))
                    try:
                        conn.execute(text(
                            "CREATE INDEX IF NOT EXISTS ix_drivers_rl_enrollment_state "
                            "ON drivers_reinke_logistics (enrollment_state)"))
                    except Exception:
                        pass

                # ── Three enrollment states, two statuses ──────────────────
                # States: FULLY_ENROLLED, AUTO_ENROLLED, UNIDENTIFIED.
                # Statuses: active, deleted. Older values are folded in:
                #   MERGED   -> the state it had before the merge, status deleted
                #   EXPIRED  -> UNIDENTIFIED, status deleted
                #   REJECTED ("unauthorised") -> UNIDENTIFIED
                #   inactive / suspended -> deleted
                # Auto-enrolled drivers without a name are labelled "Auto
                # Enrolled <vehicle> <date>" (was "Confirmed ..."). Idempotent.
                _migrate_states_and_statuses(db.engine, inspector.get_table_names())

            # ── Migrate users: rename viewer->client if needed ──────────────
            if 'users_facial_recognition_reinke_logistics' in inspector.get_table_names():
                with db.engine.begin() as conn:
                    conn.execute(text(
                        "UPDATE users_facial_recognition_reinke_logistics SET role='client' WHERE role='viewer'"
                    ))

            # ── Trips: driver_source (Trust Track / facial recognition / state) ──
            if 'trips_reinke_logistics_v2' in inspector.get_table_names():
                tcols = {c['name'] for c in inspector.get_columns('trips_reinke_logistics_v2')}
                with db.engine.begin() as conn:
                    if 'driver_source' not in tcols:
                        conn.execute(text("ALTER TABLE trips_reinke_logistics_v2 "
                                          "ADD COLUMN driver_source VARCHAR(20)"))
                    if 'driver_code' not in tcols:
                        conn.execute(text("ALTER TABLE trips_reinke_logistics_v2 "
                                          "ADD COLUMN driver_code VARCHAR(30)"))

            # Note: the trips table is named trips_reinke_logistics_v2 — a fresh
            # name — specifically so db.create_all() above just creates it
            # cleanly with the correct schema. No ALTER-based migration is
            # needed for it. (An older, incompatible trips_reinke_logistics
            # table may still exist from an earlier attempt; it's simply no
            # longer used and can be dropped manually whenever convenient.)
        except Exception as _mig_err:
            import logging
            logging.getLogger(__name__).warning(
                f"Driver role migration skipped: {_mig_err}"
            )

        # ── Create a default admin user on first run ──────────────────────────
        from app.models import User
        if User.query.count() == 0:
            admin = User(username='admin', email='admin@edgetelematics.local',
                         role='admin', is_active=True)
            admin.set_password('Admin@1234')      # CHANGE THIS after first login
            db.session.add(admin)
            db.session.commit()
            import logging
            logging.getLogger(__name__).warning(
                "Default admin created: username=admin  password=Admin@1234 — CHANGE IT NOW!"
            )

    # Blueprints
    from app.routes import main
    app.register_blueprint(main)

    from app.auth import auth
    app.register_blueprint(auth)

    # ── Inject session into all templates ─────────────────────────────────────
    from flask import session as _session
    @app.context_processor
    def inject_session():
        return dict(session=_session)

    # Driver permissions of the logged-in user, for showing/hiding buttons:
    # {% if perms.can_modify_driver %} ...  (the routes enforce them too)
    @app.context_processor
    def inject_permissions():
        from app.models import User
        empty = {p: False for p in User.DRIVER_PERMISSIONS}
        if 'user_id' not in _session:
            return dict(perms=empty)
        try:
            user = User.query.filter_by(user_id=_session['user_id']).first()
            return dict(perms=user.driver_permissions() if user else empty)
        except Exception:
            return dict(perms=empty)

    # Nav badge: drivers waiting for OCC review
    @app.context_processor
    def inject_unidentified_count():
        if 'user_id' not in _session:
            return {}
        try:
            from app.models import Driver
            return dict(unidentified_count=Driver.query.filter_by(enrollment_state='UNIDENTIFIED', status='active').count())
        except Exception:
            return dict(unidentified_count=0)

    return app
