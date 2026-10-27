"""
Driver enrollment lifecycle - web-app side (DATABASE ONLY).

The web app runs at the client and is kept separate from the recognizer
scripts. It never touches the recognizer's R2 data (embeddings_buffalo_l/,
provisional/) and never calls Trust Track. It only:

  1. changes the driver row / enrollment_state in Postgres (so the UI and the
     recognizer see OCC's decision immediately), and
  2. queues an action in driver_lifecycle_actions_reinke_logistics.

The recognizer scripts poll that queue and do the side effects:
  CONFIRM / REJECT / SYNC_TT  rename or create the Trust Track driver
                              (REJECT = back to unidentified)
  MERGE                       move face templates in R2 to the real driver,
                              rename the provisional Trust Track driver
  ENROLL / PROMOTE            build buffalo_l templates from the enrollment
                              photos (keeping cab templates), Trust Track driver
  DELETE                      purge the recognizer's R2 data for the driver

Enrollment states: FULLY_ENROLLED, AUTO_ENROLLED, UNIDENTIFIED.
Statuses: active, deleted.

    UNIDENTIFIED ──confirm──▶ AUTO_ENROLLED ──full enrollment──▶ FULLY_ENROLLED
         ▲                        │
         └──mark unidentified─────┘
    merge: the provisional record keeps its state and gets status 'deleted'
"""

import json
import logging
from datetime import datetime

from flask import session

from app import db
from app.models import (Driver, DriverStateHistory, DriverLifecycleAction,
                        RecognitionEvent, RecognitionEventLatest)

log = logging.getLogger(__name__)

UNIDENTIFIED, AUTO_ENROLLED, FULLY_ENROLLED = 'UNIDENTIFIED', 'AUTO_ENROLLED', 'FULLY_ENROLLED'
STATES = (FULLY_ENROLLED, AUTO_ENROLLED, UNIDENTIFIED)
ACTIVE, DELETED = 'active', 'deleted'
STATUSES = (ACTIVE, DELETED)


def _actor():
    return session.get('username') or 'system'


def enqueue(driver_id: str, action: str, payload: dict = None):
    """Queue work for the recognizer scripts. Caller commits."""
    db.session.add(DriverLifecycleAction(
        driver_id=driver_id, action=action,
        payload=json.dumps(payload) if payload else None,
        requested_by=_actor(), status='PENDING'))


def record_state(driver: Driver, to_state: str, note: str = None):
    frm = driver.state
    driver.enrollment_state = to_state
    db.session.add(DriverStateHistory(driver_id=driver.driver_id, from_state=frm, to_state=to_state,
                                      changed_by=_actor(), note=note))


def pending_actions(driver_id: str):
    return (DriverLifecycleAction.query.filter_by(driver_id=driver_id)
            .filter(DriverLifecycleAction.status.in_(['PENDING', 'FAILED']))
            .order_by(DriverLifecycleAction.id).all())


# ── OCC actions ───────────────────────────────────────────────────────────────

def confirm(driver: Driver, client_name, note=None, first_name=None, last_name=None,
            phone=None, email=None, role=None):
    """UNIDENTIFIED -> AUTO_ENROLLED.

    `client_name` (the client contact who authorized the driver) is mandatory;
    the note and the driver's name are optional: without it the driver
    is still auto-enrolled (alerts stop) and flagged name_pending, showing as
    "Auto Enrolled <vehicle> <date>" until someone fills the name in."""
    if driver.state != UNIDENTIFIED:
        raise ValueError(f'{driver.driver_id} is {driver.enrollment_state}; only unidentified drivers can be confirmed')
    client_name, note = (client_name or '').strip(), (note or '').strip()
    if not client_name:
        raise ValueError('Enter the client contact who authorized the driver')
    first_name, last_name = (first_name or '').strip(), (last_name or '').strip()
    if first_name and last_name:
        driver.first_name, driver.last_name = first_name, last_name
        driver.name_pending = False
    elif first_name or last_name:
        raise ValueError('Enter both first and last name, or leave both empty')
    else:
        # Keep the "<vehicle> <date>" label the recognizer gave it; Trust Track
        # shows "Auto Enrolled / R014 - CJ73834 2026-10-14" until a name is added.
        driver.first_name = Driver.AUTO_ENROLLED_PLACEHOLDER
        driver.name_pending = True
    if phone:
        driver.phone = phone
    if email:
        driver.email = email
    if role:
        driver.role = role
    driver.confirmed_by, driver.confirmed_at = _actor(), datetime.utcnow()
    driver.confirmed_by_client = client_name
    driver.state_note = note or None
    driver.status = ACTIVE
    record_state(driver, AUTO_ENROLLED, f'Authorized by client contact {client_name}' + (f': {note}' if note else ''))
    enqueue(driver.driver_id, 'CONFIRM')
    db.session.commit()


def reject(driver: Driver, note=None):
    """AUTO_ENROLLED -> UNIDENTIFIED (OCC withdraws the confirmation; alerts resume)."""
    if driver.state != AUTO_ENROLLED:
        raise ValueError(f'{driver.driver_id} is {driver.state_label}; only auto-enrolled drivers '
                         f'can be marked unidentified')
    if driver.name_pending and driver.first_name == Driver.AUTO_ENROLLED_PLACEHOLDER:
        driver.first_name = 'Unidentified'
    record_state(driver, UNIDENTIFIED, note or 'Marked unidentified by OCC')
    enqueue(driver.driver_id, 'REJECT')
    db.session.commit()


def mark_fully_enrolled(driver: Driver, note=None):
    """Called by the enroll route when an UNIDENTIFIED / AUTO_ENROLLED driver
    completes enrollment. Caller enqueues PROMOTE and commits."""
    if driver.state != FULLY_ENROLLED:
        if not driver.confirmed_at:
            driver.confirmed_by, driver.confirmed_at = _actor(), datetime.utcnow()
        record_state(driver, FULLY_ENROLLED, note or 'Full enrollment completed in the web app')


def merge(provisional: Driver, target: Driver, note=None):
    """The provisional driver turned out to be `target`. The DB side happens
    here (state, history, sightings); moving the face templates in R2 and
    renaming the provisional Trust Track driver is queued for the scripts.
    The provisional record keeps its enrollment state and is set to 'deleted'."""
    if provisional.id == target.id:
        raise ValueError('Cannot merge a driver into itself')
    if provisional.state not in (UNIDENTIFIED, AUTO_ENROLLED) or provisional.is_deleted:
        raise ValueError(f'{provisional.driver_id} is {provisional.state_label} / '
                         f'{provisional.status_label}; cannot merge')
    if target.is_deleted or target.merged_into:
        raise ValueError(f'{target.driver_id} is deleted; pick an active driver')

    RecognitionEvent.query.filter_by(driver_id=provisional.id).update(
        {'driver_id': target.id, 'driver_id_string': target.driver_id}, synchronize_session=False)
    RecognitionEventLatest.query.filter_by(driver_id_string=provisional.driver_id).update(
        {'driver_id': target.id, 'driver_id_string': target.driver_id}, synchronize_session=False)

    provisional.merged_into = target.driver_id
    provisional.status = DELETED
    target.sighting_count = (target.sighting_count or 0) + (provisional.sighting_count or 0)
    if provisional.last_seen_at and (not target.last_seen_at or provisional.last_seen_at > target.last_seen_at):
        target.last_seen_at, target.last_seen_vehicle = provisional.last_seen_at, provisional.last_seen_vehicle
    db.session.add(DriverStateHistory(
        driver_id=provisional.driver_id, from_state=provisional.state, to_state=provisional.state,
        changed_by=_actor(), note=note or f'Merged into {target.driver_id} by OCC; record deleted'))
    enqueue(provisional.driver_id, 'MERGE', {'target_driver_id': target.driver_id})
    db.session.commit()


# ── Look-alikes for the review card (computed by the recognizer) ──────────────

def nearest_drivers(driver: Driver, k: int = 3):
    """[(driver_id, similarity, likely_same_person)] the recognizer stored in
    drivers.lookalikes ("likely" = close enough that it is probably that driver)."""
    try:
        data = json.loads(driver.lookalikes or '[]')
    except ValueError:
        return []
    return [(x.get('driver_id'), float(x.get('similarity', 0)), bool(x.get('likely')))
            for x in data[:k] if x.get('driver_id')]
