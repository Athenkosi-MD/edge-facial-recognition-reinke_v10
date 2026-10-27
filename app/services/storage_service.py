"""
Cloudflare R2 storage service — reinke-logistix bucket.

All credentials are read from the Flask app config (which loads them from
environment variables / .env via config.py).  Nothing is hardcoded here.

Key layout in the bucket
────────────────────────
  drivers/{driver_id}/photo_000.jpg   ← enrollment training photos
  uploads/{driver_id}_profile.jpg     ← display profile photo
  embeddings/{driver_id}.pkl          ← pickled face-embedding list (buffalo_sc, web app)

Owned by the recognizer scripts - the web app NEVER writes or deletes these
(it only reads provisional/ to show OCC the cab snapshots):
  embeddings_buffalo_l/{driver_id}.pkl
  provisional/{driver_id}/*.jpg
"""

import pickle
import logging
import os

import boto3
from flask import current_app

logger = logging.getLogger(__name__)

# ── Singleton boto3 client ────────────────────────────────────────────────────
# Created once per process.  Re-creating it on every request caused a new HTTPS
# connection + SSL handshake each time, which caused Gunicorn workers to time
# out during multi-photo enrollment.

_client = None
_client_config: dict = {}   # tracks which config was used to build the client


def _get_client():
    """Return (and lazily create) the module-level boto3 S3 client."""
    global _client, _client_config

    # Pull credentials from Flask app config so they come from .env / env vars.
    try:
        cfg = current_app.config
        account_id  = cfg['CF_ACCOUNT_ID']
        access_key  = cfg['R2_ACCESS_KEY_ID']
        secret_key  = cfg['R2_SECRET_ACCESS_KEY']
    except RuntimeError:
        # Outside of a Flask application context (e.g. standalone scripts).
        # Fall back to environment variables directly.
        account_id  = os.environ['CF_ACCOUNT_ID']
        access_key  = os.environ['R2_ACCESS_KEY_ID']
        secret_key  = os.environ['R2_SECRET_ACCESS_KEY']

    new_cfg = {'account': account_id, 'key': access_key}
    if _client is None or new_cfg != _client_config:
        _client = boto3.client(
            's3',
            endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='auto',
        )
        _client_config = new_cfg
        logger.info('[R2] boto3 client (re)created for account %s', account_id)

    return _client


def _bucket() -> str:
    """Return the configured bucket name."""
    try:
        return current_app.config['R2_BUCKET']
    except RuntimeError:
        return os.environ.get('R2_BUCKET', 'reinke-logistix')


# ── Convenience alias used by routes.py imports ───────────────────────────────
# Some route files do `from app.services.storage_service import BUCKET`
# We keep the name but make it a property-like function call via the alias below.
# For backward-compat we also expose a module-level BUCKET that is read at
# import time (will equal the env-var value before Flask starts).
BUCKET = os.environ.get('R2_BUCKET', 'reinke-logistix')


# ── Driver enrollment photos ──────────────────────────────────────────────────

def upload_driver_photo(driver_id: str, filename: str, file_bytes: bytes) -> str:
    """Upload a face-enrollment training photo.  Returns the R2 key."""
    key = f'drivers/{driver_id}/{filename}'
    _get_client().put_object(Bucket=_bucket(), Key=key, Body=file_bytes,
                             ContentType='image/jpeg')
    logger.debug('[R2] Uploaded enrollment photo: %s', key)
    return key


def list_driver_photos(driver_id: str) -> list:
    """Return a list of all enrollment photo keys for a driver."""
    resp = _get_client().list_objects_v2(
        Bucket=_bucket(), Prefix=f'drivers/{driver_id}/'
    )
    return [obj['Key'] for obj in resp.get('Contents', [])]


def download_driver_photo(key: str) -> bytes:
    """Download a photo by its R2 key (used for recognition processing)."""
    resp = _get_client().get_object(Bucket=_bucket(), Key=key)
    return resp['Body'].read()


def delete_driver_photo(key: str) -> None:
    """Delete a single photo by key."""
    _get_client().delete_object(Bucket=_bucket(), Key=key)
    logger.debug('[R2] Deleted photo: %s', key)


def delete_driver_all(driver_id: str) -> int:
    """Delete all R2 objects for a driver (photos + profile + embeddings).
    Returns the number of objects deleted."""
    client = _get_client()
    bucket = _bucket()
    prefixes = [
        f'drivers/{driver_id}/',
        f'uploads/{driver_id}_profile.jpg',
        f'embeddings/{driver_id}.pkl',
    ]
    # embeddings_buffalo_l/ and provisional/ belong to the recognizer; a
    # DELETE action is queued for it instead (see lifecycle_service.enqueue).
    deleted = 0
    for prefix in prefixes:
        resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in resp.get('Contents', []):
            client.delete_object(Bucket=bucket, Key=obj['Key'])
            deleted += 1
    logger.info('[R2] Deleted %d objects for driver %s', deleted, driver_id)
    return deleted


# ── Embeddings ────────────────────────────────────────────────────────────────

def save_embeddings(driver_id: str, embeddings) -> str:
    """Pickle and upload face embeddings.  Returns the R2 key."""
    key = f'embeddings/{driver_id}.pkl'
    data = pickle.dumps(embeddings)
    _get_client().put_object(Bucket=_bucket(), Key=key, Body=data)
    logger.debug('[R2] Saved embeddings: %s', key)
    return key


def load_embeddings(driver_id: str):
    """Download and unpickle embeddings for one driver.  Returns None on miss."""
    key = f'embeddings/{driver_id}.pkl'
    try:
        resp = _get_client().get_object(Bucket=_bucket(), Key=key)
        return pickle.loads(resp['Body'].read())
    except Exception:
        return None


def load_all_embeddings() -> dict:
    """Load every driver's embeddings — used by the recognition pipeline."""
    client = _get_client()
    bucket = _bucket()
    resp = client.list_objects_v2(Bucket=bucket, Prefix='embeddings/')
    all_emb = {}
    for obj in resp.get('Contents', []):
        key = obj['Key']
        driver_id = key.removeprefix('embeddings/').removesuffix('.pkl')
        try:
            data = client.get_object(Bucket=bucket, Key=key)
            all_emb[driver_id] = pickle.loads(data['Body'].read())
        except Exception as e:
            logger.warning('[R2] Could not load embedding for %s: %s', driver_id, e)
    return all_emb


# ── Profile photos (displayed in the UI) ─────────────────────────────────────

def upload_profile_photo(driver_id: str, file_bytes: bytes) -> str:
    """Upload (or replace) the profile display photo.  Returns the R2 key."""
    key = f'uploads/{driver_id}_profile.jpg'
    _get_client().put_object(Bucket=_bucket(), Key=key, Body=file_bytes,
                             ContentType='image/jpeg')
    logger.debug('[R2] Uploaded profile photo: %s', key)
    return key


def get_profile_photo_url(driver_id: str, expires_in: int = 3600) -> str | None:
    """Generate a temporary pre-signed URL for displaying a driver's profile photo.
    Returns None if the object doesn't exist or the request fails."""
    key = f'uploads/{driver_id}_profile.jpg'
    try:
        return _get_client().generate_presigned_url(
            'get_object',
            Params={'Bucket': _bucket(), 'Key': key},
            ExpiresIn=expires_in,
        )
    except Exception as e:
        logger.warning('[R2] Could not generate presigned URL for %s: %s', key, e)
        return None


def get_presigned_url(key: str, expires_in: int = 3600) -> str | None:
    """Generate a pre-signed URL for any arbitrary R2 key."""
    try:
        return _get_client().generate_presigned_url(
            'get_object',
            Params={'Bucket': _bucket(), 'Key': key},
            ExpiresIn=expires_in,
        )
    except Exception as e:
        logger.warning('[R2] Could not generate presigned URL for %s: %s', key, e)
        return None


# ── Provisional cab snapshots (READ-ONLY here; written by the recognizer) ──────

PROVISIONAL_PREFIX = 'provisional/'


def list_provisional_photos(driver_id: str) -> list:
    resp = _get_client().list_objects_v2(Bucket=_bucket(), Prefix=f'{PROVISIONAL_PREFIX}{driver_id}/')
    return sorted(o['Key'] for o in resp.get('Contents', []))
