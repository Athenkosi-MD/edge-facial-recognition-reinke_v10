import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load .env from the project root (safe no-op if file is missing)
load_dotenv(os.path.join(os.path.abspath(os.path.dirname(__file__)), '.env'))

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ── Fill these in (an environment variable / .env entry of the same name wins) ─
DB_USER     = os.environ.get('DB_USER', 'edge_admin')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'OlU01)vX0B!')
DB_HOST     = os.environ.get('DB_HOST', '173.249.46.82')
DB_PORT     = os.environ.get('DB_PORT', '5432')
DB_NAME     = os.environ.get('DB_NAME', 'postgres')


def _database_url():
    if os.environ.get('DATABASE_URL'):
        return os.environ['DATABASE_URL']
    missing = [n for n, v in (('DB_USER', DB_USER), ('DB_PASSWORD', DB_PASSWORD), ('DB_HOST', DB_HOST)) if not v]
    if missing:
        raise SystemExit(f"config.py: fill in {', '.join(missing)} (or set DATABASE_URL) - "
                         f"the web app has no database to connect to.")
    # quote_plus: passwords with ) ! @ : / etc. would otherwise break the URL
    return f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-prod-2026'

    SQLALCHEMY_DATABASE_URI = _database_url()
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Local folders kept for videos (not stored in R2)
    VIDEOS_FOLDER = os.path.join(BASE_DIR, 'data', 'videos')

    # ── Cloudflare R2 ─────────────────────────────────────────────────────────
    CF_ACCOUNT_ID        = os.environ.get('CF_ACCOUNT_ID', 'dd396ee03a6ad5acff2d6062dd00bb0b')
    R2_ACCESS_KEY_ID     = os.environ.get('R2_ACCESS_KEY_ID', 'e5bfac6080fb9f673d0d4140541a1704')
    R2_SECRET_ACCESS_KEY = os.environ.get('R2_SECRET_ACCESS_KEY', '74c217098c498e9ea18f9afdd25b77f53e89e16fae29f66cbc7cf538e1fc83db')
    R2_BUCKET            = os.environ.get('R2_BUCKET', 'reinke-logistix')

    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024

    # Local embeddings cache folder (used by video recognition service)
    EMBEDDINGS_FOLDER = os.path.join(BASE_DIR, 'data', 'embeddings')

    FACE_DETECTION_CONFIDENCE = 0.5
    FACE_RECOGNITION_THRESHOLD = 0.4
    ENROLLMENT_MIN_PHOTOS = 5
    ENROLLMENT_MAX_PHOTOS = 20
