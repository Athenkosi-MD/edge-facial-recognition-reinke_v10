# gunicorn.conf.py

import os
import multiprocessing

# ── Bind (REQUIRED for Render) ────────────────────────────────────────────────
# Ensures Gunicorn listens on the port provided by Render via $PORT
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"

# ── Workers ───────────────────────────────────────────────────────────────────
# Keep at 1 on Render free tier to avoid memory issues (especially with ML models)
workers = 1

# ── Timeout ───────────────────────────────────────────────────────────────────
# Increased to allow longer processing (e.g., image processing, uploads)
timeout = 120

# ── Keep-alive ────────────────────────────────────────────────────────────────
keepalive = 5

# ── Logging ───────────────────────────────────────────────────────────────────
loglevel = "info"
accesslog = "-"   # log to stdout
errorlog  = "-"   # log to stdout
