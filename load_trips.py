"""
FM-Track Trip Loader (pipeline)
================================
Standalone data pipeline — completely separate from the web application
process. Run this on a schedule (cron / Task Scheduler / systemd timer) to
pull trips (and geozones) from the FM-Track API and load them into Postgres.
The Flask app never talks to FM-Track directly; it only ever reads the
`trips_reinke_logistics_v2` / `geozones_reinke_logistics` tables this script
maintains.

What it does, every run:
  1. Fetches all assets (vehicles) and drivers from FM-Track
  2. Fetches every geozone from FM-Track (with full geometry) and
     upserts them into the `geozones_reinke_logistics` table
  3. Fetches every trip for every asset in the requested date/time range
     (defaults to midnight today -> right now, SAST), splitting into
     monthly chunks and following continuation tokens automatically
  4. Computes the same derived fields the reports use: KM travelled,
     trip duration, whether a driver was identified, whether the
     vehicle is a trailer/towed unit that should be excluded from the
     driver-assignment check, and which geozone (if any) the trip's
     start latitude/longitude falls inside
  5. Upserts every trip into Postgres (safe to re-run — matching trips
     are updated in place, not duplicated)
  6. Records a PipelineRun row so the dashboard can show "last synced"

Usage:
    python3 pipeline/load_trips.py                       # today, midnight -> now
    python3 pipeline/load_trips.py --from 2026-07-01 --to 2026-07-23   # backfill a range
    python3 pipeline/load_trips.py --from 2026-07-01 --to 2026-07-23 --time-to 23:59:59

Run from the project root (same folder as app.py) so it can import the app package.
"""

import argparse
import hashlib
import math
from datetime import datetime, timezone, timedelta

import requests
from dateutil.relativedelta import relativedelta

# ── Config ────────────────────────────────────────────────────────────────────
import os as _os
try:  # same .env the web app uses (project root)
    from dotenv import load_dotenv as _ld
    _here = _os.path.dirname(_os.path.abspath(__file__))
    for _d in (_here, _os.path.dirname(_here)):
        if _os.path.exists(_os.path.join(_d, ".env")):
            _ld(_os.path.join(_d, ".env"))
            break
except ImportError:
    pass
API_KEY  = _os.environ.get("FMTRACK_API_KEY", 'NYlHLkrCD5bhahZl2Z2F3GwX8mmiyqH9')
BASE_URL = "https://api.fm-track.com"
OBJECTS_URL  = f"{BASE_URL}/objects?version=1&api_key={API_KEY}"
DRIVERS_URL  = f"{BASE_URL}/drivers?version=2&limit=1000&api_key={API_KEY}"
GEOZONES_URL = f"{BASE_URL}/geozones?version=1&api_key={API_KEY}&include_geometry=true&limit=1000"

SAST = timezone(timedelta(hours=2))  # South Africa Standard Time = UTC+2

# Vehicles to exclude from the driver-assignment check entirely — these are
# trailers/fridge units that are towed and never have a driver of their own.
EXCLUDED_VEHICLES = [
    "F108 (CJ13465)",
    "F321 (CJ76411)",
    "F315 (CJ56281)",
    "F320 ( CJ78984)",
    "RR38 - TEMPPX52276 Fridge Unit",
]


# ── FM-Track API helpers (same logic as all_vehicles.py) ──────────────────────

def fetch_json(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def build_trips_url(obj_id, from_dt, to_dt, continuation_token=None, limit=1000):
    url = (
        f"{BASE_URL}/objects/{obj_id}/trips"
        f"?version=1&from_datetime={from_dt}&to_datetime={to_dt}"
        f"&limit={limit}&api_key={API_KEY}"
    )
    if continuation_token:
        url += f"&continuation_token={continuation_token}"
    return url


def fetch_all_trips(obj_id, from_dt, to_dt):
    """Fetch ALL trips for one asset for a date range, following continuation tokens."""
    all_trips = []
    continuation_token = None
    while True:
        data = fetch_json(build_trips_url(obj_id, from_dt, to_dt, continuation_token))
        trips = data.get("trips", [])
        all_trips.extend(trips)
        continuation_token = data.get("continuation_token")
        if not continuation_token or len(trips) == 0:
            break
    return all_trips


def fetch_all_drivers(max_pages=50):
    """EVERY FM-Track driver. /drivers is paged (continuation_token): reading only
    the first page left every newer driver - e.g. the UNIDENTIFIED drivers the
    assign script creates - unknown, so their trips were loaded with no driver."""
    items, token = [], None
    for _ in range(max_pages):
        url = DRIVERS_URL + (f"&continuation_token={token}" if token else "")
        data = fetch_json(url)
        if isinstance(data, list):
            items.extend(data)
            break
        items.extend((data or {}).get("items") or [])
        token = (data or {}).get("continuation_token")
        if not token:
            break
    else:
        print(f"  [WARN] drivers list still had more pages after {max_pages} pages")
    return items


def fetch_all_geozones():
    """Fetch every geozone from FM-Track, following continuation tokens if present."""
    all_zones = []
    url = GEOZONES_URL
    while True:
        data = fetch_json(url)
        items = data.get("items", data) if isinstance(data, dict) else data
        if items is None:
            items = []
        all_zones.extend(items)
        continuation_token = data.get("continuation_token") if isinstance(data, dict) else None
        if not continuation_token:
            break
        sep = "&" if "?" in GEOZONES_URL else "?"
        url = f"{GEOZONES_URL}{sep}continuation_token={continuation_token}"
    return all_zones


def format_address(loc):
    if not loc:
        return ""
    addr = loc.get("address") or {}
    parts = [addr.get("street"), addr.get("house_number"), addr.get("locality"),
             addr.get("county"), addr.get("region"), addr.get("country")]
    return ", ".join(p for p in parts if p)


def fmt_dt_sast(iso):
    """Parse an FM-Track UTC ISO timestamp -> naive SAST datetime."""
    if not iso:
        return None
    dt_utc = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return dt_utc.astimezone(SAST).replace(tzinfo=None)


def get_driver_name(driver_id, drivers):
    d = drivers.get(driver_id)
    if not d:
        return ""
    first = d.get("first_name") or ""
    last = d.get("last_name") or ""
    return f"{first} {last}".strip() or first


def is_state_profile(driver_id, drivers):
    """The assign script's state profiles (address STATE-...) say WHY no driver is
    on the vehicle (no face, not linked ...) - they are not a driver."""
    d = drivers.get(driver_id) or {}
    return str(d.get("address") or "").strip().upper().startswith("STATE-")


# ── Recognition fallback ─────────────────────────────────────────────────────
# Trust Track only has a driver on a trip if the assign script put one on the
# vehicle before/while it drove, and an assignment can't be backdated. When a
# trip comes back with no driver, the driver the facial recognition saw in that
# vehicle during the trip is shown instead (driver_source = 'recognition').
REAL_STATUSES = ("identified", "unidentified", "new_unidentified", "unauthorised")
RECOGNITION_LEAD_MIN = 10      # a sighting up to 10 min before the trip started counts (driver got in, then left)
# Enrollment states: FULLY_ENROLLED, AUTO_ENROLLED, UNIDENTIFIED (older REJECTED / EXPIRED rows count as UNIDENTIFIED)
STATE_LABEL = {"UNIDENTIFIED": "UNIDENTIFIED", "REJECTED": "UNIDENTIFIED", "EXPIRED": "UNIDENTIFIED"}

import re as _re


def vehicle_key(name):
    """'R014_-_CJ73834', 'R014 - CJ73834' and 'r014 - cj73834 (fridge)' -> 'R014CJ73834'."""
    s = _re.sub(r"\(.*?\)", "", str(name or ""))
    return _re.sub(r"[^A-Z0-9]", "", s.upper())


# Drivers the web app lists under Unidentified Drivers (not fully enrolled yet).
# Several can have the same name ("Unidentified R087 - CJ12345 2026-10-20"), so
# their DRV id is added - the id the Unidentified Drivers page shows on each card.
NOT_ENROLLED_STATES = ("UNIDENTIFIED", "AUTO_ENROLLED", "REJECTED", "EXPIRED")


def driver_display_name(row):
    """The driver's name exactly as the web app shows it (drivers_reinke_logistics
    first + last name), plus the DRV id for drivers that are not fully enrolled."""
    code = (row.get("driver_id") or "").strip()
    state = (row.get("enrollment_state") or "").upper()
    name = f"{(row.get('first_name') or '').strip()} {(row.get('last_name') or '').strip()}".strip()
    if not name:
        return code
    if state in NOT_ENROLLED_STATES and code and code.upper() not in name.upper():
        return f"{name} ({code})"
    return name


class RecognitionIndex:
    """Recognitions per vehicle from recognition_events_reinke_logistics, read once per run."""

    def __init__(self, db, start, end):
        from sqlalchemy import text, bindparam
        self.by_vehicle = {}
        self.drivers = {}
        try:
            rows = db.session.execute(text("""
                SELECT e.video_filename AS folder, upper(trim(e.driver_id_string)) AS code, lower(e.status) AS status,
                       COALESCE(e.captured_at, e.timestamp + interval '2 hours') AS seen_at
                FROM recognition_events_reinke_logistics e
                WHERE lower(e.status) IN :st
                  AND e.driver_id_string IS NOT NULL AND upper(trim(e.driver_id_string)) NOT IN ('', 'UNKNOWN', 'NONE', 'NAN')
                  AND COALESCE(e.captured_at, e.timestamp + interval '2 hours') BETWEEN :a AND :b
            """).bindparams(bindparam("st", expanding=True)),
                {"st": list(REAL_STATUSES), "a": start - timedelta(hours=1), "b": end + timedelta(hours=1)}).mappings().all()
        except Exception as e:  # noqa: BLE001 - history table missing columns etc.
            db.session.rollback()
            print(f"  [WARN] recognition history not read ({e}) - trips without a Trust Track driver stay blank")
            rows = []
        for r in rows:
            self.by_vehicle.setdefault(vehicle_key(r["folder"]), []).append((r["seen_at"], r["code"], r["status"]))
        try:
            drows = db.session.execute(text("""
                SELECT upper(trim(driver_id)) AS code, driver_id, first_name, last_name, enrollment_state,
                       merged_into, trusttrack_id, name_pending
                FROM drivers_reinke_logistics""")).mappings().all()
        except Exception as e:  # noqa: BLE001
            db.session.rollback()
            print(f"  [WARN] drivers table not read ({e})")
            drows = []
        self.all_drivers = {d["code"]: dict(d) for d in drows}
        self.by_tt_id = {str(d["trusttrack_id"]): dict(d) for d in drows if d.get("trusttrack_id")}
        print(f"  {len(rows)} recognitions of a real driver on {len(self.by_vehicle)} vehicles in range")

    def _final(self, code, depth=0):
        d = self.all_drivers.get(code)
        # A merged record keeps its state and is set to status 'deleted'; follow merged_into.
        if d and d.get("merged_into") and depth < 3:
            return self._final(str(d["merged_into"]).strip().upper(), depth + 1)
        return code, d

    def web_driver(self, tt_id, tt_driver=None):
        """(display name, final DRV code) of the web-app driver behind a Trust Track driver
        (matched by its address = DRV code, else by the trusttrack_id stored in the web app)."""
        code = str((tt_driver or {}).get("address") or "").strip().upper()
        d = self.all_drivers.get(code) if code else None
        if d is None:
            d = self.by_tt_id.get(str(tt_id))
        if d is None:
            return None, None
        final, fd = self._final(d["code"])
        fd = fd or d
        return driver_display_name(fd), final

    def name_for_code(self, code):
        final, d = self._final(str(code or "").strip().upper())
        return (driver_display_name(d) if d else None), final

    def driver_for_trip(self, vehicle, start_dt, end_dt):
        """(display name, driver code) of the driver seen most in this vehicle during the trip, or (None, None)."""
        if not start_dt:
            return None, None
        seen = self.by_vehicle.get(vehicle_key(vehicle)) or []
        a = start_dt - timedelta(minutes=RECOGNITION_LEAD_MIN)
        b = end_dt or start_dt
        hits = {}
        for t, code, _st in seen:
            if t and a <= t <= b:
                code, _ = self._final(code)
                n, last = hits.get(code, (0, None))
                hits[code] = (n + 1, max(last, t) if last else t)
        if not hits:
            return None, None
        code = max(hits, key=lambda c: hits[c])
        name, final = self.name_for_code(code)
        return (name or code), final




def parse_datetime_to_utc(date_str, time_str):
    dt_sast = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=SAST)
    return dt_sast.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def generate_monthly_chunks(date_from, date_to, time_from, time_to):
    """Split a date range into <=1-month chunks (FM-Track API limit)."""
    chunks = []
    start_dt = datetime.strptime(f"{date_from} {time_from}", "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(f"{date_to} {time_to}", "%Y-%m-%d %H:%M:%S")
    current = start_dt
    while current <= end_dt:
        month_end = (current.replace(day=1) + relativedelta(months=1)) - timedelta(seconds=1)
        chunk_end = min(month_end, end_dt)
        from_utc = parse_datetime_to_utc(current.strftime("%Y-%m-%d"), current.strftime("%H:%M:%S"))
        to_utc = parse_datetime_to_utc(chunk_end.strftime("%Y-%m-%d"), chunk_end.strftime("%H:%M:%S"))
        chunks.append((from_utc, to_utc))
        current = current.replace(day=1) + relativedelta(months=1)
    return chunks


# ── Transform helpers ───────────────────────────────────────────────────────

def parse_km(meters):
    if meters is None:
        return None
    try:
        return float(meters) / 1000.0
    except (TypeError, ValueError):
        return None


def make_external_id(vehicle, start_dt, end_dt):
    key = f"{vehicle}|{start_dt.isoformat() if start_dt else ''}|{end_dt.isoformat() if end_dt else ''}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


# ── Geozone helpers ─────────────────────────────────────────────────────────
# FM-Track /geozones returns each zone as:
#   { "id": "...", "name": "...", "notes": "...", "type": "POLYGON" | "CIRCLE",
#     "circle": {"center": {...}, "radius": ...} | null,
#     "feature": {"type": "Feature",
#                 "properties": {"envelope": <GeoJSON Polygon>, "center": <GeoJSON Point>, "radius": ...},
#                 "geometry": <GeoJSON Polygon (rings of [lon, lat])>} }
#
# We store the raw geometry rings and do point-in-polygon (ray casting) /
# point-in-circle (haversine) matching ourselves, so no extra GIS dependency
# (e.g. shapely) is required.

def _polygon_rings_from_zone(zone):
    """Return a list of rings, each a list of (lon, lat) tuples, for a POLYGON zone."""
    feature = zone.get("feature") or {}
    geometry = feature.get("geometry") or {}
    coords = geometry.get("coordinates") or []
    rings = []
    for ring in coords:
        pts = [(pt[0], pt[1]) for pt in ring if isinstance(pt, (list, tuple)) and len(pt) >= 2]
        if pts:
            rings.append(pts)
    return rings


def _bbox_from_rings(rings):
    lons = [pt[0] for ring in rings for pt in ring]
    lats = [pt[1] for ring in rings for pt in ring]
    if not lons or not lats:
        return None
    return (min(lats), max(lats), min(lons), max(lons))  # min_lat, max_lat, min_lng, max_lng


def _point_in_ring(lat, lon, ring):
    """Ray-casting point-in-polygon test. `ring` is a list of (lon, lat)."""
    inside = False
    n = len(ring)
    if n < 3:
        return False
    x, y = lon, lat
    x1, y1 = ring[-1]
    for x2, y2 in ring:
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def point_in_polygon_zone(lat, lon, rings):
    """A point is inside the polygon if inside the outer ring and not inside any hole.
    We treat the first ring as the outer boundary and any subsequent rings as holes,
    which matches standard GeoJSON polygon conventions."""
    if not rings:
        return False
    if not _point_in_ring(lat, lon, rings[0]):
        return False
    for hole in rings[1:]:
        if _point_in_ring(lat, lon, hole):
            return False
    return True


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0  # metres
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def find_zone_for_point(lat, lon, prepared_zones):
    """Return the matching zone name for (lat, lon), or None.
    `prepared_zones` is a list of dicts with bbox / rings / circle info, built once per run."""
    if lat is None or lon is None:
        return None
    for z in prepared_zones:
        bbox = z.get("bbox")
        if bbox:
            min_lat, max_lat, min_lng, max_lng = bbox
            if not (min_lat <= lat <= max_lat and min_lng <= lon <= max_lng):
                continue
        if z["zone_type"] == "CIRCLE" and z.get("center_lat") is not None:
            if haversine_m(lat, lon, z["center_lat"], z["center_lng"]) <= (z.get("radius_m") or 0):
                return z["name"]
        elif z.get("rings"):
            if point_in_polygon_zone(lat, lon, z["rings"]):
                return z["name"]
    return None


# ── Main pipeline ────────────────────────────────────────────────────────────

def run(date_from, date_to, time_from, time_to):
    # Imported here so this file can be inspected/linted without Flask installed.
    from app import create_app, db
    from app.models import Trip, PipelineRun, GeoZone

    app = create_app()
    with app.app_context():
        run_log = PipelineRun(
            status='running',
            range_from=datetime.strptime(f"{date_from} {time_from}", "%Y-%m-%d %H:%M:%S"),
            range_to=datetime.strptime(f"{date_to} {time_to}", "%Y-%m-%d %H:%M:%S"),
        )
        db.session.add(run_log)
        db.session.commit()

        try:
            print(f"Range (SAST): {date_from} {time_from} -> {date_to} {time_to}")
            monthly_chunks = generate_monthly_chunks(date_from, date_to, time_from, time_to)
            print(f"Split into {len(monthly_chunks)} monthly chunk(s)")

            print("Fetching assets...")
            objects_raw = fetch_json(OBJECTS_URL)
            assets = {o["id"]: o for o in objects_raw}
            print(f"  {len(assets)} assets found")

            print("Fetching drivers...")
            items = fetch_all_drivers()
            drivers = {d["id"]: d for d in items if d.get("id")}
            print(f"  {len(drivers)} drivers found")

            from sqlalchemy import text as _text
            try:   # new column (web app adds it too); safe to run every time
                db.session.execute(_text("ALTER TABLE trips_reinke_logistics_v2 "
                                         "ADD COLUMN IF NOT EXISTS driver_source VARCHAR(20), "
                                         "ADD COLUMN IF NOT EXISTS driver_code VARCHAR(30)"))
                db.session.commit()
            except Exception as _e:  # noqa: BLE001
                db.session.rollback()
                print(f"  [WARN] driver_source column not added: {_e}")
            rng_from = datetime.strptime(f"{date_from} {time_from}", "%Y-%m-%d %H:%M:%S")
            rng_to = datetime.strptime(f"{date_to} {time_to}", "%Y-%m-%d %H:%M:%S")
            recog = RecognitionIndex(db, rng_from, rng_to)
            from_recognition = 0

            # ── Geozones: fetch + upsert, then build the in-memory matching list ──
            print("Fetching geozones...")
            zones_raw = fetch_all_geozones()
            print(f"  {len(zones_raw)} geozones found")

            import json as _json
            prepared_zones = []
            zones_seen = 0

            for zone in zones_raw:
                zone_id = zone.get("id")
                if not zone_id:
                    continue
                name = zone.get("name") or ""
                notes = zone.get("notes") or ""
                zone_type = (zone.get("type") or "").upper()
                circle = zone.get("circle") or {}
                feature = zone.get("feature") or {}
                props = feature.get("properties") or {}

                center_lat = center_lng = radius_m = None
                rings = []
                bbox = None

                if zone_type == "CIRCLE":
                    center = circle.get("center") or props.get("center", {}).get("coordinates")
                    if isinstance(center, dict):
                        center_lng, center_lat = center.get("coordinates", [None, None])[:2] \
                            if center.get("coordinates") else (None, None)
                    elif isinstance(center, (list, tuple)) and len(center) >= 2:
                        center_lng, center_lat = center[0], center[1]
                    radius_m = circle.get("radius") or props.get("radius")
                    if center_lat is not None and center_lng is not None and radius_m:
                        # bbox approx: radius in metres -> degrees (rough, generous margin)
                        deg = (radius_m / 111000.0) * 1.2
                        bbox = (center_lat - deg, center_lat + deg, center_lng - deg, center_lng + deg)
                else:
                    rings = _polygon_rings_from_zone(zone)
                    bbox = _bbox_from_rings(rings)
                    zone_type = "POLYGON"

                geometry_json = _json.dumps(feature.get("geometry")) if feature.get("geometry") else None

                existing_zone = GeoZone.query.filter_by(external_id=zone_id).first()
                if existing_zone:
                    existing_zone.name = name
                    existing_zone.notes = notes
                    existing_zone.zone_type = zone_type
                    existing_zone.min_lat = bbox[0] if bbox else None
                    existing_zone.max_lat = bbox[1] if bbox else None
                    existing_zone.min_lng = bbox[2] if bbox else None
                    existing_zone.max_lng = bbox[3] if bbox else None
                    existing_zone.center_lat = center_lat
                    existing_zone.center_lng = center_lng
                    existing_zone.radius_m = radius_m
                    existing_zone.geometry_json = geometry_json
                    existing_zone.loaded_at = datetime.utcnow()
                else:
                    db.session.add(GeoZone(
                        external_id=zone_id,
                        name=name,
                        notes=notes,
                        zone_type=zone_type,
                        min_lat=bbox[0] if bbox else None,
                        max_lat=bbox[1] if bbox else None,
                        min_lng=bbox[2] if bbox else None,
                        max_lng=bbox[3] if bbox else None,
                        center_lat=center_lat,
                        center_lng=center_lng,
                        radius_m=radius_m,
                        geometry_json=geometry_json,
                    ))
                zones_seen += 1

                prepared_zones.append({
                    "name": name,
                    "zone_type": zone_type,
                    "bbox": bbox,
                    "rings": rings,
                    "center_lat": center_lat,
                    "center_lng": center_lng,
                    "radius_m": radius_m,
                })

            db.session.commit()
            print(f"  Upserted {zones_seen} geozones")

            trips_fetched, inserted, updated = 0, 0, 0

            for obj_id, obj in assets.items():
                obj_name = obj.get("name", str(obj_id))

                for from_dt, to_dt in monthly_chunks:
                    try:
                        trips = fetch_all_trips(obj_id, from_dt, to_dt)
                    except Exception as e:
                        print(f"  [WARN] {obj_name}: {e}")
                        continue

                    trips_fetched += len(trips)

                    for trip in trips:
                        start = trip.get("trip_start", {}) or {}
                        end = trip.get("trip_end", {}) or {}

                        start_dt = fmt_dt_sast(start.get("datetime"))
                        end_dt = fmt_dt_sast(end.get("datetime"))
                        km = parse_km(trip.get("mileage", 0))
                        duration_min = None
                        if start_dt and end_dt:
                            duration_min = (end_dt - start_dt).total_seconds() / 60

                        # Driver on the trip in Trust Track: the first id that is a real driver
                        # (several ids when the driver changed; state profiles are not drivers).
                        driver_ids = trip.get("driver_ids") or []
                        driver_name, state_name, driver_source, driver_code = "", "", None, None
                        for did in driver_ids:
                            if is_state_profile(did, drivers):
                                state_name = state_name or get_driver_name(did, drivers)
                                continue
                            # The web app's name for this driver wins over the Trust Track
                            # profile name, so trips match the web app.
                            wname, wcode = recog.web_driver(did, drivers.get(did))
                            name = wname or get_driver_name(did, drivers)
                            if not name:
                                continue
                            driver_name, driver_source, driver_code = name, "trusttrack", wcode
                            break
                        if not driver_name:
                            # Nothing on the trip in Trust Track: the driver facial recognition saw
                            rname, rcode = recog.driver_for_trip(obj_name, start_dt, end_dt)
                            if rname:
                                driver_name, driver_source, driver_code = rname, "recognition", rcode
                                from_recognition += 1
                        if not driver_name and state_name:
                            driver_name, driver_source = state_name, "state"
                        has_driver = driver_source in ("trusttrack", "recognition")
                        is_excluded = obj_name in EXCLUDED_VEHICLES

                        start_lat = start.get("latitude")
                        start_lng = start.get("longitude")
                        start_zone_name = find_zone_for_point(start_lat, start_lng, prepared_zones)

                        ext_id = make_external_id(obj_name, start_dt, end_dt)
                        existing = Trip.query.filter_by(external_id=ext_id).first()

                        if existing:
                            existing.start_time = start_dt
                            existing.end_time = end_dt
                            existing.duration_min = duration_min
                            existing.start_location = format_address(start)
                            existing.end_location = format_address(end)
                            existing.start_lat = start_lat
                            existing.start_lng = start_lng
                            existing.end_lat = end.get("latitude")
                            existing.end_lng = end.get("longitude")
                            existing.km = km
                            existing.trip_type = (trip.get("trip_type") or "").capitalize() or None
                            existing.start_zone_name = start_zone_name
                            existing.driver_name = driver_name or None
                            existing.driver_source = driver_source
                            existing.driver_code = driver_code
                            existing.has_driver = has_driver
                            existing.is_excluded = is_excluded
                            existing.loaded_at = datetime.utcnow()
                            updated += 1
                        else:
                            db.session.add(Trip(
                                external_id=ext_id,
                                vehicle=obj_name,
                                start_time=start_dt,
                                end_time=end_dt,
                                duration_min=duration_min,
                                start_location=format_address(start),
                                end_location=format_address(end),
                                start_lat=start_lat,
                                start_lng=start_lng,
                                end_lat=end.get("latitude"),
                                end_lng=end.get("longitude"),
                                km=km,
                                trip_type=(trip.get("trip_type") or "").capitalize() or None,
                                start_zone_name=start_zone_name,
                                driver_name=driver_name or None,
                                driver_source=driver_source,
                                driver_code=driver_code,
                                has_driver=has_driver,
                                is_excluded=is_excluded,
                            ))
                            inserted += 1

                    # Commit per asset/chunk so a single bad batch can't lose everything
                    db.session.commit()

            # Names change in the web app (an unidentified driver confirmed with a real
            # name, merged into another driver ...): bring EVERY stored trip in line,
            # not only the trips in this run's range.
            renamed = 0
            for (code,) in db.session.execute(_text(
                    "SELECT DISTINCT driver_code FROM trips_reinke_logistics_v2 WHERE driver_code IS NOT NULL")).all():
                name, final = recog.name_for_code(code)
                if not name:
                    continue
                res = db.session.execute(_text(
                    "UPDATE trips_reinke_logistics_v2 SET driver_name = :n, driver_code = :f "
                    "WHERE driver_code = :c AND (driver_name IS DISTINCT FROM :n OR driver_code <> :f)"),
                    {"n": name, "f": final, "c": code})
                renamed += res.rowcount or 0
            db.session.commit()
            if renamed:
                print(f"  {renamed} earlier trip(s) renamed to match the web app")

            run_log.status = 'success'
            run_log.finished_at = datetime.utcnow()
            run_log.trips_fetched = trips_fetched
            run_log.trips_inserted = inserted
            run_log.trips_updated = updated
            run_log.vehicles_seen = len(assets)
            run_log.drivers_seen = len(drivers)
            run_log.message = (f"OK: {inserted} inserted, {updated} updated, {trips_fetched} fetched, "
                               f"{from_recognition} driver(s) from facial recognition")
            db.session.commit()

            print(f"\nDone. Fetched {trips_fetched} trips -> inserted {inserted}, updated {updated}; "
                  f"{from_recognition} trip(s) without a Trust Track driver got the recognised driver.")

        except Exception as e:
            db.session.rollback()
            run_log.status = 'failed'
            run_log.finished_at = datetime.utcnow()
            run_log.message = str(e)
            db.session.commit()
            print(f"[FAILED] {e}")
            raise


def main():
    now_sast = datetime.now(SAST)
    default_date = now_sast.strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(description="Load FM-Track trips into Postgres.")
    parser.add_argument("--from", dest="date_from", default=default_date,
                         help="Start date YYYY-MM-DD (SAST). Default: today.")
    parser.add_argument("--to", dest="date_to", default=default_date,
                         help="End date YYYY-MM-DD (SAST). Default: today.")
    parser.add_argument("--time-from", dest="time_from", default="00:00:00",
                         help="Start time HH:MM:SS (SAST). Default: 00:00:00.")
    parser.add_argument("--time-to", dest="time_to",
                         default=now_sast.strftime("%H:%M:%S") if now_sast.strftime("%Y-%m-%d") == default_date else "23:59:59",
                         help="End time HH:MM:SS (SAST). Default: now (if --to is today) else 23:59:59.")
    args = parser.parse_args()

    run(args.date_from, args.date_to, args.time_from, args.time_to)


if __name__ == "__main__":
    main()
