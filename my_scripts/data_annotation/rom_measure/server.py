#!/usr/bin/env python3
"""
rom_measure — 2D joint range-of-motion annotation webapp (Flask backend).

Multi-user: each username gets its own annotation namespace under
annotations/<username>/. Re-login with the same username restores that
user's annotations.

Auth options (any combination):
  * ROM_PASSWORD=shared     -> any username + this shared password
  * ROM_USERS="a:pw1,b:pw2" -> per-user passwords (overrides shared)
  * neither set             -> any username, no password

Run:
    pip install flask
    ROM_PASSWORD=changeme python3 server.py --host 0.0.0.0 --port 8080

Internet exposure (pick one tunnel):
    cloudflared tunnel --url http://localhost:8080     # recommended, free, no account
    # or
    ngrok http 8080
"""
import argparse
import csv
import io
import json
import os
import secrets
import time
from functools import wraps

from flask import (Flask, Response, jsonify, redirect, request,
                   send_from_directory, session)
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
ANNOT_DIR = os.path.join(BASE_DIR, "annotations")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(ANNOT_DIR, exist_ok=True)

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
app.secret_key = os.environ.get("ROM_SECRET", secrets.token_hex(32))
ROM_PASSWORD = os.environ.get("ROM_PASSWORD", "")

# Optional per-user passwords: ROM_USERS="alice:pw1,bob:pw2"
ROM_USERS = {}
if os.environ.get("ROM_USERS"):
    for pair in os.environ["ROM_USERS"].split(","):
        if ":" in pair:
            u, p = pair.split(":", 1)
            ROM_USERS[u.strip()] = p


# ---------- auth ----------
def login_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if not session.get("authed") or not session.get("user"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "unauthorized"}), 401
            return redirect("/")
        return f(*a, **k)
    return wrapper


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.post("/api/login")
def login():
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username:
        return jsonify({"error": "username required"}), 401
    if ROM_USERS:
        if ROM_USERS.get(username) != password:
            return jsonify({"error": "wrong username/password"}), 401
    elif ROM_PASSWORD and password != ROM_PASSWORD:
        return jsonify({"error": "wrong password"}), 401
    session["authed"] = True
    session["user"] = username
    return jsonify({"ok": True, "user": username})


@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


# ---------- images ----------
@app.get("/api/images")
@login_required
def list_images():
    user = session["user"]
    out = []
    for name in sorted(os.listdir(IMAGES_DIR)):
        if os.path.splitext(name)[1].lower() in ALLOWED_EXT:
            anno = _load_annotation(user, name)
            out.append({"name": name,
                        "annotated": bool(anno and (anno.get("angles") or anno.get("points"))),
                        "url": f"/api/image/{name}"})
    return jsonify({"images": out})


@app.get("/api/image/<path:name>")
@login_required
def serve_image(name):
    return send_from_directory(IMAGES_DIR, name)


@app.post("/api/upload")
@login_required
def upload():
    saved = []
    for f in request.files.getlist("files"):
        name = secure_filename(f.filename or "")
        if not name or os.path.splitext(name)[1].lower() not in ALLOWED_EXT:
            continue
        dest = os.path.join(IMAGES_DIR, name)
        if os.path.exists(dest):  # avoid clobbering
            stem, ext = os.path.splitext(name)
            name = f"{stem}_{int(time.time())}{ext}"
            dest = os.path.join(IMAGES_DIR, name)
        f.save(dest)
        saved.append(name)
    return jsonify({"saved": saved})


# ---------- annotations (per-user) ----------
def _user_annot_dir(user):
    d = os.path.join(ANNOT_DIR, secure_filename(user))
    os.makedirs(d, exist_ok=True)
    return d


def _anno_path(user, name):
    return os.path.join(_user_annot_dir(user), secure_filename(name) + ".json")


def _load_annotation(user, name):
    p = _anno_path(user, name)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


@app.get("/api/annotation/<path:name>")
@login_required
def get_annotation(name):
    return jsonify(_load_annotation(session["user"], name) or {})


@app.post("/api/annotation/<path:name>")
@login_required
def save_annotation(name):
    data = request.get_json(force=True, silent=True) or {}
    data["image"] = name
    data["user"] = session["user"]
    data["updated_at"] = time.time()
    path = _anno_path(session["user"], name)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    return jsonify({"ok": True, "path": path})


# ---------- export ----------
@app.get("/api/export.csv")
@login_required
def export_csv():
    user = session["user"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["user", "image", "angle_name", "vertex", "prox", "dist",
                "interior_deg", "reported_deg", "updated_at"])
    for fn in sorted(os.listdir(_user_annot_dir(user))):
        if not fn.endswith(".json"):
            continue
        anno = _load_annotation(user, fn[:-5])
        if not anno:
            continue
        for a in anno.get("angles", []):
            w.writerow([user, anno.get("image"), a.get("name"), a.get("vertex"),
                        a.get("prox"), a.get("dist"), a.get("interior_deg"),
                        a.get("reported_deg"), anno.get("updated_at")])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=rom_export.csv"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
