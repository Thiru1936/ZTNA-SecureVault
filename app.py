"""
app.py — ZTNA (Zero Trust Network Access)
Company File Storage + Insider Threat Detection
Owner sees all user behavior. Users access files only.
"""

from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, send_from_directory, abort)
from datetime import datetime
from werkzeug.utils import secure_filename
import pickle, numpy as np, hashlib, os, random, json

app = Flask(__name__)
app.secret_key = "ztna_super_secret_2024"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "xlsx", "txt", "png", "jpg", "pptx", "csv"}
MAX_FILE_MB = 10
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── ML Model ──────────────────────────────────────────────────
model_data = None
def load_model():
    global model_data
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            model_data = pickle.load(f)
        print("[✓] ML model loaded.")
    else:
        print("[!] model.pkl missing — run train_model.py first.")
load_model()

# ── User Database ─────────────────────────────────────────────
def h(p): return hashlib.sha256(p.encode()).hexdigest()

USERS = {
    # Owner — can see ALL dashboards
    "owner": {"password": h("owner123"), "role": "Owner", "dept": "Management", "is_owner": True},
    # Regular employees
    "alice": {"password": h("alice123"), "role": "Developer",  "dept": "Engineering",  "is_owner": False},
    "bob":   {"password": h("bob123"),   "role": "Analyst",    "dept": "Finance",       "is_owner": False},
    "carol": {"password": h("carol123"), "role": "HR Manager", "dept": "Human Resources","is_owner": False},
    "david": {"password": h("david123"), "role": "Designer",   "dept": "Marketing",     "is_owner": False},
}

# ── In-memory state ───────────────────────────────────────────
AUDIT_LOG   = []          # all security events
USER_STATS  = {u: {       # per-user behavioral counters
    "file_uploads": 0, "file_downloads": 0, "file_deletes": 0,
    "failed_logins": 0, "login_count": 0,
    "data_transferred_mb": 0.0, "last_seen": "—",
    "risk_score": 0.0, "risk_level": "LOW", "status": "Offline",
} for u in USERS if not USERS[u]["is_owner"]}

FILES_DB = []   # list of dicts: {id, name, uploader, size_mb, uploaded_at, dept}

# ── Helpers ───────────────────────────────────────────────────
def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def risk_label(score):
    if score >= 80: return "CRITICAL"
    if score >= 60: return "HIGH"
    if score >= 40: return "MEDIUM"
    return "LOW"

def compute_risk(username, extra=None):
    s = USER_STATS.get(username, {})
    hour = datetime.now().hour
    features = {
        "login_hour":           hour,
        "failed_logins":        s.get("failed_logins", 0),
        "data_transferred_mb":  s.get("data_transferred_mb", 0),
        "unusual_location":     0,
        "after_hours_access":   1 if (hour < 8 or hour > 18) else 0,
        "privilege_escalation": 0,
        "multiple_sessions":    0,
        "request_rate":         s.get("file_downloads", 0) + s.get("file_uploads", 0),
    }
    if extra:
        features.update(extra)

    if model_data:
        model  = model_data["model"]
        scaler = model_data["scaler"]
        feat_order = model_data["features"]
        row = np.array([[features[f] for f in feat_order]])
        prob  = model.predict_proba(scaler.transform(row))[0][1]
        score = round(prob * 100, 1)
    else:
        score = min(
            features["failed_logins"] * 8 +
            features["after_hours_access"] * 10 +
            (features["data_transferred_mb"] / 30) +
            features["request_rate"] * 0.5, 100)

    level = risk_label(score)
    if username in USER_STATS:
        USER_STATS[username]["risk_score"] = score
        USER_STATS[username]["risk_level"] = level
    return score, level

def log_event(user, event, risk, detail):
    AUDIT_LOG.append({
        "time":   datetime.now().strftime("%H:%M:%S"),
        "date":   datetime.now().strftime("%d %b %Y"),
        "user":   user,
        "event":  event,
        "risk":   risk,
        "detail": detail,
    })

# ── AUTH ROUTES ───────────────────────────────────────────────
@app.route("/")
def index():
    if "user" in session:
        if session.get("is_owner"):
            return redirect(url_for("owner_dashboard"))
        return redirect(url_for("user_dashboard"))
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    data     = request.get_json()
    username = data.get("username", "").strip().lower()
    password = h(data.get("password", ""))

    if username not in USERS or USERS[username]["password"] != password:
        if username in USER_STATS:
            USER_STATS[username]["failed_logins"] += 1
            compute_risk(username)
        log_event(username, "Failed Login", "HIGH", "Invalid credentials")
        return jsonify({"success": False, "message": "Invalid username or password."})

    u = USERS[username]
    session.update({
        "user": username, "role": u["role"],
        "dept": u["dept"], "is_owner": u["is_owner"],
    })
    if not u["is_owner"]:
        USER_STATS[username]["login_count"] += 1
        USER_STATS[username]["failed_logins"] = 0
        USER_STATS[username]["status"] = "Online"
        USER_STATS[username]["last_seen"] = datetime.now().strftime("%H:%M:%S")
        compute_risk(username)

    log_event(username, "Login", "LOW", f"Role: {u['role']}")
    return jsonify({"success": True, "is_owner": u["is_owner"]})

@app.route("/logout")
def logout():
    user = session.get("user", "unknown")
    if user in USER_STATS:
        USER_STATS[user]["status"] = "Offline"
    log_event(user, "Logout", "LOW", "Session ended")
    session.clear()
    return redirect(url_for("index"))

# ── OWNER ROUTES ──────────────────────────────────────────────
@app.route("/owner")
def owner_dashboard():
    if not session.get("is_owner"):
        return redirect(url_for("index"))
    return render_template("owner.html", owner=session["user"])

@app.route("/api/owner/stats")
def owner_stats():
    if not session.get("is_owner"):
        return jsonify({"error": "Forbidden"}), 403

    users_data = []
    for uname, stats in USER_STATS.items():
        u = USERS[uname]
        users_data.append({
            "username": uname,
            "role":     u["role"],
            "dept":     u["dept"],
            **stats,
        })

    total_files = len(FILES_DB)
    online = sum(1 for s in USER_STATS.values() if s["status"] == "Online")
    threats = sum(1 for s in USER_STATS.values() if s["risk_level"] in ("HIGH","CRITICAL"))

    return jsonify({
        "users":       users_data,
        "audit":       AUDIT_LOG[-50:][::-1],
        "files":       FILES_DB[-20:][::-1],
        "summary": {
            "total_users":  len(USER_STATS),
            "online":       online,
            "threats":      threats,
            "total_files":  total_files,
        }
    })

# ── USER ROUTES ───────────────────────────────────────────────
@app.route("/dashboard")
def user_dashboard():
    if "user" not in session or session.get("is_owner"):
        return redirect(url_for("index"))
    return render_template("user.html",
                           user=session["user"],
                           role=session["role"],
                           dept=session["dept"])

@app.route("/api/files")
def list_files():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"files": FILES_DB})

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "user" not in session or session.get("is_owner"):
        return jsonify({"error": "Unauthorized"}), 401

    username = session["user"]
    if "file" not in request.files:
        return jsonify({"error": "No file selected"}), 400

    f = request.files["file"]
    if not f.filename or not allowed(f.filename):
        return jsonify({"error": f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    fname    = secure_filename(f.filename)
    fpath    = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    f.save(fpath)
    size_mb  = round(os.path.getsize(fpath) / (1024*1024), 2)

    file_record = {
        "id":           len(FILES_DB) + 1,
        "name":         fname,
        "uploader":     username,
        "dept":         session["dept"],
        "size_mb":      size_mb,
        "uploaded_at":  datetime.now().strftime("%d %b %Y %H:%M"),
    }
    FILES_DB.append(file_record)

    USER_STATS[username]["file_uploads"] += 1
    USER_STATS[username]["data_transferred_mb"] += size_mb
    USER_STATS[username]["last_seen"] = datetime.now().strftime("%H:%M:%S")
    score, level = compute_risk(username)

    log_event(username, f"File Upload: {fname}", level,
              f"{size_mb} MB | Risk: {level} ({score})")

    return jsonify({"success": True, "file": file_record, "risk": level, "score": score})

@app.route("/api/download/<int:file_id>")
def download_file(file_id):
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    username = session["user"]
    record   = next((f for f in FILES_DB if f["id"] == file_id), None)
    if not record:
        return jsonify({"error": "File not found"}), 404

    if not session.get("is_owner"):
        USER_STATS[username]["file_downloads"] += 1
        USER_STATS[username]["data_transferred_mb"] += record["size_mb"]
        USER_STATS[username]["last_seen"] = datetime.now().strftime("%H:%M:%S")
        score, level = compute_risk(username)
        log_event(username, f"File Download: {record['name']}", level,
                  f"{record['size_mb']} MB | Risk: {level} ({score})")

    return send_from_directory(app.config["UPLOAD_FOLDER"], record["name"], as_attachment=True)

@app.route("/api/delete/<int:file_id>", methods=["DELETE"])
def delete_file(file_id):
    if "user" not in session or session.get("is_owner"):
        return jsonify({"error": "Unauthorized"}), 401

    username = session["user"]
    record   = next((f for f in FILES_DB if f["id"] == file_id), None)
    if not record:
        return jsonify({"error": "File not found"}), 404
    if record["uploader"] != username:
        log_event(username, f"Unauthorized Delete Attempt: {record['name']}", "HIGH",
                  "Tried to delete another user's file")
        return jsonify({"error": "You can only delete your own files"}), 403

    fpath = os.path.join(app.config["UPLOAD_FOLDER"], record["name"])
    if os.path.exists(fpath):
        os.remove(fpath)
    FILES_DB.remove(record)

    USER_STATS[username]["file_deletes"] += 1
    score, level = compute_risk(username)
    log_event(username, f"File Delete: {record['name']}", level, f"Risk: {level} ({score})")
    return jsonify({"success": True})

@app.route("/api/my_stats")
def my_stats():
    if "user" not in session or session.get("is_owner"):
        return jsonify({"error": "Unauthorized"}), 401
    username = session["user"]
    stats    = USER_STATS.get(username, {})
    my_files = [f for f in FILES_DB if f["uploader"] == username]
    my_logs  = [l for l in AUDIT_LOG if l["user"] == username][-10:][::-1]
    return jsonify({"stats": stats, "my_files": my_files, "my_logs": my_logs})

if __name__ == "__main__":
    app.run(debug=True, port=5000)