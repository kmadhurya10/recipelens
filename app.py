import os, sys, warnings

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*joserfc.*")
warnings.filterwarnings("ignore", message=".*authlib.*")

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3, os, json, traceback
from datetime import datetime
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from authlib.integrations.flask_client import OAuth
from database.food_dataset import RECIPE_DATABASE, find_recipe
from food_detector import detect_food_local, check_filename, describe_rejection, identify_food_type

app = Flask(__name__)
app.secret_key = "recipeLensSecretKey2024"

from flask_cors import CORS
CORS(app)

app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("database", exist_ok=True)

GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID",     "YOUR_GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "YOUR_GOOGLE_CLIENT_SECRET")

oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE ANALYSIS PIPELINE
#  Flow: Filename Check → CLIP (food/not food) → Food101 (what food)
#        → CLIP fallback → Recipe DB → Return result
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_image(image_path, original_filename):

    # ── Step 1: Filename pre-check ────────────────────────────────────────────
    fname_result = check_filename(original_filename)
    if fname_result is False:
        return {
            "success": False,
            "reason": (
                f"The filename '{original_filename}' suggests this is not a food image.\n"
                "Please upload a photo of a dish, meal, or ingredients."
            )
        }

    # ── Step 2: CLIP food vs non-food validation ──────────────────────────────
    clip_result = detect_food_local(image_path)

    if clip_result["method"] == "error":
        err = clip_result.get("error", "Unknown error")
        if "No module named" in err or "ModuleNotFound" in err:
            return {
                "success": False,
                "reason": (
                    "AI model not installed yet!\n\n"
                    "Run this command once to install it:\n"
                    "pip install transformers torch torchvision Pillow\n\n"
                    "Then restart the app."
                )
            }
        return {"success": False, "reason": f"AI classifier error: {err}\n\nTry restarting the app."}

    if clip_result["is_food"] is False:
        description = describe_rejection(clip_result)
        food_pct    = int(clip_result.get("food_score",    0) * 100)
        nonfood_pct = int(clip_result.get("nonfood_score", 0) * 100)
        return {
            "success": False,
            "reason": (
                f"{description}\n\n"
                f"Food score: {food_pct}%  |  Non-food score: {nonfood_pct}%\n\n"
                "Only food images are accepted."
            )
        }

    if clip_result["is_food"] is None:
        return {
            "success": False,
            "reason": (
                "Could not analyze this image.\n"
                f"Reason: {clip_result.get('error', 'Unknown')}\n\n"
                "Please try a clearer food photo."
            )
        }

    # ── Step 3: Food identification (Food101 PRIMARY, CLIP FALLBACK) ──────────
    food_id = identify_food_type(image_path)

    # ── Step 4: If food identification failed completely → reject ─────────────
    if food_id is None:
        return {
            "success": False,
            "reason": (
                "Could not identify the specific dish in this image.\n\n"
                "Please try a clearer food photo with better lighting."
            )
        }

    method_used = food_id.get("method", "unknown")
    print(f"[App] Food identified via '{method_used}': '{food_id['food_name']}' "
          f"(conf={food_id['confidence']:.3f})")

    # ── Step 5: Use identified food to search recipe DB ───────────────────────
    search_name = food_id.get("db_key") or food_id.get("food_name", "")

    # ── Step 6: Fallback to filename if food_id gave no useful db_key ─────────
    if not search_name:
        candidate = (
            original_filename
            .replace("_", " ")
            .replace("-", " ")
            .rsplit(".", 1)[0]
            .lower()
            .strip()
        )
        from food_detector import FOOD_FILENAME_WORDS
        if any(w in candidate for w in FOOD_FILENAME_WORDS):
            search_name = candidate
            print(f"[App] Falling back to filename for recipe search: '{search_name}'")

    # ── Step 7: Last resort — use CLIP top_label ──────────────────────────────
    if not search_name:
        top_label   = clip_result.get("top_label", "")
        search_name = top_label.replace("a photo of", "").strip()
        print(f"[App] Last resort — using CLIP top_label: '{search_name}'")

    # ── Step 8: Recipe database lookup ───────────────────────────────────────
    recipe = find_recipe(search_name)

    print(f"[App] Final recipe: '{recipe['food_name']}' "
          f"(cuisine: {recipe.get('cuisine_type', 'Unknown')})")

    # ── Step 9: If no recipe found in DB → reject cleanly ────────────────────
    if recipe.get("food_name") == "No Recipe Found":
        return {
            "success": False,
            "reason": (
                f"'{food_id['food_name']}' was identified in the image "
                f"but is not available in our recipe database yet.\n\n"
                "Please try uploading a different food image."
            )
        }

    # ── Step 10: Success ──────────────────────────────────────────────────────
    return {
        "success":       True,
        "recipe":        recipe,
        "food_name":     recipe["food_name"],
        "confidence":    food_id.get("confidence", clip_result.get("confidence", 0)),
        "identified_as": food_id["food_name"],
        "method":        method_used,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE  +  MIGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect("database/recipes.db")
    conn.row_factory = sqlite3.Row
    return conn


def migrate_db():
    """
    Safely add any columns that were missing from an older version of the DB.
    SQLite does not support ADD COLUMN IF NOT EXISTS, so we check the existing
    columns first and only ALTER when the column is absent.
    """
    conn = get_db()
    try:
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(users)")
        }
        print(f"[DB] Existing users columns: {existing}")

        migrations = [
            ("oauth_provider", "TEXT DEFAULT NULL"),
            ("oauth_id",       "TEXT DEFAULT NULL"),
            ("avatar",         "TEXT DEFAULT NULL"),
        ]
        for col, definition in migrations:
            if col not in existing:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col} {definition}")
                print(f"[DB] ✅ Added missing column: {col}")
            else:
                print(f"[DB] Column already exists, skipping: {col}")

        conn.commit()
    except Exception as e:
        print(f"[DB] Migration error: {e}")
        print(traceback.format_exc())
    finally:
        conn.close()


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            username       TEXT NOT NULL UNIQUE,
            email          TEXT NOT NULL UNIQUE,
            password       TEXT,
            oauth_provider TEXT DEFAULT NULL,
            oauth_id       TEXT DEFAULT NULL,
            avatar         TEXT DEFAULT NULL,
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS recipes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            image_path   TEXT,
            food_name    TEXT NOT NULL,
            ingredients  TEXT NOT NULL,
            steps        TEXT NOT NULL,
            cuisine_type TEXT DEFAULT 'General',
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()
    migrate_db()


init_db()


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES — Email / Password
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return redirect(url_for("home") if "user_id" in session else url_for("auth"))

@app.route("/auth")
def auth():
    if "user_id" in session:
        return redirect(url_for("home"))
    google_configured = GOOGLE_CLIENT_ID != "YOUR_GOOGLE_CLIENT_ID"
    return render_template("auth.html", google_configured=google_configured)

@app.route("/signup", methods=["POST"])
def signup():
    d        = request.get_json()
    username = d.get("username", "").strip()
    email    = d.get("email", "").strip().lower()
    password = d.get("password", "")

    if not all([username, email, password]):
        return jsonify({"success": False, "message": "All fields are required."})
    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters."})

    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO users (username, email, password) VALUES (?,?,?)",
            (username, email, generate_password_hash(password))
        )
        conn.commit()
        conn.close()
        return jsonify({
            "success":  True,
            "message":  "Account created successfully! Please sign in.",
            "redirect": "/auth"
        })
    except sqlite3.IntegrityError as e:
        return jsonify({
            "success": False,
            "message": "Username taken." if "username" in str(e) else "Email already registered."
        })

@app.route("/login", methods=["POST"])
def login():
    d     = request.get_json()
    email = d.get("email", "").strip().lower()
    conn  = get_db()
    user  = conn.execute(
        "SELECT * FROM users WHERE email=?", (email,)
    ).fetchone()
    conn.close()
    if user and user["password"] and check_password_hash(user["password"], d.get("password", "")):
        session["user_id"]  = user["id"]
        session["username"] = user["username"]
        session["avatar"]   = user["avatar"]
        return jsonify({"success": True, "redirect": "/home"})
    if user and not user["password"]:
        return jsonify({
            "success": False,
            "message": f"This account was created with {user['oauth_provider']}. Use that to sign in."
        })
    return jsonify({"success": False, "message": "Invalid email or password."})

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth"))


# ═══════════════════════════════════════════════════════════════════════════════
#  FORGOT / RESET PASSWORD
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/forgot_password")
@app.route("/forgot-password")
def forgot_password_page():
    return render_template("forgot_password.html")


@app.route("/auth/reset-password", methods=["POST"])
def auth_reset_password():
    data = request.get_json(silent=True, force=True)
    if not data:
        return jsonify({"success": False, "message": "Invalid request. Please try again."})

    email        = (data.get("email") or "").strip().lower()
    new_password = (data.get("new_password") or "").strip()

    print(f"[Reset] email='{email}' pw_len={len(new_password)}")

    if not email:
        return jsonify({"success": False, "message": "Email is required."})
    if len(new_password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters."})

    conn = None
    try:
        conn = get_db()
        user = conn.execute(
            "SELECT id, password FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        if not user:
            return jsonify({
                "success": False,
                "message": "No account found with that email address."
            })

        if not user["password"]:
            try:
                row = conn.execute(
                    "SELECT oauth_provider FROM users WHERE id = ?", (user["id"],)
                ).fetchone()
                provider = (row["oauth_provider"] if row else None) or "a social account"
            except Exception:
                provider = "a social account"

            return jsonify({
                "success": False,
                "message": (
                    f"This account uses {provider} sign-in. "
                    "Password reset is not available for OAuth accounts."
                )
            })

        conn.execute(
            "UPDATE users SET password = ? WHERE id = ?",
            (generate_password_hash(new_password), user["id"])
        )
        conn.commit()

        print(f"[Reset] ✅ Password updated for id={user['id']} ({email})")
        return jsonify({"success": True, "message": "Password reset successfully."})

    except Exception as e:
        print(f"[Reset] ❌ {e}")
        print(traceback.format_exc())
        return jsonify({"success": False, "message": f"Server error: {str(e)}"})

    finally:
        if conn:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  GOOGLE OAUTH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/auth/google")
def google_login():
    if GOOGLE_CLIENT_ID == "YOUR_GOOGLE_CLIENT_ID":
        return redirect(url_for("auth") + "?error=google_not_configured")
    redirect_uri = url_for("google_callback", _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route("/auth/google/callback")
def google_callback():
    try:
        token     = google.authorize_access_token()
        user_info = token.get("userinfo")

        if not user_info:
            return redirect(url_for("auth") + "?error=google_failed")

        email     = user_info["email"].lower()
        name      = user_info.get("name", email.split("@")[0])
        google_id = user_info["sub"]
        avatar    = user_info.get("picture", "")
        username  = name.replace(" ", "_").lower()

        conn = get_db()
        existing = conn.execute(
            "SELECT * FROM users WHERE LOWER(email)=? OR (oauth_provider='google' AND oauth_id=?)",
            (email, google_id)
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE users SET avatar=?, oauth_id=? WHERE id=?",
                (avatar, google_id, existing["id"])
            )
            conn.commit()
            user_id  = existing["id"]
            username = existing["username"]
        else:
            base_username = username
            counter = 1
            while conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone():
                username = f"{base_username}{counter}"
                counter += 1

            conn.execute(
                """INSERT INTO users (username, email, password, oauth_provider, oauth_id, avatar)
                   VALUES (?,?,NULL,'google',?,?)""",
                (username, email, google_id, avatar)
            )
            conn.commit()
            user_id = conn.execute(
                "SELECT id FROM users WHERE email=?", (email,)
            ).fetchone()["id"]

        conn.close()
        session["user_id"]  = user_id
        session["username"] = username
        session["avatar"]   = avatar
        return redirect(url_for("home"))

    except Exception as e:
        print(f"Google OAuth error: {e}")
        return redirect(url_for("auth") + "?error=google_failed")


# ═══════════════════════════════════════════════════════════════════════════════
#  FACEBOOK — not supported on localhost
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/auth/facebook")
def facebook_login():
    return jsonify({
        "success": False,
        "message": (
            "Facebook login requires a live HTTPS domain and cannot run on localhost. "
            "Please use Google login or email/password instead."
        )
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/home")
def home():
    if "user_id" not in session:
        return redirect(url_for("auth"))
    conn  = get_db()
    total = conn.execute(
        "SELECT COUNT(*) as c FROM recipes WHERE user_id=?", (session["user_id"],)
    ).fetchone()["c"]
    conn.close()
    return render_template("home.html", username=session["username"], total_recipes=total)

@app.route("/upload")
def upload():
    if "user_id" not in session:
        return redirect(url_for("auth"))
    return render_template("upload.html", username=session["username"])

@app.route("/recipes")
def recipes():
    if "user_id" not in session:
        return redirect(url_for("auth"))
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM recipes WHERE user_id=? ORDER BY created_at DESC",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    return render_template("recipes.html",
        username=session["username"],
        recipes=[{
            "id":           r["id"],
            "image_path":   r["image_path"],
            "food_name":    r["food_name"],
            "ingredients":  json.loads(r["ingredients"]),
            "steps":        json.loads(r["steps"]),
            "cuisine_type": r["cuisine_type"],
            "created_at":   r["created_at"]
        } for r in rows]
    )

@app.route("/reports")
def reports():
    if "user_id" not in session:
        return redirect(url_for("auth"))
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM recipes WHERE user_id=? ORDER BY created_at DESC",
        (session["user_id"],)
    ).fetchall()
    conn.close()

    weekly, monthly, cuisines = {}, {}, {}
    now = datetime.now()
    for r in rows:
        dt = datetime.strptime(r["created_at"][:19], "%Y-%m-%d %H:%M:%S")
        if (now - dt).days <= 7:
            day = dt.strftime("%a")
            weekly[day] = weekly.get(day, 0) + 1
        if (now - dt).days <= 30:
            ds = dt.strftime("%d %b")
            monthly[ds] = monthly.get(ds, 0) + 1
        c = r["cuisine_type"] or "General"
        cuisines[c] = cuisines.get(c, 0) + 1

    return render_template("reports.html",
        username=session["username"],
        stats={
            "total":    len(rows),
            "weekly":   weekly,
            "monthly":  monthly,
            "cuisines": cuisines
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATE RECIPE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/generate_recipe", methods=["POST"])
def generate_recipe():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Not authenticated"})
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image uploaded"})

    file = request.files["image"]
    if not file.filename:
        return jsonify({"success": False, "message": "No file selected"})

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"png", "jpg", "jpeg", "gif", "webp"}:
        return jsonify({
            "success": False,
            "message": "Invalid file type. Use PNG, JPG, JPEG, GIF, or WEBP."
        })

    filename = secure_filename(
        f"user{session['user_id']}_{int(datetime.now().timestamp())}.{ext}"
    )
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    result = analyze_image(filepath, file.filename)

    if not result["success"]:
        try:
            os.remove(filepath)
        except Exception:
            pass
        return jsonify({
            "success":  False,
            "not_food": True,
            "message":  result["reason"]
        })

    recipe = result["recipe"]

    # ── Save recipe to DB ─────────────────────────────────────────────────────
    conn = get_db()
    conn.execute(
        """INSERT INTO recipes
           (user_id, image_path, food_name, ingredients, steps, cuisine_type)
           VALUES (?,?,?,?,?,?)""",
        (
            session["user_id"],
            filename,
            recipe["food_name"],
            json.dumps(recipe["ingredients"]),
            json.dumps(recipe["steps"]),
            recipe["cuisine_type"]
        )
    )
    conn.commit()
    conn.close()

    return jsonify({
        "success":       True,
        "redirect":      "/recipes",
        "food_name":     recipe["food_name"],
        "identified_as": result.get("identified_as", recipe["food_name"]),
        "method":        result.get("method", "unknown"),
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  DELETE RECIPE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/delete_recipe/<int:recipe_id>", methods=["POST"])
def delete_recipe(recipe_id):
    if "user_id" not in session:
        return jsonify({"success": False})
    conn = get_db()
    row  = conn.execute(
        "SELECT image_path FROM recipes WHERE id=? AND user_id=?",
        (recipe_id, session["user_id"])
    ).fetchone()
    if row:
        conn.execute(
            "DELETE FROM recipes WHERE id=? AND user_id=?",
            (recipe_id, session["user_id"])
        )
        conn.commit()
        img = os.path.join(app.config["UPLOAD_FOLDER"], row["image_path"])
        if os.path.exists(img):
            os.remove(img)
    conn.close()
    return jsonify({"success": True})


# ═══════════════════════════════════════════════════════════════════════════════
#  STATUS API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/status")
def api_status():
    from food_detector import _model_loaded, _load_error, _food101_loaded, _food101_error
    return jsonify({
        "clip_model_loaded":   _model_loaded,
        "clip_load_error":     _load_error,
        "food101_loaded":      _food101_loaded,
        "food101_error":       _food101_error,
        "dataset_size":        len(RECIPE_DATABASE),
        "api_key_required":    False,
        "google_configured":   GOOGLE_CLIENT_ID != "YOUR_GOOGLE_CLIENT_ID"
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  DEBUG ROUTE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/debug/identify", methods=["POST"])
def debug_identify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]
    ext  = file.filename.rsplit(".", 1)[-1].lower()
    tmp  = f"/tmp/debug_food.{ext}"
    file.save(tmp)

    clip_result = detect_food_local(tmp)
    food_id     = identify_food_type(tmp)

    try:
        os.remove(tmp)
    except Exception:
        pass

    return jsonify({
        # ── CLIP results (food vs non-food) ───────────────────────────────────
        "clip_is_food":          clip_result.get("is_food"),
        "clip_confidence":       clip_result.get("confidence"),
        "clip_top_label":        clip_result.get("top_label"),
        "food_score":            clip_result.get("food_score"),
        "nonfood_score":         clip_result.get("nonfood_score"),
        # ── Food identification results ───────────────────────────────────────
        "identified_db_key":     food_id["db_key"]     if food_id else None,
        "identified_food_name":  food_id["food_name"]  if food_id else None,
        "identified_confidence": food_id["confidence"] if food_id else None,
        "identification_method": food_id["method"]     if food_id else None,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)