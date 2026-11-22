# 🍴 Recipe Lens — AI Recipe Generator from Food Images
### JSS Shri Manjunatheshwara Institute | Academic Project by K. Madhurya

---

## 📁 Project Structure

RECIPELENS/
│
├── __pycache__/
├── .venv/
│
├── database/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── food_dataset.py
│   ├── recipes.db
│   └── schema.sql
│
├── static/
│   ├── css/
│   │   ├── auth.css
│   │   ├── home.css
│   │   ├── recipes.css
│   │   ├── reports.css
│   │   └── upload.css
│   │
│   ├── js/
│   │   ├── auth.js
│   │   ├── recipes.js
│   │   ├── reports.js
│   │   └── upload.js
│   │
│   └── uploads/
│
├── templates/
│   ├── auth.html
│   ├── forgot_password.html
│   ├── home.html
│   ├── recipes.html
│   ├── reports.html
│   └── upload.html
│
├── app.py
├── check_db.py
├── food_detector.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Setup & Run

### Step 1 — Install Python 3.10+
Download from https://python.org

### Step 2 — Install dependencies
Open terminal in the project folder and run:
```bash
pip install -r requirements.txt
```

### Step 3 — Run the server
```bash
python app.py
```

### Step 4 — Open in browser
Visit: **http://127.0.0.1:5000**

The SQLite database (`database/recipes.db`) and `static/uploads/` folder
are created **automatically** on first run. No setup needed.

---

## 🔄 User Flow

```
/ (root)
 └── If not logged in → /auth (Login/Signup)
      └── After login/signup → /home (Dashboard)
           ├── /upload → Upload image → Generate recipe → /recipes
           ├── /recipes → View all saved recipes, delete recipes
           ├── /reports → Charts (weekly/monthly/cuisine) + PDF download
           └── /logout → Clear session → /auth
```

---

## 🤖 AI Logic (Mock / Replace with Real API)

Currently uses **keyword-based detection** (filename matching) + a curated
recipe database. To upgrade to real AI:

1. Sign up for **Anthropic Claude API** (https://console.anthropic.com)
2. Replace the `analyze_image_and_generate_recipe()` function in `app.py`:

```python
import anthropic, base64

def analyze_image_and_generate_recipe(image_filepath):
    client = anthropic.Anthropic(api_key="YOUR_API_KEY")
    with open(image_filepath, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                  "media_type": "image/jpeg", "data": img_data}},
                {"type": "text", "text":
                  "Identify the dish and return JSON: {food_name, cuisine_type, ingredients: [], steps: []}"}
            ]
        }]
    )
    import json
    return json.loads(response.content[0].text)
```

---

## 🗄️ Database Tables

**users** — id, username, email, password (hashed), created_at  
**recipes** — id, user_id, image_path, food_name, ingredients (JSON), steps (JSON), cuisine_type, created_at

---

## 📋 Features Checklist

- [x] Login + Signup with password hashing (bcrypt)
- [x] Eye icon to show/hide password
- [x] Password strength meter
- [x] Social login buttons (UI only, configurable)
- [x] SQLite database (auto-created)
- [x] File upload + drag-and-drop
- [x] Camera capture
- [x] AI recipe generation (mock, upgradeable)
- [x] Recipe cards with ingredient preview
- [x] Full recipe modal with numbered steps
- [x] Delete recipe
- [x] Reports: bar chart, doughnut, line chart
- [x] PDF report download (browser print)
- [x] Logout with session clear
- [x] Fully responsive (mobile + desktop)
- [x] Dark theme with food photography backgrounds
