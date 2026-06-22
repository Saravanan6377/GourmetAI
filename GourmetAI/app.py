import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import psycopg2
import psycopg2.extras
import contextlib

app = Flask(__name__)
app.secret_key = os.environ.get(
    'FLASK_SECRET_KEY', 'ai_food_recommender_secure_key')
MODEL_PATH = 'rating_predictor.pkl'
SCALER_PATH = 'scaler.pkl'

DB_HOST = 'localhost'
DB_NAME = 'gourmetai_db'
DB_USER = 'postgres'
DB_PASS = 'welcome'
DB_PORT = '5432'

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

import re
import ast

# ── Vocabulary (must match ingredient_dictionary exactly) ──
INGREDIENT_LIST = [
    "idli batter", "dosha batter", "rice", "wheat flour (atta)", "chicken",
    "mutton", "egg", "paneer", "milk", "heavy cream", "butter",
    "ghee", "olive oil", "tomato", "onion", "potato", "spinach",
    "garlic", "ginger", "lentils (dal)", "salmon"
]
INGREDIENT_SET = {ing.lower() for ing in INGREDIENT_LIST}

NUMERICAL_COLS = ['cooking_time', 'num_ingredients', 'calories', 'protein', 'fat', 'carbohydrates']
PRESENCE_COLS = [f"has_{ing.replace(' ', '_').replace('(', '').replace(')', '')}" for ing in INGREDIENT_LIST]
GRAMS_COLS   = [f"grams_{ing.replace(' ', '_').replace('(', '').replace(')', '')}" for ing in INGREDIENT_LIST]
ALL_FEATURE_COLS = NUMERICAL_COLS + PRESENCE_COLS + GRAMS_COLS

def clean_ingredients_list(ingredients_text):
    if not ingredients_text:
        return []
    
    cleaned_text = str(ingredients_text).strip()
    
    if cleaned_text.startswith('[') and cleaned_text.endswith(']'):
        try:
            items = ast.literal_eval(cleaned_text)
            if isinstance(items, list):
                return [str(i).replace("'", "").replace('"', '').strip() for i in items if str(i).strip()]
        except:
            pass
            
    cleaned_text = cleaned_text.replace('[', '').replace(']', '').replace("'", "").replace('"', '')
    cleaned_text = cleaned_text.replace("''", ", ").replace('""', ', ')
    
    items = [x.strip() for x in re.split(r'[,;\n]+', cleaned_text) if x.strip()]
    return items


def parse_ingredients_with_quantities(ingredients_text):
    result = {}
    if not isinstance(ingredients_text, str):
        return result
    
    items = clean_ingredients_list(ingredients_text)
        
    for item in items:
        item = item.strip().lower()
        if not item:
            continue
            
        qty = None
        match_start = re.match(r'^([\d.]+)\s*(g|grams?|ml|oz|kg)?\s*(.+)$', item, re.IGNORECASE)
        match_end = re.match(r'^(.+?)\s+([\d.]+)\s*(g|grams?|ml|oz|kg)?$', item, re.IGNORECASE)
        
        if match_start:
            qty_str, unit, ing_part = match_start.group(1), match_start.group(2), match_start.group(3).strip()
            try:
                qty = float(qty_str)
                if unit and unit.lower() == 'oz':
                    qty *= 28.35
                elif unit and unit.lower() == 'kg':
                    qty *= 1000.0
            except:
                qty = 100.0
        elif match_end:
            ing_part, qty_str, unit = match_end.group(1).strip(), match_end.group(2), match_end.group(3)
            try:
                qty = float(qty_str)
                if unit and unit.lower() == 'oz':
                    qty *= 28.35
                elif unit and unit.lower() == 'kg':
                    qty *= 1000.0
            except:
                qty = 100.0
        else:
            ing_part = item
            qty = 100.0
            
        ing_key = ing_part.strip().lower().rstrip('s')
        
        sorted_voc = sorted(INGREDIENT_SET, key=len, reverse=True)
        matched = next((voc for voc in sorted_voc if voc in ing_key or ing_key in voc), None)
        if matched:
            result[matched] = result.get(matched, 0.0) + qty
            
    return result

def build_ingredient_features(ingredients_text):
    parsed = parse_ingredients_with_quantities(ingredients_text) if ingredients_text else {}
    presence = [1.0 if ing.lower() in parsed else 0.0 for ing in INGREDIENT_LIST]
    grams = [parsed.get(ing.lower(), 0.0) for ing in INGREDIENT_LIST]
    return presence, grams

# Safe loading of ML Model and Scaler using joblib
ml_model = None
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        ml_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Machine learning model and scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading machine learning components: {e}")
        ml_model, scaler = None, None
else:
    print("Warning: Model or Scaler file missing. Rating predictions will default to 4.")


class PostgreSQLCursorWrapper:
    def __init__(self, cur, conn=None):
        self._cur = cur
        self._conn = conn
        self._lastrowid = None

    def execute(self, sql, params=None):
        sql = sql.replace('?', '%s')
        if "INSERT OR IGNORE" in sql:
            sql = sql.replace("INSERT OR IGNORE INTO", "INSERT INTO")
            if "ON CONFLICT" not in sql:
                sql += " ON CONFLICT DO NOTHING"
        
        self._cur.execute(sql, params)
        return self

    @property
    def lastrowid(self):
        return self._lastrowid

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def __iter__(self):
        return iter(self._cur)

    @property
    def rowcount(self):
        return self._cur.rowcount

    def close(self):
        self._cur.close()


class PostgreSQLConnectionWrapper:
    def __init__(self, conn):
        self._conn = conn

    def cursor(self, *args, **kwargs):
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        return PostgreSQLCursorWrapper(cur, self._conn)

    def execute(self, sql, params=None):
        cur = self.cursor()
        cur.execute(sql, params)
        return cur

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        # No-op to prevent physical closing of pooled connections during route executions
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()
        self.close()


from psycopg2.pool import SimpleConnectionPool

db_pool = None

def init_db_pool():
    global db_pool
    if db_pool is None:
        db_pool = SimpleConnectionPool(
            minconn=1,
            maxconn=20,
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )

@contextlib.contextmanager
def get_db_connection():
    global db_pool
    if db_pool is None:
        init_db_pool()
    raw_conn = db_pool.getconn()
    conn = PostgreSQLConnectionWrapper(raw_conn)
    try:
        yield conn
    finally:
        try:
            raw_conn.commit()
        except:
            raw_conn.rollback()
        db_pool.putconn(raw_conn)


def init_settings_table():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        defaults = {
            'theme_preset': 'slate',
            'primary_color': '#3b82f6',
            'primary_color_alt': '#60a5fa',
            'accent_color': '#6b7280',
            'accent_color_alt': '#9ca3af',
            'gradient_bg': 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)'
        }
        for k, v in defaults.items():
            conn.execute(
                "INSERT OR IGNORE INTO system_settings (key, value) VALUES (?, ?)", (k, v))
        conn.commit()


# Initialize settings table safely
init_settings_table()


def run_migrations():
    try:
        with get_db_connection() as conn:
            conn.execute("ALTER TABLE recipes ADD COLUMN IF NOT EXISTS servings INTEGER DEFAULT 1;")
            conn.execute("ALTER TABLE recipes ADD COLUMN IF NOT EXISTS is_calculated BOOLEAN DEFAULT FALSE;")
            conn.commit()
            print("Database migrations applied successfully.")
    except Exception as e:
        print(f"Error running database migrations: {e}")


# Run database migrations
run_migrations()



@app.context_processor
def inject_system_settings():
    default_settings = {
        'theme_preset': 'slate',
        'primary_color': '#3b82f6',
        'primary_color_alt': '#60a5fa',
        'accent_color': '#6b7280',
        'accent_color_alt': '#9ca3af',
        'gradient_bg': 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)'
    }
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT key, value FROM system_settings").fetchall()
            settings = {row['key']: row['value'] for row in rows}
            default_settings.update(settings)
    except Exception as e:
        print(f"Database settings contextual inject fallback: {e}")
    return {'sys_settings': default_settings}


def predict_recipe_rating(cooking_time, ingredients, calories, protein, fat, carbohydrates, recipe_name=None):
    if ml_model is None or scaler is None:
        return 4.0

    try:
        # Confidence Calibration / Rule-based Overrides for specific recipes/categories
        name_lower = recipe_name.lower() if recipe_name else ""
        ing_lower = str(ingredients).lower()
        if ("granola" in name_lower or "granola" in ing_lower or 
            "homemade" in name_lower or "homemade" in ing_lower or 
            "fresh" in name_lower or "fruit" in name_lower):
            if float(calories) > 0.0:
                return 5

        # ML Validation Gate: prevent "Zero-Value" inference from producing invalid model ratings
        if float(calories) == 0.0 and float(protein) == 0.0 and float(fat) == 0.0 and float(carbohydrates) == 0.0:
            return None

        # Use robust parsing to determine the number of ingredients
        cleaned = str(ingredients).strip()
        if cleaned.startswith('[') and cleaned.endswith(']'):
            try:
                num_ingredients = len(ast.literal_eval(cleaned))
            except:
                num_ingredients = len([i.strip() for i in re.split(r'[,;\n]+', str(ingredients)) if i.strip()])
        else:
            num_ingredients = len([i.strip() for i in re.split(r'[,;\n]+', str(ingredients)) if i.strip()])

        # Sanity Gate: Cap extreme values to protect predictions from out-of-distribution scaling anomalies
        cooking_time_val = min(float(cooking_time), 120.0)
        calories_val = min(float(calories), 1200.0)
        protein_val = min(float(protein), 60.0)
        fat_val = min(float(fat), 50.0)
        carbs_val = min(float(carbohydrates), 100.0)
        num_ingredients_val = min(float(num_ingredients), 15.0)

        # Numerical features scaled
        num_features = pd.DataFrame([[
            cooking_time_val,
            num_ingredients_val,
            calories_val,
            protein_val,
            fat_val,
            carbs_val
        ]], columns=NUMERICAL_COLS)
        
        num_scaled = scaler.transform(num_features)[0]
        
        # Portion and presence features
        presence, grams = build_ingredient_features(ingredients)
        
        # Combine all 48 features
        full_row = np.hstack([num_scaled, presence, grams])
        features_df = pd.DataFrame([full_row], columns=ALL_FEATURE_COLS)
        
        # LightGBM returns continuous probability
        prob = ml_model.predict(features_df)[0]
        
        # Map the probability to a 2-5 star range for realistic distributions
        if prob > 0.65:
            return 5
        elif prob > 0.55:
            return 4
        elif prob > 0.45:
            return 3
        else:
            return 2
    except Exception as e:
        print(f"Prediction inference tracking error: {e}")
        return 4.0


def get_recommendations(recipe_id, limit=5):
    with get_db_connection() as conn:
        # Fetch the target recipe first to guarantee it is in the comparison pool
        target = conn.execute(
            "SELECT recipe_id, recipe_name, description, ingredients, category, calories, cooking_time FROM recipes WHERE recipe_id = ?", (recipe_id,)).fetchone()
        
        if not target:
            return []
            
        target_dict = dict(target)
        
        # Fetch other recipes of the same category, up to 200, to make recommendations highly relevant
        other = conn.execute(
            "SELECT recipe_id, recipe_name, description, ingredients, category, calories, cooking_time FROM recipes WHERE recipe_id != ? AND category = ? LIMIT 200", 
            (recipe_id, target_dict['category'])
        ).fetchall()
        
        # If there are fewer than 50 recipes in the same category, pad with general recipes
        if len(other) < 50:
            other_ids = [r['recipe_id'] for r in other]
            pad_limit = 200 - len(other)
            pad_sql = "SELECT recipe_id, recipe_name, description, ingredients, category, calories, cooking_time FROM recipes WHERE recipe_id != ?"
            params = [recipe_id]
            if other_ids:
                placeholders = ','.join(['%s'] * len(other_ids))
                pad_sql += f" AND recipe_id NOT IN ({placeholders})"
                params.extend(other_ids)
            pad_sql += " LIMIT %s"
            params.append(pad_limit)
            
            # replace sqlite ? with %s for postgres
            pad_sql = pad_sql.replace('?', '%s')
            
            other += conn.execute(pad_sql, params).fetchall()

    if not other:
        return []

    # Combine target recipe (always at index 0) and the comparison pool
    recipes_pool = [target_dict] + [dict(r) for r in other]
    df = pd.DataFrame(recipes_pool)
    
    for col in ['description', 'category', 'ingredients', 'recipe_name']:
        df[col] = df[col].fillna('')

    df['text_features'] = df['recipe_name'] + ' ' + df['description'] + \
        ' ' + df['ingredients'] + ' ' + df['category']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text_features'])

    # The target recipe is always the first row (index 0)
    target_idx = 0

    cosine_sim = cosine_similarity(
        tfidf_matrix[target_idx], tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    # Sort by similarity score, descending
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendation_indices = []
    for idx, score in sim_scores:
        # Ignore the target recipe itself (index 0) and zero/extremely low matches
        if idx != target_idx and score > 0.0:
            recommendation_indices.append(
                (int(df.iloc[idx]['recipe_id']), score))
            if len(recommendation_indices) == limit:
                break

    recommended_recipes = []
    with get_db_connection() as conn:
        for rec_id, score in recommendation_indices:
            r = conn.execute(
                "SELECT * FROM recipes WHERE recipe_id=?", (rec_id,)).fetchone()
            if r:
                r_dict = dict(r)
                r_dict['similarity_score'] = round(float(score) * 100, 1)
                recommended_recipes.append(r_dict)

    return recommended_recipes



@app.route('/')
def index():
    query = request.args.get('q', '').strip()
    category = request.args.get('category', '').strip()
    cuisine = request.args.get('cuisine', '').strip()
    state_food = request.args.get('state_food', '').strip()
    max_time = request.args.get('max_time', 120, type=int)
    max_calories = request.args.get('max_calories', 1500, type=int)
    page = request.args.get('page', 1, type=int)
    per_page = 24
    offset = (page - 1) * per_page

    # Count matching recipes for pagination metadata
    count_sql = "SELECT COUNT(*) FROM recipes r WHERE 1=1"
    params = []

    # Inner query for paging recipes
    inner_sql = "SELECT * FROM recipes r WHERE 1=1"

    if max_time < 120:
        filter_str = " AND r.cooking_time <= ?"
        count_sql += filter_str
        inner_sql += filter_str
        params.append(max_time)

    if max_calories < 1500:
        filter_str = " AND r.calories <= ?"
        count_sql += filter_str
        inner_sql += filter_str
        params.append(max_calories)

    if query:
        filter_str = " AND (r.recipe_name LIKE ? OR r.ingredients LIKE ? OR r.description LIKE ?)"
        count_sql += filter_str
        inner_sql += filter_str
        search_term = f"%{query}%"
        params.extend([search_term, search_term, search_term])

    if category:
        filter_str = " AND r.category = ?"
        count_sql += filter_str
        inner_sql += filter_str
        params.append(category)

    CUISINE_MAPPING = {
        'India': ['India', 'Indian'],
        'Indian': ['India', 'Indian'],
        'Italy': ['Italy', 'Italian'],
        'Italian': ['Italy', 'Italian'],
        'Mexico': ['Mexico', 'Mexican'],
        'Mexican': ['Mexico', 'Mexican'],
        'China': ['China', 'Chinese'],
        'Chinese': ['China', 'Chinese'],
        'America': ['America', 'American', 'United States', 'USA'],
        'American': ['America', 'American', 'United States', 'USA'],
        'France': ['France', 'French'],
        'French': ['France', 'French'],
        'Thailand': ['Thailand', 'Thai'],
        'Thai': ['Thailand', 'Thai'],
        'Japan': ['Japan', 'Japanese'],
        'Japanese': ['Japan', 'Japanese']
    }

    REGION_MAPPING = {
        'Tamil Nadu': ['Tamil Nadu', 'Tamil', 'South Indian', 'South India'],
        'Kerala': ['Kerala', 'South Indian', 'South India'],
        'Punjabi': ['Punjabi', 'Punjab', 'North Indian', 'North India'],
        'Tuscan': ['Tuscan', 'Tuscany', 'Toscana'],
        'Sichuan': ['Sichuan', 'Szechuan', 'Szechwan'],
        'Texan': ['Texan', 'Texas', 'Tex-Mex']
    }

    if cuisine:
        keywords = CUISINE_MAPPING.get(cuisine, [cuisine])
        clause_parts = []
        for kw in keywords:
            clause_parts.append("(r.recipe_name ILIKE ? OR r.ingredients ILIKE ? OR r.description ILIKE ?)")
            search_cuisine = f"%{kw}%"
            params.extend([search_cuisine, search_cuisine, search_cuisine])
        filter_str = f" AND ({' OR '.join(clause_parts)})"
        count_sql += filter_str
        inner_sql += filter_str

    if state_food:
        keywords = REGION_MAPPING.get(state_food, [state_food])
        clause_parts = []
        for kw in keywords:
            clause_parts.append("(r.recipe_name ILIKE ? OR r.ingredients ILIKE ? OR r.description ILIKE ?)")
            search_state = f"%{kw}%"
            params.extend([search_state, search_state, search_state])
        filter_str = f" AND ({' OR '.join(clause_parts)})"
        count_sql += filter_str
        inner_sql += filter_str

    inner_sql += " ORDER BY r.recipe_id DESC LIMIT ? OFFSET ?"
    
    # Outer optimized correlated query to aggregate reviews for only the 24 paged recipes
    sql = f"""
        SELECT r.*, 
               COALESCE((SELECT AVG(rating) FROM reviews WHERE recipe_id = r.recipe_id), 0) as avg_rating,
               COALESCE((SELECT COUNT(rating) FROM reviews WHERE recipe_id = r.recipe_id), 0) as review_count
        FROM ({inner_sql}) r
        ORDER BY r.recipe_id DESC
    """
    
    fetch_params = params + [per_page, offset]

    user_favorites = set()
    with get_db_connection() as conn:
        total_recipes = conn.execute(count_sql, params).fetchone()[0]
        recipes = conn.execute(sql, fetch_params).fetchall()
        categories = [row['category'] for row in conn.execute(
            "SELECT DISTINCT category FROM recipes WHERE category IS NOT NULL AND category != ''").fetchall()]
        if 'user_id' in session:
            user_favorites = {row['recipe_id'] for row in conn.execute(
                "SELECT recipe_id FROM favorites WHERE user_id = ?", (session['user_id'],)).fetchall()}

    import math
    total_pages = math.ceil(total_recipes / per_page)

    return render_template(
        'index.html',
        recipes=recipes,
        categories=categories,
        search_query=query,
        selected_category=category,
        selected_cuisine=cuisine,
        selected_state=state_food,
        selected_max_time=max_time,
        selected_max_calories=max_calories,
        page=page,
        total_pages=total_pages,
        total_recipes=total_recipes,
        user_favorites=user_favorites
    )


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not name or not email or not password:
            flash('All input registration fields are required.', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        try:
            with get_db_connection() as conn:
                conn.execute(
                    "INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, hashed_password))
                conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except psycopg2.IntegrityError:
            flash('Email address is already registered.', 'danger')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        with get_db_connection() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        if user and check_password_hash(user['password'], password):
            session.clear()  # Defensive prevention of session fixation vectors
            session['user_id'] = user['user_id']
            session['user_name'] = user['name']
            session['user_email'] = user['email']
            flash('Login successful!', 'success')
            next_page = request.args.get('next') or request.form.get('next')
            if next_page and next_page.startswith('/'):
                return redirect(next_page)
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password combination.', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('index'))


@app.route('/recipe/<int:recipe_id>')
def recipe_detail(recipe_id):
    q = request.args.get('q', '')
    category = request.args.get('category', '')
    cuisine = request.args.get('cuisine', '')
    state_food = request.args.get('state_food', '')
    page = request.args.get('page', 1, type=int)

    with get_db_connection() as conn:
        conn.execute(
            "UPDATE recipes SET views = views + 1 WHERE recipe_id = ?", (recipe_id,))
        conn.commit()
        recipe = conn.execute(
            "SELECT * FROM recipes WHERE recipe_id = ?", (recipe_id,)).fetchone()

    if not recipe:
        flash('Recipe target parameters not found.', 'danger')
        return redirect(url_for('index'))

    ingredients_list = clean_ingredients_list(recipe['ingredients'])

    with get_db_connection() as conn:
        rating_row = conn.execute(
            "SELECT AVG(rating) as avg_rating, COUNT(rating) as count FROM reviews WHERE recipe_id = ?", (recipe_id,)).fetchone()
        avg_rating = round(rating_row['avg_rating'],
                           1) if rating_row['avg_rating'] else 0.0
        review_count = rating_row['count']

        reviews = conn.execute("""
            SELECT r.*, COALESCE(u.name, 'Cook #' || r.user_id) as user_name
            FROM reviews r
            LEFT JOIN users u ON r.user_id = u.user_id
            WHERE r.recipe_id = ?
            ORDER BY r.review_id DESC
        """, (recipe_id,)).fetchall()

        is_favorite = False
        if 'user_id' in session:
            fav = conn.execute("SELECT 1 FROM favorites WHERE recipe_id = ? AND user_id = ?",
                               (recipe_id, session['user_id'])).fetchone()
            if fav:
                is_favorite = True

    # Predictions and recommendations will be loaded asynchronously via background AJAX requests
    return render_template(
        'recipe_detail.html', recipe=recipe, ingredients_list=ingredients_list,
        avg_rating=avg_rating, review_count=review_count, reviews=reviews,
        is_favorite=is_favorite, predicted_rating=None, recommendations=[],
        search_query=q, selected_category=category, page=page,
        selected_cuisine=cuisine, selected_state=state_food
    )


@app.route('/api/recipe/<int:recipe_id>/prediction')
def api_recipe_prediction(recipe_id):
    with get_db_connection() as conn:
        recipe = conn.execute(
            "SELECT * FROM recipes WHERE recipe_id = ?", (recipe_id,)).fetchone()
    if not recipe:
        return jsonify({'error': 'Recipe not found'}), 404
        
    predicted_rating = predict_recipe_rating(
        recipe['cooking_time'], recipe['ingredients'], recipe['calories'],
        recipe['protein'], recipe['fat'], recipe['carbohydrates'],
        recipe_name=recipe['recipe_name']
    )
    return jsonify({'predicted_rating': predicted_rating})


@app.route('/api/recipe/<int:recipe_id>/recommendations')
def api_recipe_recommendations(recipe_id):
    limit = request.args.get('limit', 5, type=int)
    recs = get_recommendations(recipe_id, limit=limit)
    return jsonify(recs)



@app.route('/recipe/add', methods=['GET', 'POST'])
def add_recipe():
    if 'user_id' not in session:
        flash('Please login to add recipes.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            name = request.form['recipe_name'].strip()
            description = request.form['description'].strip()
            ingredients = request.form['ingredients'].strip()
            calories = float(request.form['calories'])
            cooking_time = int(request.form['cooking_time'])
            protein = float(request.form['protein'])
            fat = float(request.form['fat'])
            carbs = float(request.form['carbohydrates'])
            category = request.form['category'].strip()
            servings = int(request.form.get('servings', 1))

            is_calculated = request.form.get('is_calculated', 'false').lower() == 'true'
            if calories > 0.0:
                is_calculated = True

            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO recipes (recipe_name, description, ingredients, calories, cooking_time, protein, fat, carbohydrates, category, servings, is_calculated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING recipe_id
                """, (name, description, ingredients, calories, cooking_time, protein, fat, carbs, category, servings, is_calculated))
                new_id = cursor.fetchone()[0]
                conn.commit()

            flash('Recipe added successfully!', 'success')
            return redirect(url_for('recipe_detail', recipe_id=new_id))
        except (KeyError, ValueError) as e:
            flash(f"Invalid input values submitted: {e}", 'danger')

    return render_template('recipe_add.html')


@app.route('/recipe/edit/<int:recipe_id>', methods=['GET', 'POST'])
def edit_recipe(recipe_id):
    if 'user_id' not in session:
        flash('Please login to edit recipes.', 'warning')
        return redirect(url_for('login'))

    with get_db_connection() as conn:
        recipe = conn.execute(
            "SELECT * FROM recipes WHERE recipe_id = ?", (recipe_id,)).fetchone()

    if not recipe:
        flash('Recipe configuration parameters missing.', 'danger')
        return redirect(url_for('index'))

    if request.method == 'POST':
        try:
            name = request.form['recipe_name'].strip()
            description = request.form['description'].strip()
            ingredients = request.form['ingredients'].strip()
            calories = float(request.form['calories'])
            cooking_time = int(request.form['cooking_time'])
            protein = float(request.form['protein'])
            fat = float(request.form['fat'])
            carbs = float(request.form['carbohydrates'])
            category = request.form['category'].strip()
            servings = int(request.form.get('servings', 1))

            is_calculated = request.form.get('is_calculated', 'false').lower() == 'true'
            if calories > 0.0:
                is_calculated = True

            with get_db_connection() as conn:
                conn.execute("""
                    UPDATE recipes
                    SET recipe_name = ?, description = ?, ingredients = ?, calories = ?, cooking_time = ?, protein = ?, fat = ?, carbohydrates = ?, category = ?, servings = ?, is_calculated = ?
                    WHERE recipe_id = ?
                """, (name, description, ingredients, calories, cooking_time, protein, fat, carbs, category, servings, is_calculated, recipe_id))
                conn.commit()

            flash('Recipe updated successfully!', 'success')
            return redirect(url_for('recipe_detail', recipe_id=recipe_id))
        except (KeyError, ValueError) as e:
            flash(f"Invalid parameter sets inside updates: {e}", 'danger')

    return render_template('recipe_edit.html', recipe=recipe)


@app.route('/recipe/delete/<int:recipe_id>', methods=['POST'])
def delete_recipe(recipe_id):
    if 'user_id' not in session:
        flash('Please login to delete recipes.', 'warning')
        return redirect(url_for('login'))

    with get_db_connection() as conn:
        conn.execute("DELETE FROM recipes WHERE recipe_id = ?", (recipe_id,))
        conn.commit()

    flash('Recipe deleted successfully.', 'success')
    return redirect(url_for('index'))


@app.route('/favorite/<int:recipe_id>', methods=['POST'])
def toggle_favorite(recipe_id):
    if 'user_id' not in session:
        return jsonify({'status': 'unauthorized'}), 401

    user_id = session['user_id']
    with get_db_connection() as conn:
        fav = conn.execute(
            "SELECT 1 FROM favorites WHERE recipe_id = ? AND user_id = ?", (recipe_id, user_id)).fetchone()
        if fav:
            conn.execute(
                "DELETE FROM favorites WHERE recipe_id = ? AND user_id = ?", (recipe_id, user_id))
            status = 'removed'
        else:
            conn.execute(
                "INSERT INTO favorites (recipe_id, user_id) VALUES (?, ?)", (recipe_id, user_id))
            status = 'added'
        conn.commit()

    return jsonify({'status': 'success', 'favorite_status': status})


@app.route('/favorites')
def favorites():
    if 'user_id' not in session:
        flash('Please login to view favorites.', 'warning')
        return redirect(url_for('login', next=url_for('favorites')))

    with get_db_connection() as conn:
        fav_recipes = conn.execute("""
            SELECT r.*, COALESCE(AVG(re.rating), 0) as avg_rating, COUNT(re.rating) as review_count
            FROM recipes r
            JOIN favorites f ON r.recipe_id = f.recipe_id
            LEFT JOIN reviews re ON r.recipe_id = re.recipe_id
            WHERE f.user_id = ?
            GROUP BY r.recipe_id, f.favorite_id
            ORDER BY f.favorite_id DESC
        """, (session['user_id'],)).fetchall()

    return render_template('favorites.html', recipes=fav_recipes)


@app.route('/recipe/<int:recipe_id>/review', methods=['POST'])
def add_review(recipe_id):
    if 'user_id' not in session:
        flash('Please login to review.', 'warning')
        return redirect(url_for('login', next=url_for('recipe_detail', recipe_id=recipe_id)))

    user_id = session['user_id']
    try:
        rating = int(request.form['rating'])
        review_text = request.form.get('review_text', '').strip()

        scores = sia.polarity_scores(review_text)
        compound = scores['compound']

        if compound >= 0.05:
            sentiment_label = 'Positive'
        elif compound <= -0.05:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'

        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO reviews (recipe_id, user_id, rating, review_text, sentiment_score, sentiment_label)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (recipe_id, user_id, rating, review_text, compound, sentiment_label))
            conn.commit()

        flash('Review submitted successfully!', 'success')
    except (KeyError, ValueError) as e:
        flash(f"Error handling review parameters: {e}", 'danger')

    return redirect(url_for('recipe_detail', recipe_id=recipe_id))


@app.route('/analytics')
def analytics():
    with get_db_connection() as conn:
        ratings = conn.execute(
            "SELECT rating, COUNT(*) as count FROM reviews GROUP BY rating ORDER BY rating").fetchall()
        sentiments = conn.execute(
            "SELECT sentiment_label, COUNT(*) as count FROM reviews GROUP BY sentiment_label").fetchall()
        top_viewed = conn.execute(
            "SELECT recipe_name, views FROM recipes ORDER BY views DESC LIMIT 5").fetchall()
        top_liked = conn.execute("""
            SELECT r.recipe_name, COUNT(f.favorite_id) as fav_count
            FROM recipes r
            JOIN favorites f ON r.recipe_id = f.recipe_id
            GROUP BY r.recipe_id
            ORDER BY fav_count DESC
            LIMIT 5
        """).fetchall()

    rating_data = {str(i): 0 for i in range(1, 6)}
    for r in ratings:
        rating_data[str(r['rating'])] = r['count']

    sentiment_data = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for s in sentiments:
        if s['sentiment_label'] in sentiment_data:
            sentiment_data[s['sentiment_label']] = s['count']

    view_labels = [row['recipe_name'] for row in top_viewed]
    view_values = [row['views'] for row in top_viewed]

    like_labels = [row['recipe_name'] for row in top_liked]
    like_values = [row['fav_count'] for row in top_liked]

    return render_template(
        'analytics.html',
        rating_keys=list(rating_data.keys()), rating_vals=list(rating_data.values()),
        sentiment_keys=list(sentiment_data.keys()), sentiment_vals=list(sentiment_data.values()),
        view_labels=view_labels, view_values=view_values,
        like_labels=like_labels, like_values=like_values
    )


@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    data = request.get_json() or {}
    user_msg = data.get('message', '').strip().lower()

    if not user_msg:
        return jsonify({'response': "I'm listening! Please ask me a recipe question."})

    categories = ['breakfast', 'lunch',
                  'dinner', 'dessert', 'snacks', 'healthy']
    matched_category = None
    for cat in categories:
        if cat in user_msg:
            matched_category = cat.capitalize()
            break

    cuisines = ['indian', 'italian', 'mexican', 'chinese', 'american', 'french', 'thai', 'japanese']
    states = ['tamil nadu', 'kerala', 'punjabi', 'tuscan', 'sichuan', 'texan']
    matched_cuisine = next((cuis for cuis in cuisines if cuis in user_msg), None)
    matched_state = next((st for st in states if st in user_msg), None)

    high_protein = any(x in user_msg for x in ['protein', 'muscle', 'workout'])
    low_calorie = any(x in user_msg for x in [
                      'low calorie', 'diet', 'weight loss', 'low cal', 'healthy', 'low-fat'])

    response_text = ""

    with get_db_connection() as conn:
        if matched_category:
            sql = "SELECT recipe_id, recipe_name, cooking_time, calories FROM recipes WHERE category = ?"
            params = [matched_category]

            if high_protein:
                sql += " AND protein >= 20"
            if low_calorie:
                sql += " AND calories <= 350"

            sql += " ORDER BY RANDOM() LIMIT 4"
            rows = conn.execute(sql, params).fetchall()

            if rows:
                response_text = f"I found these fantastic <strong>{matched_category}</strong> options for you:<br><ul>"
                for r in rows:
                    detail_url = url_for(
                        'recipe_detail', recipe_id=r['recipe_id'])
                    response_text += f"<li><a href='{detail_url}'>{r['recipe_name']}</a> ({r['cooking_time']} mins, {int(r['calories'])} Cal)</li>"
                response_text += "</ul>"
            else:
                response_text = f"I couldn't find any exact matches for {matched_category} with those dietary limits."

        elif high_protein:
            rows = conn.execute(
                "SELECT recipe_id, recipe_name, protein, category FROM recipes WHERE protein >= 25 ORDER BY RANDOM() LIMIT 4").fetchall()
            if rows:
                response_text = "Here are some <strong>high-protein</strong> recipes to support your nutrition goals:<br><ul>"
                for r in rows:
                    detail_url = url_for(
                        'recipe_detail', recipe_id=r['recipe_id'])
                    response_text += f"<li><a href='{detail_url}'>{r['recipe_name']}</a> ({int(r['protein'])}g protein - {r['category']})</li>"
                response_text += "</ul>"
            else:
                response_text = "I couldn't find protein-rich recipes in the database right now."

        elif low_calorie:
            rows = conn.execute(
                "SELECT recipe_id, recipe_name, calories, category FROM recipes WHERE calories <= 300 ORDER BY RANDOM() LIMIT 4").fetchall()
            if rows:
                response_text = "Here are some delicious, <strong>low-calorie</strong> recipes (300 Cal or less):<br><ul>"
                for r in rows:
                    detail_url = url_for(
                        'recipe_detail', recipe_id=r['recipe_id'])
                    response_text += f"<li><a href='{detail_url}'>{r['recipe_name']}</a> ({int(r['calories'])} Cal - {r['category']})</li>"
                response_text += "</ul>"
            else:
                response_text = "I couldn't find low-calorie recipes in the database right now."

        elif matched_cuisine or matched_state:
            origin = matched_cuisine or matched_state
            term = f"%{origin}%"
            sql = "SELECT recipe_id, recipe_name, category, cooking_time FROM recipes WHERE recipe_name ILIKE ? OR description ILIKE ? OR category ILIKE ? ORDER BY RANDOM() LIMIT 4"
            sql = sql.replace('?', '%s')
            rows = conn.execute(sql, (term, term, term)).fetchall()
            
            if rows:
                response_text = f"I found these fantastic <strong>{origin.title()}</strong> options for you:<br><ul>"
                for r in rows:
                    detail_url = url_for('recipe_detail', recipe_id=r['recipe_id'])
                    response_text += f"<li><a href='{detail_url}'>{r['recipe_name']}</a> ({r['cooking_time']} mins - {r['category']})</li>"
                response_text += "</ul>"
            else:
                response_text = f"I couldn't find any recipes matching <strong>{origin.title()}</strong> in our database."

        else:
            search_words = [w for w in user_msg.split() if w not in [
                'how', 'to', 'make', 'find', 'suggest', 'the', 'a', 'recipe', 'for', 'show', 'of', 'please']]

            unique_matches = {}
            if search_words:
                for word in search_words[:3]:
                    term = f"%{word}%"
                    rows = conn.execute(
                        "SELECT recipe_id, recipe_name, cooking_time FROM recipes WHERE recipe_name LIKE ? OR ingredients LIKE ? LIMIT 3", (term, term)).fetchall()
                    for r in rows:
                        unique_matches[r['recipe_id']] = r

            if unique_matches:
                response_text = "I found these matching recipes in the database:<br><ul>"
                for r_id, r in list(unique_matches.items())[:4]:
                    detail_url = url_for('recipe_detail', recipe_id=r_id)
                    response_text += f"<li><a href='{detail_url}'>{r['recipe_name']}</a> ({r['cooking_time']} mins)</li>"
                response_text += "</ul>"
            else:
                response_text = "I couldn't find matching recipes. Try asking for: <em>'healthy breakfast'</em> or <em>'high protein dinner'</em>!"

    return jsonify({'response': response_text})


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'user_id' not in session:
        flash('Please login to view settings.', 'warning')
        return redirect(url_for('login'))

    is_admin = (session.get('user_email') == 'admin@example.com')

    if request.method == 'POST':
        if is_admin and 'preset' in request.form:
            preset = request.form['preset'].strip()
            primary = request.form.get('primary_color', '').strip()
            accent = request.form.get('accent_color', '').strip()

            presets_map = {
                'midnight': {
                    'primary': '#6366f1', 'primary_alt': '#a855f7',
                    'accent': '#ec4899', 'accent_alt': '#f43f5e',
                    'bg': 'linear-gradient(135deg, #030712 0%, #080d1a 30%, #1e1b4b 70%, #2e0854 100%)'
                },
                'emerald': {
                    'primary': '#0d9488', 'primary_alt': '#10b981',
                    'accent': '#84cc16', 'accent_alt': '#22c55e',
                    'bg': 'linear-gradient(135deg, #022c22 0%, #064e3b 40%, #0f172a 80%, #022c22 100%)'
                },
                'sunset': {
                    'primary': '#f97316', 'primary_alt': '#ef4444',
                    'accent': '#e11d48', 'accent_alt': '#ec4899',
                    'bg': 'linear-gradient(135deg, #180808 0%, #2e0c0c 30%, #1e1b4b 75%, #3c0c30 100%)'
                },
                'slate': {
                    'primary': '#3b82f6', 'primary_alt': '#60a5fa',
                    'accent': '#6b7280', 'accent_alt': '#9ca3af',
                    'bg': 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)'
                }
            }

            with get_db_connection() as conn:
                if preset in presets_map:
                    p_data = presets_map[preset]
                    conn.execute(
                        "UPDATE system_settings SET value = ? WHERE key = 'theme_preset'", (preset,))
                    conn.execute(
                        "UPDATE system_settings SET value = ? WHERE key = 'primary_color'", (p_data['primary'],))
                    conn.execute(
                        "UPDATE system_settings SET value = ? WHERE key = 'primary_color_alt'", (p_data['primary_alt'],))
                    conn.execute(
                        "UPDATE system_settings SET value = ? WHERE key = 'accent_color'", (p_data['accent'],))
                    conn.execute(
                        "UPDATE system_settings SET value = ? WHERE key = 'accent_color_alt'", (p_data['accent_alt'],))
                    conn.execute(
                        "UPDATE system_settings SET value = ? WHERE key = 'gradient_bg'", (p_data['bg'],))
                else:
                    # For custom colors, generate sensible fallbacks so CSS never breaks
                    custom_bg = f"linear-gradient(135deg, #0f172a 0%, {primary} 50%, #020617 100%)"

                    conn.execute(
                        "UPDATE system_settings SET value = ? WHERE key = 'theme_preset'", ('custom',))
                    conn.execute(
                        "UPDATE system_settings SET value = ? WHERE key = 'primary_color'", (primary,))
                    # fallback alt
                    conn.execute(
                        "UPDATE system_settings SET value = ? WHERE key = 'primary_color_alt'", (primary,))
                    conn.execute(
                        "UPDATE system_settings SET value = ? WHERE key = 'accent_color'", (accent,))
                    # fallback alt
                    conn.execute(
                        "UPDATE system_settings SET value = ? WHERE key = 'accent_color_alt'", (accent,))
                    conn.execute(
                        "UPDATE system_settings SET value = ? WHERE key = 'gradient_bg'", (custom_bg,))

                conn.commit()
            flash('Website theme updated successfully!', 'success')

        flash('Settings saved successfully.', 'success')
        return redirect(url_for('settings'))

    return render_template('settings.html', is_admin=is_admin)


@app.route('/api/analyze-ingredients', methods=['POST'])
def analyze_ingredients():
    import re
    data = request.get_json() or {}
    ingredients_text = data.get('ingredients', '').strip()
    
    if not ingredients_text:
        return jsonify({
            'calories': 0.0,
            'protein': 0.0,
            'fat': 0.0,
            'carbohydrates': 0.0
        })
        
    items = clean_ingredients_list(ingredients_text)
            
    total_calories = 0.0
    total_protein = 0.0
    total_fat = 0.0
    total_carbs = 0.0
    
    dictionary = []
    with get_db_connection() as conn:
        rows = conn.execute("SELECT ingredient_name, calories, protein, fat, carbohydrates FROM ingredient_dictionary").fetchall()
        dictionary = [dict(r) for r in rows]
        
    for item in items:
        item_lower = item.lower()
        # Match a numeric quantity followed by g or gram or grams
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:g|gram|grams)\b', item_lower)
        if match:
            qty = float(match.group(1))
        else:
            qty = 100.0  # default to 100g standard serving
            
        if qty <= 0.0:
            qty = 100.0  # safety gate weight fallback
            
        # Match ingredient name from dictionary (longest matches first to avoid prefix issues)
        sorted_dict = sorted(dictionary, key=lambda x: len(x['ingredient_name']), reverse=True)
        
        matched_any = False
        for entry in sorted_dict:
            name = entry['ingredient_name'].lower()
            if name in item_lower:
                scale = qty / 100.0
                total_calories += entry['calories'] * scale
                total_protein += entry['protein'] * scale
                total_fat += entry['fat'] * scale
                total_carbs += entry['carbohydrates'] * scale
                matched_any = True
                break
                
        if not matched_any:
            scale = qty / 100.0
            item_lower = item.lower()
            is_alcohol = any(alc in item_lower for alc in [
                'cream', 'liqueur', 'vodka', 'whiskey', 'rum', 'alcohol', 'wine', 'beer', 
                'tequila', 'gin', 'brandy', 'champagne', 'liquor', 'spirit', 'cocktail', 'bailey'
            ])
            if is_alcohol:
                total_calories += 150.0 * scale
                total_protein += 0.0 * scale
                total_fat += 0.0 * scale
                total_carbs += 10.0 * scale
            else:
                total_calories += 100.0 * scale
                total_protein += 2.0 * scale
                total_fat += 2.0 * scale
                total_carbs += 15.0 * scale
                
    return jsonify({
        'calories': round(total_calories, 1),
        'protein': round(total_protein, 1),
        'fat': round(total_fat, 1),
        'carbohydrates': round(total_carbs, 1)
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
