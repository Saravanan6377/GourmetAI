import os
import psycopg2
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import re
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH = "rating_predictor.pkl"
SCALER_PATH = "scaler.pkl"

DB_HOST = "localhost"
DB_NAME = "gourmetai_db"
DB_USER = "postgres"
DB_PASS = "welcome"
DB_PORT = "5432"

# ── Vocabulary (must match app.py exactly) ──
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

def parse_ingredients_with_quantities(ingredients_text):
    result = {}
    if not isinstance(ingredients_text, str):
        return result
    cleaned = ingredients_text.strip()
    if cleaned.startswith('[') and cleaned.endswith(']'):
        try:
            items = ast.literal_eval(cleaned)
        except:
            items = [x.strip() for x in re.split(r'[,;\n]+', ingredients_text) if x.strip()]
    else:
        items = [x.strip() for x in re.split(r'[,;\n]+', ingredients_text) if x.strip()]
        
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

def get_num_ingredients(ingredients_text):
    if not isinstance(ingredients_text, str):
        return 0
    cleaned = ingredients_text.strip()
    if cleaned.startswith('[') and cleaned.endswith(']'):
        try:
            return len(ast.literal_eval(cleaned))
        except:
            pass
    return len([i.strip() for i in re.split(r'[,;\n]+', ingredients_text) if i.strip()])

# Load from PostgreSQL
print("[INFO] Fetching records from database...")
conn = psycopg2.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    port=DB_PORT
)
query = """
    SELECT r.cooking_time,
           r.ingredients,
           r.calories, r.protein, r.fat, r.carbohydrates,
           CASE WHEN i.rating >= 4 THEN 1 ELSE 0 END AS high_rated
    FROM recipes r JOIN reviews i ON r.recipe_id = i.recipe_id
"""
df = pd.read_sql(query, conn)
conn.close()

if df.empty:
    print("❌ Error: No database entries found.")
    exit(1)

# Downsample for fast local execution if too large
if len(df) > 150000:
    print("[INFO] Downsampling dataset to 150,000 for training...")
    df = df.sample(150000, random_state=42)

df['num_ingredients'] = df['ingredients'].apply(get_num_ingredients)

# Feature engineering
print("[INFO] Engineering portion features...")
presences = np.zeros((len(df), len(INGREDIENT_LIST)))
grams = np.zeros((len(df), len(INGREDIENT_LIST)))
for i, ing_text in enumerate(df['ingredients']):
    parsed = parse_ingredients_with_quantities(ing_text)
    for j, ing in enumerate(INGREDIENT_LIST):
        norm = ing.lower()
        if norm in parsed:
            presences[i, j] = 1
            grams[i, j] = parsed[norm]

pres_df = pd.DataFrame(presences, columns=PRESENCE_COLS)
grams_df = pd.DataFrame(grams, columns=GRAMS_COLS)

# Train split
X_num = df[NUMERICAL_COLS]
X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_num, df['high_rated'], test_size=0.2, random_state=42, stratify=df['high_rated']
)

# Fit scaler
print("Scaling features...")
scaler = StandardScaler().fit(X_train_num)
X_train_num_scaled = pd.DataFrame(scaler.transform(X_train_num), columns=NUMERICAL_COLS, index=X_train_num.index)
X_test_num_scaled = pd.DataFrame(scaler.transform(X_test_num), columns=NUMERICAL_COLS, index=X_test_num.index)

# Combine numerical + portion features
X_train_full = pd.concat([X_train_num_scaled, pres_df.loc[X_train_num.index], grams_df.loc[X_train_num.index]], axis=1)
X_test_full = pd.concat([X_test_num_scaled, pres_df.loc[X_test_num.index], grams_df.loc[X_test_num.index]], axis=1)

# Train LightGBM model
print("Training LightGBM model...")
train_data = lgb.Dataset(X_train_full, label=y_train)
test_data = lgb.Dataset(X_test_full, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42,
    'n_jobs': -1,
    'is_unbalance': True
}

gbm = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[test_data]
)

# Evaluate model
print("Evaluating model...")
y_pred_prob = gbm.predict(X_test_full)
y_pred = (y_pred_prob > 0.5).astype(int)
print("=== Model Evaluation ===")
print(classification_report(y_test, y_pred, zero_division=0))

# Save artifacts
joblib.dump(gbm, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
with open('feature_order.txt', 'w') as f:
    f.write(','.join(ALL_FEATURE_COLS))
print("Model, scaler, and feature order saved successfully.")
