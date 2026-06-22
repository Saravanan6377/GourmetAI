import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import re
import ast
from sklearn.preprocessing import StandardScaler
import psycopg2

# ── Database connection ──
DB_HOST = "localhost"
DB_NAME = "gourmetai_db"
DB_USER = "postgres"
DB_PASS = "welcome"
DB_PORT = "5432"

conn = psycopg2.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    port=DB_PORT
)
conn.set_session(autocommit=True)

# ── Vocabulary (must match ingredient_dictionary exactly) ──
INGREDIENT_LIST = [
    "idli batter", "dosha batter", "rice", "wheat flour (atta)", "chicken",
    "mutton", "egg", "paneer", "milk", "heavy cream", "butter",
    "ghee", "olive oil", "tomato", "onion", "potato", "spinach",
    "garlic", "ginger", "lentils (dal)", "salmon"
]
INGREDIENT_SET = {ing.lower() for ing in INGREDIENT_LIST}

# ── Feature definitions ──
NUMERICAL_COLS = ['cooking_time', 'num_ingredients', 'calories', 'protein', 'fat', 'carbohydrates']
PRESENCE_COLS = [f"has_{ing.replace(' ', '_').replace('(', '').replace(')', '')}" for ing in INGREDIENT_LIST]
GRAMS_COLS   = [f"grams_{ing.replace(' ', '_').replace('(', '').replace(')', '')}" for ing in INGREDIENT_LIST]
ALL_FEATURE_COLS = NUMERICAL_COLS + PRESENCE_COLS + GRAMS_COLS

# ── Helper: parse ingredients and quantities ──
def parse_ingredients_with_quantities(ingredients_text):
    result = {}
    if not isinstance(ingredients_text, str):
        return result
    
    # Clean string if it looks like a python list representation (e.g. ['chicken', 'butter'])
    cleaned_text = ingredients_text.strip()
    if cleaned_text.startswith('[') and cleaned_text.endswith(']'):
        try:
            items = ast.literal_eval(cleaned_text)
        except:
            items = [x.strip() for x in re.split(r'[,;\n]+', ingredients_text) if x.strip()]
    else:
        items = [x.strip() for x in re.split(r'[,;\n]+', ingredients_text) if x.strip()]
        
    for item in items:
        item = item.strip().lower()
        if not item:
            continue
            
        qty = None
        # Start quantity: e.g. "200g chicken", "200 g chicken", "1.5 kg beef"
        match_start = re.match(r'^([\d.]+)\s*(g|grams?|ml|oz|kg)?\s*(.+)$', item, re.IGNORECASE)
        # End quantity: e.g. "chicken 200g", "chicken 200"
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
            qty = 100.0  # default to 100g standard serving
            
        ing_key = ing_part.strip().lower().rstrip('s')
        
        # Match against vocabulary (longest matches first)
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

# ── Step 1: Fit scaler on first 100k rows (numerical only) ──
print("Fitting standard scaler on numerical distribution...")
query_sample = """
    SELECT r.cooking_time,
           r.ingredients,
           r.calories, r.protein, r.fat, r.carbohydrates
    FROM recipes r JOIN reviews i ON r.recipe_id = i.recipe_id
    LIMIT 100000
"""
df_sample = pd.read_sql(query_sample, conn)
if df_sample.empty:
    print("❌ Error: No training data found in the reviews/recipes tables. Please seed the database first.")
    conn.close()
    exit(1)

df_sample['num_ingredients'] = df_sample['ingredients'].apply(get_num_ingredients)
scaler = StandardScaler().fit(df_sample[NUMERICAL_COLS])
print("[OK] Scaler fitted successfully.")

# ── LightGBM parameters ──
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

# ── Step 2: Incremental training ──
gbm = None
batch_size = 50000
offset = 0

print("Starting incremental training with ingredient portion features...")
while True:
    query = f"""
        SELECT r.cooking_time,
               r.ingredients,
               r.calories, r.protein, r.fat, r.carbohydrates,
               CASE WHEN i.rating >= 4 THEN 1 ELSE 0 END AS high_rated
        FROM recipes r JOIN reviews i ON r.recipe_id = i.recipe_id
        ORDER BY r.recipe_id, i.review_id
        LIMIT {batch_size} OFFSET {offset}
    """
    df = pd.read_sql(query, conn)
    if df.empty:
        break

    df['num_ingredients'] = df['ingredients'].apply(get_num_ingredients)
    
    # Scale numerical features
    X_num = scaler.transform(df[NUMERICAL_COLS])
    X_num_df = pd.DataFrame(X_num, columns=NUMERICAL_COLS)

    # Compile presence and grams features
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

    # Full feature matrix matching model expectations
    X_full = pd.concat([X_num_df, pres_df, grams_df], axis=1)
    y = df['high_rated']

    lgb_train = lgb.Dataset(X_full, label=y)
    if gbm is None:
        gbm = lgb.train(params, lgb_train, num_boost_round=100)
    else:
        gbm = lgb.train(params, lgb_train, num_boost_round=50, init_model=gbm)

    offset += len(df)
    print(f"Processed {offset} rows...")
    if len(df) < batch_size:
        break

conn.close()

# ── Save model artifacts ──
joblib.dump(gbm, 'rating_predictor.pkl')
joblib.dump(scaler, 'scaler.pkl')
with open('feature_order.txt', 'w') as f:
    f.write(','.join(ALL_FEATURE_COLS))
print("[OK] Training complete. Model, scaler, and feature order saved.")
