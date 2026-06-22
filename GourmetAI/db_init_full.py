import pandas as pd
import psycopg2
import ast
import os
import time

RECIPES_CSV = r"c:\antigravity\mlfr\RAW_recipes.csv"
INTERACTIONS_CSV = r"c:\antigravity\mlfr\RAW_interactions.csv"

DB_HOST = "localhost"
DB_NAME = "gourmetai_db"
DB_USER = "postgres"
DB_PASS = "welcome"
DB_PORT = "5432"

def init_full_production_database():
    if not os.path.exists(RECIPES_CSV) or not os.path.exists(INTERACTIONS_CSV):
        print("[ERROR] Missing raw Kaggle files in c:\\antigravity\\mlfr\\")
        print("Please ensure RAW_recipes.csv and RAW_interactions.csv are placed correctly.")
        return

    start_time = time.time()
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )
    cursor = conn.cursor()

    # Re-initialize the core system tables
    print("[INFO] Purging older structural matrices...")
    cursor.execute("DROP TABLE IF EXISTS favorites CASCADE")
    cursor.execute("DROP TABLE IF EXISTS reviews CASCADE")
    cursor.execute("DROP TABLE IF EXISTS recipes CASCADE")
    cursor.execute("DROP TABLE IF EXISTS users CASCADE")
    cursor.execute("DROP TABLE IF EXISTS ingredient_dictionary CASCADE")
    cursor.execute("DROP TABLE IF EXISTS system_settings CASCADE")
    conn.commit()
    
    # Create recipes schema aligned with app.py
    cursor.execute('''
    CREATE TABLE recipes (
        recipe_id INTEGER PRIMARY KEY,
        recipe_name TEXT,
        description TEXT,
        ingredients TEXT,
        cooking_time INTEGER,
        calories DOUBLE PRECISION,
        protein DOUBLE PRECISION,
        fat DOUBLE PRECISION,
        carbohydrates DOUBLE PRECISION,
        category TEXT,
        views INTEGER DEFAULT 0,
        servings INTEGER DEFAULT 1
    )''')

    # Create reviews schema aligned with app.py
    cursor.execute('''
    CREATE TABLE reviews (
        review_id SERIAL PRIMARY KEY,
        recipe_id INTEGER,
        user_id INTEGER,
        rating INTEGER,
        review_text TEXT,
        sentiment_score DOUBLE PRECISION,
        sentiment_label TEXT
    )''')

    # Ensure users and favorites tables exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS favorites (
        favorite_id SERIAL PRIMARY KEY,
        recipe_id INTEGER,
        user_id INTEGER,
        UNIQUE(recipe_id, user_id),
        FOREIGN KEY(recipe_id) REFERENCES recipes(recipe_id) ON DELETE CASCADE,
        FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ingredient_dictionary (
        ingredient_id SERIAL PRIMARY KEY,
        ingredient_name TEXT UNIQUE NOT NULL,
        calories DOUBLE PRECISION NOT NULL,
        protein DOUBLE PRECISION NOT NULL,
        fat DOUBLE PRECISION NOT NULL,
        carbohydrates DOUBLE PRECISION NOT NULL,
        food_group TEXT
    )''')

    dictionary_seed = [
        ("idli batter", 120.0, 3.0, 0.3, 25.5, "Grains / Staple"),
        ("dosha batter", 124.0, 3.2, 0.4, 26.8, "Grains / Staple"),
        ("rice", 130.0, 2.7, 0.3, 28.0, "Grains / Staple"),
        ("wheat flour (atta)", 364.0, 10.3, 1.0, 76.3, "Grains"),
        ("chicken", 165.0, 31.0, 3.6, 0.0, "Poultry"),
        ("mutton", 294.0, 25.0, 21.0, 0.0, "Meat"),
        ("egg", 155.0, 13.0, 11.0, 1.1, "Poultry / Dairy"),
        ("paneer", 265.0, 18.3, 20.8, 1.2, "Dairy"),
        ("milk", 42.0, 3.4, 1.0, 5.0, "Dairy"),
        ("heavy cream", 340.0, 2.8, 36.1, 2.7, "Dairy"),
        ("butter", 717.0, 0.9, 81.0, 0.1, "Fats & Oils"),
        ("ghee", 884.0, 0.0, 99.5, 0.0, "Fats & Oils"),
        ("olive oil", 884.0, 0.0, 100.0, 0.0, "Fats & Oils"),
        ("tomato", 18.0, 0.9, 0.2, 3.9, "Vegetables"),
        ("onion", 40.0, 1.1, 0.1, 9.3, "Vegetables"),
        ("potato", 77.0, 2.0, 0.1, 17.0, "Starch / Vegetables"),
        ("spinach", 23.0, 2.9, 0.4, 3.6, "Green Leaves"),
        ("garlic", 149.0, 6.4, 0.5, 33.1, "Spices / Aromatics"),
        ("ginger", 80.0, 1.8, 0.8, 17.8, "Spices / Aromatics"),
        ("lentils (dal)", 116.0, 9.0, 0.4, 20.0, "Legumes"),
        ("salmon", 208.0, 20.0, 13.0, 0.0, "Seafood")
    ]
    cursor.executemany('''
    INSERT INTO ingredient_dictionary (ingredient_name, calories, protein, fat, carbohydrates, food_group)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (ingredient_name) DO NOTHING
    ''', dictionary_seed)

    conn.commit()

    # Seed default users
    print("[INFO] Seeding system users...")
    from werkzeug.security import generate_password_hash
    users = [
        ("Alice Cooper", "alice@example.com", generate_password_hash("password123")),
        ("Bob Smith", "bob@example.com", generate_password_hash("password123")),
        ("Charlie Brown", "charlie@example.com", generate_password_hash("password123")),
        ("Admin User", "admin@example.com", generate_password_hash("admin123"))
    ]
    for name, email, pw in users:
        cursor.execute('''
            INSERT INTO users (name, email, password)
            VALUES (%s, %s, %s)
            ON CONFLICT (email) DO NOTHING
        ''', (name, email, pw))
    conn.commit()

    # 1. Stream & Parse Recipes
    print("[INFO] Processing RAW_recipes.csv via memory chunks...")
    recipe_chunks = pd.read_csv(RECIPES_CSV, chunksize=50000)
    
    for chunk in recipe_chunks:
        recipe_records = []
        for _, row in chunk.iterrows():
            try:
                # Extract numerical fields from the nutrition string array
                nutr = ast.literal_eval(row['nutrition'])
                calories = float(nutr[0])
                fat = float(nutr[1])
                protein = float(nutr[4])
                carbohydrates = float(nutr[6])
            except:
                calories = fat = protein = carbohydrates = 0.0

            desc = str(row['description']) if pd.notna(row['description']) else ""
            recipe_records.append((
                int(row['id']),
                str(row['name']).title(),
                desc,
                str(row['ingredients']),
                int(row['minutes']),
                calories,
                protein,
                fat,
                carbohydrates,
                "Dinner" # Baseline system category assignment
            ))
        
        cursor.executemany('''
            INSERT INTO recipes (recipe_id, recipe_name, description, ingredients, cooking_time, calories, protein, fat, carbohydrates, category)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (recipe_id) DO UPDATE SET
                recipe_name = EXCLUDED.recipe_name,
                description = EXCLUDED.description,
                ingredients = EXCLUDED.ingredients,
                cooking_time = EXCLUDED.cooking_time,
                calories = EXCLUDED.calories,
                protein = EXCLUDED.protein,
                fat = EXCLUDED.fat,
                carbohydrates = EXCLUDED.carbohydrates,
                category = EXCLUDED.category
        ''', recipe_records)
    
    conn.commit()
    print(f"[OK] Success: Loaded recipes into database.")

    # 2. Stream & Parse Interactions (Reviews)
    print("[INFO] Processing RAW_interactions.csv interactions matrix...")
    interaction_chunks = pd.read_csv(INTERACTIONS_CSV, chunksize=100000)
    
    for chunk in interaction_chunks:
        review_records = []
        filtered_chunk = chunk[(chunk['rating'] > 0) & (chunk['review'].notna())]
        
        for _, row in filtered_chunk.iterrows():
            rating = int(row['rating'])
            if rating >= 4:
                sentiment_label = 'Positive'
                score = 0.8
            elif rating == 3:
                sentiment_label = 'Neutral'
                score = 0.0
            else:
                sentiment_label = 'Negative'
                score = -0.8

            review_records.append((
                int(row['recipe_id']),
                int(row['user_id']),
                rating,
                str(row['review']),
                score,
                sentiment_label
            ))
            
        cursor.executemany('''
            INSERT INTO reviews (recipe_id, user_id, rating, review_text, sentiment_score, sentiment_label)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', review_records)
        
    print("[INFO] Creating database indexes to optimize latency...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_recipe_id ON reviews(recipe_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_user_id ON reviews(user_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_recipes_category ON recipes(category);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_recipes_name ON recipes(recipe_name);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_favorites_user_id ON favorites(user_id);")
    
    conn.commit()
    conn.close()
    
    print(f"[OK] Complete! Full production database seeded in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    init_full_production_database()
