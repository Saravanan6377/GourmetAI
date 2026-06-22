import psycopg2

def create_indexes():
    conn = psycopg2.connect(
        host="localhost",
        user="postgres",
        password="welcome",
        port="5432",
        database="gourmetai_db"
    )
    cursor = conn.cursor()
    
    print("[INFO] Creating database indexes to optimize latency...")
    
    # 1. Index reviews on recipe_id (critical for recipe details and average rating aggregation)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_recipe_id ON reviews(recipe_id);")
    
    # 2. Index reviews on user_id (for user activity checks)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_user_id ON reviews(user_id);")
    
    # 3. Index recipes on category (for category filtering on the homepage)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_recipes_category ON recipes(category);")
    
    # 4. Index recipes on recipe_name (for search auto-complete and text matching)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_recipes_name ON recipes(recipe_name);")
    
    # 5. Index favorites on user_id (for bookmarked lists retrieval)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_favorites_user_id ON favorites(user_id);")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print("[OK] Database indexes created successfully.")

if __name__ == '__main__':
    create_indexes()
