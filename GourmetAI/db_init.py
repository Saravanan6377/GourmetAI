import psycopg2
import random
import os

DB_HOST = 'localhost'
DB_NAME = 'gourmetai_db'
DB_USER = 'postgres'
DB_PASS = 'welcome'
DB_PORT = '5432'

def init_db():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )
    cursor = conn.cursor()

    # Drop existing tables with CASCADE
    cursor.execute("DROP TABLE IF EXISTS favorites CASCADE")
    cursor.execute("DROP TABLE IF EXISTS reviews CASCADE")
    cursor.execute("DROP TABLE IF EXISTS recipes CASCADE")
    cursor.execute("DROP TABLE IF EXISTS users CASCADE")
    cursor.execute("DROP TABLE IF EXISTS ingredient_dictionary CASCADE")
    cursor.execute("DROP TABLE IF EXISTS system_settings CASCADE")
    conn.commit()

    # Create tables
    cursor.execute('''
    CREATE TABLE users (
        user_id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE recipes (
        recipe_id SERIAL PRIMARY KEY,
        recipe_name TEXT NOT NULL,
        description TEXT,
        ingredients TEXT NOT NULL,
        calories DOUBLE PRECISION,
        cooking_time INTEGER,
        protein DOUBLE PRECISION,
        fat DOUBLE PRECISION,
        carbohydrates DOUBLE PRECISION,
        category TEXT,
        views INTEGER DEFAULT 0,
        servings INTEGER DEFAULT 1
    )
    ''')

    cursor.execute('''
    CREATE TABLE reviews (
        review_id SERIAL PRIMARY KEY,
        recipe_id INTEGER,
        user_id INTEGER,
        rating INTEGER,
        review_text TEXT,
        sentiment_score DOUBLE PRECISION,
        sentiment_label TEXT,
        FOREIGN KEY(recipe_id) REFERENCES recipes(recipe_id) ON DELETE CASCADE,
        FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
    )
    ''')

    cursor.execute('''
    CREATE TABLE favorites (
        favorite_id SERIAL PRIMARY KEY,
        recipe_id INTEGER,
        user_id INTEGER,
        UNIQUE(recipe_id, user_id),
        FOREIGN KEY(recipe_id) REFERENCES recipes(recipe_id) ON DELETE CASCADE,
        FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ingredient_dictionary (
        ingredient_id SERIAL PRIMARY KEY,
        ingredient_name TEXT UNIQUE NOT NULL,
        calories DOUBLE PRECISION NOT NULL,
        protein DOUBLE PRECISION NOT NULL,
        fat DOUBLE PRECISION NOT NULL,
        carbohydrates DOUBLE PRECISION NOT NULL,
        food_group TEXT
    )
    ''')

    print("Tables created successfully.")

    # Seed recipes
    recipes_seed = [
        # Real/Famous recipes
        ("Chicken Biryani", "A classic Indian flavorful dish of cooked rice layered with spiced chicken, herbs, yogurt, and saffron.", "chicken, basmati rice, yogurt, onion, garlic, ginger, garam masala, chili powder, turmeric, mint, coriander, saffron, ghee", 650, 60, 38, 22, 75, "Lunch"),
        ("Mutton Biryani", "Rich and aromatic traditional mutton biryani, cooked with tender meat and premium basmati rice.", "mutton, basmati rice, yogurt, onion, garlic, ginger, garam masala, chili powder, turmeric, mint, coriander, saffron, oil", 750, 75, 42, 32, 72, "Lunch"),
        ("Veg Biryani", "A colorful dish of spiced basmati rice layered with mixed vegetables, herbs, and aromatics.", "basmati rice, carrots, peas, beans, potatoes, yogurt, onion, garlic, ginger, garam masala, mint, coriander, oil", 520, 45, 12, 14, 85, "Lunch"),
        ("Chocolate Fudge Cake", "Moist and decadent chocolate cake layered with rich fudge icing.", "flour, sugar, cocoa powder, baking powder, eggs, milk, vegetable oil, butter, vanilla extract, dark chocolate", 450, 45, 6, 22, 58, "Dessert"),
        ("Classic Pancakes", "Fluffy homemade pancakes served with maple syrup and melted butter.", "flour, sugar, baking powder, milk, butter, egg, maple syrup, salt", 350, 20, 8, 12, 52, "Breakfast"),
        ("Caesar Salad", "Crisp romaine lettuce tossed with creamy Caesar dressing, garlic croutons, and grated parmesan cheese.", "romaine lettuce, chicken breast, croutons, parmesan cheese, caesar dressing, olive oil, garlic, lemon juice", 290, 15, 22, 18, 10, "Healthy"),
        ("Spaghetti Bolognese", "Classic Italian pasta in a hearty, slow-cooked minced beef and tomato sauce.", "spaghetti pasta, minced beef, tomato paste, onions, garlic, celery, carrots, olive oil, oregano, parmesan cheese", 580, 50, 32, 20, 68, "Dinner"),
        ("Guacamole", "Creamy Mexican dip made from mashed ripe avocados, tomatoes, lime, and fresh cilantro.", "avocado, onion, tomato, jalapeno, lime juice, cilantro, salt", 180, 10, 3, 15, 12, "Snacks"),
        ("Paneer Tikka", "Indian cottage cheese cubes marinated in spiced yogurt and grilled to smoky perfection.", "paneer, yogurt, bell pepper, onion, ginger garlic paste, red chili powder, garam masala, lemon juice, oil", 320, 30, 18, 24, 8, "Snacks"),
        ("Tomato Soup", "Rich and velvety tomato soup, perfect with grilled cheese.", "tomatoes, heavy cream, vegetable stock, onion, garlic, olive oil, basil, salt, pepper", 210, 25, 4, 12, 22, "Healthy"),
        ("Fried Rice", "Quick stir-fried rice with assorted vegetables, soy sauce, and scrambled eggs.", "rice, egg, green peas, carrots, green onions, soy sauce, garlic, sesame oil", 410, 20, 10, 9, 72, "Lunch"),
        ("Grilled Salmon", "Flaky salmon fillet grilled with lemon juice, dill, and a dash of garlic butter.", "salmon fillet, lemon, olive oil, garlic, fresh dill, salt, black pepper", 340, 20, 34, 19, 2, "Healthy"),
        ("French Toast", "Slices of bread soaked in beaten eggs and milk, toasted golden brown, and dusted with cinnamon.", "bread slices, eggs, milk, butter, cinnamon powder, sugar, maple syrup", 380, 15, 12, 14, 48, "Breakfast"),
        ("Oatmeal Bowl", "Warm rolled oats cooked in almond milk, topped with banana, chia seeds, and honey.", "rolled oats, almond milk, banana, honey, chia seeds, cinnamon", 260, 15, 7, 5, 48, "Breakfast"),
        ("Fruit Salad", "A fresh mix of seasonal fruits with a drizzle of lime juice and honey.", "strawberry, blueberry, apple, banana, orange, lime juice, honey", 120, 10, 2, 1, 28, "Healthy"),
        ("Greek Salad", "Refreshing salad with cucumbers, tomatoes, red onions, olives, and feta cheese.", "cucumber, cherry tomatoes, red onion, kalamata olives, feta cheese, olive oil, oregano", 240, 15, 6, 19, 11, "Healthy"),
        ("Garlic Bread", "Crispy baguette slices toasted with a savory garlic-herb butter spread.", "baguette bread, butter, garlic cloves, parsley, parmesan cheese", 280, 15, 5, 14, 33, "Snacks"),
        ("Hummus Dip", "Smooth and creamy chickpea dip with tahini, olive oil, garlic, and lemon.", "chickpeas, tahini, olive oil, garlic, lemon juice, cumin, salt", 220, 10, 8, 13, 20, "Snacks"),
        ("Chicken Quesadilla", "Warm tortilla filled with melted cheese, grilled chicken, and bell peppers.", "tortilla, chicken breast, cheddar cheese, bell pepper, onion, taco seasoning, oil", 490, 25, 36, 21, 38, "Dinner"),
        ("Apple Pie", "Classic American pie with a spiced apple filling and a flaky, golden pastry crust.", "apples, sugar, cinnamon, flour, butter, egg, pie crust", 410, 55, 4, 18, 58, "Dessert"),
        ("Mango Smoothie", "Creamy blend of ripe mangoes, Greek yogurt, and honey.", "mango, greek yogurt, milk, honey, ice cubes", 230, 8, 6, 2, 48, "Breakfast"),
        ("Beef Steak", "Pan-seared tenderloin steak basted with butter, garlic, and fresh rosemary.", "beef tenderloin, butter, garlic, rosemary, olive oil, salt, black pepper", 620, 30, 48, 42, 1, "Dinner"),
        ("Chicken Alfredo", "Creamy pasta dish with fettuccine, tender chicken breast, and rich parmesan sauce.", "fettuccine pasta, chicken breast, heavy cream, butter, garlic, parmesan cheese, parsley", 720, 35, 42, 34, 62, "Dinner"),
        ("Vanilla Cupcakes", "Sweet and fluffy cupcakes topped with a rich vanilla buttercream frosting.", "flour, sugar, butter, eggs, milk, baking powder, vanilla extract", 310, 30, 4, 14, 42, "Dessert"),
        ("Minestrone Soup", "Classic Italian vegetable soup with beans, tomatoes, and pasta.", "carrots, celery, onion, zucchini, kidney beans, diced tomatoes, vegetable broth, pasta, olive oil", 220, 40, 8, 4, 38, "Healthy"),
        ("Tacos", "Crispy taco shells loaded with seasoned ground beef, lettuce, cheese, and salsa.", "taco shells, ground beef, lettuce, cheddar cheese, tomato salsa, sour cream, spices", 430, 20, 26, 23, 28, "Dinner"),
        ("Club Sandwich", "Double-decker sandwich with toasted bread, turkey bacon, chicken breast, lettuce, and mayo.", "sliced bread, chicken breast, turkey bacon, lettuce, tomato, mayonnaise, cheese", 510, 15, 30, 25, 41, "Lunch"),
        ("Brownies", "Rich, chewy chocolate brownies with a crackly top.", "butter, sugar, cocoa powder, eggs, flour, chocolate chips, vanilla", 380, 35, 5, 20, 45, "Dessert")
    ]

    # Let's generate programmatically up to 210 recipes to get a robust dataset
    categories = ["Breakfast", "Lunch", "Dinner", "Dessert", "Snacks", "Healthy"]
    
    # Vocabulary for generation
    prep_verbs = ["Grilled", "Baked", "Roasted", "Steamed", "Pan-seared", "Spicy", "Classic", "Crispy", "Garlic", "Lemon", "Herb-crusted", "Sweet", "Smoky"]
    bases = ["Chicken", "Beef", "Pork", "Tofu", "Paneer", "Salmon", "Shrimp", "Rice", "Pasta", "Quinoa", "Lentils", "Eggplant", "Potato", "Avocado", "Spinach"]
    styles = ["with Garlic", "in Tomato Sauce", "Bowl", "Salad", "Stir-fry", "Wrap", "Skewer", "Curry", "Platter", "Soup", "Casserole"]

    ingredients_pool = {
        "protein": ["chicken breast", "beef", "tofu", "paneer", "salmon fillet", "shrimp", "eggs", "chickpeas", "lentils", "greek yogurt"],
        "veg": ["spinach", "bell pepper", "onion", "garlic", "tomato", "cucumber", "broccoli", "zucchini", "carrots", "mushrooms", "avocado", "lettuce"],
        "grain_carb": ["basmati rice", "pasta", "quinoa", "potatoes", "sweet potato", "flour", "bread", "tortilla", "oats"],
        "fats_dairy": ["olive oil", "butter", "heavy cream", "cheddar cheese", "parmesan cheese", "feta cheese", "coconut milk", "ghee"],
        "flavor_spices": ["soy sauce", "ginger", "garam masala", "chili powder", "oregano", "basil", "cinnamon", "vanilla extract", "maple syrup", "honey", "lemon juice", "cilantro", "parsley"]
    }

    random.seed(42)  # For reproducible generation

    for i in range(185):
        # Pick category
        cat = random.choice(categories)
        
        # Build Name
        verb = random.choice(prep_verbs)
        base = random.choice(bases)
        style = random.choice(styles)
        recipe_name = f"{verb} {base} {style}"
        
        # Ensure name uniqueness
        existing_names = [r[0] for r in recipes_seed]
        while recipe_name in existing_names:
            verb = random.choice(prep_verbs)
            base = random.choice(bases)
            style = random.choice(styles)
            recipe_name = f"{verb} {base} {style}"
            
        description = f"A delicious and easy-to-prepare {recipe_name.lower()} that makes for a perfect {cat.lower()} dish."

        # Compile ingredients based on category and title
        ings = set()
        
        # Add primary protein based on name
        if base.lower() in ["chicken", "beef", "tofu", "paneer", "salmon", "shrimp"]:
            if base.lower() == "chicken": ings.add("chicken breast")
            elif base.lower() == "beef": ings.add("beef")
            elif base.lower() == "tofu": ings.add("tofu")
            elif base.lower() == "paneer": ings.add("paneer")
            elif base.lower() == "salmon": ings.add("salmon fillet")
            elif base.lower() == "shrimp": ings.add("shrimp")
        else:
            # Pick a protein
            ings.add(random.choice(ingredients_pool["protein"]))

        # Add vegetables
        num_vegs = random.randint(2, 4)
        for _ in range(num_vegs):
            ings.add(random.choice(ingredients_pool["veg"]))
            
        # Add grain/carb
        if "rice" in recipe_name.lower() or base == "Rice":
            ings.add("basmati rice")
        elif "pasta" in recipe_name.lower() or base == "Pasta":
            ings.add("pasta")
        elif cat == "Dessert" or "sweet" in recipe_name.lower():
            ings.add("flour")
            ings.add("sugar")
        else:
            ings.add(random.choice(ingredients_pool["grain_carb"]))

        # Add fats/dairy
        ings.add(random.choice(ingredients_pool["fats_dairy"]))
        
        # Add spices/flavorings
        num_flavors = random.randint(2, 4)
        for _ in range(num_flavors):
            ings.add(random.choice(ingredients_pool["flavor_spices"]))
            
        ingredients_str = ", ".join(list(ings))

        # Cooking time
        cooking_time = random.choice([15, 20, 25, 30, 40, 45, 50, 60, 75, 90])

        # Nutrition ranges depending on category
        if cat == "Dessert":
            calories = random.randint(300, 600)
            protein = random.randint(3, 8)
            fat = random.randint(10, 30)
            carbs = random.randint(40, 90)
        elif cat == "Healthy":
            calories = random.randint(150, 350)
            protein = random.randint(15, 35)
            fat = random.randint(4, 12)
            carbs = random.randint(10, 40)
        elif cat == "Breakfast":
            calories = random.randint(200, 500)
            protein = random.randint(6, 20)
            fat = random.randint(6, 22)
            carbs = random.randint(30, 70)
        elif cat in ["Lunch", "Dinner"]:
            calories = random.randint(400, 850)
            protein = random.randint(25, 55)
            fat = random.randint(12, 35)
            carbs = random.randint(30, 80)
        else:  # Snacks
            calories = random.randint(100, 300)
            protein = random.randint(4, 12)
            fat = random.randint(4, 18)
            carbs = random.randint(15, 45)

        recipes_seed.append((recipe_name, description, ingredients_str, float(calories), int(cooking_time), float(protein), float(fat), float(carbs), cat))

    # Insert recipes
    for r in recipes_seed:
        if len(r) == 9:
            cursor.execute('''
            INSERT INTO recipes (recipe_name, description, ingredients, calories, cooking_time, protein, fat, carbohydrates, category)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', r)
        else:
            # For the first manual ones which had 8 values, let's map them
            name, desc, ings, cals, c_time, prot, fat, carbs = r[:8]
            cat = r[8] if len(r) > 8 else "Lunch"
            cursor.execute('''
            INSERT INTO recipes (recipe_name, description, ingredients, calories, cooking_time, protein, fat, carbohydrates, category)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (name, desc, ings, cals, c_time, prot, fat, carbs, cat))

    print(f"Seeded {len(recipes_seed)} recipes.")

    # Create dummy users
    from werkzeug.security import generate_password_hash
    users = [
        ("Alice Cooper", "alice@example.com", generate_password_hash("password123")),
        ("Bob Smith", "bob@example.com", generate_password_hash("password123")),
        ("Charlie Brown", "charlie@example.com", generate_password_hash("password123")),
        ("Admin User", "admin@example.com", generate_password_hash("admin123"))
    ]
    for name, email, pw in users:
        cursor.execute('INSERT INTO users (name, email, password) VALUES (%s, %s, %s)', (name, email, pw))

    print("Seeded users.")

    # Seed reviews with rating and review_text for VADER sentiment analysis
    reviews_seed = [
        ("Excellent recipe! The instructions were easy to follow and the result was delicious. Will definitely make again.", 5, "Positive"),
        ("I loved it! My whole family enjoyed this dish. A new regular in our rotation.", 5, "Positive"),
        ("So rich and flavorful, perfect amount of spices. Incredible taste!", 5, "Positive"),
        ("Pretty good, though I added a bit more salt. Overall satisfying.", 4, "Positive"),
        ("Decent recipe, but took longer to cook than specified. Good flavor.", 4, "Positive"),
        ("It was average. Nothing special, but decent for a quick dinner.", 3, "Neutral"),
        ("Okay recipe. The texture was slightly dry but the taste was alright.", 3, "Neutral"),
        ("Not bad, but I think it needs more spice. Quite plain.", 3, "Neutral"),
        ("Very disappointing. It was way too salty and greasy. I won't make this again.", 1, "Negative"),
        ("Terrible recipe. The cooking instructions were confusing and it burnt completely.", 1, "Negative"),
        ("It tasted awful. The proportions of flour to liquid were completely off.", 2, "Negative"),
        ("Too sour and flavorless. A waste of ingredients.", 2, "Negative")
    ]

    # Seed reviews randomly across different recipes
    random.seed(99)
    for recipe_id in range(1, len(recipes_seed) + 1):
        # Retrieve recipe details
        cursor.execute("SELECT recipe_name, ingredients, calories, cooking_time, protein, fat, carbohydrates, category FROM recipes WHERE recipe_id=%s", (recipe_id,))
        row = cursor.fetchone()
        name, ingredients, calories, cooking_time, protein, fat, carbs, category = row
        num_ingredients = len([i.strip() for i in ingredients.split(",")])

        # Define a rating formula for training label
        score = 30.0
        # Protein is positive
        score += min(25.0, protein * 0.5)
        # Fat is negative if excessive
        if fat > 15:
            score -= min(25.0, (fat - 15) * 1.2)
        else:
            score += 5.0
        # Cooking time: we prefer quick cooking (under 30 min: +15, 30-60 min: +8, over 60 min: -25)
        if cooking_time < 30:
            score += 15.0
        elif cooking_time <= 60:
            score += 8.0
        else:
            score -= 25.0
        # Carbs: dessert should be high carbs (+15 if carbs > 40), others should be moderate
        if category == "Dessert":
            if carbs > 40:
                score += 15.0
            else:
                score -= 5.0
        else:
            if carbs < 40:
                score += 10.0
            elif carbs > 70:
                score -= 25.0
        # Ingredients: 5 to 10 is sweet spot (+15), otherwise less (+5)
        if 5 <= num_ingredients <= 10:
            score += 15.0
        else:
            score += 5.0

        # Convert score to 1-5 rating
        if score >= 82:
            predicted_mean_rating = 5
        elif score >= 68:
            predicted_mean_rating = 4
        elif score >= 54:
            predicted_mean_rating = 3
        elif score >= 40:
            predicted_mean_rating = 2
        else:
            predicted_mean_rating = 1

        # Create 1 to 3 reviews for this recipe
        num_reviews = random.randint(1, 3)
        for _ in range(num_reviews):
            user_id = random.randint(1, 4)
            if predicted_mean_rating >= 4:
                text, rating, label = random.choice([r for r in reviews_seed if r[1] >= 4])
            elif predicted_mean_rating == 3:
                text, rating, label = random.choice([r for r in reviews_seed if r[1] == 3])
            else:
                text, rating, label = random.choice([r for r in reviews_seed if r[1] <= 2])
            
            actual_rating = predicted_mean_rating
            score_vader = 0.8 if label == "Positive" else (-0.8 if label == "Negative" else 0.0)
            
            cursor.execute('''
            INSERT INTO reviews (recipe_id, user_id, rating, review_text, sentiment_score, sentiment_label)
            VALUES (%s, %s, %s, %s, %s, %s)
            ''', (recipe_id, user_id, actual_rating, text, score_vader, label))

    print("Seeded reviews.")

    # Seed favorites
    for i in range(1, 15):
        recipe_id = random.randint(1, len(recipes_seed))
        user_id = random.randint(1, 3)
        cursor.execute('INSERT INTO favorites (recipe_id, user_id) VALUES (%s, %s) ON CONFLICT DO NOTHING', (recipe_id, user_id))

    print("Seeded favorites.")

    # Seed ingredient dictionary
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
    print("Seeded ingredient dictionary.")

    print("Creating database indexes to optimize query latency...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_recipe_id ON reviews(recipe_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_user_id ON reviews(user_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_recipes_category ON recipes(category);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_recipes_name ON recipes(recipe_name);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_favorites_user_id ON favorites(user_id);")

    conn.commit()
    conn.close()
    print("Database seeding completed successfully.")

if __name__ == '__main__':
    init_db()
