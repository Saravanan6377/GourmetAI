import unittest
import os
import psycopg2
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
from app import app, predict_recipe_rating, get_recommendations
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class TestGourmetAI(unittest.TestCase):

    def setUp(self):
        # Configure app for testing
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        self.model_path = 'rating_predictor.pkl'
        self.db_params = {
            'host': 'localhost',
            'database': 'gourmetai_db',
            'user': 'postgres',
            'password': 'welcome',
            'port': '5432'
        }

    def test_database_and_model_exists(self):
        """1. Verify database connection and model files exist"""
        self.assertTrue(os.path.exists(self.model_path), "Model rating_predictor.pkl does not exist")
        db_ok = False
        try:
            conn = psycopg2.connect(**self.db_params)
            conn.close()
            db_ok = True
        except Exception as e:
            print("PostgreSQL connection error in tests:", e)
        self.assertTrue(db_ok, "Could not connect to PostgreSQL database gourmetai_db")

    def test_authentication(self):
        """2. Verify user registration and login flows"""
        # Register a test user
        register_data = {
            'name': 'Test Tester',
            'email': 'tester@example.com',
            'password': 'testpassword'
        }
        response = self.client.post('/register', data=register_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        
        # Verify user exists in database
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email='tester@example.com'")
        user = cursor.fetchone()
        conn.close()
        self.assertIsNotNone(user, "User was not saved to database")
        
        # Login with test user
        login_data = {
            'email': 'tester@example.com',
            'password': 'testpassword'
        }
        response = self.client.post('/login', data=login_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        
        # Check if welcome message appears or session is modified
        with self.client.session_transaction() as sess:
            self.assertEqual(sess['user_email'], 'tester@example.com')
            self.assertEqual(sess['user_name'], 'Test Tester')

    def test_ml_rating_prediction(self):
        """3. Verify LightGBM rating prediction values"""
        # Test values corresponding to a healthy/low prep recipe (highly rated)
        rating_healthy = predict_recipe_rating(
            cooking_time=20,
            ingredients="lettuce, salmon, olive oil, lemon, tomato, onion", # 6 ingredients
            calories=250,
            protein=35,
            fat=10,
            carbohydrates=12
        )
        self.assertIn(rating_healthy, [1, 2, 3, 4, 5], "Rating prediction returned invalid rating")
        print(f"LightGBM rating prediction for healthy salmon recipe: {rating_healthy} Stars")
        
        # Test values corresponding to an excessive fat, long cooking recipe (lower rated)
        rating_unhealthy = predict_recipe_rating(
            cooking_time=95,
            ingredients="beef, lard, butter, cheese, cream, starch, salt, MSG, syrup, caramel, chocolate", # 11 ingredients
            calories=890,
            protein=20,
            fat=52,
            carbohydrates=95
        )
        self.assertLessEqual(rating_unhealthy, rating_healthy, "Unhealthy recipe rated higher than healthy recipe")
        print(f"LightGBM rating prediction for greasy chocolate beef recipe: {rating_unhealthy} Stars")

    def test_tfidf_recommendation_system(self):
        """4. Verify TF-IDF content recommendation yields 5 recommendations"""
        # Dynamically fetch a valid recipe ID with 'biryani' or 'rice' in the name
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()
        cursor.execute("SELECT recipe_id, recipe_name FROM recipes WHERE recipe_name ILIKE %s OR recipe_name ILIKE %s LIMIT 1", ('%biryani%', '%rice%'))
        row = cursor.fetchone()
        
        # Fallback to any recipe if none found (defensive check)
        if not row:
            cursor.execute("SELECT recipe_id, recipe_name FROM recipes LIMIT 1")
            row = cursor.fetchone()
            
        conn.close()
        
        self.assertIsNotNone(row, "No recipes found in database to run recommendations test")
        recipe_id, recipe_name = row
        
        recs = get_recommendations(recipe_id=recipe_id, limit=5)
        self.assertEqual(len(recs), 5, "Recommendation system did not return exactly 5 matches")
        
        # Check that recommendations are related
        rec_names = [r['recipe_name'] for r in recs]
        print(f"TF-IDF Recommendations for '{recipe_name}':", rec_names)
        
        # Check for matching words (biryani, rice, or similar words from the target name)
        target_words = [w.lower() for w in recipe_name.split() if len(w) > 3]
        self.assertTrue(any(any(word in name.lower() for word in target_words) for name in rec_names), 
                        f"Recommendation list contains no related recipes for {recipe_name}")

    def test_sentiment_vader_nlp(self):
        """5. Verify VADER Sentiment Analysis labels positive and negative reviews correctly"""
        sia = SentimentIntensityAnalyzer()
        
        # Test positive review text
        pos_text = "This recipe is absolutely amazing and super easy! I loved every single bite."
        pos_score = sia.polarity_scores(pos_text)['compound']
        self.assertTrue(pos_score >= 0.05, "VADER compound score failed for positive review")
        
        # Test negative review text
        neg_text = "Tasted horrible, took too long to cook, way too salty and greasy. Disgusting."
        neg_score = sia.polarity_scores(neg_text)['compound']
        self.assertTrue(neg_score <= -0.05, "VADER compound score failed for negative review")
        
        print(f"VADER Sentiment compound score: Positive='{pos_score:.3f}', Negative='{neg_score:.3f}'")

    def test_analytics_data(self):
        """6. Verify analytics endpoint compiles data fields"""
        response = self.client.get('/analytics')
        self.assertEqual(response.status_code, 200)
        # Verify key chart parameters are in HTML template response
        html = response.data.decode('utf-8')
        self.assertIn("viewChart", html)
        self.assertIn("likeChart", html)
        self.assertIn("sentimentChart", html)
        self.assertIn("ratingChart", html)

    def test_chatbot_endpoint(self):
        """7. Verify chatbot responses for categories, names, and fallbacks"""
        # Test category query
        response = self.client.post('/chatbot', json={'message': 'give me a healthy breakfast'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("breakfast", data['response'].lower())
        print("Chatbot Category Response: ", data['response'][:100] + "...")
        
        # Test recipe query
        response = self.client.post('/chatbot', json={'message': 'how to make chicken biryani'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("biryani", data['response'].lower())
        print("Chatbot Recipe Response: ", data['response'][:100] + "...")
        
        # Test fallback
        response = self.client.post('/chatbot', json={'message': 'xyz123abc'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("couldn't find matching recipes", data['response'].lower())
        print("Chatbot Fallback Response: ", data['response'][:100] + "...")

    def test_settings_and_theme_change(self):
        """8. Verify settings loading and preset theme change flow for Admin"""
        # Login with seeded admin credentials
        login_data = {
            'email': 'admin@example.com',
            'password': 'admin123'
        }
        self.client.post('/login', data=login_data)
        
        # Get settings page
        response = self.client.get('/settings')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("Admin Theme Panel", html)
        
        # Change preset to emerald
        theme_post = {
            'preset': 'emerald',
            'primary_color': '#0d9488',
            'accent_color': '#84cc16'
        }
        response = self.client.post('/settings', data=theme_post, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        
        # Read from PostgreSQL
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM system_settings WHERE key='theme_preset'")
        preset = cursor.fetchone()[0]
        cursor.execute("SELECT value FROM system_settings WHERE key='primary_color'")
        primary = cursor.fetchone()[0]
        conn.close()
        
        self.assertEqual(preset, 'emerald')
        self.assertEqual(primary, '#0d9488')
        print("Theme Settings verification succeeded! Preset set to emerald, primary set to #0d9488")

        # Revert theme settings back to slate preset so the site background color is not permanently changed
        theme_reset = {
            'preset': 'slate',
            'primary_color': '#3b82f6',
            'accent_color': '#6b7280'
        }
        self.client.post('/settings', data=theme_reset, follow_redirects=True)


    def test_ingredient_nutrition_lookup(self):
        """9. Verify Smart Bridge: Automated Nutrition Lookup Engine matches and scales macros"""
        # Test input: 200g chicken, 50g tomato, 10g olive oil
        post_data = {
            'ingredients': '200g chicken, 50g tomato, 10g olive oil'
        }
        response = self.client.post('/api/analyze-ingredients', json=post_data)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        
        # Verify calculated values match our math (Chicken + Tomato + Olive Oil)
        self.assertAlmostEqual(data['calories'], 427.4, places=1)
        self.assertAlmostEqual(data['protein'], 62.5, places=1)
        self.assertAlmostEqual(data['fat'], 17.3, places=1)
        self.assertAlmostEqual(data['carbohydrates'], 1.9, places=1)
        print("Smart Bridge Nutrition Auto-Calculation Test Succeeded! Values matched expected USDA scaling.")

    def test_sanity_gate_capping(self):
        """10. Verify Sanity Gate: Outlier recipes with extreme values are capped and predict well"""
        # A recipe with South Indian Coconut Rice ingredients but extreme macros due to bad scaling.
        # Original: 1130 Cal, 178g Carbs. Capped: 800 Cal, 100g Carbs. Should predict 5 stars.
        rating = predict_recipe_rating(
            cooking_time=50,
            ingredients="['basmati rice', 'coconut oil', 'dried red chilies', 'channa dal', 'urad dal', 'brown mustard seeds', 'curry leaves', 'salt', 'asafoetida powder', 'coconut', 'fresh cilantro']",
            calories=1130.0,
            protein=22.7,
            fat=20.3,
            carbohydrates=178.0
        )
        self.assertEqual(rating, 5, "Capped outlier recipe failed to predict high quality rating")
        print(f"Capped outlier recipe rating prediction: {rating} Stars (Success!)")

    def test_cuisine_region_filtering(self):
        """11. Verify index filtering using country mapping and state parameters"""
        # Filter by cuisine 'India'
        response = self.client.get('/?cuisine=India')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("recipes", html.lower())
        
        # Filter by region 'Tamil Nadu'
        response = self.client.get('/?state_food=Tamil+Nadu')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("recipes", html.lower())
        print("Cuisine and region filtering tests completed successfully.")

    def test_favorites_page(self):
        """12. Verify Favorites page SQL query and toggle functionality"""
        # Login
        login_data = {
            'email': 'tester@example.com',
            'password': 'testpassword'
        }
        self.client.post('/login', data=login_data, follow_redirects=True)
        
        # Get a valid recipe ID
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()
        cursor.execute("SELECT recipe_id FROM recipes LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        self.assertIsNotNone(row, "No recipes found to test favorites")
        recipe_id = row[0]
        
        # Toggle favorite (Add)
        response = self.client.post(f'/favorite/{recipe_id}', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn(data['favorite_status'], ['added', 'removed'])
        
        # Ensure it is favorited now (if it was removed, add it again)
        if data['favorite_status'] == 'removed':
            response = self.client.post(f'/favorite/{recipe_id}', follow_redirects=True)
            data = response.get_json()
            self.assertEqual(data['favorite_status'], 'added')
            
        # Get favorites page
        response = self.client.get('/favorites')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("Favorites", html)
        print("Favorites page loading test completed successfully.")


if __name__ == '__main__':
    unittest.main()
