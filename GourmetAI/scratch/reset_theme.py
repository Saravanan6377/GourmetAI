import psycopg2

def reset_theme():
    conn = psycopg2.connect(
        host='localhost',
        database='gourmetai_db',
        user='postgres',
        password='welcome',
        port=5432
    )
    cur = conn.cursor()
    
    updates = {
        'theme_preset': 'slate',
        'primary_color': '#3b82f6',
        'primary_color_alt': '#60a5fa',
        'accent_color': '#6b7280',
        'accent_color_alt': '#9ca3af',
        'gradient_bg': 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)'
    }
    
    for k, v in updates.items():
        cur.execute("UPDATE system_settings SET value = %s WHERE key = %s", (v, k))
        
    conn.commit()
    cur.close()
    conn.close()
    print("Database theme settings updated to Slate successfully.")

if __name__ == '__main__':
    reset_theme()
