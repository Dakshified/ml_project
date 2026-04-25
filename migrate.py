import sqlite3

def migrate():
    conn = sqlite3.connect('cinematch.db')
    cursor = conn.cursor()
    
    # Check if column exists
    cursor.execute("PRAGMA table_info(user_dna)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'genre_snapshot_json' not in columns:
        print("Adding genre_snapshot_json to user_dna table...")
        cursor.execute("ALTER TABLE user_dna ADD COLUMN genre_snapshot_json TEXT")
    
    # Also check for movie_ratings table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='movie_ratings'")
    if not cursor.fetchone():
        print("Creating movie_ratings table...")
        cursor.execute("""
            CREATE TABLE movie_ratings (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                movie_id TEXT NOT NULL,
                movie_title TEXT NOT NULL,
                rating FLOAT NOT NULL,
                rated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
    
    conn.commit()
    conn.close()
    print("Migration complete!")

if __name__ == "__main__":
    migrate()
