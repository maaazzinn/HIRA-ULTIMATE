import sqlite3

def migrate_database():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    print("Checking for missing columns...")
    
    # 1. Add 'filename' column if it doesn't exist
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN filename TEXT")
        print("Added 'filename' column.")
    except sqlite3.OperationalError:
        print("'filename' column already exists.")

    # 2. Add 'file_path' column if it doesn't exist
    try:
        cursor.execute("ALTER TABLE reports ADD COLUMN file_path TEXT")
        print("Added 'file_path' column.")
    except sqlite3.OperationalError:
        print("'file_path' column already exists.")

    conn.commit()
    conn.close()
    print("Migration complete!")

if __name__ == "__main__":
    migrate_database()