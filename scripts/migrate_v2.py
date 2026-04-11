import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from backend.database_manager import get_db_connection

def migrate():
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database.")
        return
    
    try:
        c = conn.cursor()
        
        # 1. Add created_by column to users table
        print("Adding 'created_by' column to users table...")
        c.execute("""
            ALTER TABLE users 
            ADD COLUMN IF NOT EXISTS created_by INTEGER REFERENCES users(id)
        """)
        
        # 2. Add created_at column if it doesn't exist (useful for User Management)
        print("Adding 'created_at' column to users table...")
        c.execute("""
            ALTER TABLE users 
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        """)
        
        conn.commit()
        print("Migration successful.")
        
    except Exception as e:
        print(f"Migration error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
