import sys
import os
import logging

# Add the project root to sys.path
sys.path.append(os.getcwd())

from backend.database_manager import get_db_connection

def repair():
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database.")
        return
    
    try:
        c = conn.cursor()
        
        # 1. Update students with missing 'batch_year'
        print("Assigning default Batch '2023-2027' to unassigned students...")
        c.execute("""
            UPDATE users 
            SET batch_year = '2023-2027'
            WHERE role = 'student' 
              AND (batch_year IS NULL OR batch_year = '' OR batch_year = 'Unassigned')
        """)
        print(f"Updated {c.rowcount} students with default batch.")
        
        # 2. Update students with missing 'year'
        print("Assigning default Year '1' to unassigned students...")
        c.execute("""
            UPDATE users 
            SET year = 1
            WHERE role = 'student' 
              AND (year IS NULL OR year = 0)
        """)
        print(f"Updated {c.rowcount} students with default year.")

        # 3. Update students with missing 'semester'
        print("Assigning default Semester '1' to unassigned students...")
        c.execute("""
            UPDATE users 
            SET semester = 1
            WHERE role = 'student' 
              AND (semester IS NULL OR semester = 0)
        """)
        print(f"Updated {c.rowcount} students with default semester.")
        
        # 4. Cleanup 'Batch' duplication in strings (e.g. 'Batch Batch 2022' -> 'Batch 2022')
        print("Cleaning up Batch string duplication...")
        c.execute("""
            UPDATE users 
            SET batch_year = TRIM(REPLACE(batch_year, 'Batch Batch', 'Batch'))
            WHERE batch_year LIKE '%%Batch Batch%%'
        """)
        print(f"Cleaned up {c.rowcount} batch strings.")

        conn.commit()
        print("\nRepair Complete.")
        
    except Exception as e:
        print(f"Error during repair: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    repair()
