import psycopg2
from backend.database_manager import PG_CONFIG, init_db

def verify_schema():
    print("Initializing database...")
    init_db()
    
    print("\nVerifying constraints on timetable_entries...")
    conn = psycopg2.connect(**PG_CONFIG)
    c = conn.cursor()
    
    # Check for unique constraint on timetable_entries
    c.execute("""
        SELECT conname 
        FROM pg_constraint 
        WHERE conrelid = 'timetable_entries'::regclass 
        AND contype = 'u';
    """)
    constraints = c.fetchall()
    print(f"Unique constraints: {[cn[0] for cn in constraints]}")
    
    if any('unique_timetable_slot' in cn[0] for cn in constraints):
        print("SUCCESS: unique_timetable_slot found.")
    else:
        print("FAILURE: unique_timetable_slot NOT found.")
    
    conn.close()

if __name__ == "__main__":
    verify_schema()
