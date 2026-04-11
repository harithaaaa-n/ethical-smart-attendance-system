import psycopg2
import logging

PG_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'root',
    'database': 'attendance'
}

def apply_constraint():
    conn = None
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        c = conn.cursor()
        
        # Remove any duplicate period_no rows. Keep the one with the lowest ID.
        print("Cleaning up duplicates if any...")
        c.execute("""
            DELETE FROM timetable_periods a USING timetable_periods b 
            WHERE a.id > b.id AND a.period_no = b.period_no;
        """)
        print(f"Deleted {c.rowcount} duplicate rows.")
        
        print("Adding unique constraint...")
        # Drop existing constraints just in case it was created with a weird name
        try:
            c.execute("ALTER TABLE timetable_periods DROP CONSTRAINT IF EXISTS timetable_periods_period_no_key;")
        except:
            pass
        try:
            c.execute("ALTER TABLE timetable_periods DROP CONSTRAINT IF EXISTS uq_period_no;")
        except:
            pass
            
        c.execute("ALTER TABLE timetable_periods ADD CONSTRAINT uq_period_no UNIQUE (period_no);")
        conn.commit()
        print("MIGRATION SUCCESS: Unique constraint added successfully.")
        
    except Exception as e:
        print(f"MIGRATION ERROR: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    apply_constraint()
