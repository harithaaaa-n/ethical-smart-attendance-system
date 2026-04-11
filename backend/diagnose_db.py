import psycopg2
import logging

PG_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'root',
    'database': 'attendance'
}

def diagnose():
    conn = None
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        c = conn.cursor()
        
        print("--- Table Columns ---")
        c.execute("SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = 'timetable_periods'")
        for col in c.fetchall():
            print(col)
            
        print("\n--- Constraints ---")
        c.execute("SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint WHERE conrelid = 'timetable_periods'::regclass")
        for con in c.fetchall():
            print(con)
            
        print("\n--- Testing Save ---")
        data = {'period_no': 1, 'start_time': '08:30:00', 'end_time': '09:20:00'}
        try:
            c.execute("""
                INSERT INTO timetable_periods (period_no, start_time, end_time)
                VALUES (%s, %s, %s)
                ON CONFLICT (period_no) DO UPDATE
                SET start_time = EXCLUDED.start_time, end_time = EXCLUDED.end_time
            """, (data['period_no'], data['start_time'], data['end_time']))
            print("Test Save: SUCCESS (Pending commit)")
            conn.rollback()
        except Exception as e:
            print(f"Test Save: FAILED - {e}")
            conn.rollback()

    except Exception as e:
        print(f"Connection Error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    diagnose()
