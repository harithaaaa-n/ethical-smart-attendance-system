import psycopg2
import psycopg2.extras

def run():
    conn = psycopg2.connect(host='localhost', user='postgres', password='root', database='attendance')
    c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Get user 117 (Haritha)
    c.execute("SELECT id, department, batch_year, semester FROM users WHERE id=117")
    user = c.fetchone()
    print("USER:", dict(user))
    
    # Check if a timetable master exists for her
    c.execute("""
        SELECT id, timetable_name, department, batch_year, semester 
        FROM timetable_master 
        WHERE LOWER(department) = LOWER(%s) AND batch_year = %s AND semester = %s
    """, (user['department'], user['batch_year'], user['semester']))
    master = c.fetchone()
    
    if master:
        print("MASTER:", dict(master))
        c.execute("""
            SELECT te.day_of_week, te.period_no, p.start_time, p.end_time, s.name as subject 
            FROM timetable_entries te 
            JOIN timetable_periods p ON te.period_no=p.period_no 
            JOIN academic_subjects s ON te.subject_id=s.id 
            WHERE te.timetable_id=%s
            ORDER BY te.day_of_week, te.period_no
        """, (master['id'],))
        entries = c.fetchall()
        print(f"ENTRIES ({len(entries)}):")
        for e in entries:
            print("  ", dict(e))
    else:
        print("NO TIMETABLE MASTER FOUND FOR THIS USER'S CLASS.")

    conn.close()

if __name__ == '__main__':
    run()
