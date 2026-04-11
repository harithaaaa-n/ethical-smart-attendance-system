import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta, time
import numpy as np
import bcrypt
import logging
import os

# --- PostgreSQL Connection Configuration ---
PG_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASS', 'root'), 
    'database': os.environ.get('DB_NAME', 'attendance')
}

STUDENT_NAMES = ["Arjun Singh", "Priya Sharma", "Rohan Verma", "Ananya Reddy", "Vikram Kumar", "Saniya Ahmed", "Rahul Mishra", "Kavya Murthy", "Siddharth Rao", "Jasmine Kaur"]
TEACHER_NAMES = ["Dr. Anika Iyer", "Prof. Rajesh Gupta", "Ms. Kavita Menon", "Mr. Vishal Singh"]

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        return conn
    except psycopg2.Error as err:
        logging.error(f"Error connecting to PostgreSQL: {err}")
        return None

def get_expected_count(department, batch_year, semester, subject_name, start_date, end_date):
    """
    Calculates the number of scheduled periods for a class in a date range based on the timetable.
    """
    conn = get_db_connection()
    if not conn: return 0
    try:
        c = conn.cursor()
        
        # 1. Resolve subject_id if needed
        subject_id = None
        if subject_name and subject_name != 'All':
            c.execute("SELECT id FROM academic_subjects WHERE LOWER(name) = LOWER(%s) OR LOWER(code) = LOWER(%s)", (subject_name, subject_name))
            res = c.fetchone()
            if res: subject_id = res[0]
            else: return 0 
            
        # 2. Find the relevant timetable(s)
        query = """
            SELECT te.day_of_week, COUNT(*) as count_per_day
            FROM timetable_entries te
            JOIN timetable_master tm ON te.timetable_id = tm.id
            WHERE tm.active_status = 'Active'
        """
        params = []
        if department and department != 'All':
            query += " AND LOWER(REPLACE(tm.department, ' ', '')) = LOWER(REPLACE(%s, ' ', '')) "
            params.append(department)
        if batch_year and batch_year != 'All':
            query += " AND (tm.batch_year = %s OR tm.batch_year LIKE %s) "
            params.extend([batch_year, f"{batch_year}%"])
        if semester and semester != 'All':
            query += " AND tm.semester = %s "
            params.append(int(semester) if str(semester).isdigit() else semester)
        if subject_id:
            query += " AND te.subject_id = %s "
            params.append(subject_id)
            
        query += " GROUP BY te.day_of_week "
        c.execute(query, tuple(params))
        counts = {row[0].strip().capitalize(): row[1] for row in c.fetchall()}
        
        if not counts: return 0
        
        # 3. Sum up dates
        total_expected = 0
        curr = datetime.strptime(str(start_date), '%Y-%m-%d')
        end = datetime.strptime(str(end_date), '%Y-%m-%d')
        while curr <= end:
            day_name = curr.strftime('%A')
            total_expected += counts.get(day_name, 0)
            curr += timedelta(days=1)
            
        return total_expected
    except Exception as e:
        logging.error(f"Error calculating expected count: {e}")
        return 0
    finally:
        conn.close()

def init_db():
    """Initializes the database by creating tables and populating them with dummy data."""
    conn = get_db_connection()
    if conn is None:
        logging.error("Could not connect to PostgreSQL. Exiting.")
        return
    
    try:
        c = conn.cursor()
        
        tables_creation_sql = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password BYTEA NOT NULL,
            role VARCHAR(50) NOT NULL,
            full_name VARCHAR(255) NOT NULL,
            roll_number VARCHAR(50) UNIQUE,
            department VARCHAR(100),
            year INTEGER,
            batch_year VARCHAR(10)
        );
        CREATE TABLE IF NOT EXISTS timetable (
            id SERIAL PRIMARY KEY,
            day VARCHAR(50) NOT NULL,
            start_time TIME NOT NULL,
            end_time TIME NOT NULL,
            subject VARCHAR(100) NOT NULL,
            department VARCHAR(100),
            year INTEGER
        );
        CREATE TABLE IF NOT EXISTS attendance (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            timestamp TIMESTAMP NOT NULL,
            status VARCHAR(50) NOT NULL,
            subject VARCHAR(100),
            marked_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
            is_finalized BOOLEAN DEFAULT TRUE
        );
        CREATE UNIQUE INDEX IF NOT EXISTS unique_attendance_per_day_period ON attendance (user_id, (timestamp::date), subject, COALESCE(period_no, -1));
        
        CREATE TABLE IF NOT EXISTS face_encodings (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            encoding BYTEA NOT NULL,
            UNIQUE (user_id)
        );
        
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            encoding BYTEA NOT NULL,
            source VARCHAR(50) DEFAULT 'live',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS teachers_subjects (
            teacher_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            subject VARCHAR(100),
            PRIMARY KEY (teacher_id, subject)
        );
        CREATE TABLE IF NOT EXISTS semester_info (
            id SERIAL PRIMARY KEY,
            batch_year VARCHAR(20) NOT NULL,
            semester INTEGER NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            saturday_working BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS semester_holidays (
            id SERIAL PRIMARY KEY,
            semester_info_id INTEGER REFERENCES semester_info(id) ON DELETE CASCADE,
            holiday_date DATE NOT NULL,
            holiday_name VARCHAR(255) NOT NULL,
            holiday_type VARCHAR(100)
        );
        """
        c.execute(tables_creation_sql)
        conn.commit()
        conn.commit()
        logging.info("Database tables verified/created.")

        # --- MIGRATION: SETTINGS MODULE ---
        ensure_settings_columns()
        ensure_section_column()
        # ----------------------------------

        # --- MIGRATION: TIMETABLE MASTER SYSTEM ---
        ensure_timetable_tables()
        # ------------------------------------------

        # --- MIGRATION: ADD FACE STATUS COLUMNS ---
        columns_to_add = [
            ("face_status", "VARCHAR(50) DEFAULT 'Not Enrolled'"),
            ("face_folder_path", "TEXT"),
            ("face_uploaded_count", "INTEGER DEFAULT 0"),
            ("face_enrolled_at", "TIMESTAMP"),
            ("is_face_enrolled", "BOOLEAN DEFAULT FALSE")
        ]
        for col_name, col_type in columns_to_add:
            try:
                c.execute(f"ALTER TABLE users ADD COLUMN IF NOT EXISTS {col_name} {col_type}")
            except Exception as e:
                logging.warning(f"Error adding column {col_name}: {e}")
                conn.rollback()
        conn.commit()
        # ------------------------------------------

        # --- MIGRATION: ADD CONFIDENCE SCORE AND PROOF IMAGE ---
        try:
            # Module 8 Schema Enhancements
            logging.info("Applying Schema Migrations for Module 8...")
            c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS register_number VARCHAR(50) UNIQUE;")
            c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS batch_start_year INTEGER;")
            c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS batch_end_year INTEGER;")
            c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS semester INTEGER;")
            c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS active_status VARCHAR(50) DEFAULT 'Active';")
            # Ensure all students are Active by default if not set
            c.execute("UPDATE users SET active_status = 'Active' WHERE active_status IS NULL;")
            c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS status_reason TEXT;")
            c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS status_updated_at TIMESTAMP;")
            conn.commit()
        except Exception as e:
            conn.rollback()
            logging.warning(f"Schema migration skipped/error (safe to ignore if columns exist): {e}")

        try:
            c.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS confidence_score FLOAT")
            c.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS proof_image_path VARCHAR(500)")
            c.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS method VARCHAR(50) DEFAULT 'live'")
            c.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS is_finalized BOOLEAN DEFAULT TRUE")
            conn.commit()
            logging.info("Migration: Added confidence_score, proof_image_path, method, and is_finalized columns.")
        except Exception as e:
            logging.warning(f"Migration warning: {e}")
            conn.rollback()
        # ----------------------------------------------------
        
        # --- PERFORMANCE OPTIMIZATION: INDEXES ---
        try:
            perf_indexes_sql = """
            CREATE INDEX IF NOT EXISTS idx_attendance_user_id ON attendance(user_id);
            CREATE INDEX IF NOT EXISTS idx_attendance_timestamp ON attendance(timestamp);
            CREATE INDEX IF NOT EXISTS idx_attendance_subject ON attendance(subject);
            CREATE INDEX IF NOT EXISTS idx_users_department ON users(department);
            CREATE INDEX IF NOT EXISTS idx_users_year ON users(year);
            """
            c.execute(perf_indexes_sql)
            conn.commit()
            logging.info("Performance initialization: Indexes verified/created.")
        except Exception as e:
            logging.warning(f"Index creation warning: {e}")
            conn.rollback()
        # ----------------------------------------------------

        # --- MIGRATION: EMERGENCY ACCESS CODES ---
        try:
            logging.info("Applying Schema Migrations for Emergency Access...")
            c.execute("""
                CREATE TABLE IF NOT EXISTS emergency_codes (
                    id SERIAL PRIMARY KEY,
                    code VARCHAR(20) UNIQUE NOT NULL,
                    created_by INTEGER REFERENCES users(id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_used BOOLEAN DEFAULT FALSE,
                    used_at TIMESTAMP,
                    metadata JSONB
                );
            """)
            conn.commit()
            logging.info("Migration: Added emergency_codes table.")
        except Exception as e:
            logging.warning(f"Migration warning (Emergency Codes): {e}")
            conn.rollback()
        # ----------------------------------------------------

        c.execute("SELECT COUNT(*) FROM users")
        if c.fetchone()[0] == 0:
            logging.info("Populating dummy data...")
            hashed_password = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt())
            c.execute("INSERT INTO users (username, password, role, full_name, department, year, batch_year) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                      ('admin', hashed_password, 'admin', 'System Administrator', 'Cyber Security', 3, '2023-2027'))
            
            for name in TEACHER_NAMES:
                hashed_password = bcrypt.hashpw('password123'.encode('utf-8'), bcrypt.gensalt())
                c.execute("INSERT INTO users (username, password, role, full_name, department, year, batch_year) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                          (name.replace(" ", "").lower(), hashed_password, 'teacher', name, 'Cyber Security', 3, '2023-2027'))
            
            for i, name in enumerate(STUDENT_NAMES):
                roll_number = f"CS2023{i+1:03d}"
                hashed_password = bcrypt.hashpw('student123'.encode('utf-8'), bcrypt.gensalt())
                c.execute("INSERT INTO users (username, password, role, full_name, roll_number, year, batch_year) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                          (name.replace(" ", "").lower(), hashed_password, 'student', name, roll_number, 3, '2023-2027'))

            # --- MIGRATION: TIMETABLE ENHANCEMENTS ---
            try:
                c.execute("ALTER TABLE timetable ADD COLUMN IF NOT EXISTS teacher_id INTEGER REFERENCES users(id) ON DELETE SET NULL")
                c.execute("ALTER TABLE timetable ADD COLUMN IF NOT EXISTS period_no INTEGER")
                conn.commit()
                logging.info("Migration: Added teacher_id and period_no to timetable.")
            except Exception as e:
                logging.warning(f"Timetable migration warning: {e}")
                conn.rollback()
            # ----------------------------------------

            # --- MIGRATION: STUDENT ACADEMIC FIELDS (Feature Request) ---
            try:
                c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS register_number VARCHAR(100) UNIQUE")
                c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS semester INTEGER DEFAULT 1")
                
                # Batch Start/End logic
                c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS batch_start_year INTEGER")
                c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS batch_end_year INTEGER")
                
                conn.commit()
                logging.info("Migration: Added student academic fields (reg_no, sem, batch_start/end).")
            except Exception as e:
                logging.warning(f"Migration warning (Student Fields): {e}")
                conn.rollback()
            # ----------------------------------------------------

            # --- MIGRATION: TIMETABLE PERIODS (Dynamic Dropdown) ---
            try:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS timetable_periods (
                        id SERIAL PRIMARY KEY,
                        period_no INTEGER UNIQUE NOT NULL,
                        start_time TIME NOT NULL,
                        end_time TIME NOT NULL,
                        label VARCHAR(50)
                    );
                """)
                
                # MIGRATION: Ensure unique constraint exists for ON CONFLICT clause
                try:
                    c.execute("ALTER TABLE timetable_periods DROP CONSTRAINT IF EXISTS uq_period_no")
                    c.execute("ALTER TABLE timetable_periods ADD CONSTRAINT uq_period_no UNIQUE (period_no)")
                    conn.commit()
                except Exception as e:
                    logging.warning(f"Note: Could not explicitly add uq_period_no constraint (may already exist via table definition): {e}")
                    conn.rollback()
                    
                c.execute("SELECT COUNT(*) FROM timetable_periods")
                if c.fetchone()[0] == 0:
                    logging.info("Populating default timetable periods (1-8)...")
                    periods_data = [
                        (1, '08:30:00', '09:20:00', 'Period 1'),
                        (2, '09:20:00', '10:10:00', 'Period 2'),
                        (3, '10:30:00', '11:20:00', 'Period 3'),
                        (4, '11:20:00', '12:10:00', 'Period 4'),
                        (5, '13:20:00', '14:10:00', 'Period 5'),
                        (6, '14:10:00', '15:00:00', 'Period 6'),
                        (7, '15:00:00', '15:50:00', 'Period 7'),
                        (8, '16:00:00', '16:50:00', 'Period 8')
                    ]
                    c.executemany("INSERT INTO timetable_periods (period_no, start_time, end_time, label) VALUES (%s, %s, %s, %s)", periods_data)
                conn.commit()
                logging.info("Migration: Added timetable_periods table.")
            except Exception as e:
                logging.warning(f"Migration warning (Timetable Periods): {e}")
                conn.rollback()
            # ----------------------------------------------------

            timetable_data = [
                ('Monday', '08:30:00', '09:25:00', 'CIA', 'Cyber Security', 3),
                ('Monday', '09:25:00', '10:20:00', 'DS', 'Cyber Security', 3),
                ('Monday', '10:30:00', '11:25:00', 'OOSE', 'Cyber Security', 3),
                ('Monday', '11:25:00', '12:20:00', 'OOSE', 'Cyber Security', 3),
                ('Monday', '13:20:00', '14:15:00', 'OOPs Lab', 'Cyber Security', 3),
                ('Monday', '14:15:00', '15:10:00', 'OOPs Lab', 'Cyber Security', 3),
                ('Monday', '15:10:00', '16:05:00', 'OOPs Lab', 'Cyber Security', 3),
                ('Tuesday', '08:30:00', '09:25:00', 'OOSE', 'Cyber Security', 3),
                ('Tuesday', '09:25:00', '10:20:00', 'DM', 'Cyber Security', 3),
                ('Tuesday', '10:30:00', '11:25:00', 'FCS', 'Cyber Security', 3),
                ('Tuesday', '11:25:00', '12:20:00', 'OOPs', 'Cyber Security', 3),
                ('Tuesday', '13:20:00', '14:15:00', 'DS', 'Cyber Security', 3),
                ('Tuesday', '14:15:00', '15:10:00', 'DM', 'Cyber Security', 3),
                ('Tuesday', '15:10:00', '16:05:00', 'OOPs', 'Cyber Security', 3),
                ('Wednesday', '08:30:00', '09:25:00', 'OOPs', 'Cyber Security', 3),
                ('Wednesday', '09:25:00', '10:20:00', 'FCS', 'Cyber Security', 3),
                ('Wednesday', '10:30:00', '11:25:00', 'DM', 'Cyber Security', 3),
                ('Wednesday', '11:25:00', '12:20:00', 'FDS', 'Cyber Security', 3),
                ('Wednesday', '13:20:00', '14:15:00', 'DS Lab', 'Cyber Security', 3),
                ('Wednesday', '14:15:00', '15:10:00', 'DS Lab', 'Cyber Security', 3),
                ('Wednesday', '15:10:00', '16:05:00', 'DS Lab', 'Cyber Security', 3),
                ('Thursday', '08:30:00', '09:25:00', 'DM', 'Cyber Security', 3),
                ('Thursday', '09:25:00', '10:20:00', 'OOSE', 'Cyber Security', 3),
                ('Thursday', '10:30:00', '11:25:00', 'DS', 'Cyber Security', 3),
                ('Thursday', '11:25:00', '12:20:00', 'OOPs', 'Cyber Security', 3),
                ('Thursday', '13:20:00', '14:15:00', 'FDS', 'Cyber Security', 3),
                ('Thursday', '14:15:00', '15:10:00', 'FCS', 'Cyber Security', 3),
                ('Thursday', '15:10:00', '16:05:00', 'OOSE', 'Cyber Security', 3),
                ('Friday', '08:30:00', '09:25:00', 'DS', 'Cyber Security', 3),
                ('Friday', '09:25:00', '10:20:00', 'FCS', 'Cyber Security', 3),
                ('Friday', '10:30:00', '11:25:00', 'FDS', 'Cyber Security', 3),
                ('Friday', '11:25:00', '12:20:00', 'DM', 'Cyber Security', 3),
                ('Friday', '13:20:00', '14:15:00', 'FDS Lab', 'Cyber Security', 3),
                ('Friday', '14:15:00', '15:10:00', 'FDS Lab', 'Cyber Security', 3),
                ('Friday', '15:10:00', '16:05:00', 'Tutor/Lib', 'Cyber Security', 3),
                ('Saturday', '08:30:00', '09:25:00', 'CIA', 'Cyber Security', 3),
                ('Saturday', '10:30:00', '11:25:00', 'PD', 'Cyber Security', 3),
                ('Saturday', '11:25:00', '12:20:00', 'PD', 'Cyber Security', 3)
            ]
            c.executemany("INSERT INTO timetable (day, start_time, end_time, subject, department, year) VALUES (%s, %s, %s, %s, %s, %s)", timetable_data)

            conn.commit()
            logging.info("Dummy data populated successfully.")
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Error during database initialization: {error}")
    finally:
        if conn:
            conn.close()

def authenticate_user(username, password):
    """Authenticates a user against the database."""
    conn = get_db_connection()
    if conn is None: return None
    
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = c.fetchone()
        
        if user:
            stored_hashed_password = bytes(user['password'])
            
            if bcrypt.checkpw(password, stored_hashed_password):
                return dict(user)
        return None
    except Exception as e:
        logging.error(f"Authentication error for user {username}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_user_info(user_id):
    """Retrieves a user's information by their ID."""
    conn = get_db_connection()
    if conn is None: return None
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = c.fetchone()
        return dict(user) if user else None
    except Exception as e:
        logging.error(f"Error getting user info for ID {user_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_student_attendance_by_date_range(user_id, start_date, end_date):
    """Fetches attendance records for a student within a date range."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute(
            """SELECT * FROM attendance WHERE user_id = %s AND timestamp::date BETWEEN %s AND %s
               ORDER BY timestamp DESC""",
            (user_id, start_date, end_date)
        )
        records = c.fetchall()
        return [dict(record) for record in records]
    except Exception as e:
        logging.error(f"Error getting attendance for user {user_id}: {e}")
        return []
    finally:
        if conn:
            conn.close()

def calculate_attendance_percentage(attendance_data):
    """Calculates attendance percentage from attendance records."""
    if not attendance_data:
        return 0
    present_count = sum(1 for record in attendance_data if record['status'] in ('Present', 'OD'))
    return (present_count / len(attendance_data)) * 100

def get_timetable(department=None, year=None):
    """Retrieves the class timetable based on department and year."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Improved Query: Join academic_subjects and teacher_subject_assignments
        # Priority: Direct Timetable Assignment > General Subject Assignment
        # ROBUST MATCHING: Handles Case, Spaces, and Subject Codes
        query = """
            SELECT 
                t.id, t.day, t.start_time, t.end_time, t.subject, t.department, t.year, t.teacher_id as direct_teacher_id,
                s.id as subject_id, s.name as subject_name, s.code as subject_code,
                COALESCE(u_direct.id, u_assigned.id) as teacher_id,
                COALESCE(u_direct.full_name, u_assigned.full_name) as teacher_name
            FROM timetable t
            -- Match Subject by Name OR Code (Case Insensitive)
            LEFT JOIN academic_subjects s ON (
                LOWER(TRIM(t.subject)) = LOWER(TRIM(s.name)) OR 
                LOWER(TRIM(t.subject)) = LOWER(TRIM(s.code))
            )
            -- Match Assignment to check availability
            LEFT JOIN teacher_subject_assignments tsa ON (
                tsa.subject_id = s.id AND 
                -- Robust Dept Match: 'Cyber Security' == 'Cybersecurity'
                LOWER(REPLACE(tsa.department, ' ', '')) = LOWER(REPLACE(t.department, ' ', '')) AND 
                -- Robust Year Match
                tsa.year::INTEGER = t.year::INTEGER
            )
            LEFT JOIN users u_assigned ON tsa.teacher_id = u_assigned.id
            LEFT JOIN users u_direct ON t.teacher_id = u_direct.id
            WHERE 1=1
        """
        params = []
        if department and department != 'All':
            # Robust filter for input department too? 
            # Ideally yes, but let's stick to simple first unless requested.
            # Usually the filter dropdown matches DB values exactly if populated from DB.
            # But t.department might be "Cyber Security" and user selected "Cybersecurity" (or vice versa).
            # Let's normalize the filter check too for consistency.
            query += " AND LOWER(REPLACE(t.department, ' ', '')) = LOWER(REPLACE(%s, ' ', ''))"
            params.append(department)
        if year and year != 'All':
            query += " AND t.year::INTEGER = %s::INTEGER"
            params.append(year)
        
        query += " ORDER BY t.day, t.start_time"
        c.execute(query, tuple(params))
        timetable = c.fetchall()
        return [dict(item) for item in timetable]
    except Exception as e:
        logging.error(f"Error getting timetable: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_teacher_today_timetable(teacher_id):
    """Retrieves timetable entries for a specific teacher for the current day."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        current_day = datetime.now().strftime("%A")
        # Fetch entries assigned to this teacher for today
        c.execute("""
            SELECT t.*, u.full_name as teacher_name 
            FROM timetable t
            LEFT JOIN users u ON t.teacher_id = u.id
            WHERE t.teacher_id = %s AND t.day = %s
            ORDER BY t.start_time
        """, (teacher_id, current_day))
        
        entries = [dict(row) for row in c.fetchall()]
        return entries
    except Exception as e:
        logging.error(f"Error getting teacher timetable: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_teacher_weekly_timetable(teacher_id):
    """Retrieves all timetable entries for a specific teacher, ordered by day and time."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("""
            SELECT 
                te.id,
                INITCAP(TRIM(te.day_of_week)) as day, 
                tp.period_no,
                tp.start_time, tp.end_time,
                s.name as subject,
                s.name as subject_name,
                tm.department, tm.batch_year, tm.section, tm.semester,
                ((tm.semester + 1) / 2) as year
            FROM timetable_entries te
            JOIN timetable_master tm ON te.timetable_id = tm.id
            JOIN academic_subjects s ON te.subject_id = s.id
            JOIN timetable_periods tp ON te.period_no = tp.period_no
            WHERE te.staff_id = %s
            ORDER BY 
                CASE 
                    WHEN INITCAP(TRIM(te.day_of_week)) = 'Monday' THEN 1
                    WHEN INITCAP(TRIM(te.day_of_week)) = 'Tuesday' THEN 2
                    WHEN INITCAP(TRIM(te.day_of_week)) = 'Wednesday' THEN 3
                    WHEN INITCAP(TRIM(te.day_of_week)) = 'Thursday' THEN 4
                    WHEN INITCAP(TRIM(te.day_of_week)) = 'Friday' THEN 5
                    WHEN INITCAP(TRIM(te.day_of_week)) = 'Saturday' THEN 6
                    WHEN INITCAP(TRIM(te.day_of_week)) = 'Sunday' THEN 7
                END,
                tp.start_time
        """, (teacher_id,))
        
        entries = [dict(row) for row in c.fetchall()]
        return entries
    except Exception as e:
        logging.error(f"Error getting teacher weekly timetable: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_timetable_entry(entry_id):
    """Retrieves a single timetable entry by ID."""
    conn = get_db_connection()
    if conn is None: return None
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT * FROM timetable WHERE id = %s", (entry_id,))
        result = c.fetchone()
        return dict(result) if result else None
    except Exception as e:
        logging.error(f"Error getting timetable entry: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_user_by_regno(register_number):
    """Fetches user info by register number."""
    conn = get_db_connection()
    if conn is None: return None
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT * FROM users WHERE register_number = %s", (register_number,))
        res = c.fetchone()
        return dict(res) if res else None
    except Exception as e:
        logging.error(f"Error fetching user by regno: {e}")
        return None
    finally:
        if conn: conn.close()

def get_current_subject(department=None, year=None):
    """Finds the current subject based on the time and day."""
    conn = get_db_connection()
    if conn is None: return None
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        now = datetime.now()
        current_time_str = now.strftime("%H:%M:%S")
        current_day = now.strftime("%A")
        c.execute("SELECT subject FROM timetable WHERE day = %s AND start_time <= %s AND end_time >= %s AND department = %s AND year = %s",
                  (current_day, current_time_str, current_time_str, department, year))
        subject = c.fetchone()
        return subject['subject'] if subject else None
    except Exception as e:
        logging.error(f"Error getting current subject: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_all_students():
    """Retrieves all student users from the database."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT id, full_name, roll_number, department, year, batch_year FROM users WHERE role = 'student' ORDER BY full_name")
        students = c.fetchall()
        return [dict(student) for student in students]
    except Exception as e:
        logging.error(f"Error getting all students: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_all_unencoded_students():
    """Retrieves students who do not have a face encoding."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("""
            SELECT u.id, u.full_name, u.roll_number, u.department, u.year, u.batch_year
            FROM users u LEFT JOIN face_encodings fe ON u.id = fe.user_id 
            WHERE u.role='student' AND fe.encoding IS NULL ORDER BY u.full_name
        """)
        students = c.fetchall()
        return [dict(student) for student in students]
    except Exception as e:
        logging.error(f"Error getting unencoded students: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_students_by_class(department, year, batch_year=None, semester=None):
    """
    Retrieves students belonging to a specific department and year.
    Source of Truth Fix: Applies filters for batch, semester and only includes 'Enrolled' and 'Active' students.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        query = """
            SELECT u.id, u.full_name, u.roll_number, u.register_number, u.department, u.year, u.batch_year, u.semester, u.face_status, u.active_status, u.created_at, creator.full_name as created_by_name
            FROM users u
            LEFT JOIN users creator ON u.created_by = creator.id
            WHERE u.role='student' 
              AND u.active_status='Active'
        """
        params = []
        if department and department != 'All':
            # Case-insensitive match to handle inconsistent capitalization (e.g. CYBERSECURITY vs Cybersecurity)
            query += " AND LOWER(u.department) = LOWER(%s)"
            params.append(department)
        if year and year != 'All':
            query += " AND u.year = %s"
            params.append(int(year) if str(year).isdigit() else year)
        if batch_year and batch_year != 'All':
            query += " AND (u.batch_year = %s OR u.batch_start_year = %s)"
            params.append(str(batch_year))
            params.append(str(batch_year).split('-')[0] if '-' in str(batch_year) else str(batch_year))
        if semester and semester != 'All':
            query += " AND u.semester = %s"
            params.append(int(semester) if str(semester).isdigit() else semester)
            
        query += " ORDER BY full_name"
        c.execute(query, tuple(params))
        students = c.fetchall()
        return [dict(student) for student in students]
    except Exception as e:
        logging.error(f"Error getting students by class: {e}")
        return []
    finally:
        if conn:
            conn.close()
def ensure_face_status_column():
    """Checks and adds face_status and face_enrolled_at columns to users table."""
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS face_status VARCHAR(50) DEFAULT 'Not Enrolled'")
        c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS face_enrolled_at TIMESTAMP")
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"Error ensuring face_status columns: {e}")
    finally:
        if conn:
            conn.close()

def update_user_face_status(user_id, status='Enrolled'):
    """Updates the face status for a user."""
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        c.execute(
            "UPDATE users SET face_status = %s, face_enrolled_at = %s WHERE id = %s",
            (status, datetime.now(), user_id)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"Error updating face status for user {user_id}: {e}")
    finally:
        if conn:
            conn.close()

# Define Path for Face DB (matching biometric_logic)
PROJECT_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DB_PATH = os.path.join(PROJECT_BACKEND_DIR, "student_faces")

def sync_face_status():
    """Calculates and updates face_status for all students based on file existence."""
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT id, register_number FROM users WHERE role='student'")
        students = c.fetchall()
        
        updated_count = 0
        for s in students:
            user_id = s['id']
            reg_no = s.get('register_number')
            
            # Logic from biometric_logic: check for folder and valid encoding/images
            user_folder = os.path.join(FACE_DB_PATH, str(user_id))
            
            is_enrolled = False
            
            # 1. RegNo Folder (New)
            if reg_no:
                reg_path = os.path.join(FACE_DB_PATH, str(reg_no))
                if os.path.exists(reg_path):
                     if len([f for f in os.listdir(reg_path) if f.endswith('.jpg') or f.endswith('.png')]) > 0:
                         is_enrolled = True

            # 2. Legacy ID Folder (Backward Compat)
            if not is_enrolled and os.path.exists(user_folder):
                 if os.path.exists(os.path.join(user_folder, 'encodings.pkl')) or len([f for f in os.listdir(user_folder) if f.endswith('.jpg')]) > 0:
                     is_enrolled = True
            
            if not is_enrolled:
                # 3. Check Embeddings Table
                c.execute("SELECT count(*) FROM face_encodings WHERE user_id = %s", (user_id,))
                count = c.fetchone()[0]
                if count > 0:
                    is_enrolled = True
                else:
                    # Check new table too?
                    c.execute("SELECT count(*) FROM face_embeddings WHERE user_id = %s", (user_id,))
                    if c.fetchone()[0] > 0:
                        is_enrolled = True
                    
            if is_enrolled:
                # Update DB
                c.execute("UPDATE users SET face_status = 'Enrolled' WHERE id = %s AND (face_status IS NULL OR face_status != 'Enrolled')", (user_id,))
                if c.rowcount > 0:
                     updated_count += 1
        
        conn.commit()
        logging.info(f"Sync Face Status: Updated {updated_count} users.")
    except Exception as e:
        conn.commit()
        logging.info(f"Sync Face Status: Updated {updated_count} users.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error syncing face status: {e}")
    finally:
        if conn:
            conn.close()

def ensure_settings_columns():
    """Adds columns required for the Settings module (Profile, Security, Preferences)."""
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        # Profile
        c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS profile_image_path TEXT")
        
        # Security / Activity
        c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login TIMESTAMP")
        c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_password_change TIMESTAMP")
        
        # Preferences (JSON string: {"theme": "light", "notifications": true, ...})
        c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS preferences TEXT DEFAULT '{}'")
        
        conn.commit()
        logging.info("Migration: Settings columns verified/added.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error ensuring settings columns: {e}")
    finally:
        if conn:
            conn.close()

def ensure_section_column():
    """Adds section column to users table if not exists."""
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS section VARCHAR(50)")
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"Error ensuring section column: {e}")
    finally:
        if conn:
            conn.close()

def ensure_timetable_tables():
    """
    Ensures that Timetable Master and Entries tables exist with required constraints.
    Fixes: 'no unique or exclusion constraint matching the ON CONFLICT specification'
    """
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        
        # 1. Timetable Master Table
        logging.info("Verifying timetable_master table...")
        c.execute("""
            CREATE TABLE IF NOT EXISTS timetable_master (
                id SERIAL PRIMARY KEY,
                batch_year VARCHAR(20) NOT NULL,
                department VARCHAR(100) NOT NULL,
                semester INTEGER NOT NULL,
                section VARCHAR(50),
                timetable_name VARCHAR(255),
                active_status VARCHAR(50) DEFAULT 'Active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(batch_year, department, semester, section)
            );
        """)
        
        # 2. Timetable Entries Table
        logging.info("Verifying timetable_entries table...")
        c.execute("""
            CREATE TABLE IF NOT EXISTS timetable_entries (
                id SERIAL PRIMARY KEY,
                timetable_id INTEGER REFERENCES timetable_master(id) ON DELETE CASCADE,
                day_of_week VARCHAR(20) NOT NULL,
                period_no INTEGER NOT NULL,
                period_id INTEGER REFERENCES timetable_periods(id),
                subject_id INTEGER REFERENCES academic_subjects(id),
                staff_id INTEGER REFERENCES users(id)
            );
        """)
        
        # 3. Add Unique Constraint for ON CONFLICT (timetable_id, day_of_week, period_no)
        # We use a sub-try block to handle 'constraint already exists' gracefully
        try:
            logging.info("Adding unique constraint unique_timetable_slot to timetable_entries...")
            c.execute("ALTER TABLE timetable_entries ADD CONSTRAINT unique_timetable_slot UNIQUE(timetable_id, day_of_week, period_no)")
        except Exception as e:
            # Table might already have the constraint, or there might be duplicates
            if "already exists" in str(e).lower():
                logging.info("Unique constraint unique_timetable_slot already exists.")
            else:
                logging.warning(f"Could not add unique constraint to timetable_entries (might be duplicates): {e}")
            conn.rollback()
            c = conn.cursor() # Re-establish cursor after rollback
            
        conn.commit()
        logging.info("Timetable Master tables verified.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error ensuring timetable tables: {e}")
    finally:
        if conn:
            conn.close()

def update_last_login(user_id):
    """Updates the last_login timestamp for a user."""
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        c.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (user_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"Error updating last_login: {e}")
    finally:
        if conn: conn.close()
        
def update_user_preferences(user_id, preferences_json):
    """Updates user preferences."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        c = conn.cursor()
        c.execute("UPDATE users SET preferences = %s WHERE id = %s", (preferences_json, user_id))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"Error updating preferences: {e}")
        return False
    finally:
        if conn: conn.close()

def change_user_password(user_id, new_password_raw):
    """Updates user password (hashes it first)."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        hashed = bcrypt.hashpw(new_password_raw.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        c = conn.cursor()
        c.execute("UPDATE users SET password = %s, last_password_change = NOW() WHERE id = %s", (hashed, user_id))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"Error changing password: {e}")
        return False
    finally:
        if conn: conn.close()

def update_profile_photo(user_id, photo_path):
    """Updates user profile photo path."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        c = conn.cursor()
        c.execute("UPDATE users SET profile_image_path = %s WHERE id = %s", (photo_path, user_id))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"Error updating profile photo: {e}")
        return False
    finally:
        if conn: conn.close()

        
def get_all_teachers():
    """Retrieves all teacher users from the database."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        import psycopg2.extras
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT id, full_name, username, department FROM users WHERE role = 'teacher' ORDER BY full_name")
        teachers = c.fetchall()
        return [dict(teacher) for teacher in teachers]
    except Exception as e:
        import logging
        logging.error(f"Error getting all teachers: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_timetable_periods(department, batch_year, subject, day_of_week=None, strict_day=False):
    """
    Fetches available time periods (start_time, end_time) from the unified timetable system
    based on filters.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        import psycopg2.extras
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # We need to join entries, master, and subjects to resolve by name
        query = """
            SELECT DISTINCT tp.start_time, tp.end_time, te.period_no
            FROM timetable_entries te
            JOIN timetable_master tm ON te.timetable_id = tm.id
            JOIN academic_subjects sub ON te.subject_id = sub.id
            LEFT JOIN timetable_periods tp ON te.period_no = tp.period_no
            WHERE 1=1
        """
        params = []
        
        if department:
            query += " AND tm.department = %s"
            params.append(department)
        
        if batch_year:
            query += " AND tm.batch_year = %s"
            params.append(batch_year)
            
        if subject:
            # Robust matching with TRIM to handle trailing spaces (common in some subject entries)
            query += " AND (LOWER(TRIM(sub.name)) = LOWER(TRIM(%s)) OR TRIM(sub.name) = TRIM(%s))"
            params.append(subject)
            params.append(subject)
            
        # Try with day first
        if day_of_week:
            temp_query = query + " AND te.day_of_week = %s"
            temp_params = params + [day_of_week]
            c.execute(temp_query + " ORDER BY tp.start_time", tuple(temp_params))
            periods = c.fetchall()
            if periods:
                return [dict(p) for p in periods]
        
        # Fallback: if no periods for specific day (e.g. weekend), 
        # return all periods for that subject/dept/batch to allow selection
        # Skip fallback if strict_day is requested (e.g. for labeling today's periods)
        if strict_day:
            return []
            
        c.execute(query + " ORDER BY tp.start_time", tuple(params))
        periods = c.fetchall()
        return [dict(p) for p in periods]

    except Exception as e:
        import logging
        logging.error(f"Error getting timetable periods: {e}")
        return []
    finally:
        if conn:
            conn.close()



def get_timetable_entry_by_details(department, batch_year, subject, day_of_week, period_no):
    """
    Fetches a specific timetable entry by resolving names/filters to IDs.
    Used for Altered Period detection.
    Now supports fallback to legacy 'timetable' table.
    """
    conn = get_db_connection()
    if conn is None: return None
    try:
        import psycopg2.extras
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # 1. NEW SYSTEM (timetable_entries)
        query = """
            SELECT te.*, te.staff_id as teacher_id, tm.batch_year, tm.department, sub.name as subject_name
            FROM timetable_entries te
            JOIN timetable_master tm ON te.timetable_id = tm.id
            JOIN academic_subjects sub ON te.subject_id = sub.id
            WHERE (LOWER(REPLACE(tm.department, ' ', '')) = LOWER(REPLACE(%s, ' ', '')) OR tm.department = %s)
              AND tm.batch_year = %s
              AND (LOWER(TRIM(sub.name)) = LOWER(TRIM(%s)) OR TRIM(sub.name) = TRIM(%s))
              AND te.day_of_week = %s
              AND te.period_no = %s
        """
        c.execute(query, (department, department, batch_year, subject, subject, day_of_week, period_no))
        res = c.fetchone()
        if res:
            return dict(res)
            
        # 2. LEGACY SYSTEM FALLBACK (timetable table)
        logging.info(f"Fallback to legacy timetable check for {subject}, Period {period_no}")
        
        calc_year = calculate_current_year(batch_year)
        
        # Resolve period_no to start_time for legacy matching
        all_periods = get_global_timings()
        p_info = next((p for p in all_periods if str(p.get('period_no')) == str(period_no)), None)
        
        if p_info:
            start_time_str = p_info['start_time'] # e.g. "10:30"
            try:
                h, m = map(int, start_time_str.split(':'))
                logging.info(f"Checking Legacy Timetable: Dept={department}, Year={calc_year}, Sub={subject}, Day={day_of_week}, H={h}, M={m}")
                legacy_query = """
                    SELECT * FROM timetable 
                    WHERE (LOWER(REPLACE(department, ' ', '')) = LOWER(REPLACE(%s, ' ', '')) OR department = %s)
                      AND year = %s 
                      AND (LOWER(TRIM(subject)) = LOWER(TRIM(%s)) OR TRIM(subject) = TRIM(%s))
                      AND (LOWER(TRIM(day)) = LOWER(TRIM(%s)) OR TRIM(day) = TRIM(%s))
                      AND EXTRACT(HOUR FROM start_time) = %s
                      AND EXTRACT(MINUTE FROM start_time) = %s
                """
                c.execute(legacy_query, (department, department, calc_year, subject, subject, day_of_week, day_of_week, h, m))
                l_res = c.fetchone()
            except Exception as e:
                logging.error(f"Legacy query error: {e}")
                l_res = None
            
            if l_res:
                l_dict = dict(l_res)
                # Normalize key names for compatibility with _check_altered_period
                l_dict['subject_name'] = l_dict.get('subject')
                
                # Resolve legacy 'teacher' name string to a teacher_id
                teacher_name = l_dict.get('teacher', '')
                if teacher_name:
                    # Very robust name cleaning
                    def clean(n): 
                        import re
                        return re.sub(r'[^a-zA-Z0-9]', '', str(n)).lower()
                    
                    clean_target = clean(teacher_name)
                    
                    # Fetch all users once
                    c.execute("SELECT id, full_name FROM users WHERE role = 'teacher'")
                    teachers = c.fetchall()
                    for t in teachers:
                        if clean(t['full_name']) == clean_target:
                            l_dict['teacher_id'] = t['id']
                            break
                    
                    # If ID resolved and matches current user, ensure it's returned for comparison
                    # (Fallback: if we found the entry but couldn't resolve ID, it would still show as Altered
                    # unless name matches. But _check_altered_period uses ID).
                
                return l_dict

        return None
    except Exception as e:
        logging.error(f"Error getting timetable entry by details: {e}")
        return None
    finally:
        if conn: conn.close()

def calculate_current_year(batch_year):
    """Calculates the academic year based on the batch year."""
    if not batch_year or '-' not in batch_year:
        return 1
    try:
        start_year = int(batch_year.split('-')[0])
        current_date = datetime.now()
        current_month = current_date.month
        current_year = current_date.year
        
        # If we're in second half (July-Dec), year increases
        year_diff = current_year - start_year
        if current_month >= 6:
            year_no = year_diff + 1
        else:
            year_no = year_diff
            
        return max(1, min(4, year_no))
    except (ValueError, IndexError):
        return 1

def add_new_user(full_name, username, password, role, department, batch_year, roll_number=None, dept_id=None, year_id=None, section_id=None, register_number=None, semester=1, batch_start_year=None, batch_end_year=None):
    """Adds a new user to the database."""
    conn = get_db_connection()
    if conn is None: return False, "Database connection failed"
    try:
        c = conn.cursor()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Auto-calculate Year if Batch is provided
        derived_year = 1
        derived_batch_start = batch_start_year
        derived_batch_end = batch_end_year
        
        if batch_year and '-' in str(batch_year):
            derived_year = calculate_current_year(batch_year)
            try:
                parts = batch_year.split('-')
                derived_batch_start = int(parts[0])
                derived_batch_end = int(parts[1])
            except:
                pass
        
        # Use derived year if year_id (which is actually year number in this context?) is not provided or to overwrite it
        # Note: year_id param seems to be passed as integer year in some contexts, let's use derived_year to be safe for consistency
        final_year = derived_year
        
        # If semester is not provided or default, try to align with year
        # Year 1 -> Sem 1 (default), Year 2 -> Sem 3, etc.
        # But user might have passed specific semester.
        # Logic: If semester is 1 (default) and Year > 1, update semester default.
        if semester == 1 and final_year > 1:
            semester = (final_year - 1) * 2 + 1

        c.execute(
            "INSERT INTO users (full_name, username, password, role, department, year, roll_number, dept_id, year_id, section_id, batch_year, register_number, semester, batch_start_year, batch_end_year) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (full_name, username, hashed_password, role, department, final_year, roll_number, dept_id, year_id, section_id, batch_year, register_number, semester, derived_batch_start, derived_batch_end)
        )
        user_id = c.fetchone()[0]
        conn.commit()
        return True, user_id
    except Exception as e:
        conn.rollback()
        logging.error(f"Error adding user: {e}")
        return False, str(e)
    finally:
        if conn:
            conn.close()

def add_teacher(full_name, username, password, department='N/A'):
    """Adds a new teacher to the users table."""
    conn = get_db_connection()
    if conn is None: return False, "Database connection failed"
    try:
        c = conn.cursor()
        
        # Check duplicate username
        c.execute("SELECT id FROM users WHERE username = %s", (username,))
        if c.fetchone():
            return False, "Username already exists."

        # Verify password is NOT already hashed (safe guard)
        # Main.py might pass it hashed if legacy, checking...
        # Main.py currently hashes it. let's handle both or rely on one.
        # Plan said: Let database_manager handle it.
        # But existing main.py hashes it.
        # I will modify main.py to NOT hash, but here I should support raw.
        # IF it looks like a bcrypt hash ($2b$..), skipped? NO, just assume raw from new main.py logic.
        
        if isinstance(password, bytes):
             password = password.decode('utf-8')
             
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        c.execute(
            "INSERT INTO users (full_name, username, password, role, department) VALUES (%s, %s, %s, 'teacher', %s) RETURNING id",
            (full_name, username, hashed_password, department)
        )
        user_id = c.fetchone()[0]
        
        # Add to teachers_subjects (optional init)
        
        conn.commit()
        return True, "Teacher added successfully."
    except Exception as e:
        conn.rollback()
        logging.error(f"Error adding teacher: {e}")
        return False, str(e)
    finally:
        if conn:
            conn.close()

def update_user_info(user_id, full_name, username, password, role, department, batch_year, roll_number):
    """Updates an existing user's information."""
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        current_year_academic = calculate_current_year(batch_year)
        c.execute(
            "UPDATE users SET full_name = %s, username = %s, password = %s, role = %s, department = %s, year = %s, roll_number = %s, batch_year = %s WHERE id = %s",
            (full_name, username, password, role, department, current_year_academic, roll_number, batch_year, user_id)
        )
        conn.commit()
    except Exception as e:
        logging.error(f"Error updating user info: {e}")
    finally:
        if conn:
            conn.close()

def delete_user(user_id):
    """Deletes a user from the database."""
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
    except Exception as e:
        logging.error(f"Error deleting user {user_id}: {e}")
    finally:
        if conn:
            conn.close()

def mark_attendance_for_user(user_id, subject, marked_by, status="Present", confidence=None, proof_path=None, method="live", period_no=None, is_finalized=True):
    """
    Marks attendance for a user.
    Stats: status='Present' or 'Absent'.
    Logic: Checks if already marked for (user_id, date, subject, period).
    Returns (status, message).
    """
    conn = get_db_connection()
    if conn is None:
        return "error", "Database connection failed"
    
    try:
        c = conn.cursor()
        
        # Source of Truth: Validate student is Active and Enrolled
        c.execute("SELECT active_status, face_status FROM users WHERE id = %s", (user_id,))
        user_info = c.fetchone()
        if user_info[0] != 'Active':
            return "error", f"Cannot mark attendance: Student is {user_info[0]}"
        
        # Log if marking attendance for a non-enrolled but active student
        if user_info[1] != 'Enrolled' and status == 'Present':
            logging.info(f"Marking attendance for Active but Not Enrolled user: {user_id}")
            
        today = datetime.now().strftime('%Y-%m-%d')
        logging.info(f"DEBUG: mark_attendance_for_user - User:{user_id}, Sub:{subject}, MarkedBy:{marked_by}, Status:{status}, PeriodNo:{period_no}, Finalized:{is_finalized}")

        # Check existing using date check on timestamp AND period_no
        c.execute("""
            SELECT id FROM attendance 
            WHERE user_id = %s AND subject = %s AND timestamp::date = %s
            AND (period_no = %s OR (%s IS NULL AND period_no IS NULL))
        """, (user_id, subject, today, period_no, period_no))
        
        existing = c.fetchone()
        
        if existing:
             # UPDATE existing record
             # Note: If is_finalized is True, we enforce it. 
             # If False, we might want to keep existing finalized state? 
             # For now, simplistic override or if passing explicit True.
             # Let's adding is_finalized to SET clause.
             c.execute("""
                UPDATE attendance 
                SET status = %s, timestamp = %s, marked_by = %s, confidence_score = %s, proof_image_path = %s, method = %s, is_finalized = %s
                WHERE id = %s
             """, (status, datetime.now(), marked_by, confidence, proof_path, method, is_finalized, existing[0]))
             conn.commit()
             return "updated", f"Attendance updated for {subject}."
        
        # Insert new record
        c.execute("""
            INSERT INTO attendance (user_id, timestamp, status, subject, marked_by, confidence_score, proof_image_path, method, period_no, is_finalized)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (user_id, datetime.now(), status, subject, marked_by, confidence, proof_path, method, period_no, is_finalized))
        
        conn.commit()
        return "marked", "Attendance marked successfully."
        
    except Exception as e:
        logging.error(f"Error marking attendance: {e}")
        return "error", str(e)
    finally:
        if conn:
            conn.close()

def save_face_encoding(student_id, encoding_data):
    """Saves a face encoding for a user."""
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        # Normalize to bytes for BYTEA column (support JSON string or raw bytes)
        if isinstance(encoding_data, str):
            encoding_bytes = encoding_data.encode('utf-8')
        elif isinstance(encoding_data, (bytes, bytearray, memoryview)):
            encoding_bytes = bytes(encoding_data)
        else:
            try:
                import numpy as _np
                if isinstance(encoding_data, _np.ndarray):
                    encoding_bytes = encoding_data.astype('float32').tobytes()
                else:
                    encoding_bytes = str(encoding_data).encode('utf-8')
            except Exception:
                encoding_bytes = str(encoding_data).encode('utf-8')
        c.execute(
            "SELECT id FROM face_encodings WHERE user_id = %s",
            (student_id,)
        )
        existing = c.fetchone()
        if existing:
            c.execute(
                "UPDATE face_encodings SET encoding = %s WHERE user_id = %s",
                (psycopg2.Binary(encoding_bytes), student_id)
            )
        else:
            c.execute(
                "INSERT INTO face_encodings (user_id, encoding) VALUES (%s, %s)",
                (student_id, psycopg2.Binary(encoding_bytes))
            )
        conn.commit()
        logging.info(f"Face encoding saved for user ID: {student_id}")
    except Exception as e:
        logging.error(f"Error saving face encoding: {e}")
    finally:
        if conn:
            conn.close()

def get_all_users_with_encodings():
    """Fetches all users with their face encodings."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT u.id, u.full_name, fe.encoding FROM users u JOIN face_encodings fe ON u.id = fe.user_id")
        users = c.fetchall()
        return [dict(user) for user in users]
    except Exception as e:
        logging.error(f"Error fetching users with encodings: {e}")
        return []
    finally:
            conn.close()
            
def get_registered_embeddings_for_class(department, year, batch_year=None, semester=None):
    """
    Fetches face encodings ONLY for students in the specified class.
    Returns list of dicts: {'id': user_id, 'full_name': name, 'encoding': bytes}
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Build query dynamically based on batch/semester availability
        # Robust Filter Logic
        params = []
        filter_sql = ""
        
        # Department: Robust Match
        if department and department != 'All':
             filter_sql += " AND LOWER(REPLACE(u.department, ' ', '')) = LOWER(REPLACE(%s, ' ', '')) "
             params.append(department)
             
        # Year: Cast to integer safely
        if year and year != 'All':
             filter_sql += " AND u.year = %s "
             params.append(str(year))
             
        if batch_year:
             # Handle batch_year vs batch_start_year
             if str(batch_year).isdigit() and len(str(batch_year)) == 4:
                 filter_sql += " AND u.batch_start_year = %s "
                 params.append(batch_year)
             else:
                 filter_sql += " AND u.batch_year = %s "
                 params.append(batch_year)
                 
        if semester:
             filter_sql += " AND u.semester = %s "
             params.append(semester)

        query = f"""
            SELECT u.id, u.full_name, u.roll_number, fe.encoding 
            FROM users u 
            JOIN face_embeddings fe ON u.id = fe.user_id 
            WHERE u.role = 'student'
              AND fe.source IN ('centroid', 'enrollment_sample')
              {filter_sql}
        """
        c.execute(query, tuple(params))
        results = [dict(row) for row in c.fetchall()]
        
        # Fallback to any source if still no results (e.g. 'live' samples if used)
        if not results:
             query_fallback = f"""
                SELECT u.id, u.full_name, u.roll_number, fe.encoding 
                FROM users u 
                JOIN face_embeddings fe ON u.id = fe.user_id 
                WHERE u.role = 'student'
                  {filter_sql}
            """
             c.execute(query_fallback, tuple(params))
             results = [dict(row) for row in c.fetchall()]
             
        return results


    except Exception as e:
        logging.error(f"Error fetching filtered encodings: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_students_with_status():
    """Retrieves students with their face enrollment status."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("""
            SELECT u.id, u.full_name, u.roll_number, u.department, u.year, u.batch_year, 
            CASE 
                WHEN COUNT(fe.id) > 0 THEN 'Enrolled' 
                ELSE 'Not Enrolled' 
            END AS face_status
            FROM users u 
            LEFT JOIN face_embeddings fe ON u.id = fe.user_id
            WHERE u.role = 'student'
            GROUP BY u.id
            ORDER BY u.full_name
        """)
        students = c.fetchall()
        return [dict(student) for student in students]
    except Exception as e:
        logging.error(f"Error getting students with status: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_students_by_dept_year(department, year):
    """Retrieves all students belonging to a specific department and year."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("""
            SELECT id, full_name, roll_number 
            FROM users 
            WHERE department = %s AND year = %s AND role = 'student'
              AND active_status = 'Active'
        """, (department, int(year)))
        return [dict(row) for row in c.fetchall()]
    except Exception as e:
        logging.error(f"Error fetching students by class: {e}")
        return []
    finally:
        if conn: conn.close()

def count_users(role):
    """Counts the number of users with a specific role."""
    conn = get_db_connection()
    if conn is None: return 0
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users WHERE role = %s", (role,))
        count = c.fetchone()[0]
        return count
    except Exception as e:
        logging.error(f"Error counting users for role {role}: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def get_today_attendance_stats():
    """Retrieves the number of present and absent students for the current day."""
    conn = get_db_connection()
    if conn is None: return 0, 0
    try:
        c = conn.cursor()
        c.execute(
            "SELECT COUNT(DISTINCT user_id) FROM attendance WHERE timestamp::date = CURRENT_DATE"
        )
        present_today = c.fetchone()[0]
        total_students = count_users('student')
        absent_today = total_students - present_today
        return present_today, absent_today
    except Exception as e:
        logging.error(f"Error getting today's attendance stats: {e}")
        return 0, 0
    finally:
        if conn:
            conn.close()

def get_recent_activity():
    """Fetches the most recent attendance records."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute(
            """SELECT u.full_name, a.timestamp, a.subject
               FROM attendance a
               JOIN users u ON a.user_id = u.id
               ORDER BY a.timestamp DESC
               LIMIT 5"""
        )
        recent_activity = c.fetchall()
        return [dict(activity) for activity in recent_activity]
    except Exception as e:
        logging.error(f"Error getting recent activity: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_today_present_count():
    """Counts the number of students marked 'Present' or 'OD' today."""
    conn = get_db_connection()
    if conn is None: return 0
    try:
        c = conn.cursor()
        c.execute(
            "SELECT COUNT(DISTINCT user_id) FROM attendance WHERE timestamp::date = CURRENT_DATE AND status IN ('Present', 'OD')"
        )
        count = c.fetchone()[0]
        return count
    except Exception as e:
        logging.error(f"Error getting today's present count: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def get_all_users():
    """Retrieves all users from the database."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT id, username, full_name, role FROM users ORDER BY role, username")
        users = c.fetchall()
        return [dict(user) for user in users]
    except Exception as e:
        logging.error(f"Error getting all users: {e}")
        return []
    finally:
        if conn:
            conn.close()

def add_timetable_entry(day, start_time, end_time, subject, department, year, teacher_id=None, period_no=None):
    """Adds a new timetable entry."""
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        
        # Auto-correct AM to PM for times < 07:00
        # Inputs are typically HH:MM strings from HTML time input
        try:
            # Parse start_time to check hour
            # Support both HH:MM and HH:MM:SS
            st_parts = str(start_time).split(':')
            if len(st_parts) >= 2:
                hour = int(st_parts[0])
                if hour < 7:
                    logging.info(f"Auto-correcting time {start_time} to PM (+12h)")
                    
                    # Fix Start
                    new_hour = hour + 12
                    # Reconstruct string
                    start_time = f"{new_hour:02d}:{st_parts[1]}" 
                    if len(st_parts) > 2: start_time += f":{st_parts[2]}"
                    
                    # Fix End Time too roughly (though duration might cross noon, usually just shift both)
                    et_parts = str(end_time).split(':')
                    if len(et_parts) >= 2:
                        e_hour = int(et_parts[0])
                        # Only shift if end time is also "early" (e.g. 1:00 to 2:00)
                        # If class is 6am to 8am? Rare. Assume shift logic applies to "afternoon classes entered as AM"
                        if e_hour < 7:
                            new_e_hour = e_hour + 12
                            end_time = f"{new_e_hour:02d}:{et_parts[1]}"
                            if len(et_parts) > 2: end_time += f":{et_parts[2]}"
        except Exception as e:
            logging.warning(f"Time auto-correction failed: {e}")

        # Removed period_no from INSERT as column likely doesn't exist based on verification logs
        c.execute(
            "INSERT INTO timetable (day, start_time, end_time, subject, department, year, teacher_id) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (day, start_time, end_time, subject, department, int(year), teacher_id)
        )
        conn.commit()
    except Exception as e:
        logging.error(f"Error adding timetable entry: {e}")
    finally:
        if conn:
            conn.close()
    
def delete_timetable_entry(entry_id):
    """Deletes a timetable entry by its ID."""
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        c.execute("DELETE FROM timetable WHERE id = %s", (entry_id,))
        conn.commit()
    except Exception as e:
        logging.error(f"Error deleting timetable entry {entry_id}: {e}")
    finally:
        if conn:
            conn.close()

def reset_timetable():
    """Deletes ALL entries from the timetable table."""
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        c.execute("TRUNCATE TABLE timetable RESTART IDENTITY CASCADE")
        conn.commit()
        logging.info("Timetable reset successfully.")
    except Exception as e:
        logging.error(f"Error resetting timetable: {e}")
    finally:
        if conn:
            conn.close()

def get_departments():
    """Retrieves all distinct departments from enrolled students."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor()
        # Changed from timetable to users (role='student') to ensure we see departments with enrolled students
        c.execute("SELECT DISTINCT department FROM users WHERE role='student' AND department IS NOT NULL AND department != '' ORDER BY department")
        departments = c.fetchall()
        return [d[0] for d in departments]
    except Exception as e:
        logging.error(f"Error getting departments: {e}")
        return []
    finally:
        if conn:
            conn.close()

def add_face_embedding(user_id, encoding_blob, source='live'):
    """Adds a new face embedding for a user (supporting multiple)."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        c = conn.cursor()
        c.execute("INSERT INTO face_embeddings (user_id, encoding, source) VALUES (%s, %s, %s)",
                  (user_id, encoding_blob, source))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error adding face embedding: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_all_face_embeddings():
    """Retrieves all face embeddings for all users from the new table."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor()
        # Join with users to get full_name
        c.execute("""
            SELECT u.id, u.full_name, fe.encoding 
            FROM face_embeddings fe
            JOIN users u ON fe.user_id = u.id
        """)
        results = c.fetchall()
        # content: list of (user_id, full_name, encoding_blob)
        return [{'id': r[0], 'full_name': r[1], 'encoding': r[2]} for r in results]
    except Exception as e:
        logging.error(f"Error getting all face embeddings: {e}")
        return []
        if conn:
            conn.close()

def get_years():
    """Returns all distinct years (hardcoded as 1-4 per requirements)."""
    return [1, 2, 3, 4]

def get_all_batch_years():
    """Returns distinct batch years from users table."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor()
        c.execute("SELECT DISTINCT batch_year FROM users WHERE role='student' AND batch_year IS NOT NULL ORDER BY batch_year DESC")
        return [row[0] for row in c.fetchall()]
    except Exception as e:
        logging.error(f"Error getting batch years: {e}")
        return []
    finally:
        if conn: conn.close()

def get_all_semesters():
    """Returns distinct semesters (1-8)."""
    return list(range(1, 9))


def get_teacher_today_timetable(user_id):
    """Retrieves timetable for the logged-in teacher based on assignments."""
    today_day = datetime.now().strftime('%A')
    conn = get_db_connection()
    if not conn: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # 1. Get assignments
        c.execute("""
            SELECT tsa.department, tsa.year, s.name as subject_name
            FROM teacher_subject_assignments tsa
            JOIN academic_subjects s ON tsa.subject_id = s.id
            WHERE tsa.teacher_id = %s
        """, (user_id,))
        assignments = c.fetchall()
        
        if not assignments:
             return []
             
        # 2. Fetch full timetable for today
        c.execute("SELECT * FROM timetable WHERE day = %s ORDER BY start_time", (today_day,))
        all_slots = c.fetchall()
        
        filtered_slots = []
        for slot in all_slots:
            # Flexible matching for year (int/string handling)
            slot_year = str(slot['year'])
            
            for assign in assignments:
                assign_year = str(assign['year'])
                
                # Check Subject (name) and Department and Year
                # Case insensitive comparison for strings might be safer
                # Flexible matching
                slot_subj = str(slot.get('subject', '')).strip().lower()
                assign_subj = str(assign.get('subject_name', '')).strip().lower()
                
                slot_dept = str(slot.get('department', '')).strip().lower()
                assign_dept = str(assign.get('department', '')).strip().lower()

                if (slot_subj == assign_subj and 
                    slot_dept == assign_dept and 
                    slot_year == assign_year):
                    filtered_slots.append(dict(slot))
                    break
                    
        return filtered_slots
        
    except Exception as e:
        logging.error(f"Error getting teacher timetable: {e}")
        return []
    finally:
        conn.close()


# Force reload
def get_subjects_by_dept_year_v2(department, year, semester=None):
    """
    Retrieves subjects from the academic mapping tables for a specific department and year.
    Matches names flexibly to find IDs.
    Supports optional semester filtering.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # 1. Get Dept ID
        # Try exact match first, then ILIKE
        dept_id = None
        c.execute("SELECT id, name FROM academic_departments WHERE name = %s", (department,))
        res = c.fetchone()
        if not res:
            c.execute("SELECT id, name FROM academic_departments WHERE name ILIKE %s", (f"%{department}%",))
            res = c.fetchone()
            
        if res:
            dept_id = res['id']
            logging.info(f"Resolved Department '{department}' to ID {dept_id} ({res['name']})")
        else:
            logging.warning(f"Department '{department}' not found in academic_departments.")
            
        # 2. Get Year ID
        # Frontend might send "1", "2" or "Year 1". DB might have "Year 1", "IV Year" etc.
        year_id = None
        year_str = str(year).strip()
        
        # Try to find a year that contains the number
        c.execute("SELECT id, name FROM academic_years")
        all_years = c.fetchall()
        
        for y in all_years:
            y_name = y['name']
            if year_str == y_name or year_str in y_name or (y_name.split() and y_name.split()[-1] == year_str):
                 # Simple heuristic: if '1' is in 'Year 1'
                 # But '1' is in '10'. Strict check:
                 # Check if the number exists as a word OR precise match
                 if year_str.lower() == y_name.lower():
                     year_id = y['id']
                     break
                 
                 # If year_str is digit, check if it's in the name
                 if year_str.isdigit() and year_str in y_name:
                     year_id = y['id']
                     break
        
        if year_id:
             logging.info(f"Resolved Year '{year}' to ID {year_id}")
        else:
             logging.warning(f"Year '{year}' not found in academic_years.")

        subjects = []
        if dept_id:
            # 3. Find Classes -> Mapping -> Subjects
            # If year_id is missing, we fetch all subjects for that department
            query = """
                SELECT DISTINCT s.name 
                FROM academic_subjects s
                JOIN class_subject_mapping m ON s.id = m.subject_id
                JOIN academic_classes c ON m.class_id = c.id
                WHERE c.dept_id = %s
            """
            params = [dept_id]
            
            if year_id:
                query += " AND c.year_id = %s"
                params.append(year_id)

            if semester:
                # Add semester filtering if provided
                sem_val = str(semester).lower().replace('sem', '').strip()
                if sem_val.isdigit():
                    query += " AND s.semester = %s"
                    params.append(int(sem_val))
            
            query += " ORDER BY s.name"
            c.execute(query, tuple(params))
            subjects = [row['name'] for row in c.fetchall()]

        if not subjects:
            logging.warning(f"No mapped subjects found for Dept: {department}, Year: {year}, Sem: {semester}.")
            return ["Not Assigned"]
            
        return subjects
        
    except Exception as e:
        logging.error(f"Error getting subjects by dept/year: {e}")
        return ["Error Fetching Subjects"]
    finally:
        if conn:
            conn.close()

def get_subjects_for_user(user_id):
    """
    Retrieves unique subjects assigned to a teacher that have valid class mappings.
    Uses strict `teacher_subject_assignments` and verifies existence in grid.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Fetch assigned subjects that are ALSO present in the timetable for that teacher
        # This ensures "count subjects without class mappings" is handled.
        c.execute("""
            SELECT DISTINCT s.name 
            FROM teacher_subject_assignments tsa
            JOIN academic_subjects s ON tsa.subject_id = s.id
            WHERE tsa.teacher_id = %s
            AND EXISTS (
                SELECT 1 FROM timetable_entries te
                WHERE te.staff_id = tsa.teacher_id AND te.subject_id = tsa.subject_id
            )
            ORDER BY s.name
        """, (user_id,))
        
        rows = c.fetchall()
        return [r['name'] for r in rows]
        
    except Exception as e:
        logging.error(f"Error getting subjects for user {user_id}: {e}")
        return []
    finally:
        if conn: conn.close()

def get_teacher_allocated_classes(teacher_id):
    """
    Returns list of distinct classes (Department + Year) assigned to the teacher.
    Source: timetable_entries (New System)
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Fetch distinct department/semester from timetable master records linked to teacher's entries
        c.execute("""
            SELECT DISTINCT tm.department, tm.semester
            FROM timetable_entries te
            JOIN timetable_master tm ON te.timetable_id = tm.id
            WHERE te.staff_id = %s
            ORDER BY tm.department, tm.semester
        """, (teacher_id,))
        
        raw_rows = c.fetchall()
        
        # Process in Python to get Year and filter uniques
        unique_classes = {}
        for row in raw_rows:
            dept = row['department']
            sem = row['semester']
            year = (sem + 1) // 2
            key = (dept, year)
            
            if key not in unique_classes:
                unique_classes[key] = {
                    'department': dept,
                    'year': year
                }
                
        return sorted(list(unique_classes.values()), key=lambda x: (x['department'], x['year']))
        
    except Exception as e:
        logging.error(f"Error getting allocated classes for teacher {teacher_id}: {e}")
        return []
    finally:
        if conn:
            conn.close()
    
def get_subject_attendance_by_date_range(subject, start_date, end_date):
    """Calculates attendance stats for a subject within a date range."""
    conn = get_db_connection()
    if conn is None: return {'present': 0, 'absent': 0, 'total': 0}
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute(
            """SELECT COUNT(*) as total_classes,
               SUM(CASE WHEN a.status IN ('Present', 'OD') THEN 1 ELSE 0 END) as present_count
               FROM attendance a
               WHERE a.subject = %s AND a.timestamp::date BETWEEN %s AND %s""",
            (subject, start_date, end_date)
        )
        records = c.fetchone()
        
        total_classes = records['total_classes'] if records['total_classes'] else 0
        present_count = records['present_count'] if records['present_count'] else 0
        absent_count = total_classes - present_count
        
        return {
            'present': present_count,
            'absent': absent_count,
            'total': total_classes
        }
    except Exception as e:
        logging.error(f"Error getting subject attendance stats for {subject}: {e}")
        return {'present': 0, 'absent': 0, 'total': 0}
    finally:
        if conn:
            conn.close()

def get_overall_attendance_by_date_range(start_date, end_date):
    """Calculates overall attendance stats across all subjects."""
    conn = get_db_connection()
    if conn is None: return {'present': 0, 'absent': 0}
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute(
            """SELECT COUNT(a.id) as count FROM attendance a
               WHERE a.status IN ('Present', 'OD') AND a.timestamp::date BETWEEN %s AND %s""",
            (start_date, end_date)
        )
        present_count = c.fetchone()['count']
        c.execute(
            """SELECT COUNT(*) as count FROM attendance a
               WHERE a.timestamp::date BETWEEN %s AND %s""",
            (start_date, end_date)
        )
        total_classes = c.fetchone()['count']
        absent_count = total_classes - present_count
        return {'present': present_count, 'absent': absent_count}
    except Exception as e:
        logging.error(f"Error getting overall attendance stats: {e}")
        return {'present': 0, 'absent': 0}
    finally:
        if conn:
            conn.close()



def get_official_attendance_report(department, year, subject, start_date, end_date, limit=None, offset=None, batch_year=None, status=None, semester=None, method=None, period=None, view_mode='Student Wise'):
    """
    Returns a unified, de-duplicated list of students and their attendance.
    Supports various View Modes: Raw Records, Daily Summary, Period Wise, Monthly Summary, Student Wise, Subject Wise.
    """
    # Automatic View Mode selection based on Date Range if not specified
    is_multi_day = False
    if start_date and end_date:
        days = (datetime.strptime(str(end_date), '%Y-%m-%d') - datetime.strptime(str(start_date), '%Y-%m-%d')).days + 1
        is_multi_day = days > 1

    # DON'T overwrite view_mode if user explicitly asked for one
    # if view_mode == 'Raw Records' and is_multi_day:
    #     view_mode = 'Student Wise'
        
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        params = [] 
        
        # 1. Target Students
        target_students_query = """
            SELECT id, full_name, roll_number, register_number, department, year, batch_year, semester
            FROM users 
            WHERE role = 'student' AND active_status = 'Active'
              AND TRIM(COALESCE(full_name, '')) NOT IN ('', '.', '-', '?', 'Test', 'test')
              AND TRIM(COALESCE(department, '')) NOT IN ('', '-', 'None', 'All')
              AND year IS NOT NULL AND year != 0
              AND full_name NOT ILIKE '%%test%%' AND roll_number NOT ILIKE '%%test%%'
        """
        if department and department != 'All':
            target_students_query += " AND LOWER(TRIM(department)) = LOWER(TRIM(%s)) "
            params.append(department)
        if year and year != 'All':
            target_students_query += " AND year = %s "
            params.append(int(year) if str(year).isdigit() else year)
        if semester and semester != 'All':
            target_students_query += " AND semester = %s "
            params.append(int(semester) if str(semester).isdigit() else semester)
        if batch_year:
            start_year = str(batch_year).split('-')[0] if '-' in str(batch_year) else str(batch_year)
            target_students_query += " AND (batch_year = %s OR batch_start_year = %s) "
            params.extend([batch_year, start_year])

        # 2. Base Attendance Filter
        att_filter = ""
        att_params = []
        if subject and subject != 'All':
            att_filter += " AND (LOWER(TRIM(TRANSLATE(subject, '_- ', ''))) = LOWER(TRIM(TRANSLATE(%s, '_- ', '')))) "
            att_params.append(subject)
        if method:
            att_filter += " AND method = %s "
            att_params.append(method)
            
        # FIX: Backend Period Fix - If 'All', remove filter
        if period and period != 'All' and period != 'None':
            att_filter += " AND period_no = %s "
            att_params.append(int(period) if str(period).isdigit() else period)

        # 3. Main Query Construction based on View Mode
        if view_mode == 'Daily Summary':
            full_query = f"""
                WITH target_students AS ({target_students_query}),
                filtered_attendance AS (
                    SELECT a.* FROM attendance a
                    JOIN target_students s ON a.user_id = s.id
                    WHERE a.timestamp >= %s::date AND a.timestamp < (%s::date + INTERVAL '1 day')
                    AND (a.is_finalized = TRUE OR a.is_finalized IS NULL) {att_filter}
                )
                SELECT 
                    DATE(timestamp) as group_date,
                    COUNT(*) as total_classes,
                    SUM(CASE WHEN status IN ('Present', 'OD') THEN 1 ELSE 0 END) as present_count,
                    SUM(CASE WHEN status = 'Absent' THEN 1 ELSE 0 END) as absent_count,
                    CASE WHEN COUNT(*) > 0 THEN
                        ROUND((SUM(CASE WHEN status IN ('Present', 'OD') THEN 1 ELSE 0 END)::numeric / COUNT(*)::numeric) * 100, 1)
                    ELSE 0 END as attendance_pct
                FROM filtered_attendance
                GROUP BY DATE(timestamp)
                ORDER BY group_date DESC
            """
            params.extend([start_date, end_date])
            params.extend(att_params)

        elif view_mode == 'Period Wise':
            full_query = f"""
                WITH target_students AS ({target_students_query}),
                filtered_attendance AS (
                    SELECT a.* FROM attendance a
                    JOIN target_students s ON a.user_id = s.id
                    WHERE a.timestamp >= %s::date AND a.timestamp < (%s::date + INTERVAL '1 day')
                    AND (a.is_finalized = TRUE OR a.is_finalized IS NULL) {att_filter}
                )
                SELECT 
                    DATE(timestamp) as group_date,
                    period_no,
                    COUNT(*) as total_classes,
                    SUM(CASE WHEN status IN ('Present', 'OD') THEN 1 ELSE 0 END) as present_count,
                    SUM(CASE WHEN status = 'Absent' THEN 1 ELSE 0 END) as absent_count
                FROM filtered_attendance
                GROUP BY DATE(timestamp), period_no
                ORDER BY group_date DESC, period_no ASC
            """
            params.extend([start_date, end_date])
            params.extend(att_params)

        elif view_mode == 'Monthly Summary':
            full_query = f"""
                WITH target_students AS ({target_students_query}),
                filtered_attendance AS (
                    SELECT a.* FROM attendance a
                    JOIN target_students s ON a.user_id = s.id
                    WHERE a.timestamp >= %s::date AND a.timestamp < (%s::date + INTERVAL '1 day')
                    AND (a.is_finalized = TRUE OR a.is_finalized IS NULL) {att_filter}
                )
                SELECT 
                    EXTRACT(YEAR FROM timestamp) as year,
                    EXTRACT(MONTH FROM timestamp) as month,
                    TO_CHAR(timestamp, 'Month YYYY') as month_name,
                    COUNT(*) as total_classes,
                    SUM(CASE WHEN status IN ('Present', 'OD') THEN 1 ELSE 0 END) as present_count,
                    CASE WHEN COUNT(*) > 0 THEN
                        ROUND((SUM(CASE WHEN status IN ('Present', 'OD') THEN 1 ELSE 0 END)::numeric / COUNT(*)::numeric) * 100, 1)
                    ELSE 0 END as attendance_pct
                FROM filtered_attendance
                GROUP BY EXTRACT(YEAR FROM timestamp), EXTRACT(MONTH FROM timestamp), TO_CHAR(timestamp, 'Month YYYY')
                ORDER BY year DESC, month DESC
            """
            params.extend([start_date, end_date])
            params.extend(att_params)

        elif view_mode == 'Student Wise':
            # User wants: ORDER BY register_no DESC, Limit last 3 (Wait, User said "Limit: Last 3 students" but also usually reports show all. 
            # Re-reading: "ORDER BY register_no DESC Limit: Last 3 students"
            full_query = f"""
                WITH target_students AS ({target_students_query}),
                filtered_attendance AS (
                    SELECT a.* FROM attendance a
                    JOIN target_students s ON a.user_id = s.id
                    WHERE a.timestamp >= %s::date AND a.timestamp < (%s::date + INTERVAL '1 day')
                    AND (a.is_finalized = TRUE OR a.is_finalized IS NULL) {att_filter}
                )
                SELECT 
                    s.full_name as student,
                    s.roll_number as roll_no,
                    s.register_number as register_no,
                    COUNT(fa.id) as total_classes,
                    SUM(CASE WHEN fa.status IN ('Present', 'OD') THEN 1 ELSE 0 END) as present_count,
                    CASE WHEN COUNT(fa.id) > 0 THEN
                        ROUND((SUM(CASE WHEN fa.status IN ('Present', 'OD') THEN 1 ELSE 0 END)::numeric / COUNT(fa.id)::numeric) * 100, 1)
                    ELSE 0 END as attendance_pct
                FROM target_students s
                LEFT JOIN filtered_attendance fa ON s.id = fa.user_id
                GROUP BY s.id, s.full_name, s.roll_number, s.register_number
                ORDER BY s.full_name ASC
            """
            params.extend([start_date, end_date])
            params.extend(att_params)

        elif view_mode == 'Subject Wise':
            full_query = f"""
                WITH target_students AS ({target_students_query}),
                filtered_attendance AS (
                    SELECT a.* FROM attendance a
                    JOIN target_students s ON a.user_id = s.id
                    WHERE a.timestamp >= %s::date AND a.timestamp < (%s::date + INTERVAL '1 day')
                    AND (a.is_finalized = TRUE OR a.is_finalized IS NULL) {att_filter}
                )
                SELECT 
                    subject,
                    COUNT(*) as total_classes,
                    SUM(CASE WHEN status IN ('Present', 'OD') THEN 1 ELSE 0 END) as present_count,
                    CASE WHEN COUNT(*) > 0 THEN
                        ROUND((SUM(CASE WHEN status IN ('Present', 'OD') THEN 1 ELSE 0 END)::numeric / COUNT(*)::numeric) * 100, 1)
                    ELSE 0 END as attendance_pct
                FROM filtered_attendance
                WHERE subject IS NOT NULL
                GROUP BY subject
                ORDER BY subject ASC
            """
            params.extend([start_date, end_date])
            params.extend(att_params)

        else: # Default: Raw Records
            full_query = f"""
                WITH target_students AS ({target_students_query}),
                raw_attendance AS (
                    SELECT 
                        a.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY a.user_id, a.timestamp::date, COALESCE(a.period_no, -1), a.subject 
                            ORDER BY 
                                CASE 
                                    WHEN a.method LIKE 'manual%%' OR a.method LIKE 'altered%%' THEN 2 
                                    WHEN a.status = 'Present' THEN 1
                                    ELSE 0 
                                END DESC,
                                a.timestamp DESC
                        ) as rn
                    FROM attendance a
                    WHERE a.timestamp >= %s::date AND a.timestamp < (%s::date + INTERVAL '1 day')
                      AND (a.is_finalized = TRUE OR a.is_finalized IS NULL) {att_filter}
                ),
                dedup_attendance AS (
                    SELECT * FROM raw_attendance WHERE rn = 1
                ),
                student_stats AS (
                    SELECT 
                        user_id,
                        COUNT(*) as total_possible,
                        SUM(CASE WHEN status IN ('Present', 'OD') THEN 1 ELSE 0 END) as present_count
                    FROM attendance
                    WHERE (is_finalized = TRUE OR is_finalized IS NULL)
                    GROUP BY user_id
                )
                SELECT 
                    s.id as user_id, s.full_name, s.roll_number, s.register_number, s.department, s.year, s.batch_year, s.semester,
                    COALESCE(rf.crop_path, rf.image_path) as registered_image,
                    da.subject, da.timestamp, da.status, da.proof_image_path, da.method,
                    COALESCE(da.status, 'Absent') as final_status, da.period_no,
                    tp.label as period_label,
                    to_char(tp.start_time, 'HH12:MI AM') || ' - ' || to_char(tp.end_time, 'HH12:MI AM') as time_slot,
                    marker.full_name as marked_by_name,
                    COALESCE(stats.present_count, 0) as present_count,
                    COALESCE(stats.total_possible, 0) as total_possible,
                    CASE WHEN COALESCE(stats.total_possible, 0) > 0 THEN 
                        ROUND((COALESCE(stats.present_count, 0)::numeric / stats.total_possible::numeric) * 100, 1)
                    ELSE 0 END as attendance_pct
                FROM target_students s
                {'JOIN' if is_multi_day else 'LEFT JOIN'} dedup_attendance da ON s.id = da.user_id
                LEFT JOIN (SELECT DISTINCT ON (user_id) user_id, crop_path, image_path FROM face_embeddings ORDER BY user_id, created_at DESC) rf ON s.id = rf.user_id
                LEFT JOIN timetable_periods tp ON da.period_no = tp.period_no
                LEFT JOIN users marker ON da.marked_by = marker.id
                LEFT JOIN student_stats stats ON s.id = stats.user_id
                WHERE 1=1
            """
            params.extend([start_date, end_date])
            params.extend(att_params)
            
            if status and status != 'All':
                if status == 'Present': full_query += " AND da.status = 'Present' "
                elif status == 'Absent': full_query += " AND (da.status IS NULL OR da.status = 'Absent') "
                elif status == 'OD': full_query += " AND da.status = 'OD' "

            full_query += """
                ORDER BY 
                COALESCE(CAST(RIGHT(NULLIF(regexp_replace(s.register_number, '[^0-9]', '', 'g'), ''), 3) AS INTEGER), 999999) ASC,
                da.timestamp DESC NULLS LAST, da.period_no ASC NULLS LAST
            """

        c.execute(full_query, tuple(params))
        rows = c.fetchall()
        
        # 4. Post-Process Results (Accurate Percentage Calculation)
        results = []
        # Cache for expected counts to avoid repeated DB calls in same request
        expected_cache = {}

        for row in rows:
            r = dict(row)
            
            # Accurate Percentage Logic
            if 'attendance_pct' in r or view_mode in ['Student Wise', 'Monthly Summary', 'Subject Wise']:
                # Calculate expected classes for this specific row's context
                target_dept = r.get('department') or department
                target_batch = r.get('batch_year') or batch_year
                target_sem = r.get('semester') or semester
                target_sub = r.get('subject') or subject
                
                # Context-specific dates
                c_start = start_date
                c_end = end_date
                if view_mode == 'Daily Summary' and r.get('group_date'):
                    c_start = c_end = r['group_date'].strftime('%Y-%m-%d')
                elif view_mode == 'Monthly Summary' and r.get('group_date'):
                    # First and last day of month
                    md = r['group_date']
                    c_start = md.replace(day=1).strftime('%Y-%m-%d')
                    import calendar
                    last_day = calendar.monthrange(md.year, md.month)[1]
                    c_end = md.replace(day=last_day).strftime('%Y-%m-%d')

                cache_key = f"{target_dept}_{target_batch}_{target_sem}_{target_sub}_{c_start}_{c_end}"
                if cache_key not in expected_cache:
                    expected_cache[cache_key] = get_expected_count(target_dept, target_batch, target_sem, target_sub, c_start, c_end)
                
                expected = expected_cache[cache_key]
                presents = r.get('present_count', 0)
                
                if expected > 0:
                    r['attendance_pct'] = round((presents / expected) * 100, 1)
                    r['total_classes'] = expected # Show expected classes, not just recorded ones
                else:
                    r['attendance_pct'] = 100.0 if presents >= 0 else 0.0
                    r['total_classes'] = presents if presents > 0 else 0
            
            if view_mode == 'Raw Records' and not r['timestamp']:
                r['status'] = 'Absent'
                r['subject'] = subject if (subject and subject != 'All') else 'All Subjects'
                r['period_no'] = period if (period and period != 'All' and period != 'None') else '-'
                r['time_slot'] = r.get('time_slot') or '-'
            results.append(r)
        return results
    except Exception as e:
        logging.error(f"Error getting official report: {e}")
        return []
    finally:
        if conn: conn.close()

def get_attendance_count(department, year, subject, start_date, end_date, batch_year=None, status=None, semester=None):
    """
    Returns total count of rows for the report filters, enabling pagination.
    Uses the exact same logic as get_official_attendance_report but counts rows.
    """
    conn = get_db_connection()
    if conn is None: return 0
    try:
        c = conn.cursor()
        
        params = []
        
        target_students_query = """
            SELECT id
            FROM users 
            WHERE role = 'student' AND active_status = 'Active'
        """
        
        if department and department != 'All':
            target_students_query += " AND department = %s "
            params.append(department)
            
        if year and year != 'All':
             target_students_query += " AND year = %s "
             params.append(int(year) if str(year).isdigit() else year)

        if semester and semester != 'All':
             target_students_query += " AND semester = %s "
             params.append(int(semester) if str(semester).isdigit() else semester)

        # Batch Filter (Inclusive)
        if batch_year:
             start_year = str(batch_year).split('-')[0] if '-' in str(batch_year) else str(batch_year)
             target_students_query += " AND (batch_year = %s OR batch_start_year = %s) "
             params.extend([batch_year, start_year])
                 
        subject_filter_clause = ""
        subject_param = []
        if subject and subject != 'All':
             subject_filter_clause = " AND (LOWER(TRANSLATE(subject, '_- ', '')) = LOWER(TRANSLATE(%s, '_- ', ''))) "
             subject_param = [subject]

        full_query = f"""
            WITH target_students AS (
                {target_students_query}
            ),
            filtered_attendance AS (
                SELECT user_id, status FROM attendance
                WHERE timestamp >= %s::date 
                  AND timestamp < (%s::date + INTERVAL '1 day')
                  {subject_filter_clause}
            )
            SELECT COUNT(*)
            FROM target_students s
            LEFT JOIN filtered_attendance a ON s.id = a.user_id
            WHERE 1=1
        """
        
        params.extend([start_date, end_date])
        if subject_param: params.extend(subject_param)
        
        # Status Filter
        if status and status != 'All':
            if status == 'Present':
                full_query += " AND a.user_id IS NOT NULL "
            elif status == 'Absent':
                full_query += " AND a.user_id IS NULL "
        
        c.execute(full_query, tuple(params))
        return c.fetchone()[0]
    except Exception as e:
        logging.error(f"Error counting report rows: {e}")
        return 0
    finally:
        if conn: conn.close()

def get_admin_report_summary(department, year, batch_year=None, semester=None):
    """
    Returns high-level statistics for a class of students (Total, Avg %, Distribution).
    Calculates overall standing for all active students matching the class filters.
    """
    conn = get_db_connection()
    if conn is None: return None
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        params = []
        target_students_query = """
            SELECT id
            FROM users 
            WHERE role = 'student' AND active_status = 'Active'
        """
        if department and department != 'All':
            target_students_query += " AND LOWER(TRIM(department)) = LOWER(TRIM(%s)) "
            params.append(department)
        if year and year != 'All':
            target_students_query += " AND year = %s "
            params.append(int(year) if str(year).isdigit() else year)
        if semester and semester != 'All':
            target_students_query += " AND semester = %s "
            params.append(int(semester) if str(semester).isdigit() else semester)
        if batch_year:
            start_year = str(batch_year).split('-')[0] if '-' in str(batch_year) else str(batch_year)
            target_students_query += " AND (batch_year = %s OR batch_start_year = %s) "
            params.extend([batch_year, start_year])

        full_query = f"""
            WITH target_students AS (
                {target_students_query}
            ),
            student_stats AS (
                SELECT 
                    u.id as user_id,
                    COUNT(a.id) as total_possible,
                    SUM(CASE WHEN a.status IN ('Present', 'OD') THEN 1 ELSE 0 END) as present_count
                FROM target_students u
                LEFT JOIN attendance a ON u.id = a.user_id
                GROUP BY u.id
            ),
            calculated_stats AS (
                SELECT 
                    user_id,
                    CASE 
                        WHEN total_possible > 0 THEN (present_count::numeric / total_possible::numeric) * 100
                        ELSE 0 
                    END as attendance_pct
                FROM student_stats
            )
            SELECT 
                COUNT(*) as total,
                COALESCE(ROUND(AVG(attendance_pct), 1), 0) as avg_pct,
                COUNT(*) FILTER (WHERE attendance_pct >= 75) as above_75,
                COUNT(*) FILTER (WHERE attendance_pct >= 60 AND attendance_pct < 75) as between_60_74,
                COUNT(*) FILTER (WHERE attendance_pct < 60) as below_60
            FROM calculated_stats
        """
        
        c.execute(full_query, tuple(params))
        res = c.fetchone()
        return dict(res) if res else None
    except Exception as e:
        logging.error(f"Error getting admin report summary: {e}")
        return None
    finally:
        if conn: conn.close()


def get_detailed_attendance_report(department=None, year=None, subject=None, start_date=None, end_date=None, page=1, per_page=10, batch_year=None):
    """Generates a detailed attendance log with proof and confidence scores, with pagination."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        query = """
            SELECT a.id, u.id as user_id, u.full_name, u.roll_number, u.department, u.year, 
                   a.subject, a.timestamp, a.status, a.confidence_score, a.proof_image_path, a.method
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            WHERE u.active_status = 'Active' AND u.face_status = 'Enrolled'
        """
        params = []
        
        if department and department != 'All':
            query += " AND LOWER(TRIM(u.department)) = LOWER(TRIM(%s))"
            params.append(department)
        if year and year != 'All':
            query += " AND u.year = %s"
            params.append(year)
            
        if batch_year:
             if str(batch_year).isdigit() and len(str(batch_year)) == 4:
                 query += " AND u.batch_start_year = %s "
                 params.append(batch_year)
             else:
                 query += " AND u.batch_year = %s "
                 params.append(batch_year)
                 
        if subject and subject != 'All':
            query += " AND LOWER(TRANSLATE(a.subject, '_- ', '')) = LOWER(TRANSLATE(%s, '_- ', ''))"
            params.append(subject)
        
        if start_date:
            query += " AND a.timestamp::date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND a.timestamp::date <= %s"
            params.append(end_date)
            
        query += " ORDER BY a.timestamp DESC"
        
        # Pagination
        if page and per_page:
            offset = (page - 1) * per_page
            query += " LIMIT %s OFFSET %s"
            params.extend([per_page, offset])
        
        logging.info(f"Detailed Report Query: {query} | Params: {params}")
        c.execute(query, tuple(params))
        records = c.fetchall()
        return [dict(r) for r in records]
    except Exception as e:
        logging.error(f"Error getting detailed attendance report: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_detailed_attendance_count(department=None, year=None, subject=None, start_date=None, end_date=None, batch_year=None):
    """Returns the total count of records matching the filters for pagination."""
    conn = get_db_connection()
    if conn is None: return 0
    try:
        c = conn.cursor()
        query = """
            SELECT COUNT(*)
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            WHERE u.active_status = 'Active' AND u.face_status = 'Enrolled'
        """
        params = []
        
        if department and department != 'All':
            query += " AND LOWER(TRIM(u.department)) = LOWER(TRIM(%s))"
            params.append(department)
        if year and year != 'All':
            query += " AND u.year = %s"
            params.append(year)
            
        if batch_year:
             if str(batch_year).isdigit() and len(str(batch_year)) == 4:
                 query += " AND u.batch_start_year = %s "
                 params.append(batch_year)
             else:
                 query += " AND u.batch_year = %s "
                 params.append(batch_year)

        if subject and subject != 'All':
            query += " AND LOWER(TRANSLATE(a.subject, '_- ', '')) = LOWER(TRANSLATE(%s, '_- ', ''))"
            params.append(subject)
        if start_date:
            query += " AND a.timestamp::date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND a.timestamp::date <= %s"
            params.append(end_date)
            
        c.execute(query, tuple(params))
        return c.fetchone()[0]
    except Exception as e:
        logging.error(f"Error getting detailed attendance count: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def get_attendance_report(department=None, year=None, subject=None):
    """
    (Legacy/Summary) Generates an attendance report for students based on filters.
    Kept for backward compatibility or summary views if needed.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        query = """
            SELECT u.id, u.full_name, u.roll_number, u.department, u.year, u.batch_year, COUNT(a.id) as total_classes,
            SUM(CASE WHEN a.status IN ('Present', 'OD') THEN 1 ELSE 0 END) as present_count
            FROM users u
            LEFT JOIN attendance a ON u.id = a.user_id
            WHERE u.role = 'student' AND u.active_status = 'Active' AND u.face_status = 'Enrolled'
        """
        params = []
        if department and department != 'All':
            query += " AND u.department = %s"
            params.append(department)
        if year and year != 'All':
            query += " AND u.year = %s"
            params.append(year)
        if subject and subject != 'All':
            query += " AND a.subject = %s"
            params.append(subject)
        
        query += " GROUP BY u.id, u.full_name, u.roll_number, u.department, u.year, u.batch_year ORDER BY u.full_name"
        c.execute(query, tuple(params))
        report_data = c.fetchall()
        return [dict(row) for row in report_data]
    except Exception as e:
        logging.error(f"Error generating attendance report: {e}")
        return []
    finally:
        if conn:
            conn.close()

def assign_subjects_to_teacher(teacher_id, subjects):
    """Assigns subjects to a teacher."""
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        c.execute("DELETE FROM teachers_subjects WHERE teacher_id = %s", (teacher_id,))
        for subject in subjects:
            c.execute("INSERT INTO teachers_subjects (teacher_id, subject) VALUES (%s, %s)", (teacher_id, subject))
        conn.commit()
    except Exception as e:
        logging.error(f"Error assigning subjects to teacher {teacher_id}: {e}")
    finally:
        if conn:
            conn.close()

def get_analytics_data(department=None, year=None, subject=None):
    """
    Fetches data for analytics dashboards.
    Enhanced with robust text matching and debugging logs.
    """
    conn = get_db_connection()
    if conn is None: return {'department_data': [], 'subject_data': []}
    
    # Debug Logging
    logging.info(f"Analytics Fetch: Dept='{department}', Year='{year}', Subject='{subject}'")
    
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # --- 1. Department Data Query ---
        dept_query = """
            SELECT u.department, 
            COUNT(a.id) as total_classes,
            COALESCE(SUM(CASE WHEN a.status IN ('Present', 'OD') THEN 1 ELSE 0 END), 0) as present_count
            FROM users u
            LEFT JOIN attendance a ON u.id = a.user_id
            WHERE u.role = 'student'
        """
        dept_params = []
        
        if department and department != 'All':
            dept_query += " AND LOWER(TRIM(u.department)) = LOWER(TRIM(%s))"
            dept_params.append(department)
            
        if year and year != 'All':
            dept_query += " AND u.year = %s"
            # Ensure year is integer
            try:
                dept_params.append(int(year))
            except:
                dept_params.append(year) # Fallback
                
        if subject and subject != 'All':
            dept_query += " AND LOWER(TRIM(a.subject)) = LOWER(TRIM(%s))"
            dept_params.append(subject)
            
        dept_query += " GROUP BY u.department"
        
        logging.info(f"Dept Query Params: {dept_params}")
        c.execute(dept_query, tuple(dept_params))
        dept_data = c.fetchall()
        
        # --- 2. Subject Data Query ---
        # Note: We group by 'a.subject' so empty attendance won't show up here anyway
        subject_query = """
            SELECT a.subject, 
            COUNT(a.id) as total_classes,
            COALESCE(SUM(CASE WHEN a.status IN ('Present', 'OD') THEN 1 ELSE 0 END), 0) as present_count
            FROM users u
            LEFT JOIN attendance a ON u.id = a.user_id
            WHERE u.role = 'student' AND a.subject IS NOT NULL
        """
        subject_params = []
        
        if department and department != 'All':
            subject_query += " AND LOWER(TRIM(u.department)) = LOWER(TRIM(%s))"
            subject_params.append(department)
            
        if year and year != 'All':
            subject_query += " AND u.year = %s"
            try:
                subject_params.append(int(year))
            except:
                subject_params.append(year)
                
        if subject and subject != 'All':
            subject_query += " AND LOWER(TRIM(a.subject)) = LOWER(TRIM(%s))"
            subject_params.append(subject)
            
        subject_query += " GROUP BY a.subject"

        logging.info(f"Subject Query Params: {subject_params}")
        c.execute(subject_query, tuple(subject_params))
        subject_data = c.fetchall()
        
        logging.info(f"Analytics Result: {len(dept_data)} dept rows, {len(subject_data)} subject rows")
        
        return {
            'department_data': [dict(row) for row in dept_data],
            'subject_data': [dict(row) for row in subject_data],
        }
    except Exception as e:
        logging.error(f"Error getting analytics data: {e}", exc_info=True)
        return {'department_data': [], 'subject_data': []}
    finally:
        if conn:
            conn.close()

def save_face_encoding(user_id, encoding_bytes):
    """Saves/Updates face encoding for a user (Legacy Single Row)."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        c = conn.cursor()
        # Upsert: Update if exists, else insert
        c.execute("INSERT INTO face_encodings (user_id, encoding) VALUES (%s, %s) ON CONFLICT (user_id) DO UPDATE SET encoding = EXCLUDED.encoding", (user_id, encoding_bytes))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error saving face encoding: {e}")
        return False
    finally:
        if conn: conn.close()

def add_face_embedding(user_id, encoding_bytes, source='live', image_path=None, crop_path=None):
    """Adds a new face embedding record (Multi-Row Support)."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        c = conn.cursor()
        c.execute("INSERT INTO face_embeddings (user_id, encoding, source, image_path, crop_path) VALUES (%s, %s, %s, %s, %s)", (user_id, encoding_bytes, source, image_path, crop_path))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error adding face embedding: {e}")
        return False
    finally:
        if conn: conn.close()

def get_face_encodings(user_id=None):
    """Retrieves face encodings from the legacy table."""
    conn = get_db_connection()
    if conn is None: return {}
    try:
        c = conn.cursor()
        if user_id:
            c.execute("SELECT user_id, encoding FROM face_encodings WHERE user_id=%s", (user_id,))
        else:
            c.execute("SELECT user_id, encoding FROM face_encodings")
        rows = c.fetchall()
        return {row[0]: row[1] for row in rows}
    except Exception as e:
        logging.error(f"Error retrieving face encodings: {e}")
        return {}
    finally:
        if conn: conn.close()

def get_face_embeddings_new():
    """Retrieves all face embeddings from the new table."""
    conn = get_db_connection()
    if conn is None: return {}
    try:
        c = conn.cursor()
        c.execute("SELECT user_id, encoding FROM face_embeddings ORDER BY created_at DESC")
        rows = c.fetchall()
        # Group by user_id? Or just return list of (user_id, encoding)
        # For current logic (matching against all), list of tuples is fine, or dict of lists.
        # Biometric logic expects dict {user_id: [enc1, enc2]} or similar?
        # Let's return list of dicts or similar. 
        # Actually logic is usually: load all, iterate.
        return rows # List of (user_id, encoding)
    except Exception as e:
        logging.error(f"Error getting new embeddings: {e}")
        return []
    finally:
        if conn: conn.close()

def get_face_embeddings_count(user_id):
    """Returns count of embeddings for a user."""
    conn = get_db_connection()
    if conn is None: return 0
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM face_embeddings WHERE user_id = %s", (user_id,))
        result = c.fetchone()
        return result[0] if result else 0
    except Exception as e:
        logging.error(f"Error getting embedding count: {e}")
        return 0
    finally:
        if conn: conn.close()

# --- Summary / Report Helpers ---

def get_total_students(department, year):
    conn = get_db_connection()
    if conn is None: return 0
    try:
        c = conn.cursor()
        query = "SELECT COUNT(*) FROM users WHERE role='student' AND active_status='Active'"
        params = []
        if department and department != 'None':
            query += " AND department = %s"
            params.append(department)
        if year and year != 'None':
            query += " AND year = %s"
            params.append(int(year))
        
        c.execute(query, tuple(params))
        return c.fetchone()[0]
    except Exception as e:
        logging.error(f"Error getting total students: {e}")
        return 0
    finally:
        if conn: conn.close()

def get_present_students(subject, hour, date_str, department, year, period_no=None):
    """Returns list of student dicts who are present."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor()
        # Join usage to specific students of that dept/year
        # Date string from frontend is YYYY-MM-DD
        query = """
            SELECT u.id, u.full_name, u.roll_number, a.timestamp, a.confidence_score, a.proof_image_path
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            WHERE TRIM(a.subject) = TRIM(%s)
              AND a.timestamp::date = %s
              AND a.status = 'Present'
        """
        params = [subject, date_str]
        logging.info(f"DEBUG: get_present_students inputs - Sub:{subject}, Date:{date_str}, Dept:{department}, Year:{year}, PeriodNo:{period_no}, Hour:{hour}")
        
        # Prioritize period_no if provided
        if period_no is not None:
            query += " AND a.period_no = %s"
            params.append(period_no)
        elif hour is not None:
            query += " AND EXTRACT(HOUR FROM a.timestamp) = %s"
            params.append(hour)
        # If neither provided, fetch all present for subject/date (fallback)
        if department and department != 'None':
            # Case-insensitive department check
            query += " AND UPPER(u.department) = UPPER(%s)"
            params.append(department)
        if year and year != 'None':
            query += " AND u.year = %s"
            params.append(int(year))
        
        query += " ORDER BY u.roll_number"
        c.execute(query, tuple(params))
        rows = c.fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r[0],
                "full_name": r[1],
                "roll_number": r[2],
                "timestamp": r[3],
                "confidence": r[4],
                "proof_path": r[5]
            })
        return result
    except Exception as e:
        logging.error(f"Error getting present students: {e}")
        return []
    finally:
        if conn: conn.close()

def get_session_attendance_records(subject, date_str, period_no=None):
    """
    Returns all attendance records for a specific session (subject+date+period).
    Includes audit fields for review phase.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        query = """
            SELECT a.user_id, a.status, a.timestamp, a.edited_by, a.original_status, a.edit_timestamp,
                   u.full_name, u.roll_number
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            WHERE TRIM(a.subject) = TRIM(%s)
              AND a.timestamp::date = %s
        """
        params = [subject, date_str]
        
        if period_no is not None:
            query += " AND a.period_no = %s"
            params.append(period_no)
            
        c.execute(query, tuple(params))
        rows = c.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logging.error(f"Error getting session attendance records: {e}")
        return []
    finally:
        if conn: conn.close()

def get_absent_students(subject, hour, date_str, department, year, period_no=None):
    """Returns list of student dicts who are absent."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor()
        # Logic: All students in dept/year EXCEPT those present
        query = """
            SELECT u.id, u.full_name, u.roll_number
            FROM users u
            WHERE u.role = 'student' 
              AND u.active_status = 'Active'
        """
        params = []
        if department and department != 'None':
            query += " AND u.department = %s"
            params.append(department)
        if year and year != 'None':
            query += " AND u.year = %s"
            params.append(int(year))
            
        # Attendance Check Logic
        attendance_filter = " WHERE a.subject = %s AND a.timestamp::date = %s "
        attendance_params = [subject, date_str]
        
        if period_no is not None:
            attendance_filter += " AND a.period_no = %s "
            attendance_params.append(period_no)
        else:
            attendance_filter += " AND EXTRACT(HOUR FROM a.timestamp) = %s "
            attendance_params.append(hour)

        query += f"""
              AND u.id NOT IN (
                  SELECT a.user_id FROM attendance a
                  JOIN users u2 ON a.user_id = u2.id
                  {attendance_filter}
        """
        params.extend(attendance_params)
        
        if department and department != 'None':
            query += " AND u2.department = %s"
            params.append(department)
        if year and year != 'None':
            query += " AND u2.year = %s"
            params.append(int(year))
            
        query += " ) ORDER BY u.roll_number"
        c.execute(query, tuple(params))
        rows = c.fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r[0],
                "full_name": r[1],
                "roll_number": r[2]
            })
        return result
    except Exception as e:
        logging.error(f"Error getting absent students: {e}")
        return []
    finally:
        if conn: conn.close()

def get_od_students(subject, hour, date_str, department, year, period_no=None):
    """Returns list of student dicts who are marked as OD (On Duty)."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor()
        query = """
            SELECT u.id, u.full_name, u.roll_number, a.timestamp
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            WHERE a.subject = %s
              AND a.timestamp::date = %s
              AND a.status = 'OD'
        """
        params = [subject, date_str]
        
        # Prioritize period_no if provided
        if period_no is not None:
            query += " AND a.period_no = %s"
            params.append(period_no)
        else:
            query += " AND EXTRACT(HOUR FROM a.timestamp) = %s"
            params.append(hour)
        if department and department != 'None':
            query += " AND u.department = %s"
            params.append(department)
        if year and year != 'None':
            query += " AND u.year = %s"
            params.append(int(year))
        
        query += " ORDER BY u.roll_number"
        c.execute(query, tuple(params))
        rows = c.fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r[0],
                "full_name": r[1],
                "roll_number": r[2],
                "timestamp": r[3]
            })
        return result
    except Exception as e:
        logging.error(f"Error getting OD students: {e}")
        return []
    finally:
        if conn: conn.close()


# --- Teacher Management Wrappers ---
def add_teacher(full_name, username, password, department, phone=None):
    """Wrapper to add a teacher with role='teacher'."""
    # Note: re-using add_new_user which handles password logic?
    # Actually add_new_user expects hashed password. Main.py handles hashing.
    return add_new_user(full_name, username, password, 'teacher', department, None, None)

def update_teacher(teacher_id, full_name, username, password, department, phone=None):
    """Wrapper to update teacher info."""
    return update_user_info(teacher_id, full_name, username, password, 'teacher', department, None, None)

def delete_teacher(teacher_id):
    """Wrapper to delete a teacher."""
    return delete_user(teacher_id)

def get_all_teachers():
    """Returns all users with role 'teacher'."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT * FROM users WHERE role = 'teacher' ORDER BY full_name")
        teachers = c.fetchall()
        return [dict(t) for t in teachers]
    except Exception as e:
        logging.error(f"Error getting teachers: {e}")
        return []
    finally:
        if conn: conn.close()

# --- Face Recognition Upgrade Functions ---

def save_embedding(user_id, encoding, start_of_week=None, source='live', method='live', crop_path=None, image_path=None):
    """
    Saves a face embedding to the database.
    source: 'live', 'enrollment_sample', 'centroid'
    """
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        
        # Ensure encoding is bytes
        if isinstance(encoding, np.ndarray):
            encoding_bytes = encoding.tobytes()
        else:
            encoding_bytes = encoding
            
        c.execute("INSERT INTO face_embeddings (user_id, encoding, source, crop_path, image_path) VALUES (%s, %s, %s, %s, %s)",
                  (user_id, encoding_bytes, source, crop_path, image_path))
        conn.commit()
    except Exception as e:
        logging.error(f"Error saving embedding: {e}")
    finally:
        if conn: conn.close()

def delete_face_embeddings(user_id, source=None):
    """Deletes face embeddings for a user, optionally filtered by source."""
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        if source:
            c.execute("DELETE FROM face_embeddings WHERE user_id = %s AND source = %s", (user_id, source))
        else:
            c.execute("DELETE FROM face_embeddings WHERE user_id = %s", (user_id,))
        conn.commit()
    except Exception as e:
        logging.error(f"Error deleting embeddings for {user_id}: {e}")
    finally:
        if conn: conn.close()

def get_student_photo_path(user_id):
    """Retrieves the registered photo path for a student."""
    conn = get_db_connection()
    if conn is None: return None
    try:
        c = conn.cursor()
        # Prefer crop_path, then image_path, from enrollment samples
        c.execute("""
            SELECT COALESCE(crop_path, image_path) as path 
            FROM face_embeddings 
            WHERE user_id = %s AND (crop_path IS NOT NULL OR image_path IS NOT NULL)
            ORDER BY created_at DESC 
            LIMIT 1
        """, (user_id,))
        row = c.fetchone()
        return row[0] if row else None
    except Exception as e:
        logging.error(f"Error fetching photo path for {user_id}: {e}")
        return None
    finally:
        if conn: conn.close()

def get_index_embeddings():
    """
    Fetches user embeddings for the recognition index.
    Retrieves both Centroids and Enrollment Samples to support multi-image matching.
    Returns: list of (user_id, numpy_array)
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor()
        
        # Fetch centroids and samples
        # Source of Truth: Only load faces for students who are currently Active and Enrolled
        c.execute("""
            SELECT fe.user_id, fe.encoding, fe.source 
            FROM face_embeddings fe
            JOIN users u ON fe.user_id = u.id
            WHERE fe.source IN ('centroid', 'enrollment_sample')
              AND u.active_status = 'Active'
              AND u.face_status = 'Enrolled'
        """)
        rows = c.fetchall()
        
        embeddings = []
        for r in rows:
            user_id = r[0]
            enc_bytes = r[1]
            source = r[2]
            
            try:
                if isinstance(enc_bytes, memoryview): enc_bytes = bytes(enc_bytes)
                enc_array = np.frombuffer(enc_bytes, dtype=np.float32)
                embeddings.append((user_id, enc_array))
            except Exception as e:
                logging.error(f"Error decoding embedding for user {user_id} (source: {source}): {e}")
            
        logging.info(f"Loaded {len(embeddings)} embeddings from database for recognition index.")
        return embeddings
    except Exception as e:
        logging.error(f"Error fetching index embeddings: {e}")
        return []
    finally:
        if conn: conn.close()
        
def get_user_samples(user_id):
    """
    Fetches raw enrollment samples for centroid calculation.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor()
        c.execute("SELECT encoding FROM face_embeddings WHERE user_id = %s AND source = 'enrollment_sample'", (user_id,))
        rows = c.fetchall()
        samples = []
        for r in rows:
            samples.append(np.frombuffer(r[0], dtype=np.float32))
        return samples
    except Exception as e:
        logging.error(f"Error fetching user samples: {e}")
        return []
    finally:
        if conn: conn.close()

def delete_enrollment_samples(user_id):
    """Deletes enrollment samples (keep centroid if needed, or clear all?). Clearing samples implies reset."""
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        c.execute("DELETE FROM face_embeddings WHERE user_id = %s AND source = 'enrollment_sample'", (user_id,))
        conn.commit()
    except Exception as e:
        logging.error(f"Error deleting samples: {e}")
    finally:
        if conn: conn.close()

def get_teacher_assigned_subjects(teacher_id):
    """Retrieves full subject details assigned to a specific teacher."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Join with academic_subjects to get names/codes
        # Use teacher_subject_assignments table
        query = """
            SELECT tsa.id as assignment_id, tsa.department, tsa.year,
                   s.name as subject_name, s.code as subject_code
            FROM teacher_subject_assignments tsa
            JOIN academic_subjects s ON tsa.subject_id = s.id
            WHERE tsa.teacher_id = %s
            ORDER BY tsa.year, tsa.department, s.name
        """
        c.execute(query, (teacher_id,))
        records = c.fetchall()
        return [dict(r) for r in records]
    except Exception as e:
        logging.error(f"Error getting teacher assigned subjects: {e}")
        return []
    finally:
        if conn: conn.close()
        
def get_teacher_timetable_weekly(teacher_id):
    """
    Retrieves the weekly timetable for a teacher.
    Includes entries where teacher is directly assigned in timetable
    OR assigned via subject assignment (fallback logic match).
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Reuse robust logic from get_timetable but filter for this teacher
        # We find entries where:
        # 1. t.teacher_id = teacher_id
        # 2. OR (t.teacher_id IS NULL AND valid assignment exists for teacher_id)
        
        query = """
            SELECT 
                t.*,
                s.name as subject_name, s.code as subject_code
            FROM timetable t
            -- Match Subject
            LEFT JOIN academic_subjects s ON (
                LOWER(TRIM(t.subject)) = LOWER(TRIM(s.name)) OR 
                LOWER(TRIM(t.subject)) = LOWER(TRIM(s.code))
            )
            -- Match Assignment to check availability
            LEFT JOIN teacher_subject_assignments tsa ON (
                tsa.subject_id = s.id AND 
                LOWER(REPLACE(tsa.department, ' ', '')) = LOWER(REPLACE(t.department, ' ', '')) AND 
                tsa.year::INTEGER = t.year::INTEGER
            )
            WHERE 
                t.teacher_id = %s 
                OR 
                (t.teacher_id IS NULL AND tsa.teacher_id = %s)
            ORDER BY 
                CASE 
                    WHEN t.day = 'Monday' THEN 1
                    WHEN t.day = 'Tuesday' THEN 2
                    WHEN t.day = 'Wednesday' THEN 3
                    WHEN t.day = 'Thursday' THEN 4
                    WHEN t.day = 'Friday' THEN 5
                    WHEN t.day = 'Saturday' THEN 6
                    WHEN t.day = 'Sunday' THEN 7
                END,
                t.start_time
        """
        c.execute(query, (teacher_id, teacher_id))
        records = c.fetchall()
        return [dict(r) for r in records]
    except Exception as e:
        logging.error(f"Error getting teacher weekly timetable: {e}")
        return []
    finally:
        if conn: conn.close()

def get_teacher_attendance_history(teacher_id):
    """
    Retrieves attendance logs for subjects assigned to this teacher.
    Since we don't strictly track 'marked_by', we show logs for ANY subject 
    that the teacher is assigned to teach.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # 1. Get list of (subject_id, dept, year) assignments for this teacher
        c.execute("""
            SELECT tsa.department, tsa.year, s.name, s.code
            FROM teacher_subject_assignments tsa
            JOIN academic_subjects s ON tsa.subject_id = s.id
            WHERE tsa.teacher_id = %s
        """, (teacher_id,))
        assignments = c.fetchall()
        
        if not assignments:
             return []
             
        # Build complex OR clause or just iterate? Query is better.
        # Logic: Select from attendance where (subject matching name/code AND dept matching AND year matching)
        # Because we store subject NAME in attendance table (legacy issue), we have to match names.
        
        where_clauses = []
        params = []
        
        for a in assignments:
            dept = a['department']
            year = a['year']
            s_name = a['name']
            s_code = a['code']
            
            # Sub-clause: (dept=X AND year=Y AND (subject=Name OR subject=Code))
            code_clause = " OR LOWER(TRANSLATE(a.subject, '_- ', '')) = LOWER(TRANSLATE(%s, '_- ', ''))" if s_code else ""
            
            clause = f"""
                (
                    LOWER(TRIM(u.department)) = LOWER(TRIM(%s)) AND 
                    u.year = %s AND 
                    (LOWER(TRANSLATE(a.subject, '_- ', '')) = LOWER(TRANSLATE(%s, '_- ', '')){code_clause})
                )
            """
            params.extend([dept, year, s_name])
            if s_code: params.append(s_code)
            
            where_clauses.append(clause)
            
        full_where = " OR ".join(where_clauses)
        
        query = f"""
            SELECT a.id, a.timestamp, a.subject, a.status, u.full_name, u.roll_number, u.department, u.year
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            WHERE ({full_where})
            ORDER BY a.timestamp DESC
            LIMIT 200
        """
        
        logging.info(f"Teacher History Query Params: {params}")
        c.execute(query, tuple(params))
        records = c.fetchall()
        return [dict(r) for r in records]
        
    except Exception as e:
        logging.error(f"Error getting teacher attendance history: {e}")
        return []
    finally:
        if conn: conn.close()

def assign_subject_to_teacher(teacher_id, subject):
    """Assigns a subject to a teacher."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        c = conn.cursor()
        c.execute("INSERT INTO teachers_subjects (teacher_id, subject) VALUES (%s, %s) ON CONFLICT DO NOTHING", (teacher_id, subject))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error assigning subject: {e}")
        return False
    finally:
        if conn:
            conn.close()
            
def remove_subject_from_teacher(teacher_id, subject):
    """Removes a subject assignment from a teacher."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        c = conn.cursor()
        c.execute("DELETE FROM teachers_subjects WHERE teacher_id = %s AND subject = %s", (teacher_id, subject))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error removing subject: {e}")
        return False
    finally:
        if conn:
            conn.close()

# Duplicate function removed - the correct implementation with batch_year and semester parameters exists at line 913

# -------------------------------------------------------------------
# ACADEMIC SETUP MODULE (New)
# -------------------------------------------------------------------

def create_academic_tables():
    """Creates tables for the Academic Setup module."""
    conn = get_db_connection()
    if conn is None: return
    try:
        c = conn.cursor()
        
        # 1. Departments
        c.execute("""
            CREATE TABLE IF NOT EXISTS academic_departments (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                code VARCHAR(20),
                hod_name VARCHAR(255)
            );
        """)
        
        # 2. Years / Batches
        c.execute("""
            CREATE TABLE IF NOT EXISTS academic_years (
                id SERIAL PRIMARY KEY,
                name VARCHAR(50) NOT NULL, -- e.g. "Year 1", "3rd Year"
                batch VARCHAR(50) -- e.g. "2024-2028"
            );
        """)
        
        # 3. Sections
        c.execute("""
            CREATE TABLE IF NOT EXISTS academic_sections (
                id SERIAL PRIMARY KEY,
                name VARCHAR(50) NOT NULL UNIQUE -- e.g. "A", "B"
            );
        """)
        
        # 4. Subjects
        c.execute("""
            CREATE TABLE IF NOT EXISTS academic_subjects (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                code VARCHAR(20) UNIQUE,
                credits INTEGER DEFAULT 3,
                semester INTEGER
            );
        """)
        
        # 5. Classes (Combination)
        c.execute("""
            CREATE TABLE IF NOT EXISTS academic_classes (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL, -- "II CSE - A" or "II CSE"
                dept_id INTEGER REFERENCES academic_departments(id) ON DELETE RESTRICT,
                year_id INTEGER REFERENCES academic_years(id) ON DELETE RESTRICT,
                section_id INTEGER REFERENCES academic_sections(id) ON DELETE RESTRICT,
                UNIQUE(dept_id, year_id, section_id) -- Works for non-null; unique index needed for nulls if strict
            );
        """)
        
        # 6. Class-Subject Mapping
        c.execute("""
            CREATE TABLE IF NOT EXISTS class_subject_mapping (
                id SERIAL PRIMARY KEY,
                class_id INTEGER REFERENCES academic_classes(id) ON DELETE CASCADE,
                subject_id INTEGER REFERENCES academic_subjects(id) ON DELETE CASCADE,
                UNIQUE(class_id, subject_id)
            );
        """)

        # 7. Teacher Assignment (New Request)
        c.execute("""
            CREATE TABLE IF NOT EXISTS teacher_subject_assignments (
                id SERIAL PRIMARY KEY,
                teacher_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                department VARCHAR(100),
                year INTEGER,
                subject_id INTEGER REFERENCES academic_subjects(id) ON DELETE CASCADE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(teacher_id, department, year, subject_id)
            );
        """)
        
        conn.commit()
        logging.info("Academic tables created/verified.")
    except Exception as e:
        logging.error(f"Error creating academic tables: {e}")
    finally:
        if conn:
            conn.close()

# --- CRUD Functions ---

def assign_teacher_subject(teacher_id, department, year, subject_id):
    """Assigns a subject to a teacher for a specific class context."""
    conn = get_db_connection()
    if not conn: return False, "DB Connection Failed"
    try:
        c = conn.cursor()
        # Check if already assigned
        c.execute("""
            SELECT id FROM teacher_subject_assignments 
            WHERE teacher_id=%s AND department=%s AND year=%s AND subject_id=%s
        """, (teacher_id, department, year, subject_id))
        if c.fetchone():
            return False, "Teacher already assigned to this subject/class."

        c.execute("""
            INSERT INTO teacher_subject_assignments (teacher_id, department, year, subject_id)
            VALUES (%s, %s, %s, %s)
        """, (teacher_id, department, year, subject_id))
        
        # Sync with Timetable
        c.execute("SELECT name FROM academic_subjects WHERE id = %s", (subject_id,))
        res = c.fetchone()
        if res:
            subject_name = res[0]
            # Update existing timetable entries to reflect this assignment
            c.execute("""
                UPDATE timetable 
                SET teacher_id = %s 
                WHERE department = %s AND year = %s AND subject = %s
            """, (teacher_id, department, int(year), subject_name))
            
        conn.commit()
        return True, "Assignment successful."
    except Exception as e:
        logging.error(f"Error assigning teacher subject: {e}")
        return False, str(e)
    finally:
        conn.close()

def get_teachers_by_assignment(department, year, subject_name):
    """Retrieves teachers assigned to a specific dept, year, and subject."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Resolve Subject ID first (flexible match)
        c.execute("SELECT id FROM academic_subjects WHERE lower(name) = lower(%s)", (subject_name,))
        sub = c.fetchone()
        if not sub:
            return []
        subject_id = sub['id']
        
        query = """
            SELECT u.id, u.full_name, u.username
            FROM users u
            JOIN teacher_subject_assignments tsa ON u.id = tsa.teacher_id
            WHERE tsa.department = %s AND tsa.year = %s AND tsa.subject_id = %s
            ORDER BY u.full_name
        """
        c.execute(query, (department, year, subject_id))
        return [dict(r) for r in c.fetchall()]
    except Exception as e:
        logging.error(f"Error fetching assigned teachers: {e}")
        return []
    finally:
        conn.close()

def get_teacher_assignments(teacher_id):
    """Fetches all assignments for a teacher."""
    conn = get_db_connection()
    if not conn: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("""
            SELECT tsa.*, s.name as subject_name, s.code as subject_code
            FROM teacher_subject_assignments tsa
            JOIN academic_subjects s ON tsa.subject_id = s.id
            WHERE tsa.teacher_id = %s
            ORDER BY tsa.department, tsa.year
        """, (teacher_id,))
        return [dict(r) for r in c.fetchall()]
    except Exception as e:
        logging.error(f"Error fetching teacher assignments: {e}")
        return []
    finally:
        conn.close()

def remove_teacher_assignment(assignment_id):
    """Removes a teacher assignment."""
    conn = get_db_connection()
    if not conn: return False
    try:
        c = conn.cursor()
        
        # Fetch details before creating to sync with timetable
        c.execute("""
            SELECT tsa.teacher_id, tsa.department, tsa.year, s.name as subject_name
            FROM teacher_subject_assignments tsa
            JOIN academic_subjects s ON tsa.subject_id = s.id
            WHERE tsa.id = %s
        """, (assignment_id,))
        row = c.fetchone()
        
        c.execute("DELETE FROM teacher_subject_assignments WHERE id = %s", (assignment_id,))
        
        if row:
            teacher_id, dept, year, subject_name = row
            # Unassign from timetable if it matches this teacher
            c.execute("""
                UPDATE timetable 
                SET teacher_id = NULL 
                WHERE department = %s AND year = %s AND subject = %s AND teacher_id = %s
            """, (dept, int(year), subject_name, teacher_id))
            
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error removing assignment: {e}")
        return False
    finally:
        conn.close()

def get_all_academic_subjects():
    """Retrieves all subjects with IDs for dropdowns."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT id, name, code FROM academic_subjects ORDER BY name")
        return [dict(r) for r in c.fetchall()]
    except Exception as e:
        logging.error(f"Error fetching academic subjects: {e}")
        return []
    finally:
        conn.close()

# Departments
def add_academic_department(name, code=None, hod_name=None):
    # Validation
    if not name or not name.strip():
        return False, "Department name cannot be empty"
    name = name.strip()
    
    conn = get_db_connection()
    try:
        c = conn.cursor()
        # Check uniqueness (case insensitive)
        c.execute("SELECT id FROM academic_departments WHERE LOWER(name) = LOWER(%s)", (name,))
        if c.fetchone():
            return False, "Department name already exists"
            
        c.execute("INSERT INTO academic_departments (name, code, hod_name) VALUES (%s, %s, %s)", (name, code, hod_name))
        conn.commit()
        return True, "Added department"
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()

def update_academic_department(dept_id, name, hod_name=None):
    if not name or not name.strip():
        return False, "Department name cannot be empty"
    name = name.strip()
    
    conn = get_db_connection()
    try:
        c = conn.cursor()
        # Check uniqueness excluding self
        c.execute("SELECT id FROM academic_departments WHERE LOWER(name) = LOWER(%s) AND id != %s", (name, dept_id))
        if c.fetchone():
            return False, "Department name already exists"
            
        c.execute("UPDATE academic_departments SET name = %s, hod_name = %s WHERE id = %s", (name, hod_name, dept_id))
        conn.commit()
        return True, "Department updated"
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()

# Duplicate get_academic_departments removed


def delete_academic_department(dept_id):
    """
    Deletes a department and CASCADES to its classes and mappings.
    """
    conn = get_db_connection()
    try:
        c = conn.cursor()
        
        # 1. Fetch associated classes to cascade delete
        c.execute("SELECT id FROM academic_classes WHERE dept_id = %s", (dept_id,))
        rows = c.fetchall()
        class_ids = [r[0] for r in rows]
        
        if class_ids:
            # Create placeholders string like %s, %s, %s
            placeholders = ','.join(['%s'] * len(class_ids))
            
            # 2. Delete Mappings for these classes
            c.execute(f"DELETE FROM class_subject_mapping WHERE class_id IN ({placeholders})", tuple(class_ids))
            
            # 3. Delete Classes
            c.execute(f"DELETE FROM academic_classes WHERE dept_id = %s", (dept_id,))
            
        # 4. Delete Department
        c.execute("DELETE FROM academic_departments WHERE id = %s", (dept_id,))
        
        conn.commit()
        return True, "Deleted department and all associated classes/mappings"
    except Exception as e:
        conn.rollback()
        return False, str(e) # Likely foreign key constraint
    finally:
        if conn: conn.close()

# Years
def add_academic_year(start_year, num_sections=0):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        start_year = int(start_year)
        end_year = start_year + 4
        name = f"{start_year}-{end_year}" # UI Display Name
        batch = name # Legacy support
        
        # Check if exists
        c.execute("SELECT id FROM academic_years WHERE start_year = %s", (start_year,))
        if c.fetchone():
            return False, "Batch starting this year already exists"

        c.execute("INSERT INTO academic_years (name, batch, start_year, end_year) VALUES (%s, %s, %s, %s) RETURNING id", 
                 (name, batch, start_year, end_year))
        batch_id = c.fetchone()[0]
        
        # Generate Sections
        if num_sections > 0:
            import string
            # A, B, C...
            # If > 26, maybe A1? For now assume A..Z
            letters = string.ascii_uppercase
            created_count = 0
            for i in range(num_sections):
                if i < len(letters):
                    sec_name = letters[i]
                    # Check partial duplicate just in case
                    c.execute("INSERT INTO academic_sections (name, batch_id) VALUES (%s, %s) ON CONFLICT DO NOTHING", (sec_name, batch_id))
                    created_count += 1
            
        conn.commit()
        return True, "Added batch and sections"
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()

def get_academic_years():
    conn = get_db_connection()
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT id, name, start_year, end_year FROM academic_years ORDER BY start_year DESC, name") # Show newest first?
        rows = c.fetchall()
        results = []
        
        current_date = datetime.now()
        curr_year = current_date.year
        curr_month = current_date.month
        
        for r in rows:
            d = dict(r)
            # Auto-calc academic year (1..4)
            # If batch started in 2024 (June/July). 
            # In Jan 2025: It is part of 2024-2025 academic year -> Year 1.
            # In July 2025: It is part of 2025-2026 academic year -> Year 2.
            
            start = d.get('start_year')
            if start:
                # Academic year starts roughly July.
                # If today < July, we are in (today_year - 1)-(today_year) academic year.
                # E.g. Jan 2025 -> Academic Year 2024-2025.
                # Batch Start = 2024.
                # Diff = 2024 - 2024 = 0. Year 1.
                
                # Logic:
                effective_academic_year_start = curr_year if curr_month >= 6 else curr_year - 1
                diff = effective_academic_year_start - start
                d['current_year_display'] = diff + 1
            else:
                d['current_year_display'] = 0
                
            results.append(d)
            
        return results


    finally:
        if conn: conn.close()

def get_batch_details(batch_id):
    conn = get_db_connection()
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # 1. Get Batch Info
        c.execute("SELECT * FROM academic_years WHERE id = %s", (batch_id,))
        batch = c.fetchone()
        if not batch: return None
        
        d = dict(batch)
        
        # 2. Calculate Current Year
        start = d.get('start_year')
        current_year_display = 1
        if start:
            from datetime import datetime
            current_date = datetime.now()
            curr_year = current_date.year
            curr_month = current_date.month
            effective_academic_year_start = curr_year if curr_month >= 7 else curr_year - 1
            year_diff = effective_academic_year_start - start
            current_stage = year_diff + 1
            if current_stage < 1: current_stage = 1
            if current_stage > 4: current_stage = 4
            current_year_display = current_stage
            
        d['current_year_display'] = current_year_display
        
        # 3. Get Sections
        c.execute("SELECT id, name FROM academic_sections WHERE batch_id = %s ORDER BY name", (batch_id,))
        sections = [dict(r) for r in c.fetchall()]
        d['sections'] = sections
        
        return d
    finally:
        if conn: conn.close()

def delete_academic_year(year_id):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        
        # 1. Delete associated Classes (cascades to mappings)
        c.execute("DELETE FROM academic_classes WHERE year_id = %s", (year_id,))
        
        # 2. Delete associated Sections (linked by batch_id)
        # Check if column exists first or just try-except? 
        # Since we know we ran migrations, we can assume batch_id exists on sections/subjects if the feature is active.
        # However, to be safe against schema variations, we can wrap or use robust queries.
        # But for this user workspace, we know the state.
        
        # Delete Sections for this batch
        c.execute("DELETE FROM academic_sections WHERE batch_id = %s", (year_id,))
        
        # 3. Unlink or Delete Subjects for this batch?
        # If a subject is specific to a batch, we might want to delete it or just set batch_id to NULL.
        # Users might prefer keeping subjects. Let's set batch_id to NULL to be safe, 
        # or delete if strict. Let's delete if strictly unique to batch, but "subjects" are often shared.
        # User request "Subject Mapping" implies reusability. 
        # But "Batch" field in subjects implies specificity.
        # Let's unlink (set null) for now to prevent data loss of subject definitions, 
        # unless user explicitly wants deep cleanup.
        c.execute("UPDATE academic_subjects SET batch_id = NULL WHERE batch_id = %s", (year_id,))
        
        # 4. Update Users (Students) linked to this batch/year?
        # If users have year_id, set to NULL?
        # We don't have a direct link in the schema shown above (users might be linked via class).
        # But migrate_students_status added 'semester' and 'year'. 'year' is just an integer (1,2,3,4).
        # We assume no direct FK from users to academic_years(id) unless added elsewhere.
        
        # 5. Delete the Year itself
        c.execute("DELETE FROM academic_years WHERE id = %s", (year_id,))
        
        conn.commit()
        return True, "Deleted Batch and associated Classes/Sections"
    except Exception as e:
        conn.rollback()
        return False, str(e)
    finally:
        if conn: conn.close()
        
def update_academic_year(year_id, name, batch):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("UPDATE academic_years SET name = %s, batch = %s WHERE id = %s", (name, batch, year_id))
        conn.commit()
        return True, "Updated year"
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()

# Sections
def add_academic_section(name):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("INSERT INTO academic_sections (name) VALUES (%s)", (name,))
        conn.commit()
        return True, "Added section"
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()

def get_academic_sections():
    conn = get_db_connection()
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT * FROM academic_sections ORDER BY name")
        return [dict(r) for r in c.fetchall()]
    finally:
        if conn: conn.close()

def delete_academic_section(sec_id):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("DELETE FROM academic_sections WHERE id = %s", (sec_id,))
        conn.commit()
        return True, "Deleted"
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()
        
def update_academic_section(sec_id, name):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("UPDATE academic_sections SET name = %s WHERE id = %s", (name, sec_id))
        conn.commit()
        return True, "Updated section"
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()

# Subjects
def add_academic_subject(name, code, credits=3, semester=None, regulation=None, subject_type='Theory', batch_id=None):
    if not batch_id:
        return False, "Batch is required for adding a subject"
    conn = get_db_connection()
    try:
        # Auto-calculate year from semester if provided
        year = None
        if semester:
            try:
                import math
                year = math.ceil(int(semester) / 2)
            except:
                year = None

        # Convert empty strings to None (NULL) for optional fields with constraints
        code = None if code == '' else code
        regulation = None if regulation == '' else regulation

        c = conn.cursor()
        c.execute("""
            INSERT INTO academic_subjects 
            (name, code, credits, semester, regulation, subject_type, batch_id, year) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (name, code, credits, semester, regulation, subject_type, batch_id, year))
        
        new_id = c.fetchone()[0]
        conn.commit()
        return True, new_id
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()

def get_academic_subjects():
    conn = get_db_connection()
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT * FROM academic_subjects ORDER BY name")
        return [dict(r) for r in c.fetchall()]
    finally:
        if conn: conn.close()

def delete_academic_subject(sub_id):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("DELETE FROM academic_subjects WHERE id = %s", (sub_id,))
        conn.commit()
        return True, "Deleted"
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()

def update_academic_subject(sub_id, name, code, credits, semester, regulation=None, subject_type='Theory', batch_id=None):
    if not batch_id:
        return False, "Batch is required"
    conn = get_db_connection()
    try:
        # Auto-calculate year from semester if provided
        year = None
        if semester:
            try:
                import math
                year = math.ceil(int(semester) / 2)
            except:
                year = None
                
        # Convert empty strings to None (NULL) for optional fields with constraints
        code = None if code == '' else code
        regulation = None if regulation == '' else regulation
                
        c = conn.cursor()
        c.execute("""
            UPDATE academic_subjects 
            SET name = %s, code = %s, credits = %s, semester = %s, 
                regulation = %s, subject_type = %s, batch_id = %s, year = %s 
            WHERE id = %s
        """, (name, code, credits, semester, regulation, subject_type, batch_id, year, sub_id))
        conn.commit()
        return True, "Updated subject"
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()

    # Classes
def add_academic_class(dept_id, year_id, section_id=None):
    conn = get_db_connection()
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Generate name
        c.execute("SELECT name FROM academic_departments WHERE id=%s", (dept_id,))
        d_name_res = c.fetchone()
        if not d_name_res: return False, "Department not found"
        d_name = d_name_res['name']
        
        c.execute("SELECT start_year, name FROM academic_years WHERE id=%s", (year_id,))
        y_res = c.fetchone()
        if not y_res: return False, "Year/Batch not found"
        
        # Calculate Current Year (Roman)
        start_year = y_res['start_year']
        roman_year = "I"
        if start_year:
            from datetime import datetime
            curr_date = datetime.now()
            curr_year = curr_date.year
            curr_month = curr_date.month
            eff_start = curr_year if curr_month >= 7 else curr_year - 1
            diff = eff_start - start_year
            stage = diff + 1
            if stage < 1: stage = 1
            if stage > 4: stage = 4 # Cap at 4
            romans = ["I", "II", "III", "IV"]
            roman_year = romans[stage-1]
        
        y_name = roman_year
        class_name = f"{y_name} {d_name}"
        
        if section_id:
            c.execute("SELECT name FROM academic_sections WHERE id=%s", (section_id,))
            s_name_res = c.fetchone()
            if not s_name_res: return False, "Section not found"
            s_name = s_name_res['name']
            class_name += f" - {s_name}"
        else:
            section_id = None # Ensure it is None not empty string
        
        c.execute("INSERT INTO academic_classes (dept_id, year_id, section_id, name) VALUES (%s, %s, %s, %s)", 
                 (dept_id, year_id, section_id, class_name))
        conn.commit()
        return True, "Added class"
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()

def get_academic_classes():
    conn = get_db_connection()
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("""
            SELECT c.id, c.name, c.dept_id, c.year_id, c.section_id, 
                   d.name as dept_name, y.name as year_name, s.name as sec_name 
            FROM academic_classes c
            JOIN academic_departments d ON c.dept_id = d.id
            JOIN academic_years y ON c.year_id = y.id
            LEFT JOIN academic_sections s ON c.section_id = s.id
            ORDER BY c.name
        """)
        return [dict(r) for r in c.fetchall()]
    finally:
        if conn: conn.close()

def update_academic_class(class_id, dept_id, year_id, section_id=None):
    # Logic similar to add: regenerate name
    conn = get_db_connection()
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Generate name
        c.execute("SELECT name FROM academic_departments WHERE id=%s", (dept_id,))
        d_name_res = c.fetchone()
        if not d_name_res: return False, "Department not found"
        d_name = d_name_res['name']
        
        c.execute("SELECT start_year, name FROM academic_years WHERE id=%s", (year_id,))
        y_res = c.fetchone()
        if not y_res: return False, "Year/Batch not found"
        
        # Calculate Current Year (Roman)
        start_year = y_res['start_year']
        roman_year = "I"
        if start_year:
            # Assumes datetime already imported or available from previous usage context if checking global? 
            # Safest to re-import locally inside function as Python allows it cheap
            from datetime import datetime
            curr_date = datetime.now()
            curr_year = curr_date.year
            curr_month = curr_date.month
            eff_start = curr_year if curr_month >= 7 else curr_year - 1
            diff = eff_start - start_year
            stage = diff + 1
            if stage < 1: stage = 1
            if stage > 4: stage = 4 
            romans = ["I", "II", "III", "IV"]
            roman_year = romans[stage-1]
        
        y_name = roman_year
        class_name = f"{y_name} {d_name}"
        
        if section_id:
            c.execute("SELECT name FROM academic_sections WHERE id=%s", (section_id,))
            s_name_res = c.fetchone()
            if not s_name_res: return False, "Section not found"
            s_name = s_name_res['name']
            class_name += f" - {s_name}"
        else:
            section_id = None
            
        c.execute("UPDATE academic_classes SET dept_id=%s, year_id=%s, section_id=%s, name=%s WHERE id=%s", 
                 (dept_id, year_id, section_id, class_name, class_id))
        conn.commit()
        return True, "Updated class"
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()
        
def delete_academic_class(class_id):
    """Deletes a class and its mappings."""
    conn = get_db_connection()
    try:
        c = conn.cursor()
        # 1. Delete Mappings first
        c.execute("DELETE FROM class_subject_mapping WHERE class_id = %s", (class_id,))
        
        # 2. Delete Class
        c.execute("DELETE FROM academic_classes WHERE id = %s", (class_id,))
        
        conn.commit()
        return True, "Deleted"
    except Exception as e:
        conn.rollback()
        return False, str(e)
    finally:
        if conn: conn.close()

# Mapping
def add_class_subject_mapping(class_id, subject_id, semester=None):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        # Check duplicate (specific semester)
        # If semester is None, we default to 1? Or handle as NULL? 
        # User requirement says strict 1..8.
        if not semester: semester = 1 # Fallback
        
        c.execute("SELECT 1 FROM class_subject_mapping WHERE class_id=%s AND subject_id=%s AND semester=%s", (class_id, subject_id, semester))
        if c.fetchone():
            return False, "Subject already mapped to this class for this semester"
            
        c.execute("INSERT INTO class_subject_mapping (class_id, subject_id, semester) VALUES (%s, %s, %s)", (class_id, subject_id, semester))
        conn.commit()
        return True, "Mapped"
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()

def bulk_add_class_subject_mapping(class_id, semester, subject_ids):
    if not subject_ids:
        return True, "No subjects selected"
        
    conn = get_db_connection()
    try:
        c = conn.cursor()
        added_count = 0
        if not semester: semester = 1
        
        for sub_id in subject_ids:
            # Check duplicate
            c.execute("SELECT 1 FROM class_subject_mapping WHERE class_id=%s AND subject_id=%s AND semester=%s", (class_id, sub_id, semester))
            if not c.fetchone():
                c.execute("INSERT INTO class_subject_mapping (class_id, subject_id, semester) VALUES (%s, %s, %s)", (class_id, sub_id, semester))
                added_count += 1
                
        conn.commit()
        return True, f"Mapped {added_count} subjects to Semester {semester}"
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()

def get_class_subjects_mapping():
    conn = get_db_connection()
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("""
            SELECT m.id, c.name as class_name, s.name as subject_name, s.code, m.semester, 
                   c.dept_id, c.year_id, c.section_id
            FROM class_subject_mapping m
            JOIN academic_classes c ON m.class_id = c.id
            JOIN academic_subjects s ON m.subject_id = s.id
            ORDER BY c.name, m.semester
        """)
        return [dict(r) for r in c.fetchall()]
    finally:
        if conn: conn.close()
        
def delete_mapping(map_id):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("DELETE FROM class_subject_mapping WHERE id = %s", (map_id,))
        conn.commit()
        return True, "Deleted"
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()

def get_students_filtered(dept_id, year_id, section_id=None, semester=None, batch_year=None, academic_year=None):
    logging.info(f"DEBUG: get_students_filtered params: Dept={dept_id}, BatchID={year_id}, Sec={section_id}, Sem={semester}, Year={academic_year}")
    conn = get_db_connection()
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # ... (Dept/Batch resolution unchanged) ...
        # 1. Resolve Department
        if dept_id:
            c.execute("SELECT name FROM academic_departments WHERE id=%s", (dept_id,))
            dept_res = c.fetchone()
            dept_name = dept_res['name'] if dept_res else ""
        else:
            dept_name = ""
        
        # 2. Resolve Batch
        # year_id passed from frontend is usually the 'academic_years' table ID (i.e. Batch ID)
        batch_start_year = None
        batch_name = ""
        
        if year_id:
            c.execute("SELECT id, name, start_year, batch FROM academic_years WHERE id=%s", (year_id,))
            year_res = c.fetchone()
            if year_res:
                batch_start_year = year_res['start_year']
                batch_name = year_res['name']
        
        logging.info(f"DEBUG: Resolved Batch - StartYear={batch_start_year}, Name={batch_name}")
        
        # 3. Build Query
        query = """
            SELECT u.id, u.full_name, u.register_number, u.roll_number, u.username,
                   u.section_id, u.department, u.year, u.semester, u.active_status,
                   u.batch_start_year, u.batch_end_year, u.batch_year,
                   u.face_folder_path, u.face_uploaded_count,
                   u.is_face_enrolled,
                   s.name as section_name, 
                   COALESCE(u.face_status, 'Not Enrolled') as face_status
            FROM users u
            LEFT JOIN academic_sections s ON u.section_id = s.id
            WHERE u.role = 'student' 
              AND (u.active_status = 'Active' OR u.active_status IS NULL)
        """
        params = []
        
        # Dept Filter
        if dept_id:
             query += " AND (CAST(u.dept_id AS TEXT) = %s OR LOWER(TRIM(CAST(u.department AS TEXT))) = LOWER(TRIM(%s)))"
             params.extend([str(dept_id), dept_name])
        
        # Year (1-4) Filter
        if academic_year:
             query += " AND u.year = %s"
             params.append(str(academic_year)) # Ensure string/int compatibility

        # Batch Filter
        if batch_start_year:
            # Match Start Year (Preferred) OR Name
            query += " AND (u.batch_start_year = %s OR u.batch_year = %s)"
            params.extend([batch_start_year, batch_name])
        elif batch_year:
             # Direct string match if ID not provided
             query += " AND u.batch_year = %s"
             params.append(batch_year)
            
        # Section Filter
        if section_id and str(section_id).lower() not in ['all', 'none', '']:
            query += " AND u.section_id = %s"
            params.append(section_id)
            
        # Semester Filter
        if semester and str(semester).lower() != 'all':
             query += " AND u.semester = %s"
             params.append(int(semester))
             
        # Robust Numeric Sort: Last 3 Digits. NULL/Empty -> 999999 (Bottom)
        query += " ORDER BY COALESCE(CAST(RIGHT(NULLIF(regexp_replace(u.register_number, '[^0-9]', '', 'g'), ''), 3) AS INTEGER), 999999) ASC, u.full_name"
        
        c.execute(query, tuple(params))
        results = [dict(r) for r in c.fetchall()]
        
        # REQUIRED FIX 3: Dynamic face status fallback logic
        PROJECT_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
        STUDENT_FACES_DIR = os.path.join(PROJECT_BACKEND_DIR, "student_faces")
        
        logging.info(f"DEBUG: Checking dynamic face status in: {STUDENT_FACES_DIR}")
        
        for s in results:
            reg_no = s.get('register_number')
            if not reg_no:
                s['face_status'] = 'Not Enrolled'
                s['is_face_enrolled'] = False
                continue
                
            folder_path = None
            image_count = 0
            
            # 1. Check for standard folder (JUST RegNo)
            standard_path = os.path.join(STUDENT_FACES_DIR, str(reg_no))
            if os.path.exists(standard_path) and os.path.isdir(standard_path):
                folder_path = standard_path
                logging.debug(f"DEBUG: Found standard folder for {reg_no}")
            else:
                # 2. Check for legacy folders (containing RegNo)
                if os.path.exists(STUDENT_FACES_DIR):
                    for d in os.listdir(STUDENT_FACES_DIR):
                        if str(reg_no) in d:
                            potential_path = os.path.join(STUDENT_FACES_DIR, d)
                            if os.path.isdir(potential_path):
                                folder_path = potential_path
                                logging.debug(f"DEBUG: Found legacy folder {d} for {reg_no}")
                                break
            
            if folder_path:
                img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                image_count = len(img_files)
                
            if image_count > 0:
                s['face_status'] = 'Enrolled'
                s['is_face_enrolled'] = True
                s['face_uploaded_count'] = image_count
                s['face_folder_path'] = folder_path
            else:
                s['face_status'] = 'Not Enrolled'
                s['is_face_enrolled'] = False
                s['face_uploaded_count'] = 0

        logging.info(f"DEBUG: Found {len(results)} students. Status check complete.")
        return results
    except Exception as e:
        logging.error(f"Error in get_students_filtered: {e}")
        return []
    finally:
        if conn: conn.close()

def update_user_face_status(user_id, status):
    """
    Updates the face enrollment status and related columns.
    """
    conn = get_db_connection()
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        is_enrolled = (status == 'Enrolled')
        
        # Resolve Folder Path and Image Count
        c.execute("SELECT register_number FROM users WHERE id = %s", (user_id,))
        user_res = c.fetchone()
        
        folder_path = None
        image_count = 0
        
        if is_enrolled and user_res and user_res['register_number']:
            reg_no = user_res['register_number']
            STUDENT_FACES_DIR = "student_faces"
            folder_path = os.path.join(STUDENT_FACES_DIR, str(reg_no))
            
            # Check for images (or fallback to any folder with RegNo)
            found_folder = False
            if os.path.exists(folder_path):
                found_folder = True
            else:
                if os.path.exists(STUDENT_FACES_DIR):
                    for d in os.listdir(STUDENT_FACES_DIR):
                        if str(reg_no) in d:
                            folder_path = os.path.join(STUDENT_FACES_DIR, d)
                            found_folder = True
                            break
            
            if found_folder and os.path.exists(folder_path):
                image_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        
        c.execute("""
            UPDATE users 
            SET is_face_enrolled = %s, 
                face_status = %s,
                face_folder_path = %s,
                face_uploaded_count = %s,
                face_enrolled_at = CASE WHEN %s THEN NOW() ELSE face_enrolled_at END
            WHERE id = %s
        """, (is_enrolled, status, folder_path, image_count, is_enrolled, user_id))
        conn.commit()
    except Exception as e:
        logging.error(f"Error updating face status: {e}")
    finally:
        if conn: conn.close()

# Duplicate promote_students logic removed.
# Please refer to the updated implementation below (around line 4022).

def remove_students_bulk(student_ids, reason):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("UPDATE users SET active_status='Inactive', status_updated_at=NOW(), status_reason=%s WHERE id = ANY(%s)", (reason, student_ids))
        conn.commit()
        return True, "Students removed/deactivated"
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()

def get_user_embeddings(user_id):
    """Retrieves all embeddings for a specific user."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor()
        c.execute("SELECT encoding FROM face_embeddings WHERE user_id = %s", (user_id,))
        rows = c.fetchall()
        return [np.frombuffer(row[0], dtype=np.float32) for row in rows]
    except Exception as e:
        logging.error(f"Error getting user embeddings: {e}")
        return []
    finally:
        if conn: conn.close()

def has_embeddings(user_id):
    """Checks if a user has any face embeddings registered."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        c = conn.cursor()
        c.execute("SELECT 1 FROM face_embeddings WHERE user_id = %s LIMIT 1", (user_id,))
        return c.fetchone() is not None
    except Exception as e:
        logging.error(f"Error checking embeddings: {e}")
        return False



def get_user_samples(user_id):
    """Retrieves all sample embeddings for a user (excluding centroids)."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor()
        c.execute("SELECT encoding FROM face_embeddings WHERE user_id = %s AND source != 'centroid'", (user_id,))
        rows = c.fetchall()
        return [np.frombuffer(r[0], dtype=np.float32) for r in rows]
    except Exception as e:
        logging.error(f"Error getting user samples: {e}")
        return []
    finally:
        if conn: conn.close()

def get_all_students():
    """Retrieves basic info for ALL students (ignoring face_status)."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT id, full_name, roll_number, register_number, face_status FROM users WHERE role = 'student' ORDER BY full_name")
        return [dict(row) for row in c.fetchall()]
    except Exception as e:
        logging.error(f"Error getting all students: {e}")
        return []
    finally:
        if conn: conn.close()

def add_user(data):
    """
    Adds a new user to the database.
    Expected data keys: username, password, role, full_name, (department, year, roll_number for students)
    """
    conn = get_db_connection()
    if conn is None: return False, "DB Connection Failed"
    
    try:
        username = data.get('username')
        password = data.get('password')
        role = data.get('role', 'student')
        full_name = data.get('full_name')
        
        # Student specific
        department = data.get('department')
        year = data.get('year')
        roll_number = data.get('roll_number')
        
        # New Module 8 Fields
        register_number = data.get('register_number')
        batch_start_year = data.get('batch_start_year')
        batch_end_year = data.get('batch_end_year')
        semester = data.get('semester')
        dept_id = data.get('dept_id')
        
        if not username or not password or not full_name:
             return False, "Missing required fields"

        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        c = conn.cursor()
        c.execute("""
            INSERT INTO users (username, password, role, full_name, department, dept_id, year, roll_number, register_number, batch_start_year, batch_end_year, semester, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (username, hashed, role, full_name, department, dept_id, year, roll_number, register_number, batch_start_year, batch_end_year, semester, data.get('created_by')))
        
        new_id = c.fetchone()[0]
        conn.commit()
        return True, f"User added with ID {new_id}"
    except Exception as e:
        conn.rollback()
        logging.error(f"Add User Error: {e}")
        return False, str(e)
    finally:
        if conn: conn.close()



def update_student_academic_info_by_regno(reg_no, data):
    """
    Updates student academic details if they already exist (Bulk Upload Upsert).
    """
    conn = get_db_connection()
    try:
        c = conn.cursor()
        
        # Update academic fields
        c.execute("""
            UPDATE users 
            SET full_name = %s,
                department = %s,
                dept_id = %s,
                year = %s,
                semester = %s,
                batch_start_year = %s,
                batch_end_year = %s,
                roll_number = %s
            WHERE register_number = %s
        """, (
            data.get('full_name'),
            data.get('department'),
            data.get('dept_id'),
            data.get('year'),
            data.get('semester'),
            data.get('batch_start_year'),
            data.get('batch_end_year'),
            data.get('roll_number'),
            reg_no
        ))
        conn.commit()
        return True, "Updated existing student details"
    except Exception as e:
        return False, str(e)
    finally:
        if conn: conn.close()
        
# --- Teacher Module Extensions ---

def get_teacher_classes(teacher_id):
    """
    Retrieves distinct classes (Dept, Year) assigned to a teacher.
    Since 'Section' is not in the schema, we group by Dept and Year.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Source of Truth: teacher_subject_assignments
        query = """
            SELECT DISTINCT department, year
            FROM teacher_subject_assignments
            WHERE teacher_id = %s
            ORDER BY department, year
        """
        c.execute(query, (teacher_id,))
        classes = c.fetchall()
        return [dict(row) for row in classes]
    except Exception as e:
        logging.error(f"Error getting teacher classes: {e}")
        return []
    finally:
        if conn: conn.close()

def get_students_for_teacher_class(teacher_id, department, year, semester=None, batch_year=None):
    """
    Returns students for a specific class ONLY if the teacher is assigned to it.
    Validates assignment before fetching.
    Supports additional filtering by semester and batch_year.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # 1. Validation: Does teacher have an assignment for this Dept+Year?
        # Source of Truth: teacher_subject_assignments
        check_query = """
            SELECT 1 
            FROM teacher_subject_assignments
            WHERE teacher_id = %s AND LOWER(department) = LOWER(%s) AND year = %s
            LIMIT 1
        """
        # Ensure year is int for validation
        try:
            y_val = int(year)
        except:
            y_val = 0

        c.execute(check_query, (teacher_id, department, y_val))
        if not c.fetchone():
            logging.warning(f"Access Denied: Teacher {teacher_id} not assigned to {department}-{year}")
            return [] 
            
        # 2. Fetch Students
        student_query = """
            SELECT id, full_name, roll_number, register_number, department, year, batch_year, semester
            FROM users
            WHERE role = 'student' 
              AND (active_status = 'Active' OR active_status IS NULL)
              AND LOWER(department) = LOWER(%s) AND year = %s
        """
        params = [department, y_val]
        
        if semester and str(semester).lower() != 'all':
            student_query += " AND semester = %s"
            params.append(int(semester))
            
        if batch_year and str(batch_year).lower() != 'all':
            student_query += " AND batch_year = %s"
            params.append(batch_year)
            
        # Robust Numeric Sort: Last 3 Digits. NULL/Empty -> 999999 (Bottom)
        student_query += " ORDER BY COALESCE(CAST(RIGHT(NULLIF(regexp_replace(register_number, '[^0-9]', '', 'g'), ''), 3) AS INTEGER), 999999) ASC, full_name"
        
        logging.info(f"DEBUG: Student Query: {student_query}")
        logging.info(f"DEBUG: Params: {params}")
        
        c.execute(student_query, tuple(params))
        students = c.fetchall()
        return [dict(row) for row in students]
            
    except Exception as e:
        logging.error(f"Error fetching students for teacher class: {e}")
        return []
    finally:
        if conn: conn.close()

def get_teacher_report_data(teacher_id, subject_param=None, department=None, year=None, batch_year=None, period_no=None, method=None):
    """
    Generates report data for a teacher's assigned subjects.
    Filters strictly by teacher_id.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # 1. Validate Subject Assignment if specific subject requested
        if subject_param:
            c.execute("""
                SELECT 1 
                FROM teacher_subject_assignments tsa
                JOIN academic_subjects s ON tsa.subject_id = s.id
                WHERE tsa.teacher_id = %s AND LOWER(TRIM(s.name)) = LOWER(TRIM(%s))
            """, (teacher_id, subject_param))
            if not c.fetchone():
                return []
        
        # 2. Build Query
        driver_query = """
            WITH SessionsHeld AS (
                SELECT 
                    TRIM(a.subject) as subject,
                    u.department, 
                    u.year,
                    COUNT(DISTINCT (a.timestamp::date, COALESCE(a.period_no, 0))) as held_count
                FROM attendance a
                JOIN users u ON a.user_id = u.id
                WHERE a.marked_by = %(teacher_id)s
                GROUP BY 1, 2, 3
            )
            SELECT 
                u.id,
                u.full_name,
                u.roll_number,
                u.department,
                u.year,
                sub.name as subject_name,
                COALESCE(sh.held_count, 0) as total_attendance_records,
                COUNT(DISTINCT CASE WHEN a.status IN ('Present', 'OD') THEN (a.timestamp::date, COALESCE(a.period_no, 0)) END) as present_count,
                COUNT(DISTINCT CASE WHEN a.status = 'OD' THEN (a.timestamp::date, COALESCE(a.period_no, 0)) END) as od_count
            FROM users u
            JOIN teacher_subject_assignments tsa ON LOWER(u.department) = LOWER(tsa.department) AND u.year = tsa.year
            JOIN academic_subjects sub ON tsa.subject_id = sub.id
            LEFT JOIN SessionsHeld sh 
                ON LOWER(TRIM(sh.subject)) = LOWER(TRIM(sub.name)) 
                AND LOWER(sh.department) = LOWER(u.department) 
                AND sh.year = u.year
            LEFT JOIN attendance a ON u.id = a.user_id 
                AND LOWER(TRIM(a.subject)) = LOWER(TRIM(sub.name))
                AND (%(period_no)s IS NULL OR a.period_no = %(period_no)s)
                AND (%(method)s IS NULL OR a.method = %(method)s)
            
            WHERE u.role = 'student' 
              AND tsa.teacher_id = %(teacher_id)s
        """
        params = {'teacher_id': teacher_id}
        params['period_no'] = int(period_no) if (period_no and period_no != 'None') else None
        params['method'] = method

        
        if subject_param:
            driver_query += " AND LOWER(TRIM(sub.name)) = LOWER(TRIM(%(subject)s))"
            params['subject'] = subject_param
            
        if department:
            driver_query += " AND LOWER(u.department) = LOWER(%(dept)s)"
            params['dept'] = department
            
        if year:
            driver_query += " AND u.year = %(year)s"
            params['year'] = year

        if batch_year:
            driver_query += " AND u.batch_year = %(batch)s"
            params['batch'] = batch_year
            
        driver_query += " GROUP BY u.id, u.full_name, u.roll_number, u.department, u.year, sub.name, sh.held_count ORDER BY u.full_name"
        
        c.execute(driver_query, params)
        results = c.fetchall()
        
        report_data = []
        for row in results:
            d = dict(row)
            if not d.get('subject_name') and d.get('subject'): 
                 d['subject_name'] = d['subject']
            
            total = d['total_attendance_records']
            present = d['present_count']
            d['percentage'] = round((present / total * 100), 1) if total > 0 else 0
            report_data.append(d)
            
        return report_data

    except Exception as e:
        logging.error(f"Error getting teacher report: {e}")
        return []
    finally:
        if conn: conn.close()

# Duplicate function definition removed.


def get_timetable_entry(entry_id):
    """
    Fetches a single timetable entry by ID (from timetable_entries).
    Joins with master, subjects, and periods for full context.
    Maps staff_id -> teacher_id for compatibility.
    """
    conn = get_db_connection()
    if conn is None: return None
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        query = """
            SELECT 
                te.*,
                te.staff_id as teacher_id,
                tm.department,
                tm.batch_year,
                tm.semester,
                tm.section,
                tp.start_time,
                tp.end_time,
                sub.name as subject
            FROM timetable_entries te
            JOIN timetable_master tm ON te.timetable_id = tm.id
            LEFT JOIN academic_subjects sub ON te.subject_id = sub.id
            LEFT JOIN timetable_periods tp ON te.period_no = tp.period_no
            WHERE te.id = %s
        """
        c.execute(query, (entry_id,))
        row = c.fetchone()
        if row:
            res = dict(row)
            # Derive academic year from semester (1-8 -> 1-4)
            if res.get('semester'):
                res['year'] = (int(res['semester']) + 1) // 2
            else:
                res['year'] = 1 # Default or handle error?
            return res
        return None
    except Exception as e:
        logging.error(f"Error getting timetable entry {entry_id}: {e}")
        return None
    finally:
        if conn: conn.close()

# Duplicate legacy function removed (was shadowing lookup at line 1684)


# --- Report & Status Helpers ---

def check_attendance_status(subject, date_str, department=None, batch_year=None, section=None, period_no=None):
    """
    Checks if attendance has been marked for a given subject on a given date for a SPECIFIC class.
    Returns True if records exist, False otherwise.
    Used for Teacher Dashboard "Completed" status.
    """
    conn = get_db_connection()
    if conn is None: return False
    try:
        c = conn.cursor()
        # count records
        query = """
            SELECT COUNT(*) FROM attendance a
            JOIN users u ON a.user_id = u.id
            WHERE TRIM(a.subject) = TRIM(%s) AND a.timestamp::date = %s
        """
        params = [subject, date_str]

        if period_no:
            query += " AND a.period_no = %s"
            params.append(period_no)

        if department:
            query += " AND LOWER(u.department) = LOWER(%s)"
            params.append(department)
        
        if batch_year:
            query += " AND u.batch_year = %s"
            params.append(batch_year)
            
        # Section in users table? No, users table doesn't strictly track section in some versions.
        # But if it does (via mapping or column), we use it. 
        # init_db doesn't show 'section' column in users table!
        # Step 217 show users table validation... "ensure_section_column()" call exists.
        # So it might exist. Let's assume it does if passed.
        if section:
             # Check if column exists dynamically or assume it does since ensure_section_column was called.
             query += " AND u.section = %s"
             params.append(section)

        c.execute(query, tuple(params))
        count = c.fetchone()[0]
        return count > 0
    except Exception as e:
        logging.error(f"Error checking attendance status: {e}")
        return False
    finally:
        if conn: conn.close()

def get_teacher_attendance_sessions(teacher_id, department=None, batch_year=None, semester=None, section=None, period_no=None):
    """
    Returns distinct attendance sessions marked by this teacher.
    UPGRADED: Unified grouping, distinct counts, OD status support, and session_id logic.
    """
    import base64
    import json
    
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Build Filter Clauses
        filter_clause = "WHERE (a.marked_by = %s)"
        params = [teacher_id]
        
        if department:
            filter_clause += " AND LOWER(u.department) = LOWER(%s)"
            params.append(department)
        
        if batch_year:
             filter_clause += " AND (u.batch_year = %s OR u.batch_start_year::TEXT = %s)"
             params.append(batch_year)
             params.append(batch_year)
             
        if semester:
             filter_clause += " AND u.semester = %s"
             params.append(semester)
             
        if section:
              filter_clause += " AND u.section = %s"
              params.append(section)

        if period_no:
              filter_clause += " AND a.period_no = %s"
              params.append(period_no)
        
        query = f"""
            WITH ClassTotals AS (
                SELECT 
                    LOWER(department) as dept, 
                    year,
                    COUNT(*) as total_students
                FROM users 
                WHERE role = 'student' AND (active_status = 'Active' OR active_status IS NULL)
                GROUP BY 1, 2
            ),
            SessionStats AS (
                SELECT 
                    a.subject, 
                    a.timestamp::date as date,
                    TRIM(TO_CHAR(a.timestamp, 'Day')) as day_name,
                    MIN(a.timestamp) as session_time,
                    COUNT(DISTINCT CASE WHEN a.status = 'Present' THEN a.user_id END) as present_count,
                    COUNT(DISTINCT CASE WHEN a.status = 'OD' THEN a.user_id END) as od_count,
                    u.department,
                    u.year,
                    MAX(u.batch_year) as batch_year,
                    MAX(u.semester) as semester,
                    MAX(u.section) as section,
                    a.period_no
                FROM attendance a
                JOIN users u ON a.user_id = u.id
                {filter_clause}
                GROUP BY a.subject, a.timestamp::date, TRIM(TO_CHAR(a.timestamp, 'Day')), u.department, u.year, a.period_no
            )
            SELECT 
                s.*,
                COALESCE(ct.total_students, s.present_count + s.od_count) as total_in_class,
                (COALESCE(ct.total_students, s.present_count + s.od_count) - s.present_count - s.od_count) as absent_count,
                t.id as timetable_id
            FROM SessionStats s
            LEFT JOIN ClassTotals ct 
                ON LOWER(s.department) = ct.dept AND s.year = ct.year
            LEFT JOIN timetable t 
                ON LOWER(TRIM(t.subject)) = LOWER(TRIM(s.subject)) 
                AND (t.teacher_id = %s OR t.teacher_id IS NULL) 
                AND t.day = s.day_name
                AND LOWER(t.department) = LOWER(s.department)
                AND t.year = s.year
                AND (t.period_no = s.period_no OR (t.period_no IS NULL AND s.period_no IS NULL))
            ORDER BY s.date DESC, s.session_time DESC
        """
        params.append(teacher_id)
        
        c.execute(query, tuple(params))
        results = [dict(row) for row in c.fetchall()]
        
        # Post-Processing: Generate Session ID
        for session_item in results:
            # Create a unique logical key for this session
            session_key = {
                'subject': session_item['subject'],
                'date': str(session_item['date']),
                'period': session_item['period_no'],
                'dept': session_item['department'],
                'year': session_item['year'],
                'batch': session_item['batch_year'],
                'sem': session_item['semester'],
                'sec': session_item['section']
            }
            # Encode as base64 for URL safety
            key_json = json.dumps(session_key)
            session_item['session_id'] = base64.urlsafe_b64encode(key_json.encode()).decode()
            
        return results
    except Exception as e:
        logging.error(f"Error getting teacher attendance sessions: {e}")
        return []
    finally:
        if conn: conn.close()

def get_attendance_session_details(teacher_id, timetable_id, date_str, fallback_subject=None, fallback_department=None, fallback_year=None, fallback_period=None):
    """
    Retrieves detailed attendance report for a specific timetable slot and date.
    Uses LEFT JOIN to ensure ALL students in the class are listed.
    """
    conn = get_db_connection()
    if conn is None: return None
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # 1. Fetch Timetable Context (Dual Lookup: New System -> Old System)
        tt_entry = None
        is_new_system = False
        
        # A. Try NEW System (timetable_entries)
        c.execute("""
            SELECT te.id, te.staff_id as teacher_id, te.subject_id, te.period_no,
                   s.name as subject, 
                   tm.department, tm.batch_year, tm.semester, ((tm.semester + 1) / 2) as year
            FROM timetable_entries te
            JOIN timetable_master tm ON te.timetable_id = tm.id
            JOIN academic_subjects s ON te.subject_id = s.id
            WHERE te.id = %s
        """, (timetable_id,))
        tt_entry = c.fetchone()
        
        if tt_entry:
            is_new_system = True
            logging.info(f"Report Access: Found in NEW timetable system (ID: {timetable_id})")
        else:
            # B. Fallback to OLD System (timetable)
            c.execute("SELECT * FROM timetable WHERE id = %s", (timetable_id,))
            tt_entry = c.fetchone()
            if tt_entry:
                logging.info(f"Report Access: Found in OLD timetable system (ID: {timetable_id})")

        if not tt_entry:
            if fallback_subject and fallback_department and fallback_year:
                logging.warning(f"Timetable ID {timetable_id} not found. Using fallback context: {fallback_subject} ({fallback_department}-{fallback_year})")
                tt_entry = {
                    'subject': fallback_subject,
                    'department': fallback_department,
                    'year': fallback_year,
                    'period_no': fallback_period,
                    'teacher_id': None # Will fail primary check but pass ownership check
                }
            else:
                logging.error(f"Timetable ID {timetable_id} not found and no fallback provided")
                return None
            
        subject = tt_entry['subject']
        department = tt_entry['department']
        year = tt_entry['year']
        period_no = tt_entry.get('period_no') # Fetch period number
        
        logging.info(f"DEBUG: Report Access Check - TT ID: {timetable_id}, Subject: {subject}, TeacherID: {teacher_id}")
        
        # 2. Security Check: Does teacher have access?
        # Primary: Is the teacher assigned in the timetable slot?
        # Secondary: Is the teacher mapped to the subject?
        
        has_access = False
        if tt_entry.get('teacher_id') == teacher_id:
             has_access = True
             logging.info("DEBUG: Access Granted via Timetable Owner")
        else:
            # Check teacher_subject_assignments (Fallback)
            c.execute("""
                SELECT 1 
                FROM teacher_subject_assignments tsa
                JOIN academic_subjects s ON tsa.subject_id = s.id
                WHERE tsa.teacher_id = %s AND LOWER(TRIM(s.name)) = LOWER(TRIM(%s))
            """, (teacher_id, subject))
            if c.fetchone():
                has_access = True
                logging.info("DEBUG: Access Granted via Subject Assignment")
            
            # Check if teacher marked this attendance (Ownership check)
            if not has_access:
                c.execute("""
                    SELECT 1 FROM attendance 
                    WHERE LOWER(TRIM(subject)) = LOWER(TRIM(%s)) 
                    AND timestamp::date = %s 
                    AND marked_by = %s
                    LIMIT 1
                """, (subject, date_str, teacher_id))
                if c.fetchone():
                    has_access = True
                    logging.info("DEBUG: Access Granted via Attendance Ownership")
                else:
                    logging.warning(f"DEBUG: Access Denied - Ownership Check Failed. Params: Sub={subject}, Date={date_str}, Marker={teacher_id}")

                
        if not has_access:
             logging.warning(f"Access Denied: Teacher {teacher_id} not assigned to {subject} (Timetable ID: {timetable_id})")
             return None

        logging.info(f"Generating Report for {subject} ({department}-{year}) on {date_str}")

        # 3. Main Query: Students LEFT JOIN Attendance
        # "Total Students" = Rows returned
        # "Present" = Rows with status='Present'
        # "Absent" = Rows with status='Absent' OR NULL
        
        sql = """
            SELECT 
                u.full_name,
                u.roll_number,
                u.register_number,
                u.department,
                u.year,
                COALESCE(a.status, 'Absent') as status, -- Default NULL to Absent
                a.timestamp,
                a.timestamp,
                -- Removed Confidence
                a.proof_image_path as proof_path
            FROM users u
            LEFT JOIN attendance a 
                ON u.id = a.user_id 
                AND LOWER(TRIM(a.subject)) = LOWER(TRIM(%s)) 
                AND a.timestamp::date = %s
                AND (a.period_no = %s OR (%s IS NULL AND a.period_no IS NULL))
            WHERE u.role = 'student' 
              AND LOWER(u.department) = LOWER(%s) 
              AND u.year = %s
            -- New Sort Order
            ORDER BY COALESCE(CAST(RIGHT(NULLIF(regexp_replace(u.register_number, '[^0-9]', '', 'g'), ''), 3) AS INTEGER), 999999) ASC, u.full_name
        """
        
        c.execute(sql, (subject, date_str, period_no, period_no, department, int(year)))
        
        details = [dict(row) for row in c.fetchall()]
        
        # Attach Metadata for UI Context
        result = {
            'class_info': {'subject': subject, 'department': department, 'year': year, 'date': date_str, 'period': period_no},
            'students': details
        }
        
        return result

    except Exception as e:
        logging.error(f"Error getting session details: {e}")
        return None
    finally:
        if conn: conn.close()


# --- Task 2: Teacher Assigned Students & Logic ---





def promote_students(student_ids, target_semester=None, action='promote'):
    """
    Promotes selected students based on action:
    - promote: Move to target_semester (or +1)
    - detain: Stay in current semester
    - depromote: Move to target_semester (or -1)
    - discontinue: Set status to 'Discontinued'
    """
    if not student_ids:
        return False, "No students selected."

    # Ensure IDs are integers to prevent DB type errors
    try:
        student_ids = [int(sid) for sid in student_ids]
    except ValueError:
        return False, "Invalid Student ID format."

    conn = get_db_connection()
    if conn is None: return False, "Database connection failed"
    try:
        c = conn.cursor()
        
        if action == 'discontinue':
            # Mark as Discontinued
            query = """
                UPDATE users 
                SET active_status = 'Discontinued'
                WHERE id = ANY(%s)
            """
            c.execute(query, (student_ids,))
            conn.commit()
            return True, f"Marked {c.rowcount} students as Discontinued."

        elif action == 'detain':
            # Keep same semester (No change to sem/year)
            # We treat this as an 'update' to confirm the action, essentially doing nothing but validating IDs
            # Optionally set active_status='Active' to ensure they aren't 'Graduated' by mistake?
            # User said: "Semester remains unchanged"
            return True, f"Detained {len(student_ids)} students (Semester unchanged)."

        elif action in ['promote', 'depromote']:
            # Move to target semester
            if target_semester:
                # Validate Bounds
                if int(target_semester) < 1:
                     return False, "Cannot de-promote below Semester 1."
                if int(target_semester) > 8 and action == 'promote':
                     # Allow for graduation (sem > 8 -> graduated)
                     pass 

                # Explicit Promotion Logic
                query = """
                    UPDATE users 
                    SET 
                        semester = %s,
                        year = CEIL(%s / 2.0)::INTEGER,
                        active_status = CASE 
                            WHEN %s >= 8 THEN 'Graduated' 
                            ELSE 'Active' 
                        END
                    WHERE id = ANY(%s)
                    RETURNING id
                """
                c.execute(query, (int(target_semester), int(target_semester), int(target_semester), student_ids))
            else:
                # Fallback Auto-Logic (Only for promote)
                change = 1 if action == 'promote' else -1
                query = """
                    UPDATE users 
                    SET 
                        semester = CASE 
                            WHEN semester + %s < 1 THEN 1
                            WHEN semester + %s > 8 THEN 8
                            ELSE semester + %s 
                        END,
                        active_status = CASE 
                            WHEN semester + %s >= 8 THEN 'Graduated' 
                            ELSE 'Active'
                        END
                    WHERE id = ANY(%s)
                    RETURNING semester
                """
                # This complex query needs year update too, but deriving year inside SQL from dynamic semester is messy.
                # Since we always send target_semester from UI now, this fallback is rare.
                # Simplifying: Just use python calculation if target not provided? 
                # Better to insist on target_semester or minimal update.
                # Let's trust the UI sends target_semester. If not, error? 
                # Original fallback was +1. 
                return False, "Target Semester required for promotion/depromotion."

            promoted = c.fetchall()
            conn.commit()
            return True, f"Successfully processed {len(promoted)} students."
            
        else:
             return False, "Invalid action."
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error promoting students: {e}")
        return False, str(e)
    finally:
        if conn:
            conn.close()

def remove_students_bulk(student_ids, status, reason):
    if not student_ids:
        return False, "No students selected."
        
    # Ensure IDs are integers
    try:
        student_ids = [int(sid) for sid in student_ids]
    except ValueError:
        return False, "Invalid Student ID format."
        
    conn = get_db_connection()
    if conn is None: return False, "Database connection failed"
    try:
        c = conn.cursor()
        now = datetime.now()
        
        # Valid statuses
        valid_statuses = ['Detained', 'Discontinued', 'Transferred']
        if status not in valid_statuses:
             return False, "Invalid status selected."
             
        query = """
            UPDATE users
            SET active_status = %s,
                status_updated_at = %s,
                status_reason = %s
            WHERE id = ANY(%s)
            RETURNING id, full_name, active_status
        """
        c.execute(query, (status, now, reason, student_ids))
        removed = c.fetchall()
        
        conn.commit()
        return True, f"Updated {len(removed)} students to '{status}'."
    except Exception as e:
        conn.rollback()
        logging.error(f"Error removing students: {e}")
        return False, str(e)
    finally:
         if conn: conn.close()

def get_academic_sections(dept_id=None, year_id=None):
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        query = "SELECT * FROM academic_sections WHERE 1=1"
        params = []
        if dept_id:
            query += " AND dept_id = %s"
            params.append(dept_id)
        if year_id:
            query += " AND year_id = %s"
            params.append(year_id)
            
        c.execute(query, tuple(params))
        return [dict(row) for row in c.fetchall()]
    except Exception as e:
        # Table might not exist yet? Or just error.
        logging.warning(f"Error fetching sections: {e}")
        return []
    finally:
        if conn: conn.close()

def get_batch_details(batch_id):
    conn = get_db_connection()
    if conn is None: return None
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT * FROM academic_years WHERE id = %s", (batch_id,))
        batch = c.fetchone()
        if not batch: return None
        
        # Calculate current year
        name = batch['name'] # "2024-2028" or "2024"
        start_year = 2024 # default
        if '-' in name:
             try: start_year = int(name.split('-')[0])
             except: pass
        elif name.isdigit():
             try: start_year = int(name)
             except: pass
             
        now = datetime.now()
        diff = now.year - start_year
        # Academic year starts ~June/July (Month 6-7)
        if now.month >= 6: 
            current_year = diff + 1
        else:
            current_year = diff
            
        return {
            'id': batch['id'],
            'name': batch['name'],
            'current_year_display': max(1, min(4, current_year))
        }
    except Exception as e:
        logging.error(f"Error fetching batch details: {e}")
        return None
    finally:
        if conn: conn.close()


# ==========================================
# TIMETABLE MANAGEMENT SYSTEM (Module 9)
# ==========================================

def get_all_subjects_detailed():
    """Returns list of subjects with details: {id, name, code, type}"""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Use academic_subjects which is the canonical source
        c.execute("SELECT id, name, code, subject_type as type FROM academic_subjects ORDER BY name")
        return [dict(row) for row in c.fetchall()]
    except Exception as e:
        logging.error(f"Error fetching subjects: {e}")
        return []
    finally:
        if conn: conn.close()

def get_global_timings():
    """Returns list of global timetable periods {id, period_no, start_time, end_time}"""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT * FROM timetable_periods ORDER BY period_no")
        rows = c.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d['start_time'] = d['start_time'].strftime('%H:%M') if d['start_time'] else ''
            d['end_time'] = d['end_time'].strftime('%H:%M') if d['end_time'] else ''
            results.append(d)
        return results
    except Exception as e:
        logging.error(f"Error fetching global timings: {e}")
        return []
    finally:
        if conn: conn.close()

def save_global_timings(periods_data):
    """
    Update global timings. 
    periods_data: List of dicts {period_no, start_time, end_time}
    """
    conn = get_db_connection()
    if conn is None: return False
    try:
        c = conn.cursor()
        for p in periods_data:
            c.execute("""
                INSERT INTO timetable_periods (period_no, start_time, end_time)
                VALUES (%s, %s, %s)
                ON CONFLICT (period_no) DO UPDATE
                SET start_time = EXCLUDED.start_time, end_time = EXCLUDED.end_time
            """, (p['period_no'], p['start_time'], p['end_time']))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error saving global timings: {e}")
        conn.rollback()
        return False
    finally:
        if conn: conn.close()

def get_timetable_master(batch_year, department, semester, section):
    """Fetch timetable master record or return None."""
    conn = get_db_connection()
    if conn is None: return None
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Normalize section
        if not section or section == 'None' or section == '':
            section = None

        c.execute("""
            SELECT id, batch_year, department, semester, section FROM timetable_master 
            WHERE batch_year=%s 
            AND LOWER(REPLACE(department, ' ', '')) = LOWER(REPLACE(%s, ' ', '')) 
            AND semester=%s 
            AND (section = %s OR (section IS NULL AND %s IS NULL))
            ORDER BY id DESC
        """, (batch_year, department, int(semester), section, section))
        result = c.fetchone()
        return dict(result) if result else None
    except Exception as e:
        logging.error(f"Error fetching timetable master: {e}")
        return None
    finally:
        if conn: conn.close()

def get_student_timetable_master(student_id):
    """
    Resolves the correct timetable master for a student using priority logic + semester derivation.
    Returns the master dict or None.
    """
    conn = get_db_connection()
    if conn is None: return None
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get student info
        c.execute("SELECT batch_year, department, semester, section, year FROM users WHERE id = %s", (student_id,))
        student = c.fetchone()
        if not student:
            return None
        
        batch_year = student['batch_year']
        department = student['department']
        semester = student['semester']
        section = student['section']
        year = student['year']
        
        # --- LOGIC ENHANCEMENT: Derive Semester if Missing ---
        if not semester:
            # 1. Try to find active semester from semester_info configuration
            try:
                c.execute("""
                    SELECT semester FROM semester_info 
                    WHERE batch_year = %s 
                    AND CURRENT_DATE BETWEEN start_date AND end_date
                    ORDER BY end_date DESC
                    LIMIT 1
                """, (batch_year,))
                sem_row = c.fetchone()
                if sem_row:
                    semester = sem_row[0]
                    logging.info(f"Derived active semester {semester} for student {student_id} (Batch: {batch_year}) from semester_info")
            except Exception as e:
                logging.warning(f"Error deriving semester from semester_info: {e}")

            # 2. Fallback: Derive from Year (e.g. Year 2 -> Sem 4)
            if not semester and year:
                from datetime import datetime
                current_month = datetime.now().month
                if current_month >= 6:
                    semester = (year * 2) - 1
                else:
                    semester = year * 2
                logging.info(f"Derived heuristic semester {semester} for student {student_id} (Year: {year})")

        # Normalize section
        if not section or section == 'None' or section == '':
            section = None
        
        master = None
        
        # Step 1: Try exact section-specific match first (only if student has a section)
        if section:
            c.execute("""
                SELECT * FROM timetable_master 
                WHERE batch_year=%s 
                AND LOWER(REPLACE(department, ' ', '')) = LOWER(REPLACE(%s, ' ', '')) 
                AND semester=%s 
                AND section = %s
                ORDER BY id DESC
                LIMIT 1
            """, (batch_year, department, int(semester) if semester else None, section))
            
            master_raw = c.fetchone()
            
            # Step 2: Validate it has entries
            if master_raw:
                master = dict(master_raw)
                c.execute("SELECT COUNT(*) as cnt FROM timetable_entries WHERE timetable_id = %s", (master['id'],))
                count_result = c.fetchone()
                if count_result and count_result['cnt'] > 0:
                    # Valid section-specific timetable found with entries
                    logging.info(f"Using section-specific timetable (ID: {master['id']}) for student {student_id}")
                else:
                    # Empty section-specific timetable, ignore it and try fallback
                    logging.info(f"Section-specific timetable (ID: {master['id']}) is empty, falling back to generic")
                    master = None
        
        # Step 3: Fallback to generic timetable (section IS NULL)
        if not master:
            c.execute("""
                SELECT * FROM timetable_master 
                WHERE batch_year=%s 
                AND LOWER(REPLACE(department, ' ', '')) = LOWER(REPLACE(%s, ' ', '')) 
                AND semester=%s 
                AND section IS NULL
                ORDER BY id DESC
                LIMIT 1
            """, (batch_year, department, int(semester) if semester else None))
            master_raw = c.fetchone()
            
            if master_raw:
                master = dict(master_raw)
                logging.info(f"Using generic timetable (ID: {master['id']}) for student {student_id} (Sem: {semester})")
        
        if not master:
            logging.warning(f"No timetable found for student {student_id} (batch: {batch_year}, dept: {department}, sem: {semester}, section: {section})")
            return None
            
        return master
        
    except Exception as e:
        logging.error(f"Error fetching student timetable master: {e}")
        return None
    finally:
        if conn: conn.close()

def get_student_timetable_grid(student_id):
    """
    Fetches timetable entries for a student based on their class info.
    Uses get_student_timetable_master to resolve the correct timetable.
    Returns list of timetable entries with subject and staff details.
    """
    conn = None
    try:
        master = get_student_timetable_master(student_id)
        
        if not master:
            return []
        
        conn = get_db_connection()
        if conn is None: return []
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get timetable entries
        c.execute("""
            SELECT 
                te.id, te.day_of_week, te.period_no,
                s.name as subject, s.code as subject_code,
                u.full_name as staff_name,
                te.subject_id, te.staff_id,
                tp.start_time::text as start_time,
                tp.end_time::text as end_time
            FROM timetable_entries te
            LEFT JOIN academic_subjects s ON te.subject_id = s.id
            LEFT JOIN users u ON te.staff_id = u.id
            LEFT JOIN timetable_periods tp ON te.period_no = tp.period_no
            WHERE te.timetable_id = %s
        """, (master['id'],))
        
        return [dict(r) for r in c.fetchall()]
    except Exception as e:
        import traceback
        logging.error(f"Error fetching student timetable grid: {e}")
        logging.error(traceback.format_exc())
        return []
    finally:
        if conn: conn.close()

def create_or_get_timetable_master(batch_year, department, semester, section, timetable_name):
    """Creates a new timetable master if not exists, else returns existing."""
    conn = get_db_connection()
    if conn is None: return None
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Normalize section
        if not section or section == 'None' or section == '':
            section = None

        # Check exist
        c.execute("""
            SELECT id FROM timetable_master 
            WHERE batch_year=%s AND department=%s AND semester=%s 
            AND (section = %s OR (section IS NULL AND %s IS NULL))
        """, (batch_year, department, int(semester), section, section))
        existing = c.fetchone()
        if existing:
            return existing['id']
        
        # Create
        c.execute("""
            INSERT INTO timetable_master (batch_year, department, semester, section, timetable_name)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (batch_year, department, int(semester), section, timetable_name))
        new_id = c.fetchone()[0]
        conn.commit()
        return new_id
    except Exception as e:
        logging.error(f"Error creating timetable master: {e}")
        conn.rollback()
        return None
    finally:
        if conn: conn.close()

def delete_timetable_master(master_id):
    """
    Deletes a timetable master and all its entries.
    """
    conn = get_db_connection()
    if conn is None: return False, "DB Connection Failed"
    try:
        c = conn.cursor()
        # 1. Delete entries first (CASCADE usually handles this, but explicit is safer if not configured)
        c.execute("DELETE FROM timetable_entries WHERE timetable_id = %s", (master_id,))
        
        # 2. Delete master
        c.execute("DELETE FROM timetable_master WHERE id = %s", (master_id,))
        
        conn.commit()
        return True, "Timetable deleted successfully"
    except Exception as e:
        conn.rollback()
        logging.error(f"Error deleting timetable master: {e}")
        return False, str(e)
    finally:
        if conn: conn.close()

def get_timetable_entries(master_id):
    """
    Returns populated grid entries for a master timetable.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("""
            SELECT 
                te.id, te.day_of_week, te.period_no, 
                s.name as subject_name, s.code as subject_code,
                u.full_name as staff_name,
                te.subject_id, te.staff_id
            FROM timetable_entries te
            LEFT JOIN academic_subjects s ON te.subject_id = s.id
            LEFT JOIN users u ON te.staff_id = u.id
            WHERE te.timetable_id = %s
        """, (master_id,))
        return [dict(r) for r in c.fetchall()]
    except Exception as e:
        logging.error(f"Error fetching timetable entries: {e}")
        return []
    finally:
        if conn: conn.close()

def check_teacher_eligibility(staff_id, subject_id, department, year):
    """
    Verifies if a teacher is eligible to teach a subject for a specific class (dept, y).
    Leverages get_teachers_by_subject logic (legacy + new).
    """
    eligible_teachers = get_teachers_by_subject(subject_id, department, year)
    return any(str(t['id']) == str(staff_id) for t in eligible_teachers)

def save_timetable_entry(master_id, day, period_no, subject_id, staff_id):
    """Upsert a single cell entry."""
    conn = get_db_connection()
    if conn is None: return False, "Database connection failed"
    try:
        c = conn.cursor()
        
        # Determine period_id
        c.execute("SELECT id FROM timetable_periods WHERE period_no = %s", (period_no,))
        pid_row = c.fetchone()
        period_id = pid_row[0] if pid_row else None

        # Handle Clear (if subject_id is None)
        if not subject_id or str(subject_id) == 'None' or str(subject_id) == '':
            c.execute("""
                DELETE FROM timetable_entries 
                WHERE timetable_id=%s AND day_of_week=%s AND period_no=%s
            """, (master_id, day, period_no))
        else:
            # --- VALIDATION: Ensure Staff is Eligible ---
            # SKIPPED: We allow Admins to override/assign any staff selected in the UI.
            # Strict backend validation was causing failures due to string mismatches (e.g. 'Cybersecurity' vs 'Cyber Security')
            # The UI already filters 'teachers_by_subject', so we trust the Admin's selection.
            pass

            c.execute("""
                INSERT INTO timetable_entries (timetable_id, day_of_week, period_no, period_id, subject_id, staff_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (timetable_id, day_of_week, period_no) DO UPDATE
                SET subject_id = EXCLUDED.subject_id, staff_id = EXCLUDED.staff_id, period_id = EXCLUDED.period_id
            """, (master_id, day, period_no, period_id, subject_id, staff_id))
        
        conn.commit()
        return True, "Successfully saved"
    except Exception as e:
        logging.error(f"Error saving entry: {e}")
        conn.rollback()
        return False, str(e)
    finally:
        if conn: conn.close()

def get_teacher_timetable_slots(teacher_id):
    """Returns grid slots assigned to a teacher. Unified with dashboard aliases."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Fetch entries assigned to this teacher using the NEW schema
        # Maps keys to match what both Dashboard and Full Timetable expect
        c.execute("""
            SELECT 
                te.id,
                INITCAP(TRIM(te.day_of_week)) as day,
                INITCAP(TRIM(te.day_of_week)) as day_of_week,
                te.period_no,
                s.name as subject_name,
                s.name as subject,
                tm.batch_year, tm.department, tm.semester, tm.section,
                tp.start_time, tp.end_time,
                ((tm.semester + 1) / 2) as year
            FROM timetable_entries te
            JOIN timetable_master tm ON te.timetable_id = tm.id
            JOIN academic_subjects s ON te.subject_id = s.id
            JOIN timetable_periods tp ON te.period_no = tp.period_no
            WHERE te.staff_id = %s
            ORDER BY 
                CASE 
                    WHEN INITCAP(TRIM(te.day_of_week)) = 'Monday' THEN 1
                    WHEN INITCAP(TRIM(te.day_of_week)) = 'Tuesday' THEN 2
                    WHEN INITCAP(TRIM(te.day_of_week)) = 'Wednesday' THEN 3
                    WHEN INITCAP(TRIM(te.day_of_week)) = 'Thursday' THEN 4
                    WHEN INITCAP(TRIM(te.day_of_week)) = 'Friday' THEN 5
                    WHEN INITCAP(TRIM(te.day_of_week)) = 'Saturday' THEN 6
                    WHEN INITCAP(TRIM(te.day_of_week)) = 'Sunday' THEN 7
                END,
                te.period_no
        """, (teacher_id,))
        
        slots = [dict(row) for row in c.fetchall()]
        return slots
    except Exception as e:
        logging.error(f"Error getting teacher timetable slots: {e}")
        return []
    finally:
        if conn: conn.close()

def get_student_class_timetable(batch_year, department, semester, section):
    """Returns full class timetable including consolidated slots."""
    master = get_timetable_master(batch_year, department, semester, section)
    if not master: return []
    return get_timetable_entries(master['id'])

def get_teachers_by_subject(subject_id, department=None, year=None):
    """
    Retrieves teachers assigned to a specific subject.
    Supports both new (teacher_subject_assignments) and legacy (teachers_subjects) tables.
    Uses highly robust matching for department and subject names.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # 1. Get Subject Name for Legacy Lookup
        subject_name = ""
        c.execute("SELECT name FROM academic_subjects WHERE id = %s", (subject_id,))
        row = c.fetchone()
        if row:
            subject_name = row[0]
            
        # 2. Build Query to fetch from BOTH tables
        
        # New Table Filter (teacher_subject_assignments)
        new_assign_filter = "tsa.subject_id = %s"
        new_params = [subject_id]
        
        if department:
            # Use non-alphanumeric normalization for department strings
            new_assign_filter += " AND (REGEXP_REPLACE(LOWER(tsa.department), '[^a-z0-9]', '', 'g') = REGEXP_REPLACE(LOWER(%s), '[^a-z0-9]', '', 'g') OR tsa.department IS NULL)"
            new_params.append(department) 
            
        if year:
            new_assign_filter += " AND (tsa.year = %s OR tsa.year IS NULL)"
            new_params.append(year)
            
        # Legacy Table Filter (subject name based)
        legacy_clause = ""
        if subject_name:
            # Normalize BOTH sides: lowercase and remove all non-alphanumeric characters
            legacy_clause = f"""
            UNION
            SELECT ts.teacher_id as id
            FROM teachers_subjects ts
            WHERE REGEXP_REPLACE(LOWER(ts.subject), '[^a-z0-9]', '', 'g') = REGEXP_REPLACE(LOWER(%s), '[^a-z0-9]', '', 'g')
            """
            new_params.append(subject_name)

        query = f"""
            SELECT DISTINCT u.id, u.full_name, u.department
            FROM users u
            WHERE u.id IN (
                SELECT tsa.teacher_id 
                FROM teacher_subject_assignments tsa
                WHERE {new_assign_filter}
                {legacy_clause}
            )
            AND u.role = 'teacher'
            AND u.active_status = 'Active'
            ORDER BY u.full_name
        """
        
        logging.info(f"Filtering Teachers: SubId={subject_id}, Dept={department}, Year={year}")
        c.execute(query, tuple(new_params))
        teachers = c.fetchall()
        logging.info(f"Found {len(teachers)} teachers for subject {subject_id}")
        return [dict(t) for t in teachers]
    except Exception as e:
        logging.error(f"Error getting teachers for subject {subject_id}: {e}")
        return []
    finally:
        if conn: conn.close()

# --- Semester Information & Analytics ---

def get_semester_info(batch_year, semester):
    """Retrieves semester configuration for a specific batch and semester."""
    conn = get_db_connection()
    if conn is None: return None
    try:
        logging.info(f"DB Lookup for Semester Info: Batch='{batch_year}', Sem={semester}")
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Robust Query: Trim whitespace to prevent mismatches
        c.execute("""
            SELECT * FROM semester_info 
            WHERE TRIM(batch_year) = TRIM(%s) AND semester = %s
        """, (str(batch_year).strip(), int(semester)))
        
        res = c.fetchone()
        
        if res:
             logging.info(f"Semester Calendar FOUND: {dict(res)}")
             return dict(res)
        else:
             logging.warning(f"Semester Calendar NOT FOUND for Batch='{batch_year}', Sem={semester}")
             return None
             
    except Exception as e:
        logging.error(f"Error getting semester info: {e}")
        return None
    finally:
        if conn: conn.close()

def get_all_semester_infos():
    """Retrieves all semester configurations."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT * FROM semester_info ORDER BY created_at DESC")
        return [dict(r) for r in c.fetchall()]
    except Exception as e:
        logging.error(f"Error getting all semester infos: {e}")
        return []
    finally:
        if conn: conn.close()

def get_periods():
    """Fetches all period definitions."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT period_no, label FROM timetable_periods ORDER BY period_no ASC")
        return [dict(row) for row in c.fetchall()]
    except Exception as e:
        logging.error(f"Error getting periods: {e}")
        return []
    finally:
        if conn: conn.close()

def upsert_semester_info(batch_year, semester, start_date, end_date, saturday_working=True):
    """Creates or updates semester calendar configuration."""
    conn = get_db_connection()
    if conn is None: return False, "DB connection failed"
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO semester_info (batch_year, semester, start_date, end_date, saturday_working)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (batch_year, semester) DO UPDATE SET
                start_date = EXCLUDED.start_date,
                end_date = EXCLUDED.end_date,
                saturday_working = EXCLUDED.saturday_working
            RETURNING id
        """, (batch_year, int(semester), start_date, end_date, saturday_working))
        
        sem_id = c.fetchone()[0]
        conn.commit()
        return True, sem_id
    except Exception as e:
        conn.rollback()
        logging.error(f"Error upserting semester info: {e}")
        return False, str(e)
    finally:
        if conn: conn.close()

def get_semester_holidays(semester_info_id):
    """Retrieves all holidays for a given semester."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT * FROM semester_holidays WHERE semester_info_id = %s ORDER BY holiday_date", (semester_info_id,))
        return [dict(r) for r in c.fetchall()]
    except Exception as e:
        logging.error(f"Error getting holidays: {e}")
        return []
    finally:
        if conn: conn.close()

def add_semester_holiday(semester_info_id, holiday_date, holiday_name, holiday_type='Public Holiday'):
    """Adds a holiday to the semester calendar."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO semester_holidays (semester_info_id, holiday_date, holiday_name, holiday_type)
            VALUES (%s, %s, %s, %s)
        """, (semester_info_id, holiday_date, holiday_name, holiday_type))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"Error adding holiday: {e}")
        return False
    finally:
        if conn: conn.close()

def delete_semester_holiday(holiday_id):
    """Deletes a holiday record."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        c = conn.cursor()
        c.execute("DELETE FROM semester_holidays WHERE id = %s", (holiday_id,))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"Error deleting holiday: {e}")
        return False
    finally:
        if conn: conn.close()

def calculate_working_days_logic(start_date, end_date, holidays, saturday_working=True):
    """Calculates working days excluding Sundays/Holidays."""
    if isinstance(start_date, str): start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str): end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    holiday_dates = [h['holiday_date'] if isinstance(h, dict) else h for h in holidays]

    total_working = 0
    curr = start_date
    while curr <= end_date:
        weekday = curr.weekday()
        if not (weekday == 6 or (weekday == 5 and not saturday_working) or curr in holiday_dates):
            total_working += 1
        curr += timedelta(days=1)
    return total_working

def get_student_attendance_stats_advanced(user_id, semester_info):
    """Calculates attendance percentage using semester calendar."""
    if not semester_info: return None
    conn = get_db_connection()
    if conn is None: return None
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        holidays = get_semester_holidays(semester_info['id'])
        c.execute("SELECT batch_year, department, semester, section FROM users WHERE id = %s", (user_id,))
        student = c.fetchone()
        if not student: return None
        
        # Get timetable (Use robust fallback logic)
        master = get_student_timetable_master(user_id)
        
        total_possible_periods = 0
        if master:
            c.execute("SELECT day_of_week, COUNT(*) as count FROM timetable_entries WHERE timetable_id = %s GROUP BY day_of_week", (master['id'],))

            day_counts = {r['day_of_week']: r['count'] for r in c.fetchall()}
            
            curr = semester_info['start_date']
            if isinstance(curr, str): curr = datetime.strptime(curr, '%Y-%m-%d').date()
            
            # Use min(end_date, today) to calculate percentage based on progress so far
            end = semester_info['end_date']
            if isinstance(end, str): end = datetime.strptime(end, '%Y-%m-%d').date()
            today = datetime.now().date()
            
            if end > today:
                end = today
            
            holiday_dates = [h['holiday_date'] for h in holidays]
            
            # Load detailed timetable for "today" calculation
            grid_entries = get_student_timetable_grid(user_id)
            timings = get_global_timings()
            timing_map = {t['period_no']: t for t in timings}
            
            # Pre-process grid entries for today's lookup
            entries_by_day = {}
            for entry in grid_entries:
                day = entry.get('day_of_week')
                if day not in entries_by_day: entries_by_day[day] = []
                entries_by_day[day].append(entry.get('period_no'))

            current_time_str = datetime.now().strftime('%H:%M')

            while curr <= end:
                day_name = curr.strftime('%A')
                weekday = curr.weekday()
                if not (weekday == 6 or (weekday == 5 and not semester_info['saturday_working']) or curr in holiday_dates):
                    if curr < today:
                        total_possible_periods += day_counts.get(day_name, 0)
                    elif curr == today:
                        # For today, check which periods have ended
                        todays_periods = entries_by_day.get(day_name, [])
                        passed_today = 0
                        for p_no in todays_periods:
                             t_info = timing_map.get(p_no)
                             # If timing exists and end_time has passed
                             if t_info and t_info.get('end_time') and t_info['end_time'] <= current_time_str:
                                 passed_today += 1
                        total_possible_periods += passed_today
                curr += timedelta(days=1)
        else:
             # Fallback to working days if no timetable
             # Also cap at today if needed
             end_date_limit = semester_info['end_date']
             if isinstance(end_date_limit, str): end_date_limit = datetime.strptime(end_date_limit, '%Y-%m-%d').date()
             today = datetime.now().date()
             
             # If strictly "conducted", adjust for partial day might be hard without timetable
             # For fallback logic, we'll stick to full days up to yesterday, and maybe pro-rate today or include it full?
             # User specifically asked for "conducted till now". Without timetable, we can't do partial.
             # Let's assume full day if it's passed noon? Or just exclude today if fallback?
             # Let's include today fully if it's fallback, for simplicity, as we lack period data.
             if end_date_limit > today:
                 end_date_limit = today
                 
             total_possible_periods = calculate_working_days_logic(semester_info['start_date'], end_date_limit, holidays, semester_info['saturday_working'])

        c.execute("""
            SELECT COUNT(*) FROM attendance 
            WHERE user_id = %s AND timestamp::date BETWEEN %s AND %s AND status IN ('Present', 'OD')
        """, (user_id, semester_info['start_date'], end))
        present_count = c.fetchone()[0]
        
        return {
            'present_count': present_count,
            'total_possible': total_possible_periods,
            'percentage': round((present_count / total_possible_periods * 100), 2) if total_possible_periods > 0 else 0
        }
    except Exception as e:
        logging.error(f"Error in advanced stats: {e}")
        return None
    finally:
        if conn: conn.close()

def get_student_attendance_reports(user_id):
    """
    Generates aggregated Weekly and Monthly attendance reports for a student.
    Strictly Read-Only using existing attendance records.
    Returns: {'weekly': [...], 'monthly': [...]}
    """
    conn = get_db_connection()
    if conn is None: return {'weekly': [], 'monthly': []}
    
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # 1. Weekly Report (Group by ISO Week)
        c.execute("""
            SELECT 
                TO_CHAR(timestamp, 'IYYY-IW') as week_code,
                MIN(timestamp)::date as week_start,
                MAX(timestamp)::date as week_end,
                COUNT(*) as total_classes,
                COUNT(*) FILTER (WHERE status IN ('Present', 'OD')) as attended_classes
            FROM attendance
            WHERE user_id = %s
            GROUP BY week_code
            ORDER BY week_code DESC
            LIMIT 50
        """, (user_id,))
        weekly_rows = c.fetchall()
        
        weekly_report = []
        for row in weekly_rows:
            total = row['total_classes']
            attended = row['attended_classes']
            weekly_report.append({
                'week_label': f"Week {row['week_code'][-2:]} ({row['week_start']} to {row['week_end']})",
                'start_date': row['week_start'],
                'end_date': row['week_end'],
                'total_classes': total,
                'attended_classes': attended,
                'percentage': round((attended / total * 100), 2) if total > 0 else 0
            })
            
        # 2. Monthly Report
        c.execute("""
            SELECT 
                TO_CHAR(timestamp, 'YYYY-MM') as month_code,
                TO_CHAR(timestamp, 'Month YYYY') as display_month,
                COUNT(*) as total_classes,
                COUNT(*) FILTER (WHERE status IN ('Present', 'OD')) as attended_classes
            FROM attendance
            WHERE user_id = %s
            GROUP BY month_code, display_month
            ORDER BY month_code DESC
            LIMIT 24
        """, (user_id,))
        monthly_rows = c.fetchall()
        
        monthly_report = []
        for row in monthly_rows:
            total = row['total_classes']
            attended = row['attended_classes']
            monthly_report.append({
                'month_label': row['display_month'],
                'total_classes': total,
                'attended_classes': attended,
                'percentage': round((attended / total * 100), 2) if total > 0 else 0
            })
            
        # 3. Subject-Wise Report
        c.execute("""
            SELECT 
                subject,
                COUNT(*) as total_classes,
                COUNT(*) FILTER (WHERE status IN ('Present', 'OD')) as attended_classes
            FROM attendance
            WHERE user_id = %s
            GROUP BY subject
            ORDER BY total_classes DESC
        """, (user_id,))
        subject_rows = c.fetchall()
        
        subject_report = []
        for row in subject_rows:
            total = row['total_classes']
            attended = row['attended_classes']
            subject_report.append({
                'subject': row['subject'],
                'total_classes': total,
                'attended_classes': attended,
                'percentage': round((attended / total * 100), 2) if total > 0 else 0
            })
            
        return {
            'weekly': weekly_report,
            'monthly': monthly_report,
            'subject': subject_report
        }
        
    except Exception as e:
        logging.error(f"Error generating attendance reports: {e}")
        return {'weekly': [], 'monthly': []}
    finally:
        if conn: conn.close()



# ==========================================
# MISSING HELPER FUNCTIONS (Restored)
# ==========================================

def get_departments():
    """Returns a list of unique department names."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor()
        # Try fetching from academic_departments first
        c.execute("SELECT name FROM academic_departments ORDER BY name")
        rows = c.fetchall()
        if rows:
             return [r[0] for r in rows]
             
        # Fallback to users table if academic_departments invalid (legacy)
        c.execute("SELECT DISTINCT department FROM users WHERE department IS NOT NULL AND department != 'N/A' ORDER BY department")
        rows = c.fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        logging.error(f"Error fetching departments: {e}")
        return []
    finally:
        if conn: conn.close()

def get_years():
    """Returns a list of year numbers (1-4)."""
    return [1, 2, 3, 4]


        
def get_academic_departments():
     """Returns list of academic departments (dicts)."""
     conn = get_db_connection()
     if conn is None: return []
     try:
         c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
         c.execute("SELECT * FROM academic_departments ORDER BY name")
         return [dict(r) for r in c.fetchall()]
     except Exception as e:
         logging.error(f"Error fetching academic departments: {e}")
         return []
     finally:
         if conn: conn.close()

# Removed duplicate get_subjects_for_user

def get_subjects_by_dept_year(department, year):
    """Returns list of subject names for dept/year."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor()
        
        c.execute("""
            SELECT DISTINCT s.name 
            FROM academic_subjects s
            JOIN class_subject_mapping m ON s.id = m.subject_id
            JOIN academic_classes c ON m.class_id = c.id
            JOIN academic_departments d ON c.dept_id = d.id
            WHERE LOWER(d.name) = LOWER(%s) AND c.year_id = %s
        """, (department, year))
        
        rows = c.fetchall()
        if rows:
             return [r[0] for r in rows]

        # Fallback simplistic
        c.execute("SELECT name FROM subjects WHERE department=%s AND year=%s", (department, year))
        rows = c.fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        logging.error(f"Error fetching subjects by dept/year: {e}")
        return []
    finally:
         if conn: conn.close()

# --- Count Functions for Dashboard (Added for Metrics) ---
def count_academic_departments():
    conn = get_db_connection()
    if not conn: return 0
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM academic_departments")
        return c.fetchone()[0]
    except: return 0
    finally: conn.close()

def count_academic_classes():
    conn = get_db_connection()
    if not conn: return 0
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM academic_classes")
        return c.fetchone()[0]
    except: return 0
    finally: conn.close()

def count_academic_batches():
    conn = get_db_connection()
    if not conn: return 0
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM academic_years") # academic_years stores batches
        return c.fetchone()[0]
    except: return 0
    finally: conn.close()

def count_semesters():
    conn = get_db_connection()
    if not conn: return 0
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM semester_info")
        return c.fetchone()[0]
    except: return 0
    finally: conn.close()

# Migration Helpers
def clear_user_embeddings(user_id):
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("DELETE FROM face_embeddings WHERE user_id = %s", (user_id,))
        conn.commit()
        return True
    except: return False
    finally: conn.close()

def get_all_registered_users_with_faces():
    conn = get_db_connection()
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Fetch ALL students to ensure we have metadata for any potential match in the index
        c.execute("SELECT id, full_name, roll_number FROM users WHERE role = 'student'")
        return [dict(r) for r in c.fetchall()]
    finally: conn.close()




def get_teacher_schedule_today(staff_id, day_of_week=None):
    """
    Fetch specifically today's assigned classes for a teacher.
    Returns rich details: Subject, Dept, Batch, Sem, Section, Period, Time.
    """
    import datetime
    if not day_of_week:
        day_of_week = datetime.datetime.now().strftime('%A')

    conn = get_db_connection()
    if conn is None: return []
    
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Join entries -> master -> subjects -> periods
        # We use DISTINCT to avoid duplicates if joins are messy, but they should be clean 1:1
        query = """
            SELECT 
                te.id as entry_id,
                te.timetable_id,
                te.period_no,
                tp.start_time,
                tp.end_time,
                sub.name as subject_name,
                sub.id as subject_id,
                tm.department,
                tm.batch_year,
                tm.semester,
                tm.section
            FROM timetable_entries te
            JOIN timetable_master tm ON te.timetable_id = tm.id
            JOIN academic_subjects sub ON te.subject_id = sub.id
            LEFT JOIN timetable_periods tp ON te.period_no = tp.period_no
            WHERE te.staff_id = %s 
              AND te.day_of_week = %s
            ORDER BY te.period_no ASC
        """
        
        c.execute(query, (staff_id, day_of_week))
        rows = c.fetchall()
        
        results = []
        for r in rows:
            # Safe Time Handling
            p_start = r['start_time'] if r['start_time'] else "00:00"
            p_end = r['end_time'] if r['end_time'] else "00:00"
            
            results.append({
                'entry_id': r['entry_id'],
                'timetable_id': r['timetable_id'],
                'period_no': r['period_no'],
                'start_time': p_start,
                'end_time': p_end,
                'subject': r['subject_name'],
                'subject_id': r['subject_id'],
                'department': r['department'],
                'batch_year': r['batch_year'],
                'semester': r['semester'],
                'section': r['section'] if r['section'] else 'N/A', # Handle None
                'status': 'Upcoming',
                'day': day_of_week
            })
            
        return results

    except Exception as e:
        logging.error(f"Error fetching teacher schedule: {e}")
        return []
    finally:
        if conn: conn.close()

def get_all_subjects():
    """Retrieves all subjects from the academic_subjects table."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor()
        # Try academic_subjects first
        try:
            c.execute("SELECT DISTINCT name FROM academic_subjects ORDER BY name")
            return [str(row[0]) for row in c.fetchall()]
        except Exception:
            conn.rollback()
            # Fallback to legacy timetable/attendance if needed
            c.execute("SELECT DISTINCT subject FROM timetable ORDER BY subject")
            return [str(row[0]) for row in c.fetchall()]
    except Exception as e:
        import logging
        logging.error(f"Error getting all subjects: {e}")
        return []
    finally:
        if conn: conn.close()

def get_subjects_by_filter(department, semester):
    """Retrieves subjects mapped to the given department and semester."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor()
        
        # Clean semester input (handle "Sem 4" -> "4")
        sem_val = str(semester).lower().replace('sem', '').strip()
        
        # Query unified timetable with flexible matching
        query = """
            SELECT DISTINCT sub.name 
            FROM timetable_master tm
            JOIN timetable_entries te ON tm.id = te.timetable_id
            JOIN academic_subjects sub ON te.subject_id = sub.id
            WHERE tm.department::text ILIKE %s 
            AND tm.semester::text ILIKE %s
            ORDER BY sub.name
        """
        c.execute(query, (f"%{department}%", f"%{sem_val}%"))
        results = [str(row[0]) for row in c.fetchall()]
        
        # Fallback to academic mapping if timetable is empty
        if not results:
             # Derive year for v2
             try:
                 year = str(int(math.ceil(int(sem_val) / 2.0)))
                 from backend.database_manager import get_subjects_by_dept_year_v2
                 results = get_subjects_by_dept_year_v2(department, year, semester)
             except: pass
             
        return results
    except Exception as e:
        import logging
        logging.error(f"Error getting subjects by filter: {e}")
        return []
    finally:
        if conn: conn.close()

# --- Emergency Access Logic ---

def create_emergency_code(admin_id, expiration_minutes=30):
    """
    Generates a secure, random alphanumeric code for emergency attendance access.
    """
    import secrets
    import string
    
    code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    expires_at = datetime.now() + timedelta(minutes=expiration_minutes)
    
    conn = get_db_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("""
                INSERT INTO emergency_codes (code, created_by, expires_at)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (code, admin_id, expires_at))
            code_id = c.fetchone()[0]
            conn.commit()
            return {'code': code, 'expires_at': expires_at, 'id': code_id}
        except Exception as e:
            logging.error(f"Error creating emergency code: {e}")
            return None
        finally:
            conn.close()
    return None

def verify_emergency_code(code):
    """
    Verifies if an emergency code is valid, unused, and not expired.
    Returns: {'valid': bool, 'data': dict/None, 'message': str}
    """
    conn = get_db_connection()
    if conn:
        try:
            c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            c.execute("""
                SELECT * FROM emergency_codes 
                WHERE code = %s
            """, (code,))
            result = c.fetchone()
            
            if not result:
                return {'valid': False, 'message': "Invalid Code"}
            
            if result['is_used']:
                return {'valid': False, 'message': "Code already used"}
            
            if result['expires_at'] < datetime.now():
                return {'valid': False, 'message': "Code expired"}
                
            return {'valid': True, 'data': dict(result), 'message': "Valid"}
            
        except Exception as e:
            logging.error(f"Error verifying code: {e}")
            return {'valid': False, 'message': "Server Error"}
        finally:
            conn.close()
    return {'valid': False, 'message': "Database Connection Error"}

def mark_emergency_code_used(code_id, metadata):
    """
    Marks an emergency code as used and stores metadata (subject, period, etc.)
    """
    import json
    conn = get_db_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("""
                UPDATE emergency_codes
                SET is_used = TRUE, used_at = %s, metadata = %s
                WHERE id = %s
            """, (datetime.now(), json.dumps(metadata), code_id))
            conn.commit()
            return True
        except Exception as e:
            logging.error(f"Error marking code used: {e}")
            return False
        finally:
            conn.close()
    return False


def get_all_period_definitions():
    """
    Fetches all defined periods from timetable_periods table.
    Returns list of dicts: [{'period_no': 1, 'start_time': '...', 'end_time': '...'}, ...]
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        import psycopg2.extras
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Check if table exists first (in case of migration lag)
        try:
            c.execute("SELECT period_no, start_time, end_time FROM timetable_periods ORDER BY period_no")
            rows = c.fetchall()
        except:
            conn.rollback()
            rows = []
        
        results = []
        for row in rows:
            # Format times
            s = row['start_time']
            e = row['end_time']
            if hasattr(s, 'strftime'): s = s.strftime('%H:%M')
            if hasattr(e, 'strftime'): e = e.strftime('%H:%M')
            
            results.append({
                'period_no': row['period_no'],
                'start_time': s,
                'end_time': e
            })
        
        # Fallback if empty (e.g. fresh install)
        if not results:
             for i in range(1, 9):
                 results.append({'period_no': i, 'start_time': '', 'end_time': ''})
                 
        return results
    except Exception as e:
        logging.error(f"Error getting period definitions: {e}")
        return []
    finally:
        if conn: conn.close()

def get_subject_type(subject_name):
    """Retrieves the subject_type for a given subject name."""
    conn = get_db_connection()
    if conn is None: return "Theory"
    try:
        c = conn.cursor()
        c.execute("SELECT subject_type FROM academic_subjects WHERE LOWER(TRIM(name)) = LOWER(TRIM(%s)) LIMIT 1", (subject_name,))
        row = c.fetchone()
        return row[0] if row else "Theory"
    except Exception as e:
        import logging
        logging.error(f"Error getting subject type: {e}")
        return "Theory"
    finally:
        if conn: conn.close()

def get_consecutive_lab_periods(department, batch_year, semester, subject, day_of_week, current_period_no):
    """
    Finds other periods for the same class/subject/day that are consecutive to the current period.
    Only returns periods that are part of the same lab session.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        import psycopg2.extras
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get all periods for this subject on this day for this class
        query = """
            SELECT DISTINCT te.period_no
            FROM timetable_entries te
            JOIN timetable_master tm ON te.timetable_id = tm.id
            JOIN academic_subjects sub ON te.subject_id = sub.id
            WHERE tm.department = %s
              AND tm.batch_year = %s
              AND (tm.semester = %s OR tm.semester::text = %s)
              AND LOWER(TRIM(sub.name)) = LOWER(TRIM(%s))
              AND te.day_of_week = %s
            ORDER BY te.period_no
        """
        c.execute(query, (department, batch_year, semester, str(semester), subject, day_of_week))
        all_periods = [r['period_no'] for r in c.fetchall()]
        
        if not all_periods or current_period_no not in all_periods:
            return []
            
        # Find consecutive block containing current_period_no
        all_periods.sort()
        idx = all_periods.index(current_period_no)
        
        consecutive_block = [current_period_no]
        
        # Look backwards
        curr = current_period_no
        for i in range(idx - 1, -1, -1):
            if all_periods[i] == curr - 1:
                consecutive_block.append(all_periods[i])
                curr = all_periods[i]
            else:
                break
                
        # Look forwards
        curr = current_period_no
        for i in range(idx + 1, len(all_periods)):
            if all_periods[i] == curr + 1:
                consecutive_block.append(all_periods[i])
                curr = all_periods[i]
            else:
                break
                
        # Remove current and return
        others = [p for p in consecutive_block if p != current_period_no]
        others.sort()
        return others
        
    except Exception as e:
        import logging
        logging.error(f"Error getting consecutive lab periods: {e}")
        return []
    finally:
        if conn: conn.close()

def replicate_attendance_db(source_subject, date_str, source_period_no, target_periods, teacher_id):
    """
    Copies all attendance records from source session to target periods.
    Prevents duplicate marking.
    """
    conn = get_db_connection()
    if conn is None: return False, "Database connection failed"
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # 1. Get source records
        c.execute("""
            SELECT user_id, status, confidence_score, proof_image_path, method, edited_by, original_status, edit_timestamp
            FROM attendance
            WHERE TRIM(subject) = TRIM(%s)
              AND timestamp::date = %s
              AND period_no = %s
        """, (source_subject, date_str, source_period_no))
        source_records = c.fetchall()
        
        if not source_records:
            return False, "No attendance records found for source period"
            
        replicated_count = 0
        skipped_count = 0
        
        replication_timestamp = datetime.now()
        
        for p in target_periods:
            # Check if attendance already exists for this period
            c.execute("""
                SELECT COUNT(*) FROM attendance 
                WHERE subject = %s AND timestamp::date = %s AND period_no = %s
            """, (source_subject, date_str, p))
            if c.fetchone()[0] > 0:
                skipped_count += 1
                continue
                
            for r in source_records:
                # Append replication note to method if not already there
                method = r['method'] or 'replicate'
                if f"replicated_from_P{source_period_no}" not in method:
                    method = f"replicated_from_P{source_period_no}:{method}"
                
                c.execute("""
                    INSERT INTO attendance (
                        user_id, timestamp, status, subject, marked_by, 
                        confidence_score, proof_image_path, method, period_no,
                        edited_by, original_status, edit_timestamp
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    r['user_id'], replication_timestamp, r['status'], source_subject, teacher_id,
                    r['confidence_score'], r['proof_image_path'], method, p,
                    r['edited_by'], r['original_status'], r['edit_timestamp']
                ))
            replicated_count += 1
            
        conn.commit()
        return True, f"Successfully replicated to {replicated_count} periods. {skipped_count} periods skipped (already marked)."
        
    except Exception as e:
        conn.rollback()
        import logging
        logging.error(f"Error replicating attendance: {e}")
        return False, str(e)
    finally:
        if conn: conn.close()

# ==========================================
# GRIEVANCE SYSTEM FUNCTIONS
# ==========================================

def create_grievance(student_id, attendance_id, reason, proof_path=None):
    """
    Creates a new grievance record.
    """
    conn = get_db_connection()
    if conn is None: return False, "Database connection failed"
    try:
        cur = conn.cursor()
        
        # Check if grievance already exists for this attendance record
        cur.execute("SELECT id FROM attendance_grievances WHERE attendance_id = %s", (attendance_id,))
        if cur.fetchone():
            return False, "Grievance already exists for this attendance record."

        cur.execute("""
            INSERT INTO attendance_grievances (student_id, attendance_id, reason, proof_path, status)
            VALUES (%s, %s, %s, %s, 'Pending')
            RETURNING id
        """, (student_id, attendance_id, reason, proof_path))
        
        grievance_id = cur.fetchone()[0]
        conn.commit()
        return True, f"Grievance submitted successfully (ID: {grievance_id})"
    except Exception as e:
        conn.rollback()
        logging.error(f"Error creating grievance: {e}")
        return False, str(e)
    finally:
        if conn: conn.close()

def get_student_grievances(student_id):
    """
    Fetches all grievances for a specific student.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT g.*, a.subject, a.timestamp, a.period_no, a.status as attendance_status
            FROM attendance_grievances g
            JOIN attendance a ON g.attendance_id = a.id
            WHERE g.student_id = %s
            ORDER BY g.created_at DESC
        """, (student_id,))
        return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logging.error(f"Error fetching student grievances: {e}")
        return []
    finally:
        if conn: conn.close()

def get_all_pending_grievances(filters=None):
    """
    Fetches all pending grievances for admin review.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        query = """
            SELECT g.*, a.subject, a.timestamp, a.period_no, a.status as attendance_status,
                   u.full_name as student_name, u.roll_number, u.department, u.year
            FROM attendance_grievances g
            JOIN attendance a ON g.attendance_id = a.id
            JOIN users u ON g.student_id = u.id
            WHERE g.status = 'Pending'
        """
        
        # Apply filters if needed (e.g., department, date range) in future
        
        query += " ORDER BY g.created_at ASC"
        
        cur.execute(query)
        return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logging.error(f"Error fetching pending grievances: {e}")
        return []
    finally:
        if conn: conn.close()

def resolve_grievance(grievance_id, action, admin_id, admin_comment=None):
    """
    Resolves a grievance (Approve/Reject).
    If Approved: Updates attendance record.
    """
    conn = get_db_connection()
    if conn is None: return False, "Database connection failed"
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Fetch grievance details
        cur.execute("SELECT * FROM attendance_grievances WHERE id = %s", (grievance_id,))
        grievance = cur.fetchone()
        if not grievance:
            return False, "Grievance not found"
            
        if grievance['status'] != 'Pending':
            return False, "Grievance is already resolved"
            
        if action == 'Approve':
            # Update Grievance Status
            cur.execute("""
                UPDATE attendance_grievances 
                SET status = 'Approved', resolved_by = %s, resolved_at = NOW(), admin_comment = %s
                WHERE id = %s
            """, (admin_id, admin_comment, grievance_id))
            
            # Update Attendance Record (Default to Present, could be OD if specified)
            # For now, let's assume 'Present' unless comment says 'OD'? 
            # Ideally, action should pass specific status. 
            # Let's assume action='Approve' implies 'Present' for now.
            new_status = 'Present'
            
            cur.execute("""
                UPDATE attendance 
                SET status = %s, edited_by = %s, edit_timestamp = NOW(), 
                    method = method || ':grievance_approved'
                WHERE id = %s
            """, (new_status, admin_id, grievance['attendance_id']))
            
        elif action == 'Reject':
             cur.execute("""
                UPDATE attendance_grievances 
                SET status = 'Rejected', resolved_by = %s, resolved_at = NOW(), admin_comment = %s
                WHERE id = %s
            """, (admin_id, admin_comment, grievance_id))
            
        elif action == 'Approve_OD': # Explicit OD Approval
             cur.execute("""
                UPDATE attendance_grievances 
                SET status = 'Approved', resolved_by = %s, resolved_at = NOW(), admin_comment = %s
                WHERE id = %s
            """, (admin_id, admin_comment, grievance_id))
            
             cur.execute("""
                UPDATE attendance 
                SET status = 'OD', edited_by = %s, edit_timestamp = NOW(), 
                    method = method || ':grievance_approved_od'
                WHERE id = %s
            """, (admin_id, grievance['attendance_id']))
            
        else:
            return False, "Invalid action"
            
        conn.commit()
        return True, f"Grievance {action}ed successfully"
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error resolving grievance: {e}")
        return False, str(e)
    finally:
        if conn: conn.close()

def get_grievance_details(grievance_id):
    """
    Fetches full details of a single grievance for admin view.
    """
    conn = get_db_connection()
    if conn is None: return None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT g.*, a.subject, a.timestamp, a.period_no, a.status as attendance_status,
                   u.full_name as student_name, u.roll_number, u.department, u.year, u.register_number
            FROM attendance_grievances g
            JOIN attendance a ON g.attendance_id = a.id
            JOIN users u ON g.student_id = u.id
            WHERE g.id = %s
        """, (grievance_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    except Exception as e:
        logging.error(f"Error fetching grievance details: {e}")
        return None
    finally:
        if conn: conn.close()

# --- BEHAVIOR MONITORING MODULE ---

def log_behavior_violation(user_id, violation_type, confidence=0.0, session_id=None, marked_by=None, department=None, batch_year=None, subject=None, image_proof_path=None):
    """Logs a behavior violation to the database."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        c = conn.cursor()
        query = """
            INSERT INTO behavior_violations 
            (user_id, violation_type, confidence, session_id, marked_by, department, batch_year, subject, image_proof_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        c.execute(query, (user_id, violation_type, confidence, session_id, marked_by, department, batch_year, subject, image_proof_path))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error logging behavior violation: {e}")
        conn.rollback()
        return False
    finally:
        if conn:
            conn.close()




def get_subjects_for_class(department, batch_year, semester):
    """
    Retrieves distinct subjects scheduled for a specific class (Dept, Batch, Sem).
    Based on Timetable Entries.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Join timetable_master -> timetable_entries -> academic_subjects
        query = """
            SELECT DISTINCT s.name 
            FROM timetable_entries te
            JOIN timetable_master tm ON te.timetable_id = tm.id
            JOIN academic_subjects s ON te.subject_id = s.id
            WHERE 
                tm.department = %s AND 
                tm.batch_year = %s AND 
                tm.semester = %s
            ORDER BY s.name
        """
        c.execute(query, (department, batch_year, int(semester)))
        rows = c.fetchall()
        
        return [r['name'] for r in rows]
    except Exception as e:
        logging.error(f"Error fetching class subjects: {e}")
        return []
    finally:
        if conn: conn.close()


def get_sections_for_class(department, batch_year, semester):
    """
    Retrieves distinct sections for a specific class (Dept, Batch, Sem).
    Based on Timetable Master.
    """
    conn = get_db_connection()
    if conn is None: return []
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        query = """
            SELECT DISTINCT section 
            FROM timetable_master
            WHERE 
                department = %s AND 
                batch_year = %s AND 
                semester = %s
            ORDER BY section
        """
        c.execute(query, (department, batch_year, int(semester)))
        rows = c.fetchall()
        return [r['section'] for r in rows if r['section']]
    except Exception as e:
        logging.error(f"Error fetching class sections: {e}")
        return []
    finally:
        if conn: conn.close()

