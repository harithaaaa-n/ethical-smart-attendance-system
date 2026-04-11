import os
import cv2
import numpy as np
import base64
import json
import logging
import csv
import io
import math
import zipfile
import tempfile
import shutil
import re
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Report Export Imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER


from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session, flash, send_file, make_response
from functools import wraps
from datetime import datetime, timedelta, time as dt_time
import pandas as pd
import bcrypt
import psycopg2
import psycopg2.extras
from flask_cors import CORS
import cv2
import base64
import numpy as np

from . import database_manager
from . import biometric_logic
from .academic_routes import academic_bp
from .mobile_routes import mobile_bp
from .od_helper import resolve_od_roll_numbers
# Behavior monitoring removed

def decode_session_id(session_id):
    """Decodes the base64 encoded session identification dictionary."""
    try:
        if not session_id: return None
        decoded_bytes = base64.urlsafe_b64decode(session_id.encode())
        return json.loads(decoded_bytes.decode())
    except Exception as e:
        logging.error(f"Error decoding session_id: {e}")
        return None

def get_attendance_method(default_method='live'):
    """
    Checks session context and returns a formatted method string if it's an altered period.
    Format: "altered:{reason}:{assigned_teacher_id}"
    """
    context = session.get('active_attendance_context')
    if context and context.get('is_altered') and context.get('altered_reason'):
        reason = context.get('altered_reason')
        assigned_id = context.get('assigned_teacher_id', 'Unknown')
        return f"altered:{reason}:{assigned_id}"
    return default_method

def _check_altered_period(department, batch_year, subject, period):
    """
    Internal helper to detect if a period is altered for the current user.
    Returns (is_altered, assigned_staff_name, assigned_teacher_id, timetable_id)
    """
    if not all([department, subject, period]):
        return False, None, None, None
        
    day_name = datetime.now().strftime('%A')
    search_period = period
    
    # Resolve period_no for detection if period is a string (e.g. "10:30 - 11:25")
    # Resolve period_no for detection if period is a string (e.g. "10:30 - 11:25" or "Period 6 ...")
    if period and not str(period).isdigit():
        # Try to extract "Period X" or just number from string
        import re
        match = re.search(r'Period\s*(\d+)', str(period), re.IGNORECASE)
        if match:
             search_period = int(match.group(1))
             logging.info(f"DEBUG: Parsed Period No {search_period} from string '{period}'")
        else:
            # Fallback for "09:30 - 10:30" format
            pts = database_manager.get_timetable_periods(department, batch_year, subject, day_name)
            for p_test in pts:
                # Compare times if possible
                period_time_str = f"{p_test['start_time']} - {p_test['end_time']}"
                if period_time_str in str(period):
                    search_period = p_test['period_no']
                    logging.info(f"DEBUG: Matched Period {search_period} by time in '{period}'")
                    break

    is_altered = False
    assigned_staff_name = None
    assigned_teacher_id = None
    timetable_id = None

    logging.info(f"DEBUG: Checking altered period. Dept={department}, Batch={batch_year}, Sub={subject}, Per={period}, Day={day_name}")
    
    if search_period:
        # Get entry for this slot
        entry = database_manager.get_timetable_entry_by_details(department, batch_year, subject, day_name, search_period)
        logging.info(f"DEBUG: Timetable Entry Found: {entry}")
        
        # LOGIC CHANGE: 
        # 1. If entry exists AND assigned to current user -> NOT ALTERED
        # 2. If entry exists but assigned to someone else -> ALTERED
        # 3. If NO entry exists -> ALTERED (Unscheduled)
        
        current_user_id = session.get('user_id')
        
        if entry:
            timetable_id = entry['id']
            assigned_teacher_id = entry.get('teacher_id')
            
            logging.info(f"DEBUG: Comparison - Assigned: {assigned_teacher_id} (Type: {type(assigned_teacher_id)}), Current: {current_user_id} (Type: {type(current_user_id)})")
            
            # Use strict comparison for teacher ID
            # If the period is assigned to THIS teacher, it is NOT altered.
            # If the current user is an admin, and the slot is scheduled, it is NOT altered (Admin override).
            if str(assigned_teacher_id) == str(current_user_id) or session.get('role') == 'admin':
                is_altered = False
                logging.info(f"DEBUG: {'Teacher Match' if str(assigned_teacher_id) == str(current_user_id) else 'Admin Role'}! NOT ALTERED.")
            else:
                # Assigned to someone else
                is_altered = True
                logging.info("DEBUG: Teacher Mismatch (and not Admin)! ALTERED.")
                if assigned_teacher_id:
                    assigned_teacher = database_manager.get_user_info(assigned_teacher_id)
                    assigned_staff_name = assigned_teacher.get('full_name') if assigned_teacher else 'Unknown Teacher'
                else:
                    # Entry exists but no teacher assigned (rare, but possible)
                    # If I am selecting it, maybe I am taking it?
                    # Let's mark as altered to be safe, so I can claim it.
                    assigned_staff_name = 'Unassigned Slot'
        else:
            # Not in timetable at all for this day
            is_altered = True
            logging.info("DEBUG: No Entry Found in Timetable. ALTERED.")
            assigned_staff_name = 'Unscheduled Period'
    else:
        logging.info("DEBUG: Search Period could not be resolved.")
                
    return is_altered, assigned_staff_name, assigned_teacher_id, timetable_id

def _get_student_list_with_status(department, year, subject, period_no, batch_year=None, semester=None):
    """
    Helper to fetch students for a class and enrich them with their current attendance status.
    """
    try:
        # Sanitization for database matching
        def sanitize_year(y):
            if not y: return y
            y_str = str(y).lower().strip()
            if 'first' in y_str or '1st' in y_str: return 1
            if 'second' in y_str or '2nd' in y_str: return 2
            if 'third' in y_str or '3rd' in y_str: return 3
            if 'fourth' in y_str or '4th' in y_str: return 4
            # Try numeric extraction
            import re
            match = re.search(r'\d+', y_str)
            return int(match.group()) if match else y
        
        dept_clean = str(department).strip() if department else None
        year_clean = sanitize_year(year)
        sem_clean = int(semester) if semester and str(semester).isdigit() else semester

        # Robust period_no conversion
        try:
            p_no = int(period_no) if period_no is not None else None
        except (ValueError, TypeError):
            p_no = period_no

        # 1. Get all students in the class (using cleaned values)
        all_students = database_manager.get_students_by_class(
            dept_clean, year_clean, 
            batch_year=batch_year, 
            semester=sem_clean
        ) or []
        
        # 2. Get currently present/OD students in this session
        today = datetime.now().strftime('%Y-%m-%d')
        session_records = database_manager.get_session_attendance_records(subject, today, period_no=p_no) or []
        
        # Map of user_id -> status
        status_map = {r['user_id']: r['status'] for r in session_records}
        
        # 3. Check for OD students in session
        if 'od_students' in session:
            for oid in session['od_students']:
                if oid not in status_map:
                    status_map[oid] = 'OD'
        
        # Merge status
        for student in all_students:
            student['status'] = status_map.get(student['id'], 'Absent')
            
        return all_students
    except Exception as e:
        logging.error(f"Error enriching student list with status: {e}")
        return []

app = Flask(__name__, template_folder='../frontend/templates')
app.register_blueprint(academic_bp)
app.register_blueprint(mobile_bp, url_prefix='/api/mobile')

app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key_change_me')
logging.basicConfig(level=logging.DEBUG)
logging.info("STARTUP: Backend Main has been reloaded correctly.")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Optimization: Background Pre-initialization ---
import threading
def background_warmup():
    logging.info("WARMUP: Starting background initialization of FaceRecognitionSystem...")
    biometric_logic.initialize_system()
    logging.info("WARMUP: FaceRecognitionSystem Ready.")

# Start warmup thread immediately
threading.Thread(target=background_warmup, daemon=True).start()

# --- Decorators and Helper Functions ---

def login_required(role=None):
    """
    A decorator to ensure a user is logged in and has the required role(s).
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Bypass if in Emergency Mode
            if session.get('is_emergency_mode') and session.get('emergency_code_id'):
                return f(*args, **kwargs)

            if 'user_id' not in session:
                logging.warning("Access denied: User not logged in. Redirecting to login page.")
                return redirect(url_for('login_page'))
            
            user_role = session.get('role')
            if role:
                if isinstance(role, str) and user_role != role:
                    logging.warning(f"Access denied: User role '{user_role}' does not match required role '{role}'.")
                    return redirect(url_for('login_page'))
                elif isinstance(role, (list, tuple)) and user_role not in role:
                    logging.warning(f"Access denied: User role '{user_role}' not in required roles '{role}'.")
                    return redirect(url_for('login_page'))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def create_hourly_timetable(timetable_data):
    """
    Transforms a list of timetable entries into a structured dictionary for rendering.
    """
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_periods = sorted(list(set([(item['start_time'], item['end_time']) for item in timetable_data])))
    
    timetable_by_day = {day: [None] * len(time_periods) for day in days}
    timetable_by_hour = [{'start_time': start, 'end_time': end} for start, end in time_periods]
    
    for item in timetable_data:
        try:
            day_index = days.index(item['day'])
            time_index = time_periods.index((item['start_time'], item['end_time']))
            timetable_by_day[days[day_index]][time_index] = item
        except ValueError:
            logging.error(f"Error processing timetable entry: {item}")
            continue
            
    return timetable_by_day, timetable_by_hour

# --- Routes ---
@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/login-page')
def login_page():
    return render_template('login.html', error=None)

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password'].encode('utf-8')
    
    user = database_manager.authenticate_user(username, password)
    
    if user:
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['role'] = user['role']
        session['full_name'] = user['full_name']
        
        # Update Last Login
        database_manager.update_last_login(user['id'])

        # Load Preferences
        import json
        try:
            prefs = json.loads(user.get('preferences') or '{}')
            session['theme_pref'] = prefs.get('theme', 'light')
        except:
            session['theme_pref'] = 'light'
        
        if user['role'] == 'student':
            return redirect(url_for('student_dashboard'))
        elif user['role'] == 'teacher':
            return redirect(url_for('teacher_dashboard'))
        elif user['role'] == 'admin':
            return redirect(url_for('admin_dashboard'))
    
    return render_template('login.html', error='Invalid credentials')


@app.route('/student/timetable')
@login_required('student')
def student_timetable():
    user_id = session['user_id']
    
    # Grid Data
    current_entries = database_manager.get_student_timetable_grid(user_id)
    master = database_manager.get_student_timetable_master(user_id)
    timings = database_manager.get_global_timings()
    
    # Metadata
    student = database_manager.get_user_info(user_id)
    
    return render_template('student_timetable.html',
                           current_entries=current_entries,
                           master=master,
                           timings=timings,
                           student=student)

# Helper function to format timetable data
def create_hourly_timetable(timetable_entries):
    """
    Converts timetable entries into day-based and hour-based structures for template rendering.
    Returns: (timetable_by_day, timetable_by_hour)
    """
    # Get unique periods/timings
    timings = database_manager.get_global_timings()
    
    # Initialize structure
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    timetable_by_day = {day: [None] * len(timings) for day in days}
    
    # Fill in the timetable
    for entry in timetable_entries:
        day = entry.get('day_of_week')
        period_no = entry.get('period_no')
        if day in timetable_by_day and period_no and period_no <= len(timings):
            timetable_by_day[day][period_no - 1] = entry
    
    # Create hour-based structure (for rows)
    timetable_by_hour = []
    for idx, timing in enumerate(timings):
        timetable_by_hour.append({
            'start_time': timing.get('start_time'),
            'end_time': timing.get('end_time'),
            'period_no': idx + 1
        })
    
    return timetable_by_day, timetable_by_hour

@app.route('/teacher/timetable')
@login_required('teacher')
def teacher_timetable():
    if session.get('is_emergency_mode'):
        return redirect(url_for('select_attendance_mode'))
    user_id = session['user_id']
    
    # Grid Data
    current_entries = database_manager.get_teacher_weekly_timetable(user_id)
    timings = database_manager.get_global_timings()
    teacher = database_manager.get_user_info(user_id)
    
    return render_template('teacher_timetable.html', 
                           current_entries=current_entries,
                           timings=timings,
                           teacher=teacher)

@app.route('/student/dashboard')
@login_required('student')
def student_dashboard():
    student = database_manager.get_user_info(session['user_id'])
    
    start_date = request.args.get('start_date', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    
    attendance_data = database_manager.get_student_attendance_by_date_range(session['user_id'], start_date, end_date)
    percentage = database_manager.calculate_attendance_percentage(attendance_data)
    
    # timetable = database_manager.get_timetable(student['department'], student['year'])
    # Use NEW grid logic (based on Master Table)
    current_entries = database_manager.get_student_timetable_grid(session['user_id'])
    master = database_manager.get_student_timetable_master(session['user_id'])
    timings = database_manager.get_global_timings()
    
    # We no longer need create_hourly_timetable if we use the grid view directly
    # timetable_by_day, timetable_by_hour = create_hourly_timetable(timetable) (Old logic)
    
    # Advanced Analytics based on Semester Configuration
    advanced_stats = None
    sem_info = None
    if student.get('batch_year') and student.get('semester'):
        sem_info = database_manager.get_semester_info(student['batch_year'], student['semester'])
        if sem_info:
            advanced_stats = database_manager.get_student_attendance_stats_advanced(session['user_id'], sem_info)

    return render_template('student_dashboard.html', 
                            attendance=attendance_data, 
                            percentage=percentage,
                            advanced_stats=advanced_stats,
                            sem_info=sem_info,
                            current_entries=current_entries, # Renamed for grid view compatibility
                            timetable_data=current_entries, # Also pass as timetable_data for template
                            master=master, # NEW
                            timings=timings, # NEW
                            student=student,
                            grievance_map={g['attendance_id']: g['status'] for g in database_manager.get_student_grievances(session['user_id'])}, # NEW: Grievance Map
                            start_date=start_date,
                            end_date=end_date)

@app.route('/student/attendance-history')
@login_required('student')
def student_history():
    user_id = session['user_id']
    student = database_manager.get_user_info(user_id)
    
    # 1. Raw History (Last 6 months)
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    history = database_manager.get_student_attendance_by_date_range(user_id, start_date, end_date)
    
    # 2. Aggregated Reports
    reports = database_manager.get_student_attendance_reports(user_id)
    
    return render_template('student_history.html',
                           student=student,
                           history=history,
                           grievance_map={g['attendance_id']: g['status'] for g in database_manager.get_student_grievances(user_id)}, # NEW: Grievance Map
                           weekly_report=reports['weekly'],
                           monthly_report=reports['monthly'],
                           subject_report=reports.get('subject', []))

@app.route('/teacher/dashboard')
@login_required('teacher')
def teacher_dashboard():
    if session.get('is_emergency_mode'):
        return redirect(url_for('select_attendance_mode'))
    user_id = session['user_id']
    
    # 1. Get Timetable (Today) using NEW Dedicated Function (Grid Based)
    # Allows retrieving data from new Admin Timetable system
    all_slots = database_manager.get_teacher_timetable_slots(user_id)
    
    # Filter for today
    now = datetime.now()
    today_name = now.strftime('%A')
    today_date = now.strftime('%Y-%m-%d')
    current_time = now.time()
    
    today_timetable = [s for s in all_slots if s.get('day') == today_name]
    
    # 2. Stats
    total_classes_today = len(today_timetable)
    assigned_subjects = database_manager.get_subjects_for_user(user_id)
    total_subjects = len(assigned_subjects)
    
    processed_timetable = []
    
    for slot in today_timetable:
        # Check attendance status
        is_completed = database_manager.check_attendance_status(
            slot['subject'], 
            today_date,
            department=slot.get('department'),
            batch_year=slot.get('batch_year'),
            section=slot.get('section'),
            period_no=slot.get('period_no')
        )
        
        # Determine status
        try:
            # Handle string vs time objects
            start_val = slot['start_time']
            end_val = slot['end_time']
            
            if isinstance(start_val, str):
                start_t = datetime.strptime(start_val, '%H:%M:%S').time()
            else:
                start_t = start_val
                
            if isinstance(end_val, str):
                end_t = datetime.strptime(end_val, '%H:%M:%S').time()
            else:
                end_t = end_val
                
            # Smart AM/PM resolution: If hour is small (1-7) and it's an afternoon period (P5+)
            # we treat it as PM for status calculation.
            period_val = str(slot.get('period_no', ''))
            is_afternoon_period = any(p in period_val for p in ['5', '6', '7'])
            
            if is_afternoon_period and start_t.hour < 12:
                # Add 12 hours for comparison logic only
                start_t = dt_time(start_t.hour + 12, start_t.minute, start_t.second)
                end_t = dt_time(end_t.hour + 12, end_t.minute, end_t.second)
                
        except Exception as e:
            logging.error(f"Time parsing error for slot: {e}")
            start_t = dt_time(0, 0)
            end_t = dt_time(23, 59)

        
        status = 'Upcoming'
        is_active = False
        
        if is_completed:
            status = 'Completed'
        # Logic: If time passed -> Missed/Pending
        elif current_time > end_t:
            status = 'Pending' # Allow late marking
        elif current_time >= start_t and current_time <= end_t:
             status = 'Active'
             is_active = True
             
        slot['status'] = status
        slot['is_active'] = is_active
        slot['is_completed'] = is_completed
        
        # Ensure entry_id is present (some funcs return id, some entry_id)
        if 'entry_id' not in slot and 'id' in slot:
            slot['entry_id'] = slot['id']
            
        processed_timetable.append(slot)
        
    # Sort chronologically by period number
    try:
        processed_timetable.sort(key=lambda x: int(str(x.get('period_no', 0)).replace('P', '')) if x.get('period_no') else 0)
    except:
        # Fallback to current order if conversion fails
        pass

    return render_template('teacher_dashboard.html',
                           today_timetable=processed_timetable,
                           total_classes=total_classes_today,
                           total_subjects=total_subjects,
                           user=database_manager.get_user_info(user_id),
                           today_date=today_date)

@app.route('/teacher/my_subjects')
@login_required('teacher')
def teacher_my_subjects():
    if session.get('is_emergency_mode'):
        return redirect(url_for('select_attendance_mode'))
    user_id = session['user_id']
    subjects = database_manager.get_subjects_for_user(user_id)
    return render_template('teacher_my_subjects.html', subjects=subjects)



@app.route('/teacher/history')
@login_required('teacher')
def teacher_attendance_history():
    if session.get('is_emergency_mode'):
        return redirect(url_for('select_attendance_mode'))
    teacher_id = session['user_id']
    
    # Filter Params
    selected_dept = request.args.get('department')
    selected_batch = request.args.get('batch_year')
    selected_sem = request.args.get('semester')
    selected_sec = request.args.get('section')
    selected_period = request.args.get('period')
    
    sessions = database_manager.get_teacher_attendance_sessions(
        teacher_id, 
        department=selected_dept, 
        batch_year=selected_batch,
        semester=selected_sem,
        section=selected_sec,
        period_no=selected_period
    )
    
    # Dropdowns for Filter UI
    allocated_classes = database_manager.get_teacher_allocated_classes(teacher_id)
    # Extract unique depts and years (batches)
    departments = sorted(list(set(c['department'] for c in allocated_classes)))
    # For batches, we should fetch actual batch years available 
    batches = database_manager.get_academic_years() 
    sections = database_manager.get_academic_sections()
    
    return render_template('teacher_reports_list.html', 
                           sessions=sessions,
                           departments=departments, # Passed for filters
                           batches=batches,
                           sections=sections,
                           selected_dept=selected_dept,
                           selected_batch=selected_batch,
                           selected_sem=selected_sem,
                           selected_sec=selected_sec,
                           selected_period=selected_period)

@app.route('/teacher/my_students')
@login_required('teacher')
def teacher_my_students():
    if session.get('is_emergency_mode'):
        return redirect(url_for('select_attendance_mode'))
    teacher_id = session['user_id']
    classes = database_manager.get_teacher_allocated_classes(teacher_id)
    
    department = request.args.get('department')
    year = request.args.get('year')
    semester = request.args.get('semester')
    batch_year = request.args.get('batch_year')
    
    students = None
    
    if department and year:
        students = database_manager.get_students_for_teacher_class(
            teacher_id, 
            department, 
            year, 
            semester=semester, 
            batch_year=batch_year
        )
        
    return render_template('teacher_my_students.html', 
                         classes=classes, 
                         students=students, 
                         selected_dept=department, 
                         selected_year=year, 
                         selected_semester=semester,
                         selected_batch=batch_year,
                         batches=database_manager.get_academic_years(),
                         semesters=range(1, 9))


@app.route('/teacher/reports')
@login_required('teacher')
def teacher_reports():
    teacher_id = session['user_id']
    
    # Filters
    subjects = database_manager.get_subjects_for_user(teacher_id)
    assigned_classes = database_manager.get_teacher_allocated_classes(teacher_id)
    teacher_depts = sorted(list(set(c['department'] for c in assigned_classes)))
    teacher_years = sorted(list(set(c['year'] for c in assigned_classes)))
    
    selected_subject = request.args.get('subject')
    selected_dept = request.args.get('department')
    selected_year = request.args.get('year')
    selected_batch_year = request.args.get('batch_year')
    selected_period = request.args.get('period')
    method = request.args.get('method') # NEW
    
    view_mode = request.args.get('view_mode', 'Student Wise')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    report_data = []
    if selected_subject or selected_dept or selected_batch_year or selected_period:
        # User unified logic
        report_data = database_manager.get_official_attendance_report(
            selected_dept, selected_year, selected_subject, start_date, end_date, 
            batch_year=selected_batch_year, period=selected_period, view_mode=view_mode
        )
        
    return render_template('teacher_reports.html',
                           subjects=subjects,
                           departments=teacher_depts,
                           years=teacher_years,
                           batches=database_manager.get_academic_years(),
                           report_data=report_data,
                           selected_subject=selected_subject,
                           selected_dept=selected_dept,
                           selected_year=selected_year,
                           selected_batch_year=selected_batch_year,
                           selected_period=selected_period,
                           view_mode=view_mode,
                           start_date=start_date,
                           end_date=end_date)

@app.route('/api/timetable/periods')
@login_required(['admin', 'teacher'])
def get_timetable_periods_api():
    department = request.args.get('department')
    batch_year = request.args.get('batch_year')
    subject = request.args.get('subject')
    day = request.args.get('day')
    
    if not day:
        day = datetime.now().strftime('%A')
    
    logging.info(f"API Fetch Periods: Dept={department}, Batch={batch_year}, Sub={subject}, Day={day}")

    if not department or not subject:
        return jsonify([])

    # Get scheduled periods for this specific subject (To mark as "Scheduled")
    # Use strict_day=True to ensure we don't label slots from other days as scheduled today.
    scheduled_for_subject = database_manager.get_timetable_periods(department, batch_year, subject, day, strict_day=True)
    scheduled_period_nos = {p.get('period_no') for p in scheduled_for_subject}
    
    # Get ALL periods for this Class on this Day (To list valid slots)
    # This ensures we "list only that days data" (i.e., slots present in the timetable)
    all_class_periods_today = database_manager.get_timetable_periods(department, batch_year, None, day, strict_day=True)
    
    final_periods = []
    
    # Logic: Use Class Schedule as the base list if available.
    # This removes periods belonging to other days.
    source_periods = all_class_periods_today
    
    # If Class Schedule is empty (e.g. no timetable entered yet for this day), 
    # fallback to All Definitions to allow manual marking/extra classes.
    if not source_periods:
        try:
            source_periods = database_manager.get_all_period_definitions()
        except Exception as e:
            logging.error(f"Error fetching period definitions: {e}")
            source_periods = [{'period_no': i, 'start_time': '', 'end_time': ''} for i in range(1, 9)]

    # Deduplicate by period_no (just in case)
    seen_periods = set()
    
    # Sort source periods
    source_periods.sort(key=lambda x: x.get('period_no') or 0)

    for p in source_periods:
        period_no = p.get('period_no')
        if not period_no or period_no in seen_periods: continue
        seen_periods.add(period_no)
        
        is_scheduled = period_no in scheduled_period_nos
        
        s = p.get('start_time', '')
        e = p.get('end_time', '')
        if hasattr(s, 'strftime'): s = s.strftime('%H:%M')
        if hasattr(e, 'strftime'): e = e.strftime('%H:%M')
        
        if is_scheduled:
            label = f"Period {period_no} ({s} - {e}) - Scheduled"
        elif s and e:
            label = f"Period {period_no} ({s} - {e})"
        else:
             label = f"Period {period_no}"
             
        final_periods.append({
            'start_time': str(s),
            'end_time': str(e),
            'label': label,
            'period_no': period_no,
            'is_scheduled': is_scheduled
        })
            
    logging.info(f"Returning {len(final_periods)} periods for {day} (Source: {'Timetable' if all_class_periods_today else 'Definitions'}, Scheduled: {len(scheduled_period_nos)})")
    return jsonify(final_periods)

@app.route('/teacher/reports/export')
@login_required('teacher')
def teacher_reports_export():
    teacher_id = session['user_id']
    selected_subject = request.args.get('subject')
    selected_dept = request.args.get('department')
    selected_year = request.args.get('year')
    selected_batch_year = request.args.get('batch_year') # NEW
    export_format = request.args.get('format', 'csv') # NEW
    method = request.args.get('method') # NEW
    
    view_mode = request.args.get('view_mode', 'Student Wise') 
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Calculate duration for report customization
    is_multi_day = False
    if start_date and end_date and start_date != end_date:
        is_multi_day = True

    # Use the unified official report logic
    data = database_manager.get_official_attendance_report(
        selected_dept, selected_year, selected_subject, start_date, end_date, 
        batch_year=selected_batch_year, view_mode=view_mode
    )
    # Note: get_official_attendance_report doesn't currently filter by teacher_id, 
    # but for teacher reports, the filters (subject/dept) usually narrow it down.
    # If teacher_id filter is strictly needed, we should add it to get_official_attendance_report.
    
    if export_format == 'pdf':
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(A4))
        elements = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        title_style.alignment = TA_CENTER
        
        # Header Info
        header_text = f"Attendance Report\nSubject: {selected_subject or 'All'}\nDepartment: {selected_dept or 'All'}"
        elements.append(Paragraph(header_text.replace('\n', '<br/>'), title_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Table Data
        if is_multi_day or view_mode == 'Student Wise':
             # Aggregated View: Hide individual status, show percentage
             table_data = [['Student Name', 'Roll No', 'Dept', 'Year', 'Subject', 'Total', 'Present', '%']]
             for row in data:
                 table_data.append([
                     row.get('student', row.get('full_name', ''))[:20],
                     row.get('register_no', row.get('roll_number', ''))[:15],
                     row.get('department', '')[:15],
                     str(row.get('year', '')),
                     row.get('subject_name', row.get('subject', ''))[:20],
                     str(row.get('total_classes', row.get('total_attendance_records', 0))),
                     str(row.get('present_count', 0)),
                     f"{row.get('attendance_pct', row.get('percentage', 0))}%"
                 ])
        else:
             # Single Day View: Detailed Status
             table_data = [['Date', 'Student Name', 'Roll No', 'Subject', 'Status']]
             for row in data:
                 ts = row.get('timestamp')
                 table_data.append([
                     ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts),
                     row.get('full_name', '')[:20],
                     row.get('roll_number', '')[:15],
                     row.get('subject', '')[:20],
                     row.get('status', 'Absent')
                 ])
            
        # Table Style
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9), # Smaller for PDF fit
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        
        try:
            doc.build(elements)
            buffer.seek(0)
            return send_file(buffer, as_attachment=True, download_name=f"Attendance_Report_{datetime.now().strftime('%Y%m%d')}.pdf", mimetype='application/pdf')
        except Exception as e:
            logging.error(f"PDF Generation Error: {e}")
            return f"Error generating PDF: {e}", 500

    else:
        # CSV Export
        si = io.StringIO()
        cw = csv.writer(si)
        if is_multi_day or view_mode == 'Student Wise':
            cw.writerow(['Student Name', 'Register No', 'Total Classes', 'Present', 'Percentage'])
            for row in data:
                cw.writerow([
                    row.get('student', row.get('full_name', '')), 
                    row.get('register_no', row.get('roll_number', '')), 
                    row.get('total_classes', row.get('total_attendance_records', 0)), 
                    row.get('present_count', 0), 
                    f"{row.get('attendance_pct', row.get('percentage', 0))}%"
                ])
        else:
            cw.writerow(['Date', 'Student Name', 'Roll Number', 'Subject', 'Status'])
            for row in data:
                ts = row.get('timestamp')
                cw.writerow([
                    ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts), 
                    row.get('full_name', ''), 
                    row.get('roll_number', ''), 
                    row.get('subject', ''), 
                    row.get('status', 'Absent')
                ])
        
        output = make_response(si.getvalue())
        output.headers["Content-Disposition"] = f"attachment; filename=Teacher_Report.csv"
        output.headers["Content-type"] = "text/csv"
        return output

@app.route('/teacher/reports/view/<int:timetable_id>')
@app.route('/teacher/reports/view/session/<session_id>')
@login_required('teacher')
def teacher_report_view(timetable_id=0, session_id=None):
    teacher_id = session['user_id']
    
    # Context extraction
    if session_id:
        ctx = decode_session_id(session_id)
        if not ctx: return "Invalid session ID.", 400
        date_str = ctx['date']
        subject = ctx['subject']
        department = ctx['dept']
        year = ctx['year']
        period = ctx['period']
    else:
        date_str = request.args.get('date')
        subject = request.args.get('subject')
        department = request.args.get('department')
        year = request.args.get('year')
        period = request.args.get('period')
    
    if not date_str:
        return "Missing date parameter", 400
        
    result = database_manager.get_attendance_session_details(
        teacher_id, timetable_id, date_str,
        fallback_subject=subject,
        fallback_department=department,
        fallback_year=year,
        fallback_period=period
    )
    
    if result is None:
        return "Access denied or session not found (Invalid ID or You do not teach this subject).", 403
        
    details = result['students']
    class_info = result['class_info']
        
    # Calculate stats with OD support
    present_count = sum(1 for d in details if d['status'] == 'Present')
    absent_count = sum(1 for d in details if d['status'] == 'Absent')
    od_count = sum(1 for d in details if d['status'] == 'OD')
    total = len(details)
    
    return render_template('teacher_report_view.html', 
                          details=details,
                          class_info=class_info,
                          subject=class_info['subject'],
                          date=date_str,
                          department=class_info['department'],
                          year=class_info['year'],
                          present_count=present_count,
                          absent_count=absent_count,
                          od_count=od_count,
                          total=total,
                          timetable_id=timetable_id,
                          session_id=session_id)

@app.route('/teacher/reports/download/<int:timetable_id>')
@app.route('/teacher/reports/download/session/<session_id>')
@login_required('teacher')
def teacher_report_download(timetable_id=0, session_id=None):
    teacher_id = session['user_id']
    export_format = request.args.get('format', 'csv')
    
    # Context extraction
    if session_id:
        ctx = decode_session_id(session_id)
        if not ctx: return "Invalid session ID.", 400
        date_str = ctx['date']
        subject = ctx['subject']
        department = ctx['dept']
        year = ctx['year']
        period = ctx['period']
    else:
        date_str = request.args.get('date')
        subject = request.args.get('subject')
        department = request.args.get('department')
        year = request.args.get('year')
        period = request.args.get('period')
    
    if not date_str:
        return "Missing date parameter", 400
        
    result = database_manager.get_attendance_session_details(
        teacher_id, timetable_id, date_str,
        fallback_subject=subject,
        fallback_department=department,
        fallback_year=year,
        fallback_period=period
    )
    
    if result is None:
         return "Access denied or session not found.", 403
         
    details = result['students']
    info = result['class_info']
    subject = info['subject']
    department = info['department']
    year = info['year']
    period_val = info.get('period', info.get('period_no', '-'))
    
    if export_format == 'pdf':
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        title_style.alignment = TA_CENTER
        normal_style = styles['Normal']
        
        # Header Info
        header_text = f"Detailed Attendance Report<br/><font size=12>{subject}</font><br/><font size=10>Class: {department} - {year} | Period: {period_val} | Date: {date_str}</font>"
        elements.append(Paragraph(header_text, title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Stats
        present_count = sum(1 for d in details if d['status'] == 'Present')
        absent_count = sum(1 for d in details if d['status'] == 'Absent')
        od_count = sum(1 for d in details if d['status'] == 'OD')
        total = len(details)
        stats_text = f"<b>Total Students: {total}</b> | <font color='green'>Present: {present_count}</font> | <font color='red'>Absent: {absent_count}</font> | <font color='blue'>OD: {od_count}</font>"
        elements.append(Paragraph(stats_text, styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Table Data
        table_data = [['ID', 'Name', 'Status']]
        
        for row in details:
            status = row['status']
            # Use Register Number if available, else Roll Number
            student_id_val = row.get('register_number') or row.get('roll_number', 'N/A')
            
            table_data.append([
                student_id_val,
                row['full_name'][:30],
                status
            ])
            
        # Table Style
        t = Table(table_data, colWidths=[2.0*inch, 3.0*inch, 1.2*inch])
        
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(t)
        
        try:
            doc.build(elements)
            buffer.seek(0)
            fname = f"Attendance_Session_{date_str}.pdf"
            return send_file(buffer, as_attachment=True, download_name=fname, mimetype='application/pdf')
        except Exception as e:
            logging.error(f"PDF Generation Error: {e}")
            return f"Error generating PDF: {e}", 500

    else:
        # Generate CSV (Simplified)
        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow(['Attendance Session Report'])
        cw.writerow([f'Subject: {subject}', f'Date: {date_str}', f'Period: {period_val}'])
        cw.writerow([])
        cw.writerow(['Name', 'ID', 'Status'])
        for row in details:
            cw.writerow([
                row['full_name'], 
                row.get('register_number') or row.get('roll_number', ''), 
                row['status']
            ])
        
        output = make_response(si.getvalue())
        output.headers["Content-Disposition"] = f"attachment; filename=Report_{subject.replace(' ','_')}_{date_str}.csv"
        output.headers["Content-type"] = "text/csv"
        return output



@app.route('/teacher/attendance/start/<int:timetable_id>')
@login_required('teacher')
def teacher_attendance_start(timetable_id):
    """
    Opens attendance marking page for a SPECIFIC timetable slot.
    Locks filters (department, year, subject) to that slot's details.
    """
    teacher_id = session['user_id']
    
    # 1. Fetch Timetable Entry
    entry = database_manager.get_timetable_entry(timetable_id)
    if not entry:
        return "Timetable entry not found", 404
        
    # 2. Validate Ownership
    # Security: Ensure this slot belongs to the logged-in teacher
    if entry.get('teacher_id') != teacher_id:
        # Extra check: maybe it's assigned via subject? 
        # But user requested "timetable based". Assuming timetable has teacher_id.
        # IF timetable.teacher_id is missing/null, we might fall back to subject check?
        # For now, strict check.
        return "Access Denied: This class is not assigned to you.", 403
        
    # 3. Check if already completed? (Optional, but good UX)
    # The dashboard handles this visually, but backend should allow re-entry if needed (or block).
    # User said "Prevent marking attendance for the same timetable slot twice on same date"
    # Logic for that usually happens during SAVE. 
    # But we can check here to redirect? 
    # Let's let them open the page, but maybe show a warning?
    # Or rely on the "Mark" button being disabled in dashboard if completed.
    
    return render_template('teacher_mark_attendance.html',
                           timetable=entry,
                           department=entry['department'],
                           year=entry['year'],
                           subject=entry['subject'],
                           section=entry.get('section', 'A'), # Default if missing
                           period=f"{entry['start_time']} - {entry['end_time']}")

@app.route('/teacher/attendance/mode/<int:timetable_id>')
@login_required(['admin', 'teacher'])
def teacher_attendance_mode(timetable_id):
    """
    Sets up the locked context for attendance marking and renders the mode selection page.
    """
    teacher_id = session['user_id']
    entry = database_manager.get_timetable_entry(timetable_id)
    
    if not entry:
        return "Timetable entry not found", 404
    
    is_altered = False
    assigned_staff_name = None
    assigned_teacher_id = entry.get('teacher_id')

    if assigned_teacher_id != teacher_id and session.get('role') != 'admin':
        is_altered = True
        if assigned_teacher_id:
            assigned_teacher = database_manager.get_user_info(assigned_teacher_id)
            if assigned_teacher:
                assigned_staff_name = assigned_teacher.get('full_name')
                assigned_staff_name = assigned_teacher.get('full_name')
        
    # Store context in session
    # 1. Active Attendance Context (for locked UI logic)
    session['active_attendance_context'] = {
        'timetable_id': timetable_id,
        'department': entry['department'],
        'year': entry['year'], # Keep alias for compatibility
        'batch_year': entry.get('batch_year'),
        'semester': entry.get('semester'),
        'subject': entry['subject'],
        'section': entry.get('section', 'A'),
        'period': f"{entry['start_time']} - {entry['end_time']}",
        'date': datetime.now().strftime('%Y-%m-%d'),
        'is_altered': is_altered,
        'assigned_teacher_id': assigned_teacher_id,
        'assigned_staff_name': assigned_staff_name,
        'period_no': entry.get('period_no')
    }
    
    # Clear previous session data (OD list) to avoid bleeding between sessions
    session.pop('od_students', None)
    session.modified = True
    
    # 2. attendance_class (For video_feed and mark_from_face compatibility)
    session['attendance_class'] = {
        'department': entry['department'],
        'year': entry['year'],
        'subject': entry['subject'],
        'period_no': entry.get('period_no') # Crucial for mark_attendance_from_face
    }

    # Render existing mode selection page, context will be used in template
    return render_template('select_attendance_mode.html', 
                           locked_context=session['active_attendance_context'],
                           is_altered_period=is_altered,
                           assigned_staff_name=assigned_staff_name)

@app.route('/teacher/attendance/save_altered_reason', methods=['POST'])
@login_required(['admin', 'teacher'])
def save_altered_reason():
    """
    Stores the reason for marking attendance in an altered period in the session.
    """
    data = request.get_json()
    reason = data.get('reason')
    if reason:
        if 'active_attendance_context' in session:
            session['active_attendance_context']['altered_reason'] = reason
            session.modified = True
            return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Invalid reason'}), 400

@app.route('/teacher/mark_attendance/face')
@login_required(['admin', 'teacher'])
def teacher_mark_attendance_face():
    """
    Renders the live face recognition page for teachers.
    Context is already locked in session['attendance_class'] by 'teacher_attendance_mode'.
    """
    context = session.get('active_attendance_context')
    if not context:
        return redirect(url_for('teacher_dashboard'))
        
    # Render the configuration page (teacher_mark_attendance.html)
    # This page allows confirming details and entering OD students before starting.
    return render_template('teacher_mark_attendance.html', 
                          department=context['department'],
                          year=context['year'],
                          subject=context['subject'],
                          period=context.get('period'),
                          period_no=context.get('period_no'),
                          semester=context.get('semester'),
                          batch_year=context.get('batch_year'))

@app.route('/teacher/mark_attendance/face/start', methods=['POST'])
@login_required(['admin', 'teacher'])
def start_face_recognition_session():
    """
    Handles the start of the face recognition session.
    Processes OD students if any, then renders the live camera view.
    """
    context = session.get('active_attendance_context')
    if not context:
        flash("Session expired or invalid context.", "error")
        return redirect(url_for('teacher_dashboard'))

    department = request.form.get('department')
    year = request.form.get('year')
    subject = request.form.get('subject')
    od_students_str = request.form.get('od_students')

    # Process OD Students
    if od_students_str:
        try:
            # Import helper to resolve roll numbers
            from backend import database_manager
            
            # Get all students in class
            all_students = database_manager.get_students_by_class(
                department, year, 
                batch_year=context.get('batch_year'),
                semester=context.get('semester')
            )
            
            # Parse OD input (comma separated suffixes)
            od_suffixes = [s.strip() for s in od_students_str.split(',') if s.strip()]
            
            marked_count = 0
            user_id = session.get('user_id')
            timestamp = datetime.now()
            
            conn = database_manager.get_db_connection()
            if conn:
                cursor = conn.cursor()
                
                for suffix in od_suffixes:
                    # Find student where roll number ends with suffix
                    # This is a simple matching logic, usually last 2-3 digits
                    target_student = None
                    for s in all_students:
                        if s['roll_number'] and s['roll_number'].endswith(suffix):
                            target_student = s
                            break
                    
                    if target_student:
                        # Mark as OD - Check for existing record first to avoid constraint errors
                        # (Constraint names vary or might be missing, so check-first is safer)
                        check_query = """
                            SELECT id FROM attendance 
                            WHERE user_id = %s 
                            AND subject = %s 
                            AND timestamp::date = %s::date
                            AND (period_no = %s OR (period_no IS NULL AND %s IS NULL))
                        """
                        cursor.execute(check_query, (target_student['id'], subject, timestamp, context.get('period_no'), context.get('period_no')))

                        existing = cursor.fetchone()

                        if existing:
                            # Update existing
                            cursor.execute("""
                                UPDATE attendance 
                                SET status = 'OD', method = 'OD_Pre_Session', marked_by = %s, timestamp = %s
                                WHERE id = %s
                            """, (user_id, timestamp, existing[0]))
                        else:
                            # Insert new
                            cursor.execute("""
                                INSERT INTO attendance (user_id, subject, timestamp, status, method, marked_by, period_no)
                                VALUES (%s, %s, %s, 'OD', 'OD_Pre_Session', %s, %s)
                            """, (target_student['id'], subject, timestamp, user_id, context.get('period_no')))
                        
                        marked_count += 1
                
                conn.commit()
                conn.close()
                logging.info(f"Marked {marked_count} students as OD before session for {subject}")
                
        except Exception as e:
            logging.error(f"Error processing OD students: {e}")
            flash(f"Error marking OD students: {e}", "error")

    # Render the actual live face recognition page
    students = _get_student_list_with_status(department, year, subject, context.get('period_no'), 
                                            batch_year=context.get('batch_year'), semester=context.get('semester'))
    
    return render_template('mark_attendance.html', 
                          department=department,
                          year=year,
                          subject=subject,
                          period=context.get('period'),
                          period_no=context.get('period_no'),
                          semester=context.get('semester'),
                          batch_year=context.get('batch_year'),
                          students=students)


@app.route('/teacher/class_photo_attendance', methods=['GET', 'POST'])
@login_required(['admin', 'teacher'])
def teacher_class_photo_attendance():
    """
    Teacher wrapper for class photo attendance.
    """
    # Just reuse the existing logic which already handles "is_locked"
    # But we need to make sure the form action is correct.
    # In class_photo_attendance.html, the form likely posts to 'process_class_photo_attendance_route' (via JS?)
    # ACTUALLY, line 1640 is `process_class_photo_attendance_route`.
    # Let's check `class_photo_attendance.html` content again for the form action. I viewed it in Step 426.
    # I didn't verify the form handling script (JS).
    # If it's a standard form, I need to know where it posts. 
    # It seems to handle upload via JS -> /admin/class_photo_attendance/process
    # Since I can't change admin templates, I must ensure that route allows teachers. 
    # Checked line 1641: @login_required(['admin', 'teacher']) -> IT ALLOWS TEACHERS.
    # So the only issue is GETting the page with locked context.
    
    # I can just call the existing function or redirect?
    # Redirecting to /admin/class_photo_attendance?locked=true works!
    # User asked for "Add route: /teacher/class_photo_attendance"
    
    return redirect(url_for('class_photo_attendance_page', locked='true'))

@app.route('/teacher/manual_attendance')
@login_required(['admin', 'teacher'])
def teacher_manual_attendance():
    return redirect(url_for('manual_attendance', locked='true'))





@app.route('/teacher/start_scheduled_class/<int:timetable_id>')
@login_required('teacher')
def start_scheduled_class(timetable_id):
    """
    Enforces strict timetable constraints before initializing the attendance session.
    """
    entry = database_manager.get_timetable_entry(timetable_id)
    if not entry:
        return "Timetable entry not found", 404
        
    # 1. RBAC: Verify Teacher
    if entry.get('teacher_id') is not None and entry.get('teacher_id') != session['user_id']:
         logging.warning(f"Teacher {session['user_id']} tried to access timetable {timetable_id} assigned to {entry.get('teacher_id')}")
         return "Access Denied: Not your class", 403
             
    # 2. Time Window Check
    now = datetime.now()
    current_time = now.time()
    # Ensure day matches
    if entry.get('day') != now.strftime('%A'):
        return "Invalid Class: Wrong Day", 400
        
    slot_start = datetime.strptime(str(entry['start_time']), '%H:%M:%S').time()
    slot_end = datetime.strptime(str(entry['end_time']), '%H:%M:%S').time()
    
    # Requirement: Strictly within the period time
    if not (slot_start <= current_time <= slot_end):
        return "Attendance Window Closed (Class time is over or has not started)", 403
        
    # 3. Check if already completed (prevent crash, but allow re-entry if needed for resilience)
    pass 

    # 4. Valid -> Initialize Session
    session['attendance_class'] = {
        'department': entry['department'],
        'year': entry['year'],
        'subject': entry['subject'],
        'period_no': entry.get('period_no')
    }

    
    biometric_logic.resume_scanning()
    
    students = _get_student_list_with_status(entry['department'], entry['year'], entry['subject'], 
                                            entry.get('period_no'), batch_year=None, semester=None)
    
    return render_template('mark_attendance.html', 
                          subject=entry['subject'], 
                          department=entry['department'], 
                          year=entry['year'],
                          period_no=entry.get('period_no'),
                          students=students)


@app.route('/select_attendance_mode')
@login_required(['admin', 'teacher'])
def select_attendance_mode():
    # Helper to carry over query params (dept, year, subject) to the next step
    
    # Check if manual entry (Sidebar) -> Clear locked context
    if request.args.get('manual_entry') == 'true':
        session.pop('active_attendance_context', None)
        session.pop('attendance_class', None) # Also clear this just in case
        return render_template('select_attendance_mode.html', locked_context=None)
        
    return render_template('select_attendance_mode.html', 
                          locked_context=session.get('active_attendance_context'))

@app.route('/select_class')
@login_required(['admin', 'teacher'])
def select_class():
    mode = request.args.get('mode', 'face')
    
    # Get dropdown data
    departments = database_manager.get_departments()
    years = database_manager.get_years() # Keep for reference if needed, but we need batches
    batches = database_manager.get_academic_years()
    subjects = database_manager.get_all_subjects()
    
    return render_template('select_class.html', 
                          departments=departments, 
                          years=years, 
                          batches=batches,
                          subjects=subjects,
                          mode=mode)

# Route Alias/Wrapper for Face Recognition
@app.route('/mark_attendance/face', methods=['GET', 'POST'])
@login_required(['admin', 'teacher'])
def mark_attendance_face_recognition():
    if request.method == 'GET':
        dept = request.args.get('department')
        year = request.args.get('year')
        subject = request.args.get('subject')
        period = request.args.get('period')
        
        if dept and year and subject:
             session['attendance_class'] = {
                 'department': dept,
                 'year': year,
                 'subject': subject,
                 'period_no': period
             }
             students = _get_student_list_with_status(dept, year, subject, period)
             return render_template('mark_attendance.html', 
                                   department=dept, 
                                   year=year, 
                                   subject=subject,
                                   period=period,
                                   period_no=period,
                                   students=students)

        return redirect(url_for('select_class'))
        
    # POST from select_class form
    department = request.form.get('department')
    year = request.form.get('year')
    subject = request.form.get('subject')
    period = request.form.get('period')
    semester = request.form.get('semester')
    batch_year = request.form.get('batch_year')
    time_display = request.form.get('time_display', '') 
    
    if not (department and year and subject):
        flash('All fields are required', 'error')
        return redirect(url_for('select_class'))

    # --- Unified Altered Period Detection (Manual Flow) ---
    is_altered = False
    assigned_staff_name = None
    assigned_teacher_id = None
    timetable_id = None

    is_altered, assigned_staff_name, assigned_teacher_id, timetable_id = _check_altered_period(
        department, batch_year, subject, period
    )

    # Build Context
    context = {
        'timetable_id': timetable_id,
        'department': department,
        'year': year,
        'batch_year': batch_year,
        'semester': semester,
        'subject': subject,
        'period': period,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'is_altered': is_altered,
        'assigned_teacher_id': assigned_teacher_id,
        'assigned_staff_name': assigned_staff_name
    }
    session['active_attendance_context'] = context
    session['attendance_class'] = context # Sync for compatibility

    # REDIRECT if altered but no reason provided yet
    if is_altered and not context.get('altered_reason'):
        logging.info(f"Face Recog Altered Period detected. Redirecting to justification.")
        return render_template('select_attendance_mode.html', 
                             locked_context=context,
                             is_altered_period=True,
                             assigned_staff_name=assigned_staff_name)

    # Store in session for video_feed route
    try:
        p_no = int(period) if period is not None else None
    except (ValueError, TypeError):
        p_no = period

    session['attendance_class'] = {
        'department': department,
        'year': year,
        'subject': subject,
        'period_no': p_no,
        'semester': semester,
        'batch_year': batch_year  # Added batch_year
    }
        
    # Handle OD Students
    od_student_ids = []
    od_input = request.form.get('od_students', '').strip()
    
    if od_input:
        try:
            partial_rolls = [r.strip() for r in od_input.split(',') if r.strip()]
            resolved_od = resolve_od_roll_numbers(
                partial_rolls, 
                department, 
                year,
                batch_year=batch_year,
                semester=semester
            )
            od_student_ids = [student[0] for student in resolved_od]
            logging.info(f"OD Students Resolved: {len(od_student_ids)} students")
        except Exception as e:
            logging.error(f"Error resolving OD students: {e}")

    # Store in session for finish_attendance to use
    session['od_students'] = od_student_ids

    # Session already configured above (attendance_class)
    
    # Sanitize period for display (avoid "Period Period X")
    display_period = str(period).replace("Period", "").replace("period", "").strip()

    return render_template('mark_attendance.html', 
                         subject=subject, 
                         department=department,
                         year=year,
                         period=display_period,
                         semester=semester,
                         batch_year=batch_year, # Added missing param
                         period_no=period,      # Ensure period_no is available for report link
                         time_display=time_display,
                         od_count=len(od_student_ids),
                         students=_get_student_list_with_status(department, year, subject, period, 
                                                               batch_year=batch_year, semester=semester) or [])





@app.route('/admin/dashboard')
@login_required('admin')
def admin_dashboard():
    today_date = datetime.now().strftime('%Y-%m-%d')
    start_date = request.args.get('start_date', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    
    # Existing Metrics
    total_students = database_manager.count_users('student')
    total_teachers = database_manager.count_users('teacher')
    present_today = database_manager.get_today_present_count()
    
    # NEW METRICS (Departments, Classes, Batches, Semesters)
    total_departments = database_manager.count_academic_departments()
    total_classes = database_manager.count_academic_classes()
    total_batches = database_manager.count_academic_batches()
    total_semesters = database_manager.count_semesters()
    
    users = database_manager.get_all_users()
    overall_attendance_stats = database_manager.get_overall_attendance_by_date_range(start_date, end_date)
    
    # Fetch Analytics Data for Chart
    analytics_data = database_manager.get_analytics_data()

    return render_template('admin_dashboard.html',
                            total_students=total_students,
                            total_teachers=total_teachers,
                            present_today=present_today,
                            # Pass New Metrics
                            total_departments=total_departments,
                            total_classes=total_classes,
                            total_batches=total_batches,
                            total_semesters=total_semesters,
                            
                            users=users,
                            overall_attendance_stats=overall_attendance_stats,
                            analytics_data=analytics_data,
                            start_date=start_date,
                            end_date=end_date)
                            


@app.route('/admin/report/export')
@login_required('admin')
def export_attendance_report():
    try:
        format_type = request.args.get('format', 'csv')
        department = request.args.get('department', 'All').strip() or 'All'
        year = request.args.get('year', 'All').strip() or 'All'
        batch_year = request.args.get('batch_year', '').strip()
        subject = request.args.get('subject', 'All').strip() or 'All'
        semester = request.args.get('semester', 'All').strip() or 'All'
        period = request.args.get('period', 'All').strip() or 'All' # Add period
        
        today_date = datetime.now().strftime('%Y-%m-%d')
        start_date = request.args.get('start_date', today_date)
        end_date = request.args.get('end_date', today_date)
        
        # Log export request
        logging.info(f"Export Request: Format={format_type}, Dept={department}, Year={year}, Batch={batch_year}, Sub={subject}, Period={period}")

        view_mode = request.args.get('view_mode', 'Raw Records')
        
        # Calculate duration
        is_multi_day = False
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            if (end_dt - start_dt).days > 0:
                is_multi_day = True
        except: pass

        # Fetch Data
        attendance_logs = database_manager.get_official_attendance_report(
            department, year, subject, start_date, end_date, batch_year=batch_year, semester=semester, period=period, view_mode=view_mode
        )
        
        # Handle empty data safely
        if attendance_logs is None: 
            attendance_logs = []

        # --- CSV EXPORT ---
        if format_type == 'csv':
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header based on view_mode
            if view_mode == 'Daily Summary':
                writer.writerow(['Date', 'Total Periods', 'Present', 'Attendance %'])
                for log in attendance_logs:
                    writer.writerow([log.get('group_date'), log.get('total_classes'), log.get('present_count'), f"{log.get('attendance_pct', 0)}%"])
            elif view_mode == 'Period Wise':
                writer.writerow(['Date', 'Period', 'Total Students', 'Present', 'Absent'])
                for log in attendance_logs:
                    writer.writerow([log.get('group_date'), log.get('period_no'), log.get('total_classes'), log.get('present_count'), log.get('absent_count')])
            elif view_mode == 'Monthly Summary':
                writer.writerow(['Month', 'Total Periods', 'Present', 'Attendance %'])
                for log in attendance_logs:
                    writer.writerow([log.get('month_name'), log.get('total_classes'), log.get('present_count'), f"{log.get('attendance_pct', 0)}%"])
            elif view_mode == 'Student Wise':
                writer.writerow(['Student Name', 'Register No', 'Total Periods', 'Present', 'Attendance %'])
                for log in attendance_logs:
                    writer.writerow([log.get('student'), log.get('register_no'), log.get('total_classes'), log.get('present_count'), f"{log.get('attendance_pct', 0)}%"])
            elif view_mode == 'Subject Wise':
                writer.writerow(['Subject', 'Total Periods', 'Present', 'Attendance %'])
                for log in attendance_logs:
                    writer.writerow([log.get('subject'), log.get('total_classes'), log.get('present_count'), f"{log.get('attendance_pct', 0)}%"])
            else:
                # Raw Records - Hide Status/Proof if multi-day
                if is_multi_day:
                    writer.writerow(['Student Name', 'Roll Number', 'Register No', 'Subject', 'Total Classes', 'Present Classes', 'Attendance %'])
                    for log in attendance_logs:
                         writer.writerow([
                             log.get('full_name'), 
                             log.get('roll_number'), 
                             log.get('register_number'),
                             log.get('subject', '-'), 
                             log.get('total_classes', 1),
                             log.get('present_count', 0),
                             f"{log.get('attendance_pct', 0)}%"
                         ])
                else:
                    # Raw Records - Keep Mode/Status/Proof
                    writer.writerow(['Date', 'Time', 'Period', 'Student Name', 'Roll Number', 'Subject', 'Mode', 'Status'])
                    for log in attendance_logs:
                        ts = log.get('timestamp')
                        writer.writerow([
                            ts.strftime('%Y-%m-%d') if ts else '-',
                            ts.strftime('%H:%M:%S') if ts else '-',
                            log.get('period_no', '-'),
                            log.get('full_name', ''),
                            log.get('roll_number', ''),
                            log.get('subject', '-'),
                            log.get('method', 'System'),
                            log.get('final_status', log.get('status', 'Absent'))
                        ])
                
            filename = f"attendance_report_{department}_{year}_{subject}.csv".replace(' ', '_')
            return Response(output.getvalue(), mimetype="text/csv", 
                            headers={"Content-Disposition": f"attachment;filename={filename}"})

        # --- EXCEL EXPORT ---
        elif format_type == 'excel':
            from openpyxl import Workbook
            from openpyxl.utils import get_column_letter
            from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
            
            wb = Workbook()
            ws = wb.active
            ws.title = "Attendance_Report"
            
            # Freeze Header
            ws.freeze_panes = "A4" 
            
            # 1. Title (A-K)
            ws.merge_cells('A1:K1')
            ws['A1'] = "Attendance Report"
            ws['A1'].font = Font(size=16, bold=True, color="000000")
            ws['A1'].alignment = Alignment(horizontal='center')
            
            # Generated Timestamp
            ws.merge_cells('L1:M1')
            generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ws['L1'] = f"Generated: {generated_at}"
            ws['L1'].font = Font(size=9, italic=True, color="555555")
            ws['L1'].alignment = Alignment(horizontal='right', vertical='center')
            
            # 2. Filter Info
            ws.merge_cells('A2:M2')
            filter_text = f"Filters: Dept: {department} | Batch: {batch_year} | Sem: {semester} | Date: {start_date} to {end_date}"
            ws['A2'].value = filter_text
            ws['A2'].alignment = Alignment(horizontal='center')
            ws['A2'].font = Font(italic=True, size=10)

            # 3. Headers
            # Dynamic Headers based on view_mode
            if view_mode == 'Daily Summary': headers = ["Date", "Total Periods", "Present", "Attendance %"]
            elif view_mode == 'Period Wise': headers = ["Date", "Period", "Total Students", "Present", "Absent"]
            elif view_mode == 'Monthly Summary': headers = ["Month", "Total Periods", "Present", "Attendance %"]
            elif view_mode == 'Student Wise': headers = ["Student Name", "Register No", "Total Periods", "Present", "Attendance %"]
            elif view_mode == 'Subject Wise': headers = ["Subject", "Total Periods", "Present", "Attendance %"]
            else: 
                if is_multi_day:
                    headers = ["Student Name", "Roll Number", "Register No", "Subject", "Total Classes", "Present Classes", "Attendance %"]
                else:
                    headers = ["Date", "Period", "Student Name", "Roll Number", "Subject", "Mode", "Status"]
            
            for col_idx, header in enumerate(headers, 1):
                cell = ws.cell(row=3, column=col_idx, value=header)
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                cell.alignment = Alignment(horizontal='center', vertical='center')

            # 4. Data Rows
            for idx, record in enumerate(attendance_logs, 1):
                row_num = 3 + idx
                if view_mode == 'Daily Summary':
                    row_data = [record.get('group_date'), record.get('total_classes'), record.get('present_count'), f"{record.get('attendance_pct', 0)}%"]
                elif view_mode == 'Period Wise':
                    row_data = [record.get('group_date'), record.get('period_no'), record.get('total_classes'), record.get('present_count'), record.get('absent_count')]
                elif view_mode == 'Monthly Summary':
                    row_data = [record.get('month_name'), record.get('total_classes'), record.get('present_count'), f"{record.get('attendance_pct', 0)}%"]
                elif view_mode == 'Student Wise':
                    row_data = [record.get('student'), record.get('register_no'), record.get('total_classes'), record.get('present_count'), f"{record.get('attendance_pct', 0)}%"]
                elif view_mode == 'Subject Wise':
                    row_data = [record.get('subject'), record.get('total_classes'), record.get('present_count'), f"{record.get('attendance_pct', 0)}%"]
                else:
                    if is_multi_day:
                        row_data = [
                            record.get('full_name'),
                            record.get('roll_number'),
                            record.get('register_number'),
                            record.get('subject', '-'),
                            record.get('total_classes', 1),
                            record.get('present_count', 0),
                            f"{record.get('attendance_pct', 0)}%"
                        ]
                    else:
                        ts = record.get('timestamp')
                        row_data = [
                            ts.strftime('%Y-%m-%d') if ts else '-',
                            record.get('period_no', '-'),
                            record.get('full_name', ''),
                            record.get('roll_number', ''),
                            record.get('subject', '-'),
                            record.get('method', 'System'),
                            record.get('final_status', record.get('status', 'Absent'))
                        ]
                
                for col_idx, val in enumerate(row_data, 1):
                    cell = ws.cell(row=row_num, column=col_idx, value=val)
                    cell.alignment = Alignment(horizontal='center', vertical='center')
                    # Basic status coloring for Raw Records
                    if view_mode == 'Raw Records' and not is_multi_day and col_idx == 6:
                        if val == 'Present': cell.fill = PatternFill(start_color="C6EFCE", fill_type="solid")
                        elif val == 'Absent': cell.fill = PatternFill(start_color="FFC7CE", fill_type="solid")

            # Auto-adjust column widths
            for col in ws.columns:
                max_length = 0
                column = col[0].column_letter # Get the column name
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                ws.column_dimensions[column].width = adjusted_width
                
            output = io.BytesIO()
            wb.save(output)
            output.seek(0)
            
            filename = f"attendance_report_{department}_{datetime.now().strftime('%Y%m%d')}.xlsx"
            return Response(output.read(), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                            headers={"Content-Disposition": f"attachment;filename={filename}"})


        # --- PDF EXPORT ---
        elif format_type == 'pdf':
            output = io.BytesIO()
            doc = SimpleDocTemplate(output, pagesize=landscape(A4)) # Use Landscape
            elements = []
            
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Heading1'],
                alignment=TA_CENTER,
                fontSize=16,
                spaceAfter=12
            )
            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=styles['Normal'],
                alignment=TA_CENTER,
                fontSize=12,
                spaceAfter=6
            )
            
            # Header
            elements.append(Paragraph("Attendance Management System", title_style))
            
            filter_summary = f"<b>Filters:</b> Dept: {department} | Batch: {batch_year} | Sem: {semester} | Date: {start_date} to {end_date}<br/>"
            filter_summary += f"<b>Generated At:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            elements.append(Paragraph(filter_summary, styles['Normal']))
            elements.append(Spacer(1, 0.2 * inch))
            
            # Table Data
            if is_multi_day or view_mode != 'Raw Records':
                # Aggregated Headers
                headers = ["S.No", "Register No", "Student Name", "Dept", "Batch", "Total", "Present", "Percentage"]
                col_widths = [35, 90, 160, 90, 70, 60, 60, 70]
                data = [headers]
                
                for idx, record in enumerate(attendance_logs, 1):
                    row = [
                        str(idx),
                        str(record.get('register_no') or record.get('register_number', '')),
                        record.get('student') or record.get('full_name', ''),
                        record.get('department', ''),
                        str(record.get('batch_year', '')),
                        str(record.get('total_classes') or record.get('total_attendance_records', 1)),
                        str(record.get('present_count', 0)),
                        f"{record.get('attendance_pct') or record.get('percentage', 0)}%"
                    ]
                    data.append(row)
            else:
                # Individual Headers - Remove Registered By
                headers = ["S.No", "Register No", "Student Name", "Dept", "Batch", "Date", "Status"]
                col_widths = [35, 100, 180, 100, 70, 80, 70]
                data = [headers]
                
                for idx, record in enumerate(attendance_logs, 1):
                    ts = record.get('timestamp')
                    date_val = ts.strftime('%Y-%m-%d') if ts else '-'
                    status = record.get('final_status', record.get('status', 'Absent'))
                    
                    row = [
                        str(idx),
                        str(record.get('register_number', '')),
                        record.get('full_name', ''),
                        record.get('department', ''),
                        str(record.get('batch_year', '')),
                        date_val,
                        status
                    ]
                    data.append(row)
                
            # Table Style
            table = Table(data, colWidths=col_widths)
            
            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ])
            
            # Row Styling (Green/Red text logic tricky in TableStyle per cell, stick to simple)
            # Or alternates
            # style.add('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            
            table.setStyle(style)
            elements.append(table)
            
            doc.build(elements)
            output.seek(0)
            
            filename = f"Attendance_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            return send_file(output, 
                             mimetype='application/pdf',
                             as_attachment=True, 
                             download_name=filename)

        return "Invalid Format", 400
            
    except Exception as e:
        logging.error(f"Export Error: {e}")
        return f"Error exporting report: {e}", 500


@app.route('/admin/attendance_report')
@login_required('admin')
def admin_attendance_report():
    departments = database_manager.get_academic_departments()
    years = database_manager.get_years()
    batches = database_manager.get_academic_years() # Fetch Batches
    subjects = database_manager.get_academic_subjects()
    periods = database_manager.get_periods() # Fetch Periods

    selected_dept = request.args.get('department', 'All') or 'All'
    selected_year = request.args.get('year', 'All') or 'All'
    selected_subject = request.args.get('subject', 'All') or 'All'
    batch_year = request.args.get('batch_year', '').strip()
    semester = request.args.get('semester', 'All') or 'All'
    status_filter = request.args.get('status', 'All') or 'All'
    method = request.args.get('method')
    selected_period = request.args.get('period', 'All') or 'All' # Period Filter
    view_mode = request.args.get('view_mode', 'Raw Records') or 'Raw Records'
    
    today_date = datetime.now().strftime('%Y-%m-%d')
    start_date = request.args.get('start_date', today_date)
    end_date = request.args.get('end_date', today_date)
    import sys
    # DEBUG: Print incoming request args
    print(f"DEBUG ADMIN REPORT ARGS: {request.args}")
    sys.stdout.flush()

    page = request.args.get('page', 1, type=int)
    limit = 50
    offset = (page - 1) * limit
    
    logs = database_manager.get_official_attendance_report(
             selected_dept, selected_year, selected_subject, start_date, end_date, 
             limit, offset, 
             batch_year=batch_year, 
             semester=semester, 
             status=status_filter,
             method=method,
             period=selected_period,
             view_mode=view_mode
         )
         
    # Calculate Pagination
    total_records = len(logs) if limit is None else database_manager.get_attendance_count(
        selected_dept, selected_year, selected_subject, start_date, end_date, 
        batch_year=batch_year, 
        semester=semester,
        status=status_filter,
        # Period count support? Not critical for count but ideally yes.
        # Assuming count function isn't updated yet, keep as is or accept slight mismatch in pagination count.
        # For now, simplistic approach:
    )
    # The count usage here is tricky if get_attendance_count doesn't support period.
    # But since we limit at query level, pagination works.
    
    import math
    total_pages = math.ceil(total_records / limit)
    
    sem_info = None
    class_summary = None
    if batch_year and semester and semester != 'All':
        sem_info = database_manager.get_semester_info(batch_year, semester)
        # Fetch Class Summary for the dashboard cards
        class_summary = database_manager.get_admin_report_summary(
            selected_dept, selected_year, batch_year=batch_year, semester=semester
        )
    
    return render_template('admin_attendance_report.html',
                            departments=departments,
                            years=years,
                            batches=batches, 
                            subjects=subjects,
                            periods=periods, # Pass periods
                            selected_dept=selected_dept,
                            selected_year=selected_year,
                            selected_subject=selected_subject,
                            selected_period=selected_period, # Pass selected_period
                            start_date=start_date,
                            end_date=end_date,                    
                            attendance_logs=logs, # Use correct variable name
                            page=page,
                            limit=limit,
                            total_pages=total_pages,
                            batch_year=batch_year,
                            selected_semester=semester,
                            status_filter=status_filter,
                            method=method,
                            sem_info=sem_info,
                            class_summary=class_summary,
                            view_mode=view_mode)

@app.route('/admin/attendance_report/print')
@login_required('admin')
def print_attendance_report():
    try:
        # Same logic as export filters
        department = request.args.get('department', 'All').strip() or 'All'
        year = request.args.get('year', 'All').strip() or 'All'
        batch_year = request.args.get('batch_year', '').strip()
        subject = request.args.get('subject', 'All').strip() or 'All'
        semester = request.args.get('semester', 'All').strip() or 'All'
        status_filter = request.args.get('status', 'All').strip() or 'All'
        
        today_date = datetime.now().strftime('%Y-%m-%d')
        start_date = request.args.get('start_date', today_date)
        end_date = request.args.get('end_date', today_date)
        
        # New: Method Filter (e.g. 'class_photo')
        method = request.args.get('method')
        
        # Fetch Data (Official Report Logic Only for consistency)
        logs = database_manager.get_official_attendance_report(
                 department, year, subject, start_date, end_date, 
                 batch_year=batch_year, 
                 semester=semester,
                 status=status_filter,
                 method=method, # Pass method here
                 limit=100000, offset=0 # No pagination for print
             )
             
        return render_template('print_report.html',
            logs=logs,
            department=department,
            batch_year=batch_year,
            semester=semester,
            subject=subject,
            start_date=start_date,
            end_date=end_date,
            status_filter=status_filter,
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            generated_by=session.get('username', 'Admin')
        )
    except Exception as e:
        logging.error(f"Print Report Error: {e}")
        return f"Error generating print view: {e}", 500





@app.route('/admin/attendance-report')
@login_required('admin')
def admin_attendance_report_alias():
    return redirect(url_for('admin_attendance_report', **request.args))

@app.route('/admin/registered_face/<int:user_id>', endpoint='registered_face')
@app.route('/api/image/student/<int:user_id>')
@login_required(['admin', 'teacher'])
def api_get_student_image(user_id):
    try:
        logging.info(f"DEBUG: Serving image for user_id={user_id}")
        faces_dir = biometric_logic.get_student_folder(user_id)
        logging.info(f"DEBUG: faces_dir resolved to: {faces_dir}")
        
        if not os.path.exists(faces_dir):
             logging.error(f"DEBUG: faces_dir does not exist: {faces_dir}")
             return redirect("https://via.placeholder.com/150?text=No+Folder")

        files = os.listdir(faces_dir)
        logging.info(f"DEBUG: Files in dir: {files}")
        images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
             logging.error("DEBUG: No images found in folder")
             return redirect("https://via.placeholder.com/150?text=No+Images")

        priority_names = ['registered.jpg', 'face.jpg', '0.jpg']
        selected_image = None
        for name in priority_names:
            if name in images:
                selected_image = name
                break
        
        if not selected_image:
            images.sort(key=lambda x: os.path.getmtime(os.path.join(faces_dir, x)), reverse=True)
            selected_image = images[0]
        
        logging.info(f"DEBUG: Selected image: {selected_image}")
        from flask import send_from_directory
        return send_from_directory(faces_dir, selected_image)

    except Exception as e:
        logging.error(f"Error serving student image: {e}", exc_info=True)
        return redirect("https://via.placeholder.com/150?text=Error")

@app.route('/api/image/serve_path')
@login_required(['admin', 'teacher'])
def serve_proof_by_path():
    path = request.args.get('path')
    if not path:
        return "Missing path", 400
    
    # Handle relative paths from class photos
    if not os.path.isabs(path):
        # Check if it's a class photo proof (TIMESTAMP/FILENAME)
        class_proof_path = os.path.join(biometric_logic.ATTENDANCE_PROOFS_DIR, "class_crops", path)
        if os.path.exists(class_proof_path):
            return send_file(class_proof_path)
        
        # Check if it's a standard proof
        standard_proof_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance_proofs", path)
        if os.path.exists(standard_proof_path):
            return send_file(standard_proof_path)
            
    # Handle absolute paths
    if not os.path.isabs(path):
         # If still not absolute, maybe it was meant to be relative to CWD
         path = os.path.abspath(path)

    if not os.path.exists(path):
        # Return a placeholder instead of 404 to avoid broken images in report
        return redirect("https://via.placeholder.com/150?text=No+Proof")
        
    return send_file(path)

@app.route('/admin/analytics')
@login_required('admin')
def admin_analytics():
    departments = database_manager.get_departments()
    years = database_manager.get_years()
    subjects = database_manager.get_all_subjects()
    selected_department = request.args.get('department', 'All')
    selected_year = request.args.get('year', 'All')
    selected_subject = request.args.get('subject', 'All')
    analytics_data = database_manager.get_analytics_data(selected_department, selected_year, selected_subject)
    return render_template('admin_analytics.html',
                            departments=departments,
                            years=years,
                            subjects=subjects,
                            selected_department=selected_department,
                            selected_year=selected_year,
                            selected_subject=selected_subject,
                            analytics_data=analytics_data)

@app.route('/admin/behavior_logs')
@login_required(['admin', 'teacher'])
def behavior_logs():
    """
    Renders the Behavior Monitoring Logs page.
    Available to Admin and Teachers.
    """
    departments = database_manager.get_academic_departments()
    batches = database_manager.get_academic_years()
    subjects = database_manager.get_academic_subjects()
    
    selected_dept = request.args.get('department', 'All')
    selected_batch = request.args.get('batch_year', 'All')
    selected_subject = request.args.get('subject', 'All')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    violations = database_manager.get_behavior_violations(
        department=selected_dept,
        batch_year=selected_batch,
        subject=selected_subject,
        start_date=start_date,
        end_date=end_date
    )
    
    return render_template('behavior_logs.html',
                          violations=violations,
                          departments=departments,
                          batches=batches,
                          subjects=subjects,
                          selected_dept=selected_dept,
                          selected_batch=selected_batch,
                          selected_subject=selected_subject,
                          start_date=start_date,
                          end_date=end_date)

@app.route('/api/behavior-violations', methods=['POST'])
@login_required(['admin', 'teacher'])
def api_log_behavior_violation():
    """
    API endpoint to manually log or allow frontend-triggered logs for behavior violations.
    """
    try:
        data = request.json
        user_id = data.get('user_id') or data.get('student_id')
        v_type = data.get('violation_type')
        confidence = data.get('confidence', 0.0)
        session_id = data.get('session_id')
        marked_by = session.get('user_id')
        
        # Additional context from data or session
        dept = data.get('department')
        batch = data.get('batch_year')
        subject = data.get('subject')
        
        success = database_manager.log_behavior_violation(
            user_id=user_id,
            violation_type=v_type,
            confidence=confidence,
            session_id=session_id,
            marked_by=marked_by,
            department=dept,
            batch_year=batch,
            subject=subject
        )
        
        if success:
            return jsonify({"success": True, "message": "Violation logged successfully"}), 201
        return jsonify({"success": False, "message": "Failed to log violation"}), 500
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400

@app.route('/admin/user_management')
@app.route('/admin/user-management')
@login_required('admin')
def admin_user_management():
    active_tab = request.args.get('tab', 'students')
    
    # Context Data
    try:
        departments = database_manager.get_academic_departments()
        years = database_manager.get_academic_years() # Batches
        sections = database_manager.get_academic_sections(None, None) 
    except AttributeError:
        departments = []
        years = []
        sections = []
    
    selected_dept_id = request.args.get('dept_id')
    selected_year_id = request.args.get('year_id') # Legacy Batch ID
    selected_batch_year = request.args.get('batch_year') # New Batch String
    selected_section_id = request.args.get('section_id')
    selected_semester = request.args.get('semester') # New Filter
    
    students = []
    
    # Filter if Dept is present (Relaxed condition)
    if selected_dept_id:
        # Prefer Batch Year over logical Year ID if both present? 
        # User said "If batch_year is provided, derive year automatically and ignore old year parameter."
        # backend's get_students_filtered handles batch_year. We just ensure we don't conflict.
        # If batch_year is set, we can ignore year_id to be safe, or pass it if it's relevant.
        # For now, let's pass what we have, but rely on get_students_filtered.
        
        students = database_manager.get_students_filtered(
            dept_id=selected_dept_id, 
            year_id=selected_year_id, 
            section_id=selected_section_id, 
            semester=selected_semester,
            batch_year=selected_batch_year
        )
             
        # REQUIRED FIX 3: Ensure we use the dynamic status from get_students_filtered
        for s in students:
             if s.get('batch_start_year') and s.get('batch_end_year'):
                 s['batch_year'] = f"{s['batch_start_year']}-{s['batch_end_year']}"

    return render_template('admin_user_management.html', 
                         active_tab=active_tab,
                         departments=departments,
                         years=years,
                         sections=sections,
                         students=students,
                         selected_dept_id=selected_dept_id,
                         selected_year_id=selected_year_id,
                         selected_section_id=selected_section_id,
                         selected_semester=selected_semester,
                         selected_batch_year=selected_batch_year)
@app.route('/admin/bulk_enrollment')
@login_required('admin')
def bulk_enrollment_page():
    return render_template('bulk_enrollment.html')

@app.route('/admin/bulk_enrollment/upload', methods=['POST'])
@login_required('admin')
def bulk_enrollment_upload():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'success': False, 'message': 'Invalid file type. Please upload a CSV.'})

    try:
        # Read file as text lines
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        lines = stream.readlines()
        
        # 1. Header Detection Logic
        header_index = -1
        valid_headers = ['name', 'full_name', 'regno', 'register', 'register_number', 'batch', 'sem', 'semester']
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # aggressive check: must contain at least "name" AND ("reg" or "batch") to be a candidate
            if 'name' in line_lower and ('reg' in line_lower or 'batch' in line_lower):
                header_index = i
                break
        
        if header_index == -1:
            return jsonify({'success': False, 'message': 'Could not detect valid header row. Ensure columns Name, RegNo, Batch, Sem exist.'})
            
        # Parse CSV from header row
        try:
            # Re-join lines from header onwards for DictReader
            csv_content = "".join(lines[header_index:])
            csv_reader = csv.DictReader(io.StringIO(csv_content))
        except Exception as e:
             return jsonify({'success': False, 'message': f'CSV Parsing Error: {str(e)}'})

        normalized_rows = []
        for row in csv_reader:
             # Normalize Keys
             new_row = {}
             # Remove BOM if present in keys
             clean_row = {k.strip().replace('\ufeff', ''): v for k, v in row.items() if k}
             
             for k, v in clean_row.items():
                 k_lower = k.lower().strip()
                 val = v.strip() if v else ""
                 
                 if 'register' in k_lower or 'regno' in k_lower: new_row['register_number'] = val
                 elif 'full_name' in k_lower or 'name' in k_lower and 'user' not in k_lower: new_row['full_name'] = val
                 elif 'batch' in k_lower: new_row['batch'] = val
                 elif 'sem' in k_lower: new_row['semester'] = val.replace(',', '').strip() # Handle trailing comma
                 elif 'roll' in k_lower: new_row['roll_number'] = val
                 elif 'dept' in k_lower or 'department' in k_lower: new_row['department'] = val
             
             # Only add if at least one field is present
             if any(new_row.values()):
                 normalized_rows.append(new_row)
             
        success_count = 0
        skipped_count = 0
        errors = []
        
        for idx, row in enumerate(normalized_rows):
            real_row_num = header_index + idx + 2 # +1 for 0-index, +1 for header itself
            
            try:
                reg_no = row.get('register_number')
                name = row.get('full_name')
                batch_str = row.get('batch') 
                sem = row.get('semester')
                
                if not reg_no or not name or not batch_str or not sem:
                    errors.append(f"Row {real_row_num}: Skipped - Missing required fields")
                    continue
                    
                # Fix Batch
                if batch_str and '-' not in batch_str and len(batch_str) == 4:
                    batch_str = f"{batch_str}-{int(batch_str)+4}"
                
                if batch_str and '-' in batch_str:
                     try:
                         batch_start = int(batch_str.split('-')[0])
                         batch_end = int(batch_str.split('-')[1])
                     except:
                         batch_start = 2024
                         batch_end = 2028
                else:
                     batch_start = 2024
                     batch_end = 2028

                # Fix Semester/Year
                # Fix Semester/Year
                try:
                    sem_val = int(sem)
                    year_val = math.ceil(sem_val / 2)
                    year_val = max(1, min(4, year_val))
                except:
                    errors.append(f"Row {real_row_num}: Invalid Semester '{sem}'")
                    continue
                
                # Department Lookup (Resolve ID from Name)
                dept_name = row.get('department') or 'General'
                dept_id = None
                try:
                    conn_temp = database_manager.get_db_connection()
                    if conn_temp:
                         cur_temp = conn_temp.cursor()
                         # Flexible Search
                         cur_temp.execute("SELECT id, name FROM academic_departments WHERE LOWER(TRIM(name)) = LOWER(TRIM(%s)) OR LOWER(TRIM(code)) = LOWER(TRIM(%s))", (dept_name, dept_name))
                         res_d = cur_temp.fetchone()
                         if res_d: 
                             dept_id = res_d[0]
                             dept_name = res_d[1] # Unify name
                         conn_temp.close()
                except Exception as ex:
                    logging.error(f"Dept Lookup Error: {ex}")

                user_data = {
                    'username': reg_no,
                    'password': reg_no, # Default password = RegNo
                    'full_name': name,
                    'role': 'student',
                    'register_number': reg_no,
                    'roll_number': row.get('roll_number') or reg_no, # Default Roll = RegNo
                    'batch_start_year': batch_start,
                    'batch_end_year': batch_end,
                    'semester': sem_val,
                    'year': year_val,
                    'department': dept_name,
                    'dept_id': dept_id 
                }
                
                success, msg = database_manager.add_user(user_data)
                
                if success: 
                    success_count += 1
                else: 
                    if "duplicate" in msg.lower() or "already exists" in msg.lower():
                        # UPSERT: Update the existing user with new details (fixing missing batch/sem)
                        up_success, up_msg = database_manager.update_student_academic_info_by_regno(reg_no, user_data)
                        if up_success:
                             skipped_count += 1
                             errors.append(f"Row {real_row_num}: Updated existing record (Duplicate) - {reg_no}")
                        else:
                             errors.append(f"Row {real_row_num}: Failed to update duplicate - {up_msg}")
                    else:
                        errors.append(f"Row {real_row_num}: Failed - {msg}")
                
            except Exception as e:
                errors.append(f"Row {real_row_num}: Error - {str(e)}")
                
        return jsonify({
            'success': True,
            'total_processed': len(normalized_rows),
            'inserted': success_count,
            'skipped': skipped_count,
            'errors': errors
        })

    except Exception as e:
        logging.error(f"Bulk Upload Error: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/admin/teachers')
@login_required('admin')
def admin_teachers():
    teachers = database_manager.get_all_teachers()
    return render_template('admin_teachers.html', teachers=teachers)

@app.route('/admin/teachers/add', methods=['POST'])
@login_required('admin')
def add_teacher():
    try:
        full_name = request.form['full_name']
        username = request.form['username']
        password = request.form['password']
        department = 'N/A' # Teachers might not be strictly bound
        
        # database_manager.add_teacher handles hashing and uniqueness checks
        success, message = database_manager.add_teacher(
            full_name=full_name,
            username=username,
            password=password,
            department=department
        )
        
        if success:
            flash(message, 'success')
            return redirect(url_for('admin_teachers'))
        else:
            flash(message, 'error')
            return redirect(url_for('admin_teachers'))
            
    except Exception as e:
        logging.error(f"Error adding teacher: {e}")
        flash(f"Error adding teacher: {str(e)}", 'error')
        return redirect(url_for('admin_teachers'))

@app.route('/admin/teachers/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required('admin')
def edit_teacher(user_id):
    user = database_manager.get_user_info(user_id)
    if not user:
        return "User not found", 404

    if request.method == 'POST':
        full_name = request.form['full_name']
        username = request.form['username']
        password = request.form.get('password')
        department = request.form.get('department', 'N/A')
        
        if password:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        else:
            # Keep existing password if not provided (requires fetching it? 
            # database_manager.update_teacher needs the password to update.
            # user info dict has 'password' (bytea).
            hashed_password = user['password']

        database_manager.update_teacher(
            teacher_id=user_id, 
            full_name=full_name, 
            username=username, 
            password=hashed_password, 
            department=department
        )
        return redirect(url_for('admin_teachers'))
    
    # GET: Render Edit Page (Creating a simple template string or reusing a generic edit template would be best, 
    # but user asked for "edit_teacher.html" or "admin_teachers.html" flow.
    # The dangling code suggested 'edit_teacher.html'.
    departments = database_manager.get_departments()
    return render_template('edit_teacher.html', user=user, departments=departments)

@app.route('/admin/teachers/<int:user_id>/delete', methods=['POST'])
@login_required('admin')
def delete_teacher(user_id):
    database_manager.delete_teacher(user_id)
    return redirect(url_for('admin_teachers'))

@app.route('/admin/teachers/<int:teacher_id>/assign')
@login_required('admin')
def assign_subjects_page(teacher_id):
    teacher = database_manager.get_user_info(teacher_id)
    if not teacher: return "Teacher not found", 404
    
    departments = database_manager.get_departments()
    years = database_manager.get_years() # ["Year 1", "Year 2"...] or similar
    subjects = database_manager.get_all_academic_subjects()
    assignments = database_manager.get_teacher_assignments(teacher_id)
    
    return render_template('admin_assign_subjects.html', 
                           teacher=teacher, 
                           departments=departments, 
                           years=years, 
                           subjects=subjects, 
                           assignments=assignments)

@app.route('/admin/teachers/assign', methods=['POST'])
@login_required('admin')
def assign_teacher_subject_route():
    teacher_id = request.form.get('teacher_id')
    department = request.form.get('department')
    year = request.form.get('year')
    subject_id = request.form.get('subject_id')
    
    success, message = database_manager.assign_teacher_subject(teacher_id, department, year, subject_id)
    if success:
        flash(message, 'success')
    else:
        flash(f"Error: {message}", 'error')
        
    return redirect(url_for('assign_subjects_page', teacher_id=teacher_id))

@app.route('/admin/teachers/assignment/<int:assignment_id>/remove', methods=['POST'])
@login_required('admin')
def remove_teacher_assignment_route(assignment_id):
    teacher_id = request.form.get('teacher_id')
    success = database_manager.remove_teacher_assignment(assignment_id)
    if success:
        flash("Assignment removed.", 'success')
    else:
        flash("Failed to remove assignment.", 'error')
    return redirect(url_for('assign_subjects_page', teacher_id=teacher_id))

@app.route('/admin/teachers/<int:user_id>/enroll_face')
@login_required('admin')
def enroll_student_face(user_id):
    user = database_manager.get_user_info(user_id)
    if not user: return "User Not Found", 404
    
    # Stateless Return URL Logic
    return_url = request.args.get('return_url')
    # If not provided, try to construct from individual params (legacy support)
    if not return_url:
        dept_id = request.args.get('dept_id')
        year_id = request.args.get('year_id')
        section_id = request.args.get('section_id')
        if dept_id and year_id:
             return_url = url_for('admin_user_management', tab='students', dept_id=dept_id, year_id=year_id, section_id=section_id)
        else:
             return_url = url_for('admin_user_management', tab='students')

    return render_template('enroll_student_face.html', user=user, student_name=user['full_name'], user_id=user_id, return_url=return_url)


@app.route('/admin/edit_user/<int:user_id>', methods=['GET', 'POST'])
@login_required('admin')
def edit_user(user_id):
    user = database_manager.get_user_info(user_id)
    if not user:
        return "User not found", 404
        
    departments = database_manager.get_departments()
    years = database_manager.get_years()
    
    if request.method == 'POST':
        full_name = request.form['full_name']
        username = request.form['username']
        password = request.form.get('password')
        role = request.form['role']
        department = request.form['department']
        batch_year = request.form.get('batch_year')
        roll_number = request.form.get('roll_number')
        
        if password:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        else:
            hashed_password = user['password']

        database_manager.update_user_info(user_id, full_name, username, hashed_password, role, department, batch_year, roll_number)
        return redirect(url_for('admin_user_management'))
    
    return render_template('edit_user.html', user=user, departments=departments, years=years)

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required('admin')
def delete_user_route(user_id):
    database_manager.delete_user(user_id)
    return redirect(url_for('admin_user_management'))



@app.route('/api/get_assigned_teachers')
@login_required('admin')
def get_assigned_teachers_api():
    department = request.args.get('department')
    year = request.args.get('year')
    subject = request.args.get('subject')
    
    if not department or not year or not subject:
        return jsonify([])
        
    teachers = database_manager.get_teachers_by_assignment(department, year, subject)
    return jsonify(teachers)


    
@app.route('/video_feed')
@login_required(['teacher', 'admin'])
def video_feed():
    class_info = session.get('attendance_class')
    if not class_info:
        return Response("Error: Class information not found.", status=400)
    subject = class_info['subject']
    marked_by_id = session.get('user_id')
    resp = Response(biometric_logic.generate_frames(subject, marked_by_id), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route('/enrollment_video_feed/<int:user_id>')
@login_required(['teacher', 'admin'])
def enrollment_video_feed(user_id):
    resp = Response(biometric_logic.generate_enrollment_frames(user_id), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route('/enrollment_status/<int:user_id>')
@login_required(['teacher', 'admin'])
def enrollment_status(user_id):
    status = biometric_logic.get_enrollment_status(user_id)
    return jsonify(status)

@app.route('/api/recognition/latest')
@login_required(['teacher', 'admin'])
def latest_recognition_event():
    return jsonify(biometric_logic.get_latest_recognition())

@app.route('/api/recognition/process_frame', methods=['POST'])
@login_required(['teacher', 'admin'])
def process_client_frame():
    """Endpoint for processing frames from client-side camera."""
    if not biometric_logic.is_system_ready():
        return jsonify({"success": False, "message": "Biometric system is initializing. Please wait...", "status": "initializing"}), 503
        
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "message": "No image data"}), 400

        # Decode Base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "message": "Image decode failed"}), 400

        # Get context
        class_info = session.get('attendance_class')
        subject = class_info['subject'] if class_info else data.get('subject')
        marked_by_id = session.get('user_id')

        # Use session-based spoof counter to handle multiple clients
        if 'spoof_counter' not in session:
            session['spoof_counter'] = [0]
        
        result = biometric_logic.process_client_frame(
            frame, subject, marked_by_id, 
            spoof_counter=session['spoof_counter']
        )
        
        # Save updated spoof counter back to session
        session.modified = True
        
        # Store latest recognition in session for this specific client
        if result.get('success'):
            session['latest_client_recognition'] = result
        
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error in process_frame endpoint: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# --- END OF MONITORING MODULES ---

# -----------------------------------------------

@app.route('/api/live/class_students')
@login_required(['teacher', 'admin'])
def live_class_students():
    """
    Returns the full class student list with their current attendance status
    for the live Attendance Progress panel.
    """
    try:
        # Context extraction - be extremely robust
        context = session.get('active_attendance_context') or {}
        att_class = session.get('attendance_class') or {}
        
        # Merge contexts (prefer active_attendance_context for live data)
        merged_context = {**att_class, **context}
        
        # Data sanitization for robust lookup
        def clean_year(y):
            if not y: return y
            y_s = str(y).lower().strip()
            if '1st' in y_s or 'first' in y_s: return 1
            if '2nd' in y_s or 'second' in y_s: return 2
            if '3rd' in y_s or 'third' in y_s: return 3
            if '4th' in y_s or 'fourth' in y_s: return 4
            import re
            m = re.search(r'\d+', y_s)
            return int(m.group()) if m else y

        department = (merged_context.get('department') or request.args.get('department'))
        if department: department = department.strip()
        
        year = clean_year(merged_context.get('year') or request.args.get('year'))
        subject = merged_context.get('subject') or request.args.get('subject')
        batch_year = merged_context.get('batch_year') or request.args.get('batch_year')
        semester = merged_context.get('semester') or request.args.get('semester')
        period_no = merged_context.get('period_no') or merged_context.get('period') or request.args.get('period')

        if not department or not year or not subject:
            # Last ditch effort: Check if there's any active session at all in the DB for this teacher
            # But for now, we'll log and return a helpful message
            logging.warning(f"live_class_students: Missing key context. dept={department}, year={year}, sub={subject}")
            return jsonify({
                'success': False, 
                'message': 'No active session found. Please start attendance from the dashboard.',
                'students': []
            }), 200 # Return 200 to avoid console noise, but success=False

        logging.info(f"live_class_students fetch: dept={department}, year={year}, batch={batch_year}, sem={semester}, sub={subject}, period={period_no}")

        students_raw = _get_student_list_with_status(
            department, year, subject, period_no,
            batch_year=batch_year, semester=semester
        )

        # Map DB status to frontend color tokens
        STATUS_MAP = {'Present': 'green', 'OD': 'blue'}
        result = []
        for s in students_raw:
            result.append({
                'id':          s.get('id'),
                'full_name':   s.get('full_name') or "Unnamed Student",
                'roll_number': s.get('roll_number') or "N/A",
                'status':      STATUS_MAP.get(s.get('status'), 'red'),
            })

        return jsonify({
            'success': True, 
            'students': result, 
            'count': len(result),
            'context': {
                'department': department,
                'year': year,
                'semester': semester,
                'subject': subject
            }
        })

    except Exception as e:
        logging.error(f"Error in live_class_students API: {e}", exc_info=True)
        return jsonify({'success': False, 'message': 'Internal Server Error'}), 500


@app.route('/api/attendance/mark_from_face', methods=['POST'])
@login_required(['teacher', 'admin'])
def mark_attendance_from_face():
    if not biometric_logic.is_system_ready():
        return jsonify({"status": "error", "message": "Biometric system is initializing. Please wait..."}), 503
        
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        subject = data.get('subject')
        
        if not user_id:
            return jsonify({"status": "error", "message": "No student ID provided."}), 400

        # Use session-based recognition for client-side camera
        latest = session.get('latest_client_recognition')
        
        confidence = None
        proof_path = None
        if latest and str(latest.get('user_id')) == str(user_id):
            confidence = latest.get('confidence')
            proof_path = latest.get('proof_path')
            
            # Anti-Spoofing check
            if latest.get('is_live') == False: 
                return jsonify({
                    "status": "error", 
                    "message": f"Anti-Spoofing Alert: {latest.get('liveness_msg', 'Verification failed')}. Please show a live face."
                }), 403

        marked_by = session.get('user_id')

        # Robust period_no resolution from session
        period_no = session.get('attendance_class', {}).get('period_no')
        if period_no is None:
            period_no = session.get('active_attendance_context', {}).get('period_no')

        # Marking Logic
        status, msg = database_manager.mark_attendance_for_user(
            user_id, subject, marked_by, 
            status="Present", 
            confidence=confidence, 
            proof_path=proof_path,
            method=get_attendance_method('live'),
            period_no=period_no,
            is_finalized=True
        )

        # Get full user info for immediate frontend update
        user = database_manager.get_user_info(user_id)
        if not user:
            return jsonify({"status": "error", "message": f"Student ID {user_id} not found in database."}), 404
        
        # Gender from the latest recognition result OR database info
        gender = "N/A"
        if latest and latest.get('gender'):
            gender = latest.get('gender')
        elif user.get('gender'):
            gender = user.get('gender')

        response = {
            "status": status,
            "message": msg,
            "student": {
                "id": user['id'],
                "full_name": user.get('full_name', 'Unknown Student'),
                "roll_number": user.get('roll_number', 'N/A'),
                "department": user.get('department', 'N/A'),
                "year": user.get('year', 'N/A'),
                "gender": gender
            },
            "subject": subject,
            "timestamp": datetime.now().timestamp()
        }
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error in mark_attendance_from_face API: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/attendance/resume', methods=['POST'])
@login_required(['teacher', 'admin'])
def resume_scan():
    biometric_logic.resume_scanning()
    return jsonify({"success": True, "message": "Scanning resumed"})

@app.route('/api/attendance/finish', methods=['POST'])
@login_required(['teacher', 'admin'])
def finish_scan():
    try:
        # Get OD students from session
        od_student_ids = session.get('od_students', [])
        current_class = session.get('attendance_class', {})
        
        # Mark OD students with status='OD'
        if od_student_ids and current_class:
            conn = database_manager.get_db_connection()
            if conn:
                try:
                    cur = conn.cursor()
                    for user_id in od_student_ids:
                        timestamp = datetime.now()
                        cur.execute("""
                            INSERT INTO attendance (user_id, status, timestamp, subject, period_no, marked_by)
                            VALUES (%s, 'OD', %s, %s, %s, %s)
                            ON CONFLICT (user_id, subject, DATE(timestamp)) 
                            DO UPDATE SET status = 'OD'
                        """, (user_id, timestamp, current_class.get('subject'), current_class.get('period_no'), session.get('user_id')))
                    conn.commit()
                    logging.info(f"Marked {len(od_student_ids)} students as OD")
                except Exception as e:
                    logging.error(f"Error marking OD students: {e}")
                    conn.rollback()
                finally:
                    conn.close()
        
        # Finish scanning (this will mark remaining students as absent, excluding OD)
        biometric_logic.finish_scanning()
        
        # Clean up session
        session.pop('od_students', None)
        session.pop('attendance_class', None)
        
        return jsonify({"success": True, "message": "Session finished successfully"})
    except Exception as e:
        logging.error(f"Error finishing attendance session: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/attendance/current_session_stats', methods=['GET'])
@login_required(['teacher', 'admin'])
def get_current_session_stats():
    """
    Returns statistics and student lists for the current active session.
    Used for the End Session Confirmation Modal.
    """
    try:
        # 1. Get Session Context
        context = session.get('active_attendance_context')
        if not context:
            return jsonify({'success': False, 'message': 'No active session found'}), 404

        subject = context.get('subject')
        department = context.get('department')
        year = context.get('year')
        batch_year = context.get('batch_year')
        semester = context.get('semester')
        period = context.get('period')
        
        # 2. Get All Students (Total Strength) - Fetch first to use as filter
        all_students = database_manager.get_students_by_class(department, year, batch_year=batch_year, semester=semester)
        class_uids = {str(s.get('id')) for s in all_students}

        # 3. Get All Records from DB (Present, OD, etc.)
        today_str = datetime.now().strftime('%Y-%m-%d')
        period_no = context.get('period_no')
        
        db_records_raw = database_manager.get_session_attendance_records(
            subject=subject, 
            date_str=today_str, 
            period_no=period_no
        )
        
        # Deduplicate records by user_id AND ensure they belong to the current class
        deduped_db = {}
        for r in db_records_raw:
            uid = str(r.get('user_id'))
            
            # STRICT FILTER: Only include students from the active class
            if uid not in class_uids:
                continue

            if uid not in deduped_db:
                deduped_db[uid] = r
            else:
                curr = deduped_db[uid]
                # Priority 1: Edited records always win
                if r.get('edited_by') and not curr.get('edited_by'):
                    deduped_db[uid] = r
                    continue
                elif curr.get('edited_by') and not r.get('edited_by'):
                    continue
                
                # Priority 2: Present > OD > Absent
                curr_status = curr.get('status')
                new_status = r.get('status')
                
                status_weight = {'Present': 3, 'OD': 2, 'Absent': 1}
                w_curr = status_weight.get(curr_status, 0)
                w_new = status_weight.get(new_status, 0)
                
                if w_new > w_curr:
                    deduped_db[uid] = r
                elif w_new == w_curr:
                    # Priority 3: Latest timestamp
                    ts_curr = curr.get('timestamp')
                    ts_new = r.get('timestamp')
                    if ts_new and ts_curr and ts_new > ts_curr:
                        deduped_db[uid] = r
        
        db_records = list(deduped_db.values())
        
        present_records = [r for r in db_records if r.get('status') == 'Present']
        od_db_records = [r for r in db_records if r.get('status') == 'OD']
        
        # 4. Get OD Students (From Session) - Uncommitted yet
        raw_od_session_ids = session.get('od_students', [])
        # Deduplicate and filter by class scope
        od_student_ids = [str(oid) for oid in list(set(raw_od_session_ids)) if str(oid) in class_uids]
        od_records = od_db_records # Start with what's in DB
        
        if od_student_ids:
             # Resolve IDs to names for session ODs not yet in DB
             # We filter out those already in DB to avoid duplicates
             db_od_ids = {str(r.get('user_id')) for r in od_db_records}
             new_od_ids = [oid for oid in od_student_ids if str(oid) not in db_od_ids]
             
             if new_od_ids:
                 conn = database_manager.get_db_connection()
                 if conn:
                    try:
                        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                        # Ensure IDs are integers to avoid type mismatch with users.id (integer)
                        cur.execute("SELECT id, full_name, roll_number FROM users WHERE id = ANY(%s)", ([int(oid) for oid in new_od_ids],))
                        session_od_records = []
                        for r in cur.fetchall():
                            d = dict(r)
                            d['user_id'] = d.pop('id') # Map id to user_id
                            session_od_records.append(d)
                        od_records.extend(session_od_records)
                    finally:
                        conn.close()

        # 5. Calculate Absent
        # Use IDs for robust comparison
        present_ids = {str(r.get('user_id')) for r in present_records}
        
        # Filter OD: If someone was Present in DB but also in session OD (unlikely), 
        # normally Present (DB) wins. 
        od_records = [r for r in od_records if str(r.get('user_id')) not in present_ids]
        od_ids = {str(r.get('user_id')) for r in od_records}
        
        absent_records = []
        for s in all_students:
            uid = str(s.get('id'))
            if uid not in present_ids and uid not in od_ids:
                # Map to user_id for consistency in review modal
                student_data = dict(s)
                student_data['user_id'] = student_data.pop('id')
                absent_records.append(student_data)
                
        # Sort Lists
        present_records.sort(key=lambda x: x.get('roll_number') or '')
        od_records.sort(key=lambda x: x.get('roll_number') or '')
        absent_records.sort(key=lambda x: x.get('roll_number') or '')

        stats = {
            'session_info': {
                'subject': subject,
                'period': period,
                'department': department,
                'year': year,
                'date': today_str,
                'teacher_name': session.get('full_name'),
                'is_altered': context.get('is_altered', False),
                'period_no': period_no,
                'is_lab': False,
                'other_lab_periods': []
            },
            'counts': {
                'total': len(all_students),
                'present': len(present_records),
                'od': len(od_records),
                'absent': len(absent_records)
            },
            'lists': {
                'present': present_records,
                'od': od_records,
                'absent': absent_records
            }
        }

        # 6. Check for LAB replication possibility
        try:
            subject_type = database_manager.get_subject_type(subject)
            if subject_type == 'Laboratory':
                stats['session_info']['is_lab'] = True
                day_name = datetime.now().strftime('%A')
                others = database_manager.get_consecutive_lab_periods(
                    department, batch_year, semester, subject, day_name, period_no
                )
                stats['session_info']['other_lab_periods'] = others
        except Exception as e:
            logging.error(f"Error checking lab replication: {e}")

        return jsonify({'success': True, 'data': stats})

    except Exception as e:
        logging.error(f"Error getting session stats: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/attendance/replicate', methods=['POST'])
@login_required(['teacher', 'admin'])
def replicate_attendance():
    """API endpoint to replicate attendance to other periods."""
    try:
        data = request.get_json()
        target_periods = data.get('target_periods', [])
        
        if not target_periods:
            return jsonify({'success': False, 'message': 'No target periods selected'}), 400
            
        context = session.get('active_attendance_context')
        if not context:
            return jsonify({'success': False, 'message': 'No active session context'}), 404
            
        subject = context.get('subject')
        period_no = context.get('period_no')
        date_str = datetime.now().strftime('%Y-%m-%d')
        teacher_id = session.get('user_id')
        
        success, message = database_manager.replicate_attendance_db(
            subject, date_str, period_no, target_periods, teacher_id
        )
        
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        logging.error(f"Error in replication API: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/attendance/manual_save', methods=['POST'])
@login_required(['teacher', 'admin'])
def manual_save_attendance():
    """
    Saves a batch of attendance records from manual or class photo modes.
    """
    try:
        data = request.get_json()
        attendance = data.get('attendance', {}) # { student_id: status }
        subject = data.get('subject')
        period_no = data.get('period')
        marked_by = session.get('user_id')
        
        # Robust period parsing
        if isinstance(period_no, str):
             import re
             match = re.search(r'(\d+)', period_no)
             period_no = int(match.group(1)) if match else None

        for student_id, status in attendance.items():
            database_manager.mark_attendance_for_user(
                int(student_id), 
                subject, 
                marked_by, 
                status=status,
                method=get_attendance_method('manual'),
                period_no=period_no
            )
            
        return jsonify({'success': True, 'message': f'Saved {len(attendance)} records'})
    except Exception as e:
        logging.error(f"Error in manual_save API: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500




@app.route('/api/attendance/update_status', methods=['POST'])
@login_required(['teacher', 'admin'])
def update_attendance_status():
    """
    Updates the status of a specific attendance record during the review phase.
    """
    conn = database_manager.get_db_connection()
    if conn is None:
        return jsonify({'success': False, 'message': 'Database connection failed'}), 500

    try:
        data = request.get_json()
        # Be resilient to different frontend key names (student_id/user_id, status/new_status)
        student_id = data.get('student_id') or data.get('user_id')
        new_status = data.get('status') or data.get('new_status') # 'Present', 'Absent', 'OD'

        if not student_id or not new_status:
             logging.error(f"Missing fields in update_status: student_id={student_id}, status={new_status}, received={data}")
             return jsonify({'success': False, 'message': 'Missing required fields (student_id/user_id and status/new_status)'}), 400

        # Get Session Context to ensure we are updating the right record
        context = session.get('active_attendance_context')
        if not context:
            return jsonify({'success': False, 'message': 'No active session'}), 400
        
        subject = context.get('subject')
        today = datetime.now().strftime('%Y-%m-%d')
        period_no = context.get('period_no') # Important for uniqueness
        
        marked_by = session.get('user_id')

        cur = conn.cursor()
        
        # 1. Find existing record for this session
        query = """
            SELECT id, status, is_finalized, original_status 
            FROM attendance 
            WHERE user_id = %s AND subject = %s AND timestamp::date = %s
        """
        params = [student_id, subject, today]

        if period_no is not None:
             query += " AND period_no = %s"
             params.append(period_no)
        
        cur.execute(query, tuple(params))
        record = cur.fetchone()

        if record:
            # Update Existing Record
            rec_id, current_status, is_finalized, original_status = record
            
            if is_finalized:
                return jsonify({'success': False, 'message': 'Attendance is finalized and cannot be edited.'}), 403
            
            # If status unchanged, do nothing
            if current_status == new_status:
                return jsonify({'success': True, 'message': 'Status unchanged'})

            # Set original_status if not already set
            update_orig = ""
            update_params = []
            if original_status is None:
                update_orig = "original_status = %s, "
                update_params.append(current_status)
            
            # Update fields
            # Note: We put new_status first
            update_params.insert(0, new_status)
            
            # Append other params
            update_params.append(marked_by)
            update_params.append(datetime.now())
            update_params.append(rec_id)
            
            update_sql = f"""
                UPDATE attendance 
                SET status = %s, {update_orig} edited_by = %s, edit_timestamp = %s, is_finalized = TRUE
                WHERE id = %s
            """
            
            cur.execute(update_sql, tuple(update_params))
            conn.commit()
            
        else:
            # Create NEW record
            if new_status == 'Absent':
                return jsonify({'success': True, 'message': 'Student is already absent'})
            
                cur.execute("""
                    INSERT INTO attendance (user_id, subject, timestamp, status, marked_by, method, period_no, original_status, edited_by, edit_timestamp, is_finalized)
                    VALUES (%s, %s, %s, %s, %s, 'Manual_Edit', %s, 'Absent', %s, %s, TRUE)
                """, (student_id, subject, datetime.now(), new_status, marked_by, period_no, marked_by, datetime.now()))
            conn.commit()

        return jsonify({'success': True, 'message': 'Status updated'})

    except Exception as e:
        logging.error(f"Error updating status: {e}")
        conn.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/attendance/finalize_session', methods=['POST'])
@login_required(['teacher', 'admin'])
def finalize_session():
    """
    Locks the attendance for the current session by setting is_finalized = TRUE.
    """
    conn = database_manager.get_db_connection()
    if conn is None:
        return jsonify({'success': False, 'message': 'Database connection failed'}), 500

    try:
        data = request.get_json(silent=True) or {}
        context = session.get('active_attendance_context') or {}
        
        # Priority: JSON Body > Session Context
        subject = data.get('subject') or context.get('subject')
        date_val = data.get('date') or datetime.now().strftime('%Y-%m-%d')
        period_no = data.get('period_no') or context.get('period_no')
        
        if not subject:
             return jsonify({'success': False, 'message': 'Missing subject information'}), 400

        cur = conn.cursor()
        
        # We update strictly matching records.
        # Note: If a record doesn't exist (Absent), it remains non-existent, so implicitly final?
        # Yes, becuase future edits check 'is_finalized' ONLY if record exists.
        # BUT if we create a new record later, is it finalized?
        # Defaults to FALSE. So we might need to be careful.
        # Actually, if we try to Insert new record for a past finalized session, we should block it keying off session state?
        # OR we block strictly on Existing records.
        # Ideally, we should create records for ALL Absents too if we want to lock them?
        # NO, 'Absent' is implied.
        # Logic: When 'finalize_session' is called, we update all existing records.
        # If someone tries to add a NEW record (mark Present) later for this session, we must check if session is finalized.
        # But 'mark_attendance_for_user' doesn't check 'is_finalized'.
        # We should probably check there too? Or just rely on UI hiding buttons.
        # For now, let's just lock existing. User constraints say "Do NOT change existing attendance marking logic".
        # So we can't change 'mark_attendance_for_user'.
        # But 'update_attendance_status' DOES check it.
        
        query = """
            UPDATE attendance 
            SET is_finalized = TRUE 
            WHERE subject = %s AND timestamp::date = %s
        """
        params = [subject, date_val]
        
        if period_no is not None:
            query += " AND period_no = %s"
            params.append(period_no)
            
        cur.execute(query, tuple(params))
        conn.commit()
        
        # CLEAR session context and OD list after finalization
        session.pop('active_attendance_context', None)
        session.pop('od_students', None)
        session.modified = True
        
        return jsonify({'success': True, 'message': 'Session finalized successfully'})
        
    except Exception as e:
        logging.error(f"Error finalizing session: {e}")
        conn.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/resolve_od_students', methods=['POST'])
@login_required(['teacher', 'admin'])
def resolve_od_students_api():
    """
    Resolves partial roll numbers to full student records for OD marking.
    Returns the list of matched students without actually marking attendance.
    """
    try:
        data = request.get_json()
        
        od_input = data.get('od_students', '').strip()
        department = data.get('department', '')
        year = data.get('year', '')
        batch_year = data.get('batch_year', '')
        semester = data.get('semester', '')
        
        logging.info(f"API Resolve OD - Input: '{od_input}', Dept: {department}, Year: {year}, Batch: {batch_year}, Sem: {semester}")
        
        if not od_input:
            return jsonify({'success': False, 'message': 'No OD input provided'})
        
        if not all([department, year]):
            return jsonify({'success': False, 'message': 'Department and Year are required'})
        
        from backend.od_helper import resolve_od_roll_numbers
        
        partial_rolls = [r.strip() for r in od_input.split(',') if r.strip()]
        
        resolved_od = resolve_od_roll_numbers(
            partial_rolls,
            department,
            year,
            batch_year=batch_year,
            semester=semester
        )
        
        # Convert to list of dicts for JSON response
        od_students = [
            {
                'id': student[0],
                'roll_number': student[1],
                'full_name': student[2]
            }
            for student in resolved_od
        ]
        
        # Save to session if requested (For Live Attendance)
        if data.get('save_to_session'):
             od_ids = [s['id'] for s in od_students]
             session['od_students'] = od_ids
             session.modified = True
             logging.info(f"API Resolve OD - Saved {len(od_ids)} OD students to session")
        
        logging.info(f"API Resolve OD - Resolved {len(od_students)} students")
        
        return jsonify({
            'success': True,
            'od_students': od_students,
            'count': len(od_students)
        })
        
    except Exception as e:
        logging.error(f"Error resolving OD students via API: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/live/class_students')
@login_required(['teacher', 'admin'])
def get_live_class_students():
    try:
        context = session.get('active_attendance_context')
        if not context:
            # Fallback to current attendance class if context is missing
            context = session.get('attendance_class')
            
        if not context:
            return jsonify({'success': False, 'message': 'No active session'}), 400
            
        dept = context.get('department')
        year = context.get('year')
        batch = context.get('batch_year')
        # Handle semester being optional or derived
        sem = context.get('semester')
        subject = context.get('subject')
        
        # 1. Get All Students
        # Ensure we have enough info to fetch students
        if not dept or not year:
             return jsonify({'success': False, 'message': 'Missing class context'}), 400
             
        all_students = database_manager.get_students_by_class(dept, year, batch_year=batch, semester=sem)
        
        # 2. Get Present Students (Database)
        # We need to know who is ALREADY marked in DB
        today = datetime.now().strftime('%Y-%m-%d')
        # Using period_no is safer if available
        period_no = context.get('period_no')
        
        present_list = database_manager.get_present_students(
            subject=subject, hour=None, date_str=today, 
            department=dept, year=year, period_no=period_no
        )
        # distinct IDs
        present_ids = {str(s['id']) for s in present_list}
        
        # 3. Get OD Students (Session)
        od_ids = set(map(str, session.get('od_students', [])))
        
        # 4. Merge & Prepare List
        student_data = []
        for s in all_students:
            uid = str(s['id'])
            status = 'red' # Default: Unmarked
            if uid in present_ids:
                status = 'green'
            elif uid in od_ids:
                status = 'green' # Feature Request says: "Already marked students must remain GREEN" (OD counts as marked/handled?) 
                                 # Actually, let's keep OD distinct color or Green? 
                                 # Request says "Change their name color from RED -> GREEN".
                                 # Let's use Green for simplest compliance, or maybe a distinct shade if needed.
                                 # Let's stick to GREEN as per request for "Marked".
            
            student_data.append({
                'id': uid,
                'roll_number': s['roll_number'],
                'full_name': s['full_name'],
                'status': status
            })
            
        # Sort by last 3 digits of roll number (numeric sort)
        def sort_key(s):
            try:
                # Extract last 3 digits
                return int(str(s['roll_number']).strip()[-3:])
            except:
                return 0
                
        student_data.sort(key=sort_key)
        
        return jsonify({'success': True, 'students': student_data})
        
    except Exception as e:
        logging.error(f"Error fetching live class students: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/attendance_summary')
@login_required(['teacher', 'admin'])
def attendance_summary():
    subject = request.args.get('subject')
    hour = request.args.get('hour')
    date_str = request.args.get('date') 
    department = request.args.get('department') 
    year = request.args.get('year')
    semester = request.args.get('semester')
    batch_year = request.args.get('batch_year')

    if 'attendance_class' in session:
         info = session['attendance_class']
         if not subject: subject = info.get('subject')
         if not department: department = info.get('department')
         if not year: year = info.get('year')
         if not hour: hour = info.get('hour')
         if not semester: semester = info.get('semester')
         if not batch_year: batch_year = info.get('batch_year')
    
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')
    if not hour:
        hour = datetime.now().strftime('%H')

    # Normalize department for DB matching (e.g., CYBERSECURITY -> Cybersecurity)
    if department:
        department = department.strip().title()


    try:
        total = database_manager.get_total_students(department, year)
        
        # Robust period resolution for summary filtering
        # Prioritize request param, then fallback to active session context
        summary_period_no = request.args.get('period')
        if not summary_period_no:
            summary_period_no = session.get('active_attendance_context', {}).get('period_no')
        
        # Convert period to integer if it's a string
        if summary_period_no:
            try:
                summary_period_no = int(summary_period_no)
            except (ValueError, TypeError):
                logging.warning(f"Could not convert period '{summary_period_no}' to int")
                summary_period_no = None
        
        present_list = database_manager.get_present_students(subject, hour, date_str, department, year, period_no=summary_period_no)
        absent_list = database_manager.get_absent_students(subject, hour, date_str, department, year, period_no=summary_period_no)
        od_list = database_manager.get_od_students(subject, hour, date_str, department, year, period_no=summary_period_no)


        return render_template('attendance_summary.html', 
             subject=subject, hour=hour, date=date_str, 
             total=total, present_count=len(present_list), absent_count=len(absent_list), od_count=len(od_list),
             present=present_list, absent=absent_list, od=od_list,
             department=department, year=year
        )
    except Exception as e:
        logging.error(f"Summary error: {e}")
        return f"Error generating summary: {e}", 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/teacher/student_attendance/<int:student_id>')
@login_required('teacher')
def teacher_student_attendance(student_id):
    student = database_manager.get_user_info(student_id)
    attendance_data = database_manager.get_student_attendance(student_id)
    return render_template('teacher_student_attendance.html', student=student, attendance=attendance_data)


@app.route('/get_user_image/<int:user_id>')
@login_required('admin')
def get_user_image(user_id):
    return redirect(url_for("get_student_image", student_id=user_id))

# Redundant serve_proof_by_path removed (consolidated above)



# --- Class Photo Attendance Routes ---

@app.route('/admin/class_photo_attendance', methods=['GET', 'POST'])
@login_required(['admin', 'teacher'])
def class_photo_attendance_page():
    # Fetch lists for dropdowns
    departments = database_manager.get_departments()
    years = database_manager.get_years()
    batches = database_manager.get_academic_years() # Fetch available batches

    
    # Determine Selected Filters Early
    selected_dept = request.form.get('department') or request.args.get('department') or ''
    selected_semester = request.form.get('semester') or request.args.get('semester') or ''
    selected_subject = request.form.get('subject') or request.args.get('subject') or ''
    selected_year = request.form.get('year') or request.args.get('year') or ''
    selected_batch_year = request.form.get('batch_year') or request.args.get('batch_year') or ''
    selected_period = request.form.get('period') or request.args.get('period') or ''
    selected_time_display = request.form.get('time_display') or request.args.get('time_display') or ''


    # CHECK FOR LOCKED CONTEXT (Teacher Mode)
    locked_context = session.get('active_attendance_context')
    is_locked = False
    

    if locked_context:
        if request.args.get('locked') == 'true':
            # ROBUST FIX: Force-fetch fresh details if timetable_id exists
            # This handles stale sessions where batch/sem might be missing or year is wrong.
            if 'timetable_id' in locked_context:
                fresh_entry = database_manager.get_timetable_entry(locked_context['timetable_id'])
                if fresh_entry:
                    # Update variables from fresh DB fetch
                    selected_dept = fresh_entry['department']
                    selected_year = fresh_entry['year'] # Will be int (1, 2, etc)
                    selected_subject = fresh_entry['subject']
                    selected_batch_year = fresh_entry.get('batch_year', '')
                    selected_semester = fresh_entry.get('semester', '')
                    selected_period = f"{fresh_entry['start_time']} - {fresh_entry['end_time']}"
                    
                    # Optional: Update session execution context to prevent future drifts
                    # locked_context['year'] = selected_year
                    # locked_context['batch_year'] = selected_batch_year
                    # locked_context['semester'] = selected_semester
                    # session.modified = True
                else:
                     # Fallback to session if fetch fails
                    selected_dept = locked_context['department']
                    selected_year = locked_context['year']
                    selected_subject = locked_context['subject']
                    selected_batch_year = locked_context.get('batch_year', '')
                    selected_semester = locked_context.get('semester', '')
                    selected_period = locked_context.get('period', '')
                    selected_time_display = locked_context.get('time_display', '')
            else:
                selected_dept = locked_context['department']
                selected_year = locked_context['year']
                selected_subject = locked_context['subject']
                selected_batch_year = locked_context.get('batch_year', '')
                selected_semester = locked_context.get('semester', '')
                selected_period = locked_context.get('period', '')
                
            is_locked = True  # Lock when coming from timetable
        elif request.method != 'POST':
            pass

    if request.method == 'POST':
        if is_locked:
             selected_dept = locked_context['department']
             selected_year = locked_context['year']
             selected_subject = locked_context['subject']
             selected_batch_year = locked_context.get('batch_year', '')
             selected_semester = locked_context.get('semester', '')
             selected_period = locked_context.get('period', '')
        else:
            # Already extracted at the top, just ensure year derivation if needed
            if not selected_year and selected_semester:
                try:
                    selected_year = (int(selected_semester) + 1) // 2
                except: pass

    # --- Unified Altered Period Detection (Manual Flow) ---
    is_altered = False
    assigned_staff_name = None
    assigned_teacher_id = None
    timetable_id = None

    if not is_locked and selected_dept and selected_subject and selected_period:
        is_altered, assigned_staff_name, assigned_teacher_id, timetable_id = _check_altered_period(
            selected_dept, selected_batch_year, selected_subject, selected_period
        )
        
        # Build Context
        context = {
            'timetable_id': timetable_id,
            'department': selected_dept,
            'year': selected_year,
            'batch_year': selected_batch_year,
            'semester': selected_semester,
            'subject': selected_subject,
            'period': selected_period,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'is_altered': is_altered,
            'assigned_teacher_id': assigned_teacher_id,
            'assigned_staff_name': assigned_staff_name
        }
        session['active_attendance_context'] = context
        session['attendance_class'] = context

        # REDIRECT if altered but no reason provided yet
        if is_altered and not context.get('altered_reason'):
            logging.info(f"Altered Period detected for {session.get('user_id')}. Redirecting to justification.")
            return render_template('select_attendance_mode.html', 
                                 locked_context=context,
                                 is_altered_period=True,
                                 assigned_staff_name=assigned_staff_name)
    # --- Final Logic Prep: Fetch Subjects based on Resolved Filters ---
    if session.get('role') == 'teacher':
        subjects = database_manager.get_subjects_for_user(session.get('user_id'))
    else:
        # Admin: Filter if Dep is selected (Sem optional now)
        if selected_dept:
            subjects = database_manager.get_subjects_by_dept_year_v2(selected_dept, selected_year, selected_semester)
            if not subjects:
                 subjects = database_manager.get_all_subjects()
        else:
            subjects = database_manager.get_all_subjects()

    # --- BLOCK ENTRY if mandatory fields are missing (unless it's a GET for Admin) ---
    if not (selected_dept and selected_subject):
        # Only Redirect if it's an attempt to access the page without context
        # Admin might just be navigating here, so we send them to selection
        logging.warning("Class Photo Attendance accessed without Dept/Subject. Redirecting to selection.")
        return redirect(url_for('select_class', mode='class_photo'))

    return render_template('class_photo_attendance.html',
                           departments=departments,
                           years=years, 
                           batches=batches,
                           subjects=subjects,
                           department=selected_dept,
                           year=selected_year,
                           subject=selected_subject,
                           batch_year=selected_batch_year,
                           semester=selected_semester,
                           period=selected_period,
                           time_display=selected_time_display,
                           is_locked=is_locked)


@app.route('/admin/class_photo_attendance/process', methods=['POST'])
@login_required(['admin', 'teacher'])
def process_class_photo_attendance_route():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})
            
        uploaded_files = request.files.getlist('file')
        if not uploaded_files or uploaded_files[0].filename == '':
             return jsonify({'success': False, 'message': 'No selected file'})

        subject = request.form.get('subject')
        department = request.form.get('department')
        year = request.form.get('year')
        batch_year = request.form.get('batch_year')
        semester = request.form.get('semester')
        
        logging.info(f"DEBUG: Processing Class Photo Batch ({len(uploaded_files)} files) - Dept: {department}, Year: {year}, Sem: {semester}, Subject: {subject}")
        if not all([subject, department, year]):
            return jsonify({'success': False, 'message': 'Missing fields'})

        # Handle OD Students
        # Note: class_photo_attendance form sends od_students as comma-separated string
        od_student_ids = []
        od_input = request.form.get('od_students', '').strip()
        
        logging.info(f"OD INPUT RECEIVED: '{od_input}'")
        
        if od_input:
            try:
                # Import helper locally to avoid circular imports
                from backend.od_helper import resolve_od_roll_numbers
                
                partial_rolls = [r.strip() for r in od_input.split(',') if r.strip()]
                logging.info(f"OD PARTIAL ROLLS: {partial_rolls}")
                logging.info(f"OD QUERY PARAMS - Dept: {department}, Year: {year}, Batch: {batch_year}, Sem: {semester}")
                
                resolved_od = resolve_od_roll_numbers(
                    partial_rolls, 
                    department, 
                    year,
                    batch_year=batch_year,
                    semester=semester
                )
                od_student_ids = [student[0] for student in resolved_od]
                logging.info(f"OD Students Resolved for Class Photo: {len(od_student_ids)} students - IDs: {od_student_ids}")
                if resolved_od:
                    for student in resolved_od:
                        logging.info(f"  - {student[1]} ({student[2]})")
            except Exception as e:
                logging.error(f"Error resolving OD students for Class Photo: {e}", exc_info=True)


        # Initialize Aggregation Containers
        all_present_students = {}  # Dict for deduplication: {id: student_dict}
        all_other_students = {}    # Dict for deduplication
        all_unknown_faces = []
        total_faces_detected = 0
        
        marked_by = session.get('user_id')
        
        # Prepare Directory
        timestamp = datetime.now()
        date_str = timestamp.strftime('%Y-%m-%d')
        hour = timestamp.strftime('%H')
        safe_subject = "".join([c for c in subject if c.isalnum() or c in (' ', '_', '-')]).strip()
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "class_photos", date_str, safe_subject, hour)
        os.makedirs(base_dir, exist_ok=True)

        # Loop through files
        for i, file in enumerate(uploaded_files):
            if file and file.filename:
                # Preserve original extension
                from werkzeug.utils import secure_filename
                original_filename = secure_filename(file.filename)
                file_ext = os.path.splitext(original_filename)[1].lower()
                
                # Validate image extension
                allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
                if file_ext not in allowed_extensions:
                    logging.warning(f"Skipping file with unsupported extension: {original_filename}")
                    continue
                
                # Save with original extension
                filename = f"class_{int(timestamp.timestamp())}_{i}{file_ext}"
                filepath = os.path.join(base_dir, filename)
                
                # Save file in binary mode to prevent corruption
                # Read file data into memory first, then write to disk
                file.seek(0)  # Reset file position to start
                file_data = file.read()  # Read all bytes into memory
                
                with open(filepath, 'wb') as f:
                    f.write(file_data)
                
                # Ensure file is fully written to disk
                time.sleep(0.1)  # Small delay to ensure file system sync
                
                # Verify file was saved successfully
                if not os.path.exists(filepath):
                    logging.error(f"File save failed: {filepath} does not exist after save")
                    continue
                
                file_size = os.path.getsize(filepath)
                if file_size == 0:
                    logging.error(f"File save failed: {filepath} is empty (0 bytes)")
                    continue
                
                logging.info(f"Saved class photo: {filepath} ({file_size} bytes)")
                
                # Process (Skip Absent Marking)
                # Resolve period_no from period string
                period_val = request.form.get('period')
                period_int = None
                if period_val:

                    if period_val.isdigit():
                        period_int = int(period_val)
                    else:
                        # Period might be a time string - Resolve it
                        try:
                            day_of_week = datetime.now().strftime('%A')
                            periods = database_manager.get_timetable_periods(department, batch_year, subject, day_of_week)
                            for p in periods:
                                time_range = f"{p['start_time']} - {p['end_time']}"
                                # Handle partial match or missing seconds
                                if period_val in [time_range, time_range.replace(':00', '')]:
                                    period_int = p.get('period_no')
                                    break
                        except Exception as e:
                            logging.warning(f"Could not resolve period_no from string '{period_val}': {e}")

                result = biometric_logic.process_class_photo_attendance(
                    filepath, subject, hour, department, year, marked_by,
                    batch_year=batch_year, semester=semester, 
                    mark_absent=False, # CRITICAL: Don't mark absent yet!
                    period_no=period_int,
                    method=get_attendance_method('class_photo')
                )
                
                # Aggregate
                if result.get('success'):
                    total_faces_detected += result.get('total_faces', 0)
                    
                    for s in result.get('present_students', []):
                        all_present_students[s['id']] = s
                        
                    for s in result.get('other_students', []):
                        all_other_students[s['id']] = s
                        
                    all_unknown_faces.extend(result.get('unknown_faces', []))



        # Final Absenteeism Calculation
        # 1. Fetch Full Class List
        class_students = database_manager.get_students_by_class(department, year, batch_year=batch_year, semester=semester)
        
        # 2. Mark OD Students FIRST (before marking absent)
        od_student_ids_set = set(od_student_ids)  # Convert to set for fast lookup
        final_od_list = []
        
        if od_student_ids:
            logging.info(f"Marking {len(od_student_ids)} students as OD before absent calculation")
            for s in class_students:
                if s['id'] in od_student_ids_set:
                    final_od_list.append(s)
                    # Mark as OD in database
                    try:
                        database_manager.mark_attendance_for_user(
                            s['id'], subject, marked_by, 
                            status='OD', 
                            method=get_attendance_method('class_photo'),
                            period_no=period_int
                        )
                    except Exception as e:
                        logging.error(f"Error marking OD student {s['id']}: {e}")
        
        # 3. Determine Absentees (Class List - Present List - OD List)
        present_ids = set(all_present_students.keys())
        final_absent_list = []
        
        for s in class_students:
            # Skip if student is present OR on OD
            if s['id'] not in present_ids and s['id'] not in od_student_ids_set:
                final_absent_list.append(s)
                # Mark Absent in DB
                try:
                    database_manager.mark_attendance_for_user(s['id'], subject, marked_by, status='Absent', method=get_attendance_method('class_photo'), period_no=period_int)
                except Exception as e:
                    logging.error(f"Error marking absent student {s['id']}: {e}")

        logging.info(f"DEBUG: Batch Process Complete. Present: {len(present_ids)}, Absent: {len(final_absent_list)}, OD: {len(od_student_ids)}")



        aggregated_results = {
            "success": True,
            "total_faces": total_faces_detected,
            "matched_count": len(all_present_students),
            "other_count": len(all_other_students),
            "unknown_count": len(all_unknown_faces),
            "od_count": len(od_student_ids),  # Add OD count
            "present_students": list(all_present_students.values()),
            "other_students": list(all_other_students.values()),
            "absent_students": final_absent_list,
            "od_students": final_od_list,  # Add OD students list
            "unknown_faces": all_unknown_faces
        }

        
        return jsonify({'success': True, 'result': aggregated_results})
        
    except Exception as e:
        logging.error(f"Class Photo Batch Error: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/class_photo/proof/<path:filename>')
@login_required(['admin', 'teacher'])
def serve_class_photo_proof(filename):
    """
    Serves proof crops for class photo attendance.
    Expects filename to be a relative path from ATTENDANCE_PROOFS_DIR/class_crops
    e.g., '20240118_120000/proof_123.jpg'
    """
    try:
        # Construct path relative to class_crops (where logic saves them)
        # biometric_logic saves to: ATTENDANCE_PROOFS_DIR/class_crops/TIMESTAMP/filename
        # Logic returns relative path: TIMESTAMP/filename
        
        # We need to know where ATTENDANCE_PROOFS_DIR is.
        # It is defined in biometric_logic.py. 
        # We can access it via biometric_logic.ATTENDANCE_PROOFS_DIR
        
        base_dir = os.path.join(biometric_logic.ATTENDANCE_PROOFS_DIR, "class_crops")
        
        # Security check
        safe_path = os.path.abspath(os.path.join(base_dir, filename))
        if not safe_path.startswith(os.path.abspath(base_dir)):
            return "Invalid path", 403
            
        if not os.path.exists(safe_path):
            return "File not found", 404
            
        return send_file(safe_path, mimetype='image/jpeg')
        
    except Exception as e:
        logging.error(f"Error serving class proof: {e}")
        return "Error serving image", 500


@app.route('/admin/attendance/update_individual', methods=['POST'])
@login_required(['admin', 'teacher'])
def update_attendance_individual_route():
    """
    Updates attendance for a single student. 
    Expects JSON: {user_id: 123, subject: '...', status: '...', period_no: 1}
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        subject = data.get('subject')
        status = data.get('status')
        period_no = data.get('period_no')
        
        marked_by = session.get('user_id')
        
        if not all([user_id, subject, status]):
            return jsonify({'success': False, 'message': 'Missing required fields'})

        # Mark attendance in database
        # We reuse the existing database_manager function
        success, msg = database_manager.mark_attendance_for_user(
            user_id, subject, marked_by,
            status=status,
            method=get_attendance_method('manual'), # Mark as manual edit
            period_no=period_no,
            is_finalized=True # Auto-finalize individual edits
        )
        
        return jsonify({'success': success, 'message': msg})

    except Exception as e:
        logging.error(f"Error updating individual attendance: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/admin/class_photo_attendance/full_report')
@login_required(['admin', 'teacher'])
def class_photo_full_report():
    """
    Displays full formatted attendance report for a specific class session.
    Triggered by 'View Full Report' button in class photo attendance page.
    """
    try:
        # Get session parameters from query string
        department = request.args.get('department', '')
        year = request.args.get('year', '')
        semester = request.args.get('semester', '')
        batch_year = request.args.get('batch_year', '')
        subject = request.args.get('subject', '')
        period = request.args.get('period', '')
        date_str = request.args.get('date', '')
        source = request.args.get('source', '')
        
        # Validate required parameters
        if not date_str:
            from datetime import datetime
            date_str = datetime.now().strftime('%Y-%m-%d')
            logging.info(f"Fallback: Using today's date ({date_str}) for full report")

        # Fallback: Calculate Year from Semester if missing
        if not year and semester:
            try:
                year = str((int(semester) + 1) // 2)
            except:
                pass

        if not all([department, year, subject, date_str]):
            return f"Missing required parameters for attendance report. Please ensure all class details are provided. (Missing: {[k for k in ['department', 'year', 'subject', 'date'] if not locals().get(k)]})", 400
        
        # Resolve period_no from period string
        period_no = None
        if period:
            # Check if period is already a number
            if period.isdigit():
                period_no = int(period)
            else:
                # Try Regex first (Robust)
                import re
                match = re.search(r'Period\s*(\d+)', str(period), re.IGNORECASE)
                if match:
                     period_no = int(match.group(1))
                else:
                    # Period might be a time string like "08:30:00 - 09:20:00"
                    # Fetch valid periods for this class to match
                    try:
                        from datetime import datetime
                        day_of_week = datetime.strptime(date_str, '%Y-%m-%d').strftime('%A')
                        periods = database_manager.get_timetable_periods(department, batch_year, subject, day_of_week)
                        
                        # Try to match the period string to a time range
                        for p in periods:
                            start_str = str(p.get('start_time', ''))
                            end_str = str(p.get('end_time', ''))
                            time_range = f"{start_str} - {end_str}"
                            # Also try without seconds and check if period string is contained
                            time_range_short = f"{start_str[:5]} - {end_str[:5]}"
                            
                            if period == time_range or period == time_range_short:
                                period_no = p.get('period_no')
                                break
                            # Lenient check: if the period string contains the time range or vice versa
                            if (time_range in period) or (time_range_short in period) or (period in time_range):
                                period_no = p.get('period_no')
                                break
    
                    except Exception as e:
                        logging.warning(f"Could not resolve period_no from period string '{period}': {e}")
        
        # Get detailed attendance data using unified logic
        logs = database_manager.get_official_attendance_report(
            department=department,
            year=int(year),
            subject=subject,
            start_date=date_str,
            end_date=date_str,
            batch_year=batch_year,
            semester=int(semester) if semester else None,
            period=period_no,
            view_mode='Raw Records'
        )
        
        # Structure the data for full_attendance_report.html
        present_count = sum(1 for log in logs if log.get('status') == 'Present' or log.get('final_status') == 'Present')
        od_count = sum(1 for log in logs if log.get('status') == 'OD' or log.get('final_status') == 'OD')
        total_count = len(logs)
        
        # Determine if session is already finalized
        is_finalized = any(log.get('is_finalized') is True for log in logs)

        report_data = {
            'students': logs,
            'summary': {
                'total': total_count,
                'present': present_count,
                'absent': total_count - present_count - od_count,
                'od': od_count,
                'attendance_percentage': round((present_count / total_count * 100), 2) if total_count > 0 else 0
            },
            'session_info': {
                'department': department,
                'year': year,
                'semester': semester,
                'batch_year': batch_year,
                'subject': subject,
                'date': date_str,
                'period': period,
                'is_finalized': is_finalized
            }
        }
        
        if not report_data:
            return "Failed to generate attendance report. Please check the class details and try again.", 500
        
        # Get current user info for "Marked By"
        current_user_name = session.get('full_name', 'Unknown')
        
        # Render full report template
        return render_template('full_attendance_report.html',
            students=report_data['students'],
            summary=report_data['summary'],
            session_info=report_data['session_info'],
            period=period,
            period_no=period_no,  # Pass resolved period_no to template
            marked_by=current_user_name,
            report_date=date_str,
            source=source
        )
        
    except Exception as e:
        logging.error(f"Error generating full attendance report: {e}")
        return f"Error generating report: {str(e)}", 500



    
    # Decoded automatically by Flask?
    # Actually, sending absolute path in URL param is risky/ugly. 
    # Better to serve by ID or relative path. 
    # But for now, let's assume the client sends the path provided by backend.
    
    # Decode double-encoded? No, Flask handles url decoding.
    # filename here might be the full path if client passed full path.
    
    if os.path.exists(filename):
        return send_file(filename)
    else:
        # Fallback using 'clean' logic if filename is just the basename
        # Search in recent proofs? detailed logic is complex.
        # User requirement: "GET /class_photo/proof/<path:filename>"
        return "File not found", 404





@app.route('/api/get_subjects')
@login_required(['admin', 'teacher'])
def api_get_subjects():
    department = request.args.get('department')
    year = request.args.get('year')
    semester = request.args.get('semester')
    
    logging.info(f"API Fetch Subjects: Dept={department}, Year={year}, Sem={semester}")

    if not year and semester:
        try:
            # Handle "Sem 4" or "4"
            sem_digit = str(semester).lower().replace('sem', '').strip()
            if sem_digit.isdigit():
                year = str(int(math.ceil(int(sem_digit) / 2.0)))
                logging.info(f"Derived Year {year} from Sem {semester}")
        except Exception as e:
            logging.error(f"Error deriving year: {e}")

    if not department:
        logging.warning("Missing Dept for subjects fetch")
        return jsonify([])
    
    # If year is missing, we still fetch subjects for the department (and optionally semester)
    subjects = database_manager.get_subjects_by_dept_year_v2(department, year, semester)
    
    # Filter for teachers
    if session.get('role') == 'teacher':
        teacher_subjects = database_manager.get_subjects_for_user(session.get('user_id'))
        subjects = [s for s in subjects if s in teacher_subjects]
        
    logging.info(f"Returning {len(subjects)} subjects")
    return jsonify(subjects)



@app.route('/admin/manual_attendance', methods=['GET', 'POST'])
@login_required(['admin', 'teacher'])
def manual_attendance():
    # 1. Determine Context (Locked vs Open)
    is_locked = request.args.get('locked') == 'true'
    context = {}
    
    if is_locked:
        # Try to get from active context first (most detailed)
        context = session.get('active_attendance_context')
        if not context:
            # Fallback to general class context
            context = session.get('attendance_class')
            
        if not context and request.method == 'GET':
             # Only flash error on GET to avoid breaking POSTs if session expired but form has data
            flash("No active class session found.", "error")
            return redirect(url_for('teacher_dashboard'))
    
    # 2. Extract Parameters
    # Priority: Context (if locked) > Request Args/Form
    if is_locked and context:
        department = context.get('department')
        year = context.get('year')
        subject = context.get('subject')
        batch_year = context.get('batch_year')
        semester = context.get('semester')
        period = context.get('period')
    else:
        department = request.values.get('department')
        year = request.values.get('year')
        subject = request.values.get('subject')
        batch_year = request.values.get('batch_year')
        semester = request.values.get('semester')
        period = request.values.get('period')

    # --- Unified Altered Period Detection (Manual Selection) ---
    is_altered = False
    assigned_staff_name = None
    assigned_teacher_id = None
    timetable_id = None

    if not is_locked and department and subject and period:
        is_altered, assigned_staff_name, assigned_teacher_id, timetable_id = _check_altered_period(
            department, batch_year, subject, period
        )
        
        # Build/Update Context
        context = {
            'timetable_id': timetable_id,
            'department': department,
            'year': year,
            'batch_year': batch_year,
            'semester': semester,
            'subject': subject,
            'period': period,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'is_altered': is_altered,
            'assigned_teacher_id': assigned_teacher_id,
            'assigned_staff_name': assigned_staff_name
        }
        session['active_attendance_context'] = context
        session['attendance_class'] = context

        # REDIRECT if altered but no reason provided yet
        if is_altered and not context.get('altered_reason'):
            logging.info(f"Manual Altered Period detected. Redirecting to justification.")
            return render_template('select_attendance_mode.html', 
                                 locked_context=context,
                                 is_altered_period=True,
                                 assigned_staff_name=assigned_staff_name)
             
    # 3. Fetch Students
    students = []
    if department and year:
        # Robust fetch: pass batch and semester if available
        students = database_manager.get_students_by_class(
            department, year, 
            batch_year=batch_year, 
            semester=semester
        )
        if not students:
             # Fallback to broader search if class-specific fails
             students = database_manager.get_students_by_dept_year(department, year)

    # Sort students by last 3 digits of roll number
    if students:
        try:
            # Sort by last 3 digits cast to int
            students.sort(key=lambda x: int(str(x.get('roll_number', '000'))[-3:]))
        except Exception as e:
            logging.error(f"Error sorting students: {e}")

    return render_template('manual_attendance.html', 
                           students=students, 
                           department=department, 
                           year=year, 
                           subject=subject,
                           period=period,
                           semester=semester,
                           batch_year=batch_year)

@app.route('/admin/manual_attendance/submit', methods=['POST'])
@login_required(['admin', 'teacher'])
def submit_manual_attendance():
    try:
        department = request.form.get('department')
        year = request.form.get('year')
        subject = request.form.get('subject')
        timestamp = datetime.now()
        marked_by = session.get('user_id')
        
        # Extract period ONCE outside the loop
        period_val = request.form.get('period')
        period_int = None
        if period_val:
            try:
                # Extract period number from formats like:
                # "3", "Period 3", "Period 3 (10:30 - 11:25)"
                import re
                match = re.search(r'(?:Period\s+)?(\d+)', str(period_val))
                if match:
                    period_int = int(match.group(1))
                    logging.info(f"Extracted period_no={period_int} from '{period_val}'")
                else:
                    logging.warning(f"No period number found in '{period_val}'")
            except Exception as e:
                logging.error(f"Could not parse period from '{period_val}': {e}")
                period_int = None
        else:
            logging.warning("No period value provided in form")
        
        # Mark attendance for each student
        for key, value in request.form.items():
            if key.startswith('status-'):
                student_id = int(key.split('-')[1])
                status = value 

                database_manager.mark_attendance_for_user(
                    student_id, 
                    subject.strip() if subject else subject, # Normalize subject
                    marked_by, 
                    status=status, 
                    method=get_attendance_method('manual'),
                    period_no=period_int,
                    is_finalized=True # Auto-finalize manual submissions
                )

        logging.info(f"Manual attendance submitted: {subject}, period={period_int}")
        
        # Redirect to full report (same format as class photo report)
        return redirect(url_for('class_photo_full_report', 
                                department=department, 
                                year=year, 
                                subject=subject,
                                semester=request.form.get('semester', ''),
                                batch_year=request.form.get('batch_year', ''),
                                period=period_int,
                                date=datetime.now().strftime('%Y-%m-%d')))
    except Exception as e:
        logging.error(f"Manual Submit Error: {e}")
        return f"Error: {e}", 500










# --- ALIAS ROUTES TO FIX BUILD ERRORS ---


@app.route('/admin/delete_user/alias/<int:user_id>', endpoint='delete_user', methods=['POST'])
@login_required('admin')
def delete_user_alias(user_id):
    """Alias for delete_user_route to match potential user assumptions"""
    return delete_user_route(user_id)








@app.route('/finalize_enrollment', methods=['POST'])
@login_required('admin')
def finalize_enrollment():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'message': 'Missing User ID'})
            
        # Rebuild index / Finalize
        result = biometric_logic.finalize_enrollment_logic(user_id)
        
        if result.get('success'):
             # Redirect to stored return URL if exists. 
             # If NOT exists, return None (or omit key) so frontend uses its fallback (which has current params).
             return_url = session.pop('enrollment_return_url', None)
             if return_url:
                 result['redirect_url'] = return_url
        
        return jsonify(result)
    except Exception as e:
        logging.error(f"Finalize Enrollment Error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/enroll/upload', methods=['POST'])
@login_required('admin')
def enroll_face_upload():
    try:
        user_id = request.form.get('user_id')
        if 'photo' not in request.files:
            return jsonify({'success': False, 'message': 'No file part'})
        file = request.files['photo']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'})
            
        if file and user_id:
            # Use new Augmentation Logic
            result = biometric_logic.enroll_face_from_upload(user_id, file)
            return jsonify(result)
            
        return jsonify({'success': False, 'message': 'Invalid data'})
    except Exception as e:
        logging.error(f"Enroll Upload Error: {e}")
        return jsonify({'success': False, 'message': str(e)})

# --- Fixed Routes for BuildErrors (Task A) ---

@app.route('/admin/enroll_user', methods=['GET', 'POST'])
@login_required('admin')
def enroll_user():
    if request.method == 'POST':
        try:
            full_name = request.form['full_name']
            
            # Use get for these ensuring safety
            roll_number = request.form.get('roll_number', '')  
            
            # Department: Try 'department' first, then lookup from 'dept_id'
            department = request.form.get('department')
            if not department:
                dept_id = request.form.get('dept_id')
                if dept_id:
                     # Lookup Name
                     conn_temp = database_manager.get_db_connection()
                     if conn_temp:
                         cur_temp = conn_temp.cursor()
                         cur_temp.execute("SELECT name FROM academic_departments WHERE id=%s", (dept_id,))
                         res = cur_temp.fetchone()
                         if res: department = res[0]
                         conn_temp.close()
            
            if not department:
                 return jsonify({'success': False, 'message': 'Department is required (missing department or valid dept_id)'})
            
            # Username/Password
            username = request.form.get('username')
            password = request.form.get('password')
            
            # New Fields
            register_number = request.form.get('register_number')
            semester = request.form.get('semester')
            batch_start = request.form.get('batch_start_year')
            
            # Derived Year (e.g. 2)
            year = request.form.get('year') 
            
            if not username or not password:
                 return jsonify({'success': False, 'message': 'Username/Register Number and Password are required.'})

            # Auto-calculate derived fields
            try:
                semester = int(semester) if semester else 1
                batch_start_year = int(batch_start) if batch_start else None
                # Batch End = Start + 4
                batch_end_year = batch_start_year + 4 if batch_start_year else None
                
                # Year derived from Semester? 
                if semester:
                    derived_year = math.ceil(semester / 2.0)
                    year = derived_year # Override request.form['year'] if derived
            except ValueError:
                return jsonify({'success': False, 'message': 'Invalid numeric values for academic fields.'})
            
            # Fallback for batch_year string (e.g. 2024-2028)
            batch_year_str = f"{batch_start_year}-{batch_end_year}" if batch_start_year else request.form.get('batch_year', '2024-2028')


            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            from backend.database_manager import get_db_connection
            from backend.od_helper import resolve_od_roll_numbers
            conn = database_manager.get_db_connection()
            user_id = None
            if conn:
                cur = conn.cursor()
                try:
                    # UPDATED SQL to include new columns
                    cur.execute(
                        """INSERT INTO users (
                            username, password, role, full_name, department, year, roll_number, 
                            register_number, semester, batch_start_year, batch_end_year, batch_year, created_by
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                        (
                            username, hashed_password, 'student', full_name, department, year, roll_number,
                            register_number, semester, batch_start_year, batch_end_year, batch_year_str, session.get('user_id')
                        )
                    )
                    user_id = cur.fetchone()[0]
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    logging.error(f"Enroll User DB Error: {e}")
                    return jsonify({'success': False, 'message': str(e)})
                finally:
                    conn.close()
            
            if user_id:
                biometric_logic.create_face_directory(user_id)
                return jsonify({'success': True, 'user_id': user_id, 'message': 'Student details saved successfully.'})
            else:
                return jsonify({'success': False, 'message': 'Failed to save student details.'})

        except Exception as e:
            logging.error(f"Enroll User Error: {e}")
            return jsonify({'success': False, 'message': f"Error: {str(e)}"})

    departments = database_manager.get_departments()
    years = database_manager.get_years()
    return render_template('enroll_user.html', departments=departments, years=years)

@app.route('/api/students/promote', methods=['POST'])
@login_required('admin')
def promote_students_route():
    try:
        data = request.get_json()
        student_ids = data.get('student_ids', [])
        target_semester = data.get('target_semester')
        action = data.get('action', 'promote') # Default to promote
        
        if not student_ids:
             return jsonify({'success': False, 'message': 'No students selected.'})
             
        success, message = database_manager.promote_students(student_ids, target_semester, action)
        
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        logging.error(f"Promote Students API Error: {e}")
        return jsonify({'success': False, 'message': f"Server Error: {str(e)}"})

@app.route('/admin/students/promote')
@login_required('admin')
def admin_promote_students():
    departments = database_manager.get_academic_departments() # Use academic departments for full objects
    years = database_manager.get_years() # Keep getting years 1-4 for reference if needed
    batches = database_manager.get_academic_years() # Fetch batches
    return render_template('promote_students.html', departments=departments, years=years, batches=batches)

@app.route('/api/students/search', methods=['GET'])
@login_required('admin')
def api_search_students():
    dept_id = request.args.get('dept_id')
    batch_year = request.args.get('batch_year')
    semester = request.args.get('semester')
    year_id = request.args.get('year_id') # Optional Year 1-4 filter
    
    # Validation: Need at least one filter? Or allow all?
    # Default to preventing all for performance, unless requested.
    
    students = database_manager.get_students_filtered(dept_id=dept_id, year_id=year_id, semester=semester, batch_year=batch_year)
    return jsonify({'success': True, 'students': students})

@app.route('/api/enroll/capture', methods=['POST'])
@login_required('admin')
def capture_face():
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        image_data = data.get('image_data')

        if not student_id or not image_data:
            return jsonify({'success': False, 'message': 'Missing data'})

        if ',' in image_data:
            header, encoded = image_data.split(',', 1)
        else:
            encoded = image_data

        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
             return jsonify({'success': False, 'message': 'Failed to decode image'})

        # Save to Student Folder
        save_dir = biometric_logic.get_student_directory_path(student_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        count = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')])
        filename = f"face_{count+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        cv2.imwrite(filepath, img)
        

        # Update Encodings
        biometric_logic.update_student_face_encodings(student_id)
        
        return jsonify({'success': True, 'message': 'Face captured successfully'})

    except Exception as e:
        logging.error(f"Face Capture Error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/admin/enroll_user_page')
@login_required('admin')
def enroll_user_page():
    return enroll_user()

@app.route('/admin/users/add', methods=['POST'])
@login_required('admin')
def add_user():
    try:
        data = request.form.to_dict()
        success, msg = database_manager.add_user(data)
        flash(msg, "success" if success else "error")
    except Exception as e:
        flash(str(e), "error")
    return redirect(url_for('admin_user_management'))





@app.route('/api/proof/<path:filepath>')
@login_required(['admin', 'teacher'])
def get_proof_image(filepath):
    if os.path.exists(filepath):
         return send_file(filepath)
    return "Not Found", 404

# --- System Reset Routes (Task G) ---
@app.route('/admin/reset_system', methods=['GET'])
@login_required('admin')
def admin_reset_system_page():
    return render_template('admin_reset.html')

@app.route('/admin/reset_system/execute', methods=['POST'])
@login_required('admin')
def admin_reset_system_execute():
    success, message = system_reset.perform_full_system_reset()
    if success:
        biometric_logic.initialize_system()
        return render_template('admin_dashboard.html', 
                               success_message="SYSTEM RESET COMPLETE. All data has been wiped.")
    else:
        return f"Reset Failed: {message}", 500

@app.route('/api/admin/students/bulk-face-upload', methods=['POST'])
@login_required('admin')
def bulk_face_upload_route():
    if 'file' not in request.files and 'zip_file' not in request.files:
         return jsonify({'success': False, 'message': 'No file uploaded'})
    
    file = request.files.get('file') or request.files.get('zip_file')
    if not file or not file.filename.lower().endswith('.zip'):
         return jsonify({'success': False, 'message': 'Invalid file. Please upload a ZIP file.'})
    
    # Log Upload Details
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    logging.info(f"Bulk Face Upload: Filename={file.filename}, Size={file_size} bytes")
         
    temp_dir = os.path.join(tempfile.gettempdir(), f"bulk_faces_{int(datetime.now().timestamp())}")
    os.makedirs(temp_dir, exist_ok=True)
    zip_path = os.path.join(temp_dir, 'upload.zip')
    file.save(zip_path)
    
    report = []
    stats = {'val_success': 0, 'val_failed': 0, 'images_processed': 0}
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
             namelist = zf.namelist()
             logging.info(f"Bulk Face: ZIP contains {len(namelist)} items.")
             # Security check for Zip Slip
             for member in namelist:
                 if member.startswith('/') or '..' in member:
                     raise Exception("Security Error: Invalid ZIP path (Zip Slip detected)")
             zf.extractall(temp_dir)
             
        # Collection phase
        # We process file by file
        processed_regnos = set()
        
        # Valid Extensions
        valid_exts = ['.jpg', '.jpeg', '.png', '.webp']
        
        for root, dirs, files in os.walk(temp_dir):
            for filename in files:
                # 1. Check Extension
                ext = os.path.splitext(filename)[1].lower()
                if ext not in valid_exts:
                    continue
                
                file_path = os.path.join(root, filename)
                
                # 2. Extract Register Number
                # Strategy A: Filename
                name_no_ext = os.path.splitext(filename)[0]
                reg_no = None
                
                if re.match(r'^\d{10,20}$', name_no_ext):
                    reg_no = name_no_ext
                else:
                    # Strategy B: Parent Folder
                    parent_folder = os.path.basename(root)
                    if re.match(r'^\d{10,20}$', parent_folder):
                        reg_no = parent_folder
                
                if not reg_no:
                    # Skip file if no regno associated
                    continue
                    
                # 3. Process the file
                # Lookup User
                user = database_manager.get_user_by_regno(reg_no)
                if not user:
                    # Deduplicate error reporting for same regno
                    if reg_no not in processed_regnos: 
                        report.append({'reg_no': reg_no, 'status': 'Failed', 'reason': 'Student not found in DB', 'count': 0})
                        stats['val_failed'] += 1
                        processed_regnos.add(reg_no)
                    continue
                
                user_id = user['id']
                
                try:
                    with open(file_path, 'rb') as f_img:
                         res = biometric_logic.process_uploaded_photo(user_id, f_img)
                         if res['success']:
                             # Check if we already have a success entry for this regno in report
                             existing_entry = next((item for item in report if item['reg_no'] == reg_no and item['status'] == 'Success'), None)
                             if existing_entry:
                                 existing_entry['count'] += 1
                             else:
                                 report.append({'reg_no': reg_no, 'status': 'Success', 'reason': 'Enrolled', 'count': 1})
                                 stats['val_success'] += 1 
                             
                             stats['images_processed'] += 1
                             processed_regnos.add(reg_no)
                         else:
                            logging.warning(f"Bulk: Image failed for {reg_no}/{filename}: {res.get('message')}")
                except Exception as e:
                     logging.error(f"Bulk processing error for {reg_no} file {filename}: {e}")

        shutil.rmtree(temp_dir)
        
        # Final validation
        if stats['images_processed'] == 0:
             return jsonify({
                 'success': False, 
                 'message': 'No valid images with Register Numbers (10-20 digits) were processed. Check filenames or folder names.',
                 'report': report
             })
             
        logging.info(f"Bulk Face processing complete. Stats: {stats}")
        return jsonify({'success': True, 'report': report, 'stats': stats})

    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        logging.error(f"Bulk Upload Error: {e}")
        return jsonify({'success': False, 'message': str(e)})






# ==========================================
# TIMETABLE MANAGEMENT ROUTES
# ==========================================

@app.route('/admin/timetable', methods=['GET'])
@login_required('admin')
def admin_timetable():
    try:
        # Fetch Filter Options
        batches = database_manager.get_academic_years()
        departments = database_manager.get_academic_departments()
        subjects = database_manager.get_all_subjects_detailed()
        teachers = database_manager.get_all_teachers()
        
        # Get Global Timings
        timings = database_manager.get_global_timings()
        all_classes_data = database_manager.get_academic_classes()
        
        # Pre-serialize to JSON and back to ensure all data is JSON-safe
        import json
        teachers = json.loads(json.dumps(teachers, default=str))
        all_classes_data = json.loads(json.dumps(all_classes_data, default=str))
        
        # Selected Filters
        selected_batch = request.args.get('batch_year', '')
        selected_dept = request.args.get('department', '')
        selected_sem = request.args.get('semester', '')
        selected_sec = request.args.get('section', '')
        
        timetable_master = None
        current_entries = []
        
        if selected_batch and selected_dept and selected_sem:
            # Try to fetch existing master
            timetable_master = database_manager.get_timetable_master(selected_batch, selected_dept, selected_sem, selected_sec)
            
            if timetable_master:
                current_entries = database_manager.get_timetable_entries(timetable_master['id'])
            
            # If not exists, we don't auto-create here. We let the Create button do it, or handle it via API.
            # Actually, to make UI smooth, if all filters selected, we can show "Create Timetable" button or empty grid if we auto-create.
            # Let's rely on an explicit "Load/Create" action in JS or automatic check.
            
        return render_template('admin_timetable.html',
                               batches=batches,
                               departments=departments,
                               subjects=subjects,
                               teachers=teachers,
                               timings=timings,
                               selected_batch=selected_batch,
                               selected_dept=selected_dept,
                               selected_sem=selected_sem,
                               selected_sec=selected_sec,
                               timetable_master=timetable_master,
                               current_entries=current_entries,
                               academic_classes=all_classes_data)
    except Exception as e:
        logging.error(f"Error in admin_timetable: {e}")
        return f"Error loading timetable page: {str(e)}", 500

@app.route('/admin/timetable/config', methods=['GET', 'POST'])
@login_required('admin')
def admin_timetable_config():
    try:
        if request.method == 'POST':
            # Save configs
            # Expecting form data: period_1_start, period_1_end, etc.
            periods_data = []
            for i in range(1, 8):
                start = request.form.get(f'period_{i}_start')
                end = request.form.get(f'period_{i}_end')
                if start and end:
                    periods_data.append({'period_no': i, 'start_time': start, 'end_time': end})
            
            if database_manager.save_global_timings(periods_data):
                flash('Timings saved successfully.', 'success')
            else:
                flash('Error saving timings.', 'error')
            return redirect(url_for('admin_timetable_config'))
            
        timings = database_manager.get_global_timings()
        return render_template('admin_timetable_config.html', timings=timings)
    except Exception as e:
        logging.error(f"Error in timetable config: {e}")
        return "Error", 500

@app.route('/api/timetable/create_master', methods=['POST'])
@login_required('admin')
def api_create_timetable_master():
    try:
        data = request.json
        batch = data.get('batch_year')
        dept = data.get('department')
        sem = data.get('semester')
        sec = data.get('section')
        name = data.get('timetable_name')
        
        if not all([batch, dept, sem, name]):
            return jsonify({'success': False, 'message': 'Missing fields'})
            
        master_id = database_manager.create_or_get_timetable_master(batch, dept, sem, sec, name)
        if master_id:
            return jsonify({'success': True, 'master_id': master_id})
        else:
            return jsonify({'success': False, 'message': 'Database error'})
    except Exception as e:
        logging.error(f"Error creating master: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/timetable/delete/<int:master_id>', methods=['POST'])
@login_required('admin')
def api_delete_timetable_master(master_id):
    success, msg = database_manager.delete_timetable_master(master_id)
    return jsonify({'success': success, 'message': msg})

@app.route('/api/timetable/save_entry', methods=['POST'])
@login_required('admin')
def api_save_timetable_entry():
    try:
        data = request.json
        master_id = data.get('master_id')
        day = data.get('day')
        period_no = data.get('period_no')
        subject_id = data.get('subject_id')
        staff_id = data.get('staff_id')
        
        if not all([master_id, day, period_no]):
             return jsonify({'success': False, 'message': 'Missing key fields'})
             
        success, message = database_manager.save_timetable_entry(master_id, day, period_no, subject_id, staff_id)
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': message})
    except Exception as e:
        logging.error(f"Error saving entry: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/timetable/teachers_by_subject/<int:subject_id>')
@login_required('admin')
def api_get_teachers_by_subject(subject_id):
    try:
        department = request.args.get('department')
        year = request.args.get('year')
        
        # Determine if year is valid int
        if year and year.isdigit():
            year = int(year)
        else:
            year = None
            
        teachers = database_manager.get_teachers_by_subject(subject_id, department, year)
        return jsonify({'success': True, 'teachers': teachers})
    except Exception as e:
        logging.error(f"Error fetching teachers by subject: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/admin/semester-info', methods=['GET', 'POST'])
@login_required('admin')
def admin_semester_info():
    try:
        if request.method == 'POST':
            batch = request.form.get('batch_year')
            semester = request.form.get('semester')
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            saturday_working = request.form.get('saturday_working') == 'on'
            
            success, res = database_manager.upsert_semester_info(batch, semester, start_date, end_date, saturday_working)
            if success:
                flash("Semester information updated successfully.", "success")
            else:
                flash(f"Error: {res}", "error")
            return redirect(url_for('admin_semester_info'))

        semester_infos = database_manager.get_all_semester_infos()
        batches = database_manager.get_academic_years()
        
        # If a specific semester_id is in args, fetch its holidays
        selected_sem_id = request.args.get('id')
        holidays = []
        selected_sem = None
        if selected_sem_id:
            for s in semester_infos:
                if str(s['id']) == selected_sem_id:
                    selected_sem = s
                    break
            holidays = database_manager.get_semester_holidays(selected_sem_id)

        return render_template('admin_semester_info.html', 
                               semester_infos=semester_infos, 
                               batches=batches,
                               selected_sem=selected_sem,
                               holidays=holidays)
    except Exception as e:
        logging.error(f"Error in admin_semester_info: {e}")
        return "Internal Server Error", 500

@app.route('/api/semester/holiday/add', methods=['POST'])
@login_required('admin')
def api_add_holiday():
    try:
        data = request.json
        sem_info_id = data.get('semester_info_id')
        h_date = data.get('holiday_date')
        h_name = data.get('holiday_name')
        h_type = data.get('holiday_type', 'Public Holiday')
        
        if database_manager.add_semester_holiday(sem_info_id, h_date, h_name, h_type):
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Failed to add holiday'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/semester/holiday/delete/<int:holiday_id>', methods=['POST'])
@login_required('admin')
def api_delete_holiday(holiday_id):
    try:
        if database_manager.delete_semester_holiday(holiday_id):
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Failed to delete holiday'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


    except Exception as e:
        logging.error(f"Error loading teacher timetable: {e}")
        return "Error", 500




# --- Settings Module Routes ---

@app.route('/admin/settings')
@login_required('admin')
def admin_settings():
    return render_settings_page()

@app.route('/teacher/settings')
@login_required('teacher')
def teacher_settings():
    return render_settings_page()

@app.route('/student/settings')
@login_required('student')
def student_settings():
    return render_settings_page()

def render_settings_page():
    if session.get('is_emergency_mode'):
        return redirect(url_for('select_attendance_mode'))
    user_id = session['user_id']
    user = database_manager.get_user_info(user_id)
    return render_template('settings.html', user=user)

@app.route('/api/settings/change_password', methods=['POST'])
@login_required()
def change_password():
    user_id = session['user_id']
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    # 1. Check if New == Old
    if current_password == new_password:
        flash('New password cannot be the same as the current password.', 'error')
        return redirect_to_settings()

    # 2. Strict Complexity Rules
    if len(new_password) < 8:
        flash('Password must be at least 8 characters long.', 'error')
        return redirect_to_settings()
        
    if not re.search(r"[A-Z]", new_password):
        flash('Password must contain at least one uppercase letter.', 'error')
        return redirect_to_settings()
        
    if not re.search(r"\d", new_password):
        flash('Password must contain at least one number.', 'error')
        return redirect_to_settings()
        
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", new_password):
        flash('Password must contain at least one special character.', 'error')
        return redirect_to_settings()
    
    if new_password != confirm_password:
        flash('New passwords do not match.', 'error')
        return redirect_to_settings()
        
    # 3. Verify Current Password
    username = session.get('username')
    auth_user = database_manager.authenticate_user(username, current_password)
    
    if not auth_user:
        flash('Current password is incorrect.', 'error')
        return redirect_to_settings()

    # 4. Update
    success = database_manager.change_user_password(user_id, new_password)
    if success:
        flash('Password updated successfully.', 'success')
    else:
        flash('Failed to update password. Try again.', 'error')

    return redirect_to_settings()

@app.route('/api/settings/update_preferences', methods=['POST'])
@login_required()
def update_preferences():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
            
        user_id = session['user_id']
        current_prefs_json = database_manager.get_user_info(user_id).get('preferences') or '{}'
        
        # Merge
        import json
        try:
            prefs = json.loads(current_prefs_json)
        except:
            prefs = {}
            
        prefs.update(data)
        
        # Save
        success = database_manager.update_user_preferences(user_id, json.dumps(prefs))
        return jsonify({'success': success})
    except Exception as e:
        logging.error(f"Error updating preferences: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/settings/update_profile_photo', methods=['POST'])
@login_required()
def update_profile_photo():
    if 'photo' not in request.files:
        flash('No file part', 'error')
        return redirect_to_settings()
    
    file = request.files['photo']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect_to_settings()
    
    if file:
        # Secure filename and save
        from werkzeug.utils import secure_filename
        from datetime import datetime
        import os
        filename = secure_filename(f"profile_{session['user_id']}_{int(datetime.now().timestamp())}.jpg")
        
        # Save path: static/uploads/profiles
        # PROJECT_BACKEND_DIR is in main.py? No, typically main.py is in backend/
        # app.root_path points to backend/ .
        # We want frontend/static/uploads/profiles
        base_dir = os.path.dirname(app.root_path) # Project root (d:\Project\attendance app)
        upload_folder = os.path.join(base_dir, 'frontend', 'static', 'uploads', 'profiles')
        os.makedirs(upload_folder, exist_ok=True)
        
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        
        # Web path
        web_path = f"/static/uploads/profiles/{filename}"
        
        database_manager.update_profile_photo(session['user_id'], web_path)
        flash('Profile photo updated.', 'success')
        
    return redirect_to_settings()

def redirect_to_settings():
    role = session.get('role')
    if role == 'admin': return redirect(url_for('admin_settings'))
    elif role == 'teacher': return redirect(url_for('teacher_settings'))
    elif role == 'student': return redirect(url_for('student_settings'))
    return redirect(url_for('login_page'))

    return redirect_to_settings()

# --- Emergency Access Routes ---

@app.route('/admin/generate-emergency-code', methods=['POST'])
@login_required(role='admin')
def generate_emergency_code_route():
    try:
        user_id = session.get('user_id')
        code_data = database_manager.create_emergency_code(user_id)
        if code_data:
            return jsonify({
                'success': True, 
                'code': code_data['code'],
                'expires_at': code_data['expires_at'].isoformat()
            })
        return jsonify({'success': False, 'message': 'Failed to generate code'}), 500
    except Exception as e:
        logging.error(f"Error generating code: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/emergency-access')
def emergency_access():
    if session.get('is_emergency_mode'):
        # If already verified, allow them to see the form
        pass
    
    # Fetch Data for Dropdowns
    departments = database_manager.get_academic_departments()
    batches = database_manager.get_academic_years()
    subjects = database_manager.get_all_subjects_detailed() # Returns {id, name, code, type}
    
    return render_template('emergency_access.html', 
                           departments=departments, 
                           batches=batches, 
                           subjects=subjects)

@app.route('/emergency-access/verify', methods=['POST'])
def verify_emergency_access():
    data = request.get_json()
    code = data.get('code')
    
    result = database_manager.verify_emergency_code(code)
    
    if result['valid']:
        return jsonify({'valid': True})
    else:
        return jsonify({'valid': False, 'message': result['message']})

@app.route('/emergency-access/start', methods=['POST'])
def emergency_start_session():
    code = request.form.get('access_code')
    department = request.form.get('department')
    batch_year = request.form.get('batch_year')
    semester = request.form.get('semester')
    period = request.form.get('period')
    subject = request.form.get('subject') # Using subject name for compatibility
    reason = request.form.get('reason')
    staff_name = request.form.get('staff_name', 'Unknown')
    
    # Re-verify just in case
    result = database_manager.verify_emergency_code(code)
    if not result['valid']:
         flash(f"Error: {result['message']}", 'error')
         return redirect(url_for('emergency_access'))
    
    # Set Session for Emergency Mode
    session['is_emergency_mode'] = True
    session['emergency_code_id'] = result['data']['id']
    session['emergency_code'] = code
    session['role'] = 'teacher' # Temporary role for compatibility with routes
    
    # Store context for attendance marking
    session['context_department'] = department
    session['context_batch_year'] = batch_year
    session['context_semester'] = semester
    session['context_period'] = period
    session['context_subject'] = subject
    session['context_reason'] = reason
    session['staff_name'] = staff_name
    session['full_name'] = f"Emergency: {staff_name} ({reason})"
    
    # NEW: Bridging to Active Attendance Context (Locked UI)
    # Derive academic year (1-4) from batch for compatibility
    batches_data = database_manager.get_academic_years()
    selected_batch = next((b for b in batches_data if b['name'] == batch_year), None)
    year_no = selected_batch['current_year_display'] if selected_batch else 1
    
    session['active_attendance_context'] = {
        'department': department,
        'year': year_no, 
        'batch_year': batch_year,
        'semester': semester,
        'subject': subject,
        'period': period,
        'reason': reason,
        'staff_name': staff_name,
        'is_emergency': True,
        'date': datetime.now().strftime('%Y-%m-%d')
    }
    
    # 2. attendance_class (Legacy support for logic checking this key)
    session['attendance_class'] = {
        'department': department,
        'year': year_no,
        'subject': subject,
        'period_no': period,
        'semester': semester,
        'batch_year': batch_year
    }
    
    # Mark code as "In Use"
    flag_code_usage = database_manager.mark_emergency_code_used(
        result['data']['id'], 
        {
            'department': department, 
            'batch_year': batch_year,
            'semester': semester,
            'period': period, 
            'subject': subject, 
            'reason': reason,
            'staff_name': staff_name
        }
    )
    
    if flag_code_usage:
        return redirect(url_for('select_attendance_mode'))
    else:
        flash("System Error: Could not start emergency session.", 'error')
        return redirect(url_for('emergency_access'))

# ==========================================
# GRIEVANCE SYSTEM ROUTES
# ==========================================

@app.route('/api/student/grievance/submit', methods=['POST'])
@login_required('student')
def student_submit_grievance():
    student_id = session['user_id']
    attendance_id = request.form.get('attendance_id')
    reason = request.form.get('reason')
    
    if not attendance_id or not reason:
        return jsonify({'success': False, 'message': 'Missing required fields'})
        
    # Optional Proof
    proof_path = None
    if 'proof' in request.files:
        file = request.files['proof']
        if file.filename != '':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"grievance_{student_id}_{attendance_id}_{timestamp}_{file.filename}"
            save_dir = os.path.join(ATTENDANCE_PROOFS_DIR, 'grievances')
            os.makedirs(save_dir, exist_ok=True)
            proof_path = os.path.join(save_dir, filename)
            file.save(proof_path)
            # Store relative path for DB/UI
            proof_path = f"/grievances/{filename}"
            
    success, msg = database_manager.create_grievance(student_id, attendance_id, reason, proof_path)
    return jsonify({'success': success, 'message': msg})

@app.route('/admin/grievances')
@login_required('admin')
def admin_grievances():
    grievances = database_manager.get_all_pending_grievances()
    return render_template('admin_grievances.html', grievances=grievances)

@app.route('/api/admin/grievance/resolve', methods=['POST'])
@login_required('admin')
def admin_resolve_grievance():
    data = request.json
    grievance_id = data.get('grievance_id')
    action = data.get('action') # Approve, Reject, Approve_OD
    comment = data.get('comment')
    admin_id = session['user_id']
    
    if not grievance_id or not action:
         return jsonify({'success': False, 'message': 'Missing fields'})
         
    success, msg = database_manager.resolve_grievance(grievance_id, action, admin_id, comment)
    return jsonify({'success': success, 'message': msg})

@app.route('/api/grievance/proof/<path:filename>')
@login_required(['admin', 'student'])
def serve_grievance_proof(filename):
    directory = os.path.join(ATTENDANCE_PROOFS_DIR, 'grievances')
    return send_from_directory(directory, filename)


if __name__ == '__main__':
    # Ensure tables exist
    database_manager.init_db()
    # Check background init handled by thread above
    
    # Run
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
