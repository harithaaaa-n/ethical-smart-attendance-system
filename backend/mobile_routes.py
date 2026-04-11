from flask import Blueprint, request, jsonify, session
import logging
from datetime import datetime
from . import database_manager

mobile_bp = Blueprint('mobile', __name__)

# All mobile API routes are registered here
# They are accessible at /api/mobile/* (prefix set in main.py)

@mobile_bp.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok', 'message': 'Mobile API is running'})

def get_auth_user_id():
    """Helper to extract user_id from the mobile dummy token."""
    auth_header = request.headers.get('Authorization')
    if auth_header and 'mobile_auth_token_' in auth_header:
        try:
            # Bearer mobile_auth_token_7 -> 7
            return int(auth_header.split('_')[-1])
        except:
            return None
    return None

@mobile_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request'}), 400
        
    username = data.get('username') or data.get('student_id')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
        
    user = database_manager.authenticate_user(username, password.encode('utf-8'))
    
    if user:
        # Update Last Login
        database_manager.update_last_login(user['id'])
        
        # Prepare user data for app
        user_data = {
            'id': user['id'],
            'username': user['username'],
            'full_name': user['full_name'],
            'role': user['role'],
            'department': user.get('department', 'General'),
            'year': user.get('year'),
            'roll_number': user.get('roll_number'),
            'profile_photo': user.get('profile_image_path'), # Map database column to app field
            'institution': {
                'name': 'Adhiyamaan College of Engineering',
                'logo_url': '/static/images/college_logo.png', # Placeholder
                'dept_name': f"Department of {user.get('department', 'Cyber Security')}",
                'dept_logo_url': '/static/images/dept_logo.png' # Placeholder
            }
        }
        return jsonify({
            'token': f"mobile_auth_token_{user['id']}",
            'user': user_data,
            'institution': user_data.get('institution')
        })
        
    return jsonify({'error': 'Invalid credentials'}), 401

@mobile_bp.route('/profile', methods=['GET'])
def profile():
    # In a real app, this would use JWT. For now, we take from query param if available or error
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
        
    user = database_manager.get_user_info(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
        
    # Return same structure as login
    user_data = {
        'id': user['id'],
        'username': user['username'],
        'full_name': user['full_name'],
        'role': user['role'],
        'department': user.get('department'),
        'year': user.get('year'),
        'roll_number': user.get('roll_number'),
        'profile_photo': user.get('profile_image_path'),
        'institution': {
            'name': 'Adhiyamaan College of Engineering',
            'logo_url': '/static/images/college_logo.png',
            'dept_name': f"Department of {user.get('department', 'Cyber Security')}",
            'dept_logo_url': '/static/images/dept_logo.png'
        }
    }
    return jsonify(user_data)

@mobile_bp.route('/timetable', methods=['GET'])
def timetable():
    user_id = request.args.get('user_id') or get_auth_user_id()
    if not user_id:
        return jsonify({'error': 'User ID required'}), 400
        
    entries = database_manager.get_student_timetable_grid(user_id)
    return jsonify(entries)

@mobile_bp.route('/attendance-history', methods=['GET'])
def attendance_history():
    user_id = request.args.get('user_id') or get_auth_user_id()
    if not user_id:
        return jsonify({'error': 'User ID required'}), 400
        
    # Default to last 30 days
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    history = database_manager.get_student_attendance_by_date_range(
        user_id, 
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')
    )
    
    # Format dates for JSON
    for record in history:
        if record['timestamp']:
            record['timestamp'] = record['timestamp'].isoformat()
            
    return jsonify(history)

@mobile_bp.route('/challenge', methods=['GET'])
def challenge():
    # Placeholder for liveness detection challenge
    import random
    challenges = ["Blink your eyes", "Turn your head left", "Turn your head right", "Smile"]
    return jsonify({
        'id': random.randint(1000, 9999), 
        'challenge': random.choice(challenges)
    })

@mobile_bp.route('/check-attendance', methods=['GET'])
def check_attendance():
    user_id = request.args.get('user_id') or get_auth_user_id()
    period = request.args.get('period')
    
    if not user_id or not period:
        return jsonify({'error': 'User ID and period required'}), 400
        
    date_str = datetime.now().strftime('%Y-%m-%d')
    conn = database_manager.get_db_connection()
    if conn is None: return jsonify({'error': 'DB Error'}), 500
    try:
        c = conn.cursor()
        c.execute("""
            SELECT status, subject 
            FROM attendance 
            WHERE user_id = %s AND timestamp::date = %s AND period_no = %s
        """, (user_id, date_str, period))
        res = c.fetchone()
        if res:
            return jsonify({'already_marked': True, 'subject': res[1], 'status': res[0]})
        return jsonify({'already_marked': False})
    finally:
        conn.close()

@mobile_bp.route('/mark-attendance', methods=['POST'])
def mark_attendance():
    import numpy as np
    import cv2
    from . import biometric_logic
    
    user_id = request.form.get('student_id') or get_auth_user_id()
    period = request.form.get('period')
    frames = request.files.getlist('frames[]')
    
    if not user_id or not period or not frames:
        return jsonify({'success': False, 'message': 'Missing data'}), 400
        
    date_str = datetime.now().strftime('%Y-%m-%d')
    day_name = datetime.now().strftime('%A')
    
    # Get user subject
    conn = database_manager.get_db_connection()
    if conn is None: return jsonify({'success': False, 'message': 'DB Error'}), 500
    subject_name = "Classroom Attendance"
    try:
        import psycopg2.extras
        c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        c.execute("SELECT department, batch_year, semester FROM users WHERE id = %s", (user_id,))
        uinfo = c.fetchone()
        if uinfo:
            c.execute("""
                SELECT id FROM timetable_master 
                WHERE LOWER(department) = LOWER(%s) AND batch_year = %s AND semester = %s
            """, (uinfo[0], uinfo[1], uinfo[2]))
            tm = c.fetchone()
            if tm:
                c.execute("""
                    SELECT s.name 
                    FROM timetable_entries te
                    JOIN academic_subjects s ON te.subject_id = s.id
                    WHERE te.timetable_id = %s AND LOWER(te.day_of_week) = LOWER(%s) AND te.period_no = %s
                """, (tm['id'], day_name, period))
                te = c.fetchone()
                if te:
                    subject_name = te['name']
    finally:
        conn.close()

    try:
        frame_file = frames[0]
        file_bytes = frame_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'message': 'Invalid image format'})
            
        result = biometric_logic.sys.process_client_frame(img, subject_name, user_id)
        
        if not result.get('success'):
             return jsonify({'success': False, 'message': result.get('message', 'Face detection failed')})
             
        if not result.get('is_live'):
             return jsonify({'success': False, 'message': result.get('liveness_msg', 'Please show a live face')})
             
        if str(result.get('user_id')) != str(user_id):
             return jsonify({'success': False, 'message': 'Face verification failed! Does not match profile.'})
             
        # Save attendance
        save_status, msg = database_manager.mark_attendance_for_user(
            user_id=user_id,
            subject=subject_name,
            marked_by=user_id,
            status="Present",
            confidence=result.get('score'),
            method="mobile_ble",
            period_no=period,
            is_finalized=True
        )
        
        return jsonify({
            'success': True if save_status in ['marked', 'updated'] else False,
            'message': msg
        })
    except Exception as e:
        logging.error(f"Mobile Attendance Error: {e}", exc_info=True)
        return jsonify({'success': False, 'message': 'An unexpected error occurred'})
