"""
OD (On Duty) Helper Functions
Provides utilities for resolving partial roll numbers for OD marking.
"""
import psycopg2
import psycopg2.extras
import logging
from .database_manager import get_db_connection

def resolve_od_roll_numbers(partial_rolls, department, year, batch_year=None, semester=None):
    """
    Resolves partial roll numbers (last 2-3 digits) to full student records.
    Used for OD (On Duty) marking feature.
    
    Args:
        partial_rolls: List of strings like ['25', '032', '48']
        department: Department name
        year: Academic year (1-4)
        batch_year: Batch year (e.g., '2024-2028')
        semester: Semester number (optional)
    
    Returns:
        List of tuples: [(user_id, full_roll_number, full_name), ...]
    """
    if not partial_rolls:
        return []
    
    conn = get_db_connection()
    if conn is None:
        return []
    
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Build query to get students for the class
        # Note: year might be stored as INTEGER or STRING (e.g., "Year 2")
        # Note: department is case-insensitive
        query = """
            SELECT id, roll_number, full_name
            FROM users
            WHERE role = 'student'
              AND active_status = 'Active'
              AND LOWER(department) = LOWER(%s)
              AND (year = %s OR year::text = %s)
        """
        year_string = str(year)
        params = [department, year, year_string]
        
        logging.info(f"OD Helper - Searching for students: dept={department}, year={year}, batch={batch_year}, sem={semester}")
        
        if batch_year:
            query += " AND LOWER(batch_year) = LOWER(%s)"
            params.append(batch_year)
        
        if semester:
            # Re-enable semester filter for precision
            try:
                sem_int = int(str(semester).replace('Sem ', '').strip())
                query += " AND semester = %s"
                params.append(sem_int)
            except:
                query += " AND semester::text = %s"
                params.append(str(semester))
        
        cursor.execute(query, params)
        students = cursor.fetchall()
        
        logging.info(f"OD Helper - Found {len(students)} students in class")
        
        resolved = []
        for partial in partial_rolls:
            partial = partial.strip()
            if not partial:
                continue
            
            logging.info(f"OD Helper - Looking for match with partial: '{partial}'")
            matched = False
            
            # Match students whose roll number ends with this partial
            for student in students:
                roll = student['roll_number'] or ''
                if roll.endswith(partial):
                    resolved.append((
                        student['id'],
                        student['roll_number'],
                        student['full_name']
                    ))
                    logging.info(f"OD Helper - MATCHED: '{partial}' -> {student['roll_number']} ({student['full_name']})")
                    matched = True
                    break  # Only match first occurrence
            
            if not matched:
                logging.warning(f"OD Helper - NO MATCH for partial '{partial}'")
        
        logging.info(f"OD Helper - Total resolved: {len(resolved)} students")
        return resolved

        
    except Exception as e:
        logging.error(f"Error resolving OD roll numbers: {e}")
        return []
    finally:
        if conn:
            conn.close()
