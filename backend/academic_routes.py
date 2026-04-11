from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from . import database_manager
from functools import wraps
import logging

academic_bp = Blueprint('academic', __name__, 
                        url_prefix='/admin/user-management/academic',
                        template_folder='../frontend/templates')

# --- RBAC Helper (Reused or Importable) ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# ---------------------------------------------------
# 1. DEPARTMENTS
# ---------------------------------------------------
@academic_bp.route('/departments', methods=['GET', 'POST'])
@admin_required
def departments():
    if request.method == 'POST':
        dept_id = request.form.get('dept_id')
        name = request.form.get('name')
        hod_name = request.form.get('hod_name')
        
        if dept_id:
             # Edit
             success, msg = database_manager.update_academic_department(dept_id, name, hod_name)
             action = "updated"
        else:
             # Add (Code no longer supported)
             success, msg = database_manager.add_academic_department(name, code=None, hod_name=hod_name)
             action = "added"
             
        if not success:
            flash(f"Error: {msg}", "error")
        else:
            flash(f"Department {action} successfully", "success")
        return redirect(url_for('academic.departments'))
        
    departments = database_manager.get_academic_departments()
    return render_template('academic/departments.html', departments=departments, active_tab='departments')

@academic_bp.route('/departments/delete/<int:id>', methods=['POST'])
@admin_required
def delete_department(id):
    success, msg = database_manager.delete_academic_department(id)
    if not success:
        flash(f"Error: {msg}", "error")
    else:
        flash("Department deleted", "success")
    return redirect(url_for('academic.departments'))

# ---------------------------------------------------
# 2. YEARS / BATCHES
# ---------------------------------------------------
@academic_bp.route('/years', methods=['GET', 'POST'])
@admin_required
def years():
    if request.method == 'POST':
        year_id = request.form.get('year_id')
        
        if year_id:
            # Edit
            name = request.form.get('name')
            batch = request.form.get('batch') # Legacy edit
            success, msg = database_manager.update_academic_year(year_id, name, batch)
            action = "updated"
        else:
            # Add New Batch
            start_year = request.form.get('start_year')
            num_sections = int(request.form.get('num_sections', 0))
            
            success, msg = database_manager.add_academic_year(start_year, num_sections)
            action = "added"
            
        if not success:
            flash(f"Error: {msg}", "error")
        else:
            flash(f"Batch {action} successfully", "success")
        return redirect(url_for('academic.years'))
        
    years = database_manager.get_academic_years()
    return render_template('academic/years.html', years=years, active_tab='years')

@academic_bp.route('/years/delete/<int:id>', methods=['POST'])
@admin_required
def delete_year(id):
    success, msg = database_manager.delete_academic_year(id)
    if not success:
        flash(f"Error: {msg}", "error")
    else:
        flash("Year deleted", "success")
    return redirect(url_for('academic.years'))

# ---------------------------------------------------
# 3. SECTIONS
# ---------------------------------------------------
@academic_bp.route('/sections', methods=['GET', 'POST'])
@admin_required
def sections():
    if request.method == 'POST':
        section_id = request.form.get('section_id')
        name = request.form.get('name')
        
        if section_id:
            success, msg = database_manager.update_academic_section(section_id, name)
            action = "updated"
        else:
            success, msg = database_manager.add_academic_section(name)
            action = "added"
            
        if not success:
            flash(f"Error: {msg}", "error")
        else:
            flash(f"Section {action} successfully", "success")
        return redirect(url_for('academic.sections'))
        
    sections = database_manager.get_academic_sections()
    return render_template('academic/sections.html', sections=sections, active_tab='sections')

@academic_bp.route('/sections/delete/<int:id>', methods=['POST'])
@admin_required
def delete_section(id):
    success, msg = database_manager.delete_academic_section(id)
    if not success:
        flash(f"Error: {msg}", "error")
    else:
        flash("Section deleted", "success")
    return redirect(url_for('academic.sections'))

# ---------------------------------------------------
# 4. SUBJECTS
# ---------------------------------------------------
@academic_bp.route('/subjects', methods=['GET', 'POST'])
@admin_required
def subjects():
    if request.method == 'POST':
        subject_id = request.form.get('subject_id')
        name = request.form.get('name')
        code = request.form.get('code')
        credits = request.form.get('credits')
        credits = None if credits == '' else credits
        
        semester = request.form.get('semester')
        semester = None if semester == '' else semester
        
        regulation = request.form.get('regulation')
        subject_type = request.form.get('subject_type', 'Theory')
        batch_id = request.form.get('batch_id')
        
        # Inline Mapping Data
        map_dept_id = request.form.get('map_dept_id')
        map_section_id = request.form.get('map_section_id')
        map_scope = request.form.get('map_scope', 'GLOBAL')

        if subject_id:
            success, msg = database_manager.update_academic_subject(subject_id, name, code, credits, semester, regulation, subject_type, batch_id)
            action = "updated"
            target_sub_id = subject_id
        else:
            success, result = database_manager.add_academic_subject(name, code, credits, semester, regulation, subject_type, batch_id)
            action = "added"
            target_sub_id = result if success else None
            msg = result if not success else "Subject added successfully"

        if success and map_dept_id and map_dept_id.strip():
            # Handle Inline Mapping (Shared for Add and Edit)
            try:
                classes = database_manager.get_academic_classes()
                mapped_count = 0
                target_classes = []
                
                for cls in classes:
                    # Filter by Dept and Batch
                    if str(cls['dept_id']) == str(map_dept_id) and str(cls['year_id']) == str(batch_id):
                        # Filter by Section/Scope
                        if map_scope == 'GLOBAL':
                            target_classes.append(cls)
                        elif map_scope == 'SECTION' and map_section_id:
                            if str(cls['section_id']) == str(map_section_id):
                                target_classes.append(cls)
                
                for cls in target_classes:
                    m_success, m_msg = database_manager.add_class_subject_mapping(cls['id'], target_sub_id, semester)
                    if m_success:
                        mapped_count += 1
                        
                if mapped_count > 0:
                    msg = f"Subject {action} and mapped to {mapped_count} classes."
                else:
                    msg = f"Subject {action}, but no matching classes found for mapping."
            except Exception as e:
                logging.error(f"Inline Mapping Error: {e}")
                msg = f"Subject {action}, but mapping failed: {str(e)}"

        if not success:
            flash(f"Error: {msg}", "error")
        else:
            flash(msg if 'msg' in locals() else f"Subject {action} successfully", "success")
        return redirect(url_for('academic.subjects'))
        
    subjects = database_manager.get_academic_subjects()
    mappings = database_manager.get_class_subjects_mapping()
    classes = database_manager.get_academic_classes()
    departments = database_manager.get_academic_departments()
    batches = database_manager.get_academic_years() 
    sections = database_manager.get_academic_sections()
    
    return render_template('academic/subjects_mapping.html', 
                          subjects=subjects, 
                          mappings=mappings,
                          classes=classes,
                          departments=departments,
                          batches=batches, 
                          sections=sections,
                          active_tab='subjects')

@academic_bp.route('/subjects/delete/<int:id>', methods=['POST'])
@admin_required
def delete_subject(id):
    success, msg = database_manager.delete_academic_subject(id)
    if not success:
        flash(f"Error: {msg}", "error")
    else:
        flash("Subject deleted", "success")
    return redirect(url_for('academic.subjects'))

# ---------------------------------------------------
# 5. CLASSES
# ---------------------------------------------------
@academic_bp.route('/classes', methods=['GET', 'POST'])
@admin_required
def classes():
    if request.method == 'POST':
        class_id = request.form.get('class_id')
        dept_id = request.form.get('dept_id')
        year_id = request.form.get('year_id')
        section_id = request.form.get('section_id')
        
        # Handle empty string for optional section
        if section_id == "":
            section_id = None
            
        if class_id:
            success, msg = database_manager.update_academic_class(class_id, dept_id, year_id, section_id)
            action = "updated"
        else:
            success, msg = database_manager.add_academic_class(dept_id, year_id, section_id)
            action = "created"
        
        if not success:
            flash(f"Error: {msg}", "error")
        else:
            flash(f"Class {action} successfully", "success")
        return redirect(url_for('academic.classes'))
        
    classes = database_manager.get_academic_classes()
    departments = database_manager.get_academic_departments()
    years = database_manager.get_academic_years()
    sections = database_manager.get_academic_sections()
    
    return render_template('academic/classes.html', 
                          classes=classes, 
                          departments=departments, 
                          years=years, 
                          sections=sections,
                          active_tab='classes')

@academic_bp.route('/classes/delete/<int:id>', methods=['POST'])
@admin_required
def delete_class(id):
    success, msg = database_manager.delete_academic_class(id)
    if not success:
        flash(f"Error: {msg}", "error")
    else:
        flash("Class deleted", "success")
    return redirect(url_for('academic.classes'))

# ---------------------------------------------------
# 6. MAPPING
# ---------------------------------------------------
@academic_bp.route('/mapping', methods=['GET', 'POST'])
@admin_required
def mapping():
    if request.method == 'POST':
        class_id = request.form.get('class_id')
        
        if not class_id or not class_id.strip():
             flash("Please select a valid class (Department, Batch, and Section)", "error")
             return redirect(url_for('academic.mapping'))
             
        subject_ids = request.form.getlist('subject_ids') # Bulk checkboxes
        
        semester = request.form.get('semester')
        
        if subject_ids:
             success, msg = database_manager.bulk_add_class_subject_mapping(class_id, semester, subject_ids)
        else:
             # Fallback or single
             subject_id = request.form.get('subject_id')
             if subject_id:
                success, msg = database_manager.add_class_subject_mapping(class_id, subject_id, semester)
             else:
                success, msg = False, "No subjects selected"

        if not success:
            flash(f"Error: {msg}", "error")
        else:
            flash(f"{msg}", "success")
        return redirect(url_for('academic.mapping'))
        
    subjects = database_manager.get_academic_subjects()
    mappings = database_manager.get_class_subjects_mapping()
    classes = database_manager.get_academic_classes()
    departments = database_manager.get_academic_departments()
    batches = database_manager.get_academic_years()
    sections = database_manager.get_academic_sections()
    
    return render_template('academic/subjects_mapping.html', 
                          subjects=subjects,
                          mappings=mappings,
                          classes=classes,
                          departments=departments,
                          batches=batches,
                          sections=sections,
                          active_tab='mapping')

@academic_bp.route('/mapping/delete/<int:id>', methods=['POST'])
@admin_required
def delete_mapping(id):
    success, msg = database_manager.delete_mapping(id)
    if not success:
        flash(f"Error: {msg}", "error")
    else:
        flash("Mapping removed", "success")
    return redirect(url_for('academic.mapping'))

# # ---------------------------------------------------
# # 7. PROMOTION & REMOVAL
# # ---------------------------------------------------
# @academic_bp.route('/promote', methods=['GET', 'POST'])
# @admin_required
# def promote_class():
#     students = []
#     selected_class_id = None
#     departments = database_manager.get_academic_departments()
#     years = database_manager.get_academic_years()
#     classes = database_manager.get_academic_classes()
#     
#     # Handle GET with class_id (e.g. after redirect or link)
#     if request.method == 'GET' and request.args.get('class_id'):
#         try:
#             selected_class_id = int(request.args.get('class_id'))
#             cls = next((c for c in classes if c['id'] == selected_class_id), None)
#             if cls:
#                  students = database_manager.get_students_filtered(cls['dept_id'], cls['year_id'], cls['section_id'])
#         except ValueError:
#             pass
# 
#     if request.method == 'POST':
#         action = request.form.get('action')
#         class_id = request.form.get('class_id')
#         
#         try:
#             selected_class_id = int(class_id) if class_id else None
#         except:
#             selected_class_id = None
# 
#         if action == 'fetch':
#              # Handle filtering
#              dept_id = request.form.get('dept_id')
#              year_id = request.form.get('year_id')
#              section_id = request.form.get('section_id')
#              semester = request.form.get('semester')
#              
#              # Flexible Filter Logic: Require Dept AND (Batch OR Semester OR Year)
#              # This allows users to search by Dept+Semester without knowing the exact Batch.
#              academic_year = request.form.get('academic_year')
#              
#              if dept_id and (year_id or semester or academic_year):
#                   logging.info(f"Promote Filter: Dept={dept_id}, Batch={year_id}, Sem={semester}, Year={academic_year}")
#                   
#                   # Pass semester and academic_year directly to DB
#                   students = database_manager.get_students_filtered(
#                       dept_id=dept_id, 
#                       year_id=year_id, 
#                       section_id=section_id, 
#                       semester=semester,
#                       academic_year=academic_year
#                   )
#              elif selected_class_id:
#                   # Fallback to single class selection
#                   cls = next((c for c in classes if c['id'] == selected_class_id), None)
#                   if cls:
#                        students = database_manager.get_students_filtered(cls['dept_id'], cls['year_id'], cls['section_id'])
#                   
#         elif action == 'promote':
#              student_ids = request.form.getlist('student_ids')
#              if student_ids:
#                  success, msg = database_manager.promote_students(student_ids)
#                  if success: 
#                      flash(msg, "success")
#                  else: 
#                      flash(msg, "error")
#              else:
#                  flash("No students selected", "error")
#              # Reload page
#              return redirect(url_for('academic.promote_class'))
#              
#         elif action == 'remove':
#              student_ids = request.form.getlist('student_ids')
#              status = request.form.get('status')
#              reason = request.form.get('reason')
#              
#              if student_ids:
#                  success, msg = database_manager.remove_students_bulk(student_ids, status, reason)
#                  if success: 
#                      flash(msg, "success")
#                  else: 
#                      flash(msg, "error")
#              else:
#                  flash("No students selected", "error")
#              return redirect(url_for('academic.promote_class'))
# 
#     return render_template('academic/promote_class.html', classes=classes, students=students, selected_class_id=selected_class_id, departments=departments, years=years, active_tab='promotion')

# ---------------------------------------------------
# 7. API Routes
# ---------------------------------------------------

@academic_bp.route('/api/batch/<int:batch_id>/details')
@admin_required
def get_batch_details_route(batch_id):
    details = database_manager.get_batch_details(batch_id)
    if not details:
        return jsonify({'error': 'Batch not found'}), 404
    return jsonify(details)