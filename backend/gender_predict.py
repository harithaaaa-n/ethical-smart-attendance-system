import logging

def predict_gender(face_obj):
    """
    Utility function to map InsightFace gender predictions.
    1 = Male, 0 = Female in standard buffalo_l model.
    """
    try:
        # Check if the object has a gender attribute (provided by InsightFace's genderage module)
        gender_val = getattr(face_obj, 'gender', -1)
        
        if gender_val == 1:
            return "Male"
        elif gender_val == 0:
            return "Female"
        else:
            return "Unknown"
            
    except Exception as e:
        logging.warning(f"Error in predict_gender helper: {e}")
        return "Unknown"
