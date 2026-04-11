import cv2
import numpy as np
import warnings
import gc
import logging
import os
import time
from datetime import datetime
from PIL import Image, ImageOps
import io
# from .gender_predict import predict_gender # Removed in favor of integrated FaceAnalysis properties
from .liveness_detection import AntiSpoofingValidator
# Suppress benign FutureWarnings from insightface/numpy
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning) # Also UserWarnings just in case
import base64
from flask import session
import time
from datetime import datetime
import logging
import io
import json
import os
import sys
import re
import insightface
from insightface.app import FaceAnalysis
from sklearn.preprocessing import normalize
from . import database_manager
from . import face_config
from PIL import Image, ImageOps

# Configure logging
logging.basicConfig(level=logging.INFO)

# Directories
PROJECT_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_FACES_DIR = os.path.join(PROJECT_BACKEND_DIR, "student_faces")
ATTENDANCE_PROOFS_DIR = os.path.join(PROJECT_BACKEND_DIR, "attendance_proofs")

# dedicated debug log
DEBUG_LOG_PATH = os.path.join(PROJECT_BACKEND_DIR, "biometric_debug.log")
def log_debug(msg):
    with open(DEBUG_LOG_PATH, "a") as f:
        f.write(f"[{datetime.now()}] {msg}\n")

os.makedirs(STUDENT_FACES_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_PROOFS_DIR, exist_ok=True)


class FaceRecognitionSystem:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceRecognitionSystem, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self, force_recompute=False):
        if self.initialized and not force_recompute:
            return

        log_debug("Starting FaceRecognitionSystem initialization...")
        
        # 1. Initialize InsightFace (Single Instance Optimization)
        # Optimized for: 640x640 detection (Fast & Accurate balance)
        # Strategy: Try GPU first, fallback to CPU quietly on failure.
        
        provider_options = [
            (['CUDAExecutionProvider', 'CPUExecutionProvider'], "GPU Mode"),
            (['CPUExecutionProvider'], "CPU Fallback Mode")
        ]
        
        init_success = False
        last_error = ""

        for providers, mode_name in provider_options:
            try:
                log_debug(f"Attempting initialization in {mode_name}...")
                self.app = FaceAnalysis(
                    name=face_config.MODEL_NAME,
                    root=face_config.MODELS_DIR,
                    allowed_modules=['detection', 'recognition', 'genderage'],
                    providers=providers
                )
                
                # Use ctx_id=0 for GPU, -1 for CPU
                ctx_id = 0 if 'CUDAExecutionProvider' in providers else -1
                
                # GOLDEN RESOLUTION: 640x640
                # Large enough for group photos, small enough for real-time single face.
                self.app.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.25)
                
                log_debug(f"FaceAnalysis initialized in {mode_name}.")
                init_success = True
                break # Success!
            except Exception as e:
                last_error = str(e)
                log_debug(f"Failed to init {mode_name}: {last_error}")
                continue

        if not init_success:
            log_debug(f"CRITICAL: All initialization attempts failed. Last error: {last_error}")
            logging.error(f"CRITICAL: FaceRecognitionSystem failed to initialize: {last_error}")
            self.initialized = False
            return # Don't raise, let endpoints handle it

        try:
            # 2. Sync Shared Objects
            self.app_high_res = self.app # Legacy mapping to keep code compatible

            # 3. Initialize in-memory index structures
            self.known_embeddings = []
            self.known_ids = []

            # 4. Mandatory Standardization Migration
            migration_file = os.path.join(PROJECT_BACKEND_DIR, ".face_standardization_done")
            if not os.path.exists(migration_file) or force_recompute:
                logging.info("MIGRATION: Starting face embedding standardization...")
                self.recompute_all_embeddings()
                with open(migration_file, "w") as f:
                    f.write(str(datetime.now()))
                logging.info("MIGRATION: Standardization complete.")

            # 5. Load FAISS / Embeddings Index
            self._rebuild_index()
            
            # 6. Initialize Recognition States
            self.recognition_cooldowns = {}
            self.recognition_queue = []
            self.MAX_QUEUE_SIZE = 50

            # 7. Initialize Anti-Spoofing
            self.liveness_validator = AntiSpoofingValidator()
            
            self._liveness_history = [] 
            self._MAX_LIVENESS_HISTORY = 10
            self._spoof_counter = [0]

            self.initialized = True
            log_debug("FaceRecognitionSystem fully initialized.")
            logging.info("FaceRecognitionSystem initialized successfully.")

        except Exception as e:
            log_debug(f"CRITICAL ERROR during final setup: {str(e)}")
            logging.error(f"CRITICAL: FaceRecognitionSystem setup failed: {e}")
            self.initialized = False

    def process_enrollment_image(self, user_id, img, source='enrollment_sample', image_path=None):
        """
        Unified handler for enrollment images (captured or uploaded).
        STRICT RULES:
        1. Exactly ONE face per image.
        2. Consistent normalization and model usage.
        3. Validation of embedding dimension.
        Returns: (success, message, embedding_saved)
        """
        try:
            # 1. Detect Faces
            faces = self.app.get(img)
            
            # STRICT: Single Face Check
            if len(faces) == 0:
                # Try enhancement as fallback
                enhanced = self.enhance_image_clahe(img)
                faces = self.app.get(enhanced)
                if len(faces) == 0:
                    return False, "No face detected in the image.", False
                img = enhanced # Use enhanced if it worked

            if len(faces) > 1:
                logging.warning(f"Multiple faces ({len(faces)}) detected. Selecting the largest face.")
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)

            face = faces[0]
            
            # 2. Quality Check (Optional but recommended)
            passed, reason = self.check_face_quality(face, img.shape)
            if not passed:
                logging.warning(f"Face Quality Alert for User {user_id}: {reason}")
                # We might still allow it if it's the only face, but let's be strict if needed.
                # For now, let's just log and proceed unless it's too bad.

            # 3. Standardized Preprocessing & Embedding
            processed_face = self.preprocess_face(img, face, target_size=(224, 224))
            v_img_uint8 = (processed_face * 255).astype(np.uint8)
            v_faces = self.app.get(v_img_uint8)
            
            if not v_faces:
                return False, "Failed to extract embedding from processed face.", False
                
            # Take largest (should be only) face from processed image
            v_faces = sorted(v_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            raw_emb = v_faces[0].embedding.reshape(1, -1)
            
            # 4. Normalization (L2) - Ensure Unit Vector
            norm = np.linalg.norm(raw_emb)
            if norm == 0:
                return False, "Zero-length embedding detected.", False
            emb = (raw_emb / norm).flatten()
            
            # 5. Consistency Validation
            if len(emb) != face_config.EMBEDDING_DIM:
                 logging.error(f"DIMENSION_MISMATCH: Got {len(emb)}, Expected {face_config.EMBEDDING_DIM}")
                 return False, f"Invalid embedding dimension: {len(emb)}", False

            # 6. Save to DB
            database_manager.save_embedding(
                user_id, emb, source=source, image_path=image_path
            )
            
            return True, "Face processed and saved successfully.", True

        except Exception as e:
            logging.error(f"Error in process_enrollment_image: {e}")
            return False, str(e), False

    def recompute_all_embeddings(self):
        """
        MANDATORY STANDARDIZATION (STRICT IDENTITY MAPPING):
        Iterates through ALL registered students.
        If a face folder exists -> Process & Enroll.
        Ensures strict rules (1 face, normalized).
        """
        all_students = database_manager.get_all_students()
        total_students = len(all_students)
        logging.info(f"STRICT TRAINING: Checking {total_students} students for face data...")
        
        enrolled_count = 0
        
        for user in all_students:
            user_id = user['id']
            database_manager.delete_face_embeddings(user_id)
            
            folder = self.get_student_folder(user_id)
            if not os.path.exists(folder):
                continue
                
            valid_images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            samples_count = 0
            
            for filename in valid_images:
                file_path = os.path.join(folder, filename)
                try:
                    with open(file_path, 'rb') as f:
                        img = self._correct_exif_rotation(f.read())
                    if img is None: continue
                    
                    success, msg, saved = self.process_enrollment_image(user_id, img, source='enrollment_sample', image_path=file_path)
                    if saved:
                        samples_count += 1
                    else:
                        logging.warning(f"Skipping {filename} for User {user_id}: {msg}")
                except Exception as e:
                    logging.error(f"Error processing {filename} for user {user_id}: {e}")
            
            if samples_count > 0:
                self.finalize_enrollment(user_id)
                enrolled_count += 1
        
        self._rebuild_index()
        logging.info(f"STRICT TRAINING COMPLETE: Enrolled {enrolled_count} students.")



    def update_student_face_encodings(self, user_id):
        """
        Refreshes the in-memory embeddings for a specific student.
        Called after enrollment to ensure immediate availability.
        """
        try:
            # Full Rebuild for consistency (fast enough for 100s users)
            self._rebuild_index() 
            return True
        except Exception as e:
            logging.error(f"Error updating encodings for {user_id}: {e}")
            return False

    def match_face(self, query_embedding, thresh_strict=None, thresh_relaxed=None, margin=None, allowed_user_ids=None):
        """
        Unified Matching Logic (Strict Requirements):
        1. Compare against all known embeddings.
        2. Find max similarity per student.
        3. RULE: Accept IF best_score >= THRESHOLD AND (best_score - second_best) >= MARGIN.
        4. Centralized debug logging for validation.
        """
        t_strict = thresh_strict or face_config.MATCH_THRESHOLD_STRICT
        m = margin or face_config.MATCH_MARGIN
        
        if not self.known_embeddings:
            logging.warning("MATCH_FAILURE: No embeddings in memory.")
            return None, 0.0, False

        # --- Scope Filtering ---
        search_embeddings = self.known_embeddings
        search_ids = self.known_ids
        
        if allowed_user_ids is not None:
             allowed_set = set(str(uid) for uid in allowed_user_ids)
             indices = [i for i, uid in enumerate(self.known_ids) if str(uid) in allowed_set]
             if not indices: return None, 0.0, False
             search_embeddings = [self.known_embeddings[i] for i in indices]
             search_ids = [self.known_ids[i] for i in indices]

        # 1. Calculate Similarities
        sims = np.dot(search_embeddings, query_embedding)
        
        # 2. Maximum Similarity per Student (Handles multiple embeddings)
        user_scores = {}
        for idx, score in enumerate(sims):
            uid = search_ids[idx]
            if uid not in user_scores or score > user_scores[uid]:
                user_scores[uid] = score
        
        if not user_scores: return None, 0.0, False

        # 3. Sort results to find Best and Second Best
        ranked = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
        top_uid, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        
        diff = top_score - second_score

        # --- STRICT RULE DECISION (As requested by user) ---
        # Rule: Accept IF best_score >= THRESHOLD AND (best_score - second_best) >= MARGIN
        
        # 1. Threshold Check
        if top_score < t_strict:
            log_debug(f"DECISION: REJECTED (THRESHOLD) - Top Candidate {top_uid} Score: {top_score:.4f} < {t_strict}")
            return None, float(top_score), False
            
        # 2. Margin Check
        if diff < m:
            log_debug(f"DECISION: REJECTED (AMBIGUOUS) - Margin {diff:.4f} < Required {m} (Best: {top_score:.4f}, 2nd: {second_score:.4f})")
            return None, float(top_score), False

        # SUCCESS
        log_debug(f"DECISION: ACCEPTED - Matched {top_uid} with score {top_score:.4f} (Margin {diff:.4f})")
        return top_uid, float(top_score), False

    def save_student_face_samples(self, user_id, images):
        """
        Saves provided images as samples for the student.
        """
        save_dir = self.get_student_folder(user_id)
        os.makedirs(save_dir, exist_ok=True)
        count = 0
        for idx, img in enumerate(images):
            path = os.path.join(save_dir, f"sample_{idx}_{int(time.time())}.jpg")
            cv2.imwrite(path, img)
            # detect and save embedding logic should be here or separate
            count += 1
        return count

    def get_student_face_status(self, user_id):
        """Returns 'Enrolled' or 'Not Enrolled'."""
        info = database_manager.get_user_info(user_id)
        if info:
             # Check if status is explicitly set
             if info.get('face_status') == 'Enrolled':
                 return 'Enrolled'
             # Double check DB embeddings presence
             if database_manager.has_embeddings(user_id):
                 return 'Enrolled'
        return 'Not Enrolled'


    def get_student_folder(self, user_id):
        """
        Determines the correct folder path for a student.
        Format: Name_RollNumber
        Backward Compatibility: Renames old ID-based folder if found.
        """
        try:
            # 1. Fetch User Info
            user_info = database_manager.get_user_info(user_id)
            if not user_info:
                # Fallback
                return os.path.join(STUDENT_FACES_DIR, str(user_id))

            register_number = user_info.get('register_number')

            # 1. Preferred: Strict Register Number Folder
            if register_number:
                reg_path = os.path.join(STUDENT_FACES_DIR, str(register_number))
                if os.path.exists(reg_path):
                    return reg_path

                # 2. Backward Compatibility: Search all folders for this register number
                if os.path.exists(STUDENT_FACES_DIR):
                    for d in os.listdir(STUDENT_FACES_DIR):
                        if str(register_number) in d:
                            legacy_path = os.path.join(STUDENT_FACES_DIR, d)
                            if os.path.isdir(legacy_path):
                                logging.info(f"Biometric: Found legacy folder {d} for RegNo {register_number}")
                                return legacy_path
                
                # 3. Target Path for NEW enrollments
                return reg_path

            # Fallback (ID)
            return os.path.join(STUDENT_FACES_DIR, str(user_id))

            try:
                 for d in os.listdir(STUDENT_FACES_DIR):
                     if d.startswith(f"{safe_name}_") and os.path.isdir(os.path.join(STUDENT_FACES_DIR, d)):
                         return os.path.join(STUDENT_FACES_DIR, d)
            except Exception:
                 pass

            # 5. Check for Old ID folder (Backward Compatibility)
            old_path = os.path.join(STUDENT_FACES_DIR, str(user_id))
            if os.path.exists(old_path):
                # Migrate!
                try:
                    os.rename(old_path, target_path)
                    logging.info(
                        f"MIGRATION: Renamed face folder {old_path} -> {target_path}")
                    return target_path
                except OSError as e:
                    logging.error(f"Migration Failed (Rename Error): {e}")
                    return old_path  # Use old path if rename fails

            return target_path

        except Exception as e:
            logging.error(f"Error resolving student folder: {e}")
            return os.path.join(STUDENT_FACES_DIR, str(user_id))

    def _rebuild_index(self):
        """Fetches ALL available embeddings (centroids + samples) from DB."""
        logging.info("Rebuilding multi-embedding face index...")
        data = database_manager.get_index_embeddings()

        self.known_ids = []
        self.known_embeddings = []

        for user_id, emb in data:
            if emb.shape[0] != face_config.EMBEDDING_DIM:
                logging.error(f"EMBEDDING_DIM_MISMATCH: Stored Dim {emb.shape[0]} != Expected {face_config.EMBEDDING_DIM} for user ID {user_id}")
                continue

            # Ensure normalized
            emb = emb.reshape(1, -1)
            emb = normalize(emb, norm='l2', axis=1).flatten()

            self.known_ids.append(user_id)
            self.known_embeddings.append(emb)

        logging.info(f"Index rebuilt. Total Identities={len(set(self.known_ids))}, Total Embeddings={len(self.known_ids)}")

    def check_face_quality(self, face, image_shape, min_confidence=None):
        """
        STRICT QUALITY FILTER
        image_shape: (h, w)
        min_confidence: Optional override for detection score threshold.
        Returns: (bool_pass, reason)
        """
        threshold = min_confidence if min_confidence is not None else face_config.MIN_DETECTION_CONFIDENCE

        # 1. Detection Confidence
        if face.det_score < threshold:
            return False, f"Low detection score: {face.det_score:.2f} (Threshold: {threshold})"

        # 2. Face Size
        bbox = face.bbox
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w < face_config.MIN_FACE_SIZE[0] or h < face_config.MIN_FACE_SIZE[1]:
            return False, f"Face too small: {int(w)}x{int(h)}"

        # 3. Pose (Yaw, Pitch)
        if face.pose is not None:
            pitch, yaw, roll = face.pose
            if abs(pitch) > face_config.MAX_PITCH:
                return False, f"Extreme pitch: {pitch:.1f}"
            if abs(yaw) > face_config.MAX_YAW:
                return False, f"Extreme yaw: {yaw:.1f}"

        return True, "OK"

    def is_image_blurry(self, img_crop, threshold=100.0):
        """
        Check image blur using Laplacian Variance.
        Low variance = Blurry.
        """
        try:
            gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            return score < threshold, score
        except:
            return True, 0.0

    def enhance_image_clahe(self, img):
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return enhanced
        except:
            return img

    def _correct_exif_rotation(self, file_bytes):
        """Corrects image rotation based on EXIF orientation."""
        try:
            img_pil = Image.open(io.BytesIO(file_bytes))
            img_pil = ImageOps.exif_transpose(img_pil)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logging.warning(f"EXIF rotation correction failed: {e}")
            # Fallback to standard decode
            nparr = np.frombuffer(file_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def preprocess_face(self, img, face_obj, target_size=(224, 224)):
        """
        MANDATORY STANDARDIZED PREPROCESSING:
        1. Alignment (Eyes Horizontal)
        2. Resize to 224x224
        3. Pixel Normalization
        """
        try:
            # 1. Alignment using Keypoints
            if face_obj.kps is not None:
                # InsightFace keypoints order: [left_eye, right_eye, nose, left_mouth, right_mouth]
                kps = face_obj.kps
                eye_left = kps[0]
                eye_right = kps[1]
                
                # Calculate angle (standard eyes-level alignment)
                dx = eye_right[0] - eye_left[0]
                dy = eye_right[1] - eye_left[1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Center of eyes
                eye_center = ((eye_left[0] + eye_right[0]) / 2, (eye_left[1] + eye_right[1]) / 2)
                
                # Rotation Matrix
                M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
                
                # Align
                h, w = img.shape[:2]
                aligned_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
            else:
                aligned_img = img

            # 2. Smart Crop & Resize
            bbox = face_obj.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            # Add some margin
            bh = y2 - y1
            bw = x2 - x1
            margin = int(max(bh, bw) * 0.2)
            
            h_img, w_img = aligned_img.shape[:2]
            nx1 = max(0, x1 - margin)
            ny1 = max(0, y1 - margin)
            nx2 = min(w_img, x2 + margin)
            ny2 = min(h_img, y2 + margin)
            
            face_crop = aligned_img[ny1:ny2, nx1:nx2]
            if face_crop.size == 0:
                face_crop = img[max(0,y1):min(h_img,y2), max(0,x1):min(w_img,x2)]

            # Resize to target
            resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
            
            # 3. Consistently Normalize (0-1 or -1 to 1)
            # Most modern models prefer float32
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized
        except Exception as e:
            logging.error(f"Preprocessing failed: {e}")
            return img # Fallback to original if alignment fails

    def _crop_face_smart(self, image, box, padding_pct, target_size=(300, 300)):
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = box

        # Calculate padding
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * padding_pct)
        pad_y = int(bh * padding_pct)

        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(w_img, x2 + pad_x)
        ny2 = min(h_img, y2 + pad_y)

        cropped = image[ny1:ny2, nx1:nx2]

        if cropped.size == 0:
            return image

        return cv2.resize(cropped, target_size)

    def process_uploaded_photo(self, user_id, file_obj):
        """
        Process uploaded photo with SMART FACE CROPPING.
        Generates 3 samples: 
          1. Original
          2. Zoom 1 (25% padding)
          3. Zoom 2 (10% padding)
        Enforces STRICT single face check on the original image.
        """
        try:
            # Read file
            file_bytes = file_obj.read()
            img = self._correct_exif_rotation(file_bytes)

            if img is None:
                return {"success": False, "message": "Failed to decode image"}

            # Step 1: Strict Detection on Original
            faces = self.app.get(img)
            if len(faces) == 0:
                enhanced = self.enhance_image_clahe(img)
                faces = self.app.get(enhanced)
                if len(faces) > 0:
                    img = enhanced

            if len(faces) == 0:
                return {"success": False, "message": "No face detected in the uploaded photo."}
            if len(faces) > 1:
                logging.warning(f"Multiple faces ({len(faces)}) detected. Selecting the largest face for enrollment.")
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)

            primary_face = faces[0]
            bbox = primary_face.bbox.astype(int)

            # Generate Variants
            variants = []
            variants.append(('original', img))
            variants.append(('zoom_1', self._crop_face_smart(img, bbox, 0.25)))
            variants.append(('zoom_2', self._crop_face_smart(img, bbox, 0.10)))

            encodings_count = 0
            save_dir = self.get_student_folder(user_id)
            os.makedirs(save_dir, exist_ok=True)

            for name, img_variant in variants:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = f"{name}_{timestamp}.jpg"
                filepath = os.path.join(save_dir, filename)
                
                # We use process_enrollment_image which handles detection, normalization, and DB saving
                success, msg, saved = self.process_enrollment_image(user_id, img_variant, source='enrollment_sample', image_path=filepath)
                
                if saved:
                    # Also save the image file for reference
                    cv2.imwrite(filepath, img_variant)
                    encodings_count += 1
                else:
                    logging.warning(f"Variant {name} failed for user {user_id}: {msg}")

            if encodings_count == 0:
                return {"success": False, "message": "Failed to extract embeddings from any image variants."}

            # Finalize
            res = self.finalize_enrollment(user_id)
            finalized_success = res.get('success', False)
            
            return {"success": True, "message": f"Enrollment successful. Generated {encodings_count} embeddings.", "count": encodings_count, "finalized": finalized_success}

        except Exception as e:
            logging.error(f"Enroll Upload Logic Error: {e}")
            return {"success": False, "message": str(e)}

    def finalize_enrollment(self, user_id):
        """
        Computes centroid from samples, checks duplicates, validates, saves centroid.
        """
        samples = database_manager.get_user_samples(user_id)
        if len(samples) == 0:
            return {"success": False, "message": "No face samples found. Please capture or upload a photo."}

        # Calculate Centroid
        samples_arr = np.array(samples)
        centroid = np.mean(samples_arr, axis=0)
        
        # Normalize Centroid
        norm = np.linalg.norm(centroid)
        if norm == 0:
             return {"success": False, "message": "Resulting centroid is zero-length."}
        centroid = (centroid / norm).flatten()

        # Consistency Validation
        if len(centroid) != face_config.EMBEDDING_DIM:
             return {"success": False, "message": f"Centroid dimension mismatch: {len(centroid)} != {face_config.EMBEDDING_DIM}"}

        # Duplicate Identity Check
        if len(self.known_embeddings) > 0:
            known_arr = np.array(self.known_embeddings)
            sims = np.dot(known_arr, centroid)
            max_sim = np.max(sims)

            if max_sim > face_config.DUPLICATE_IDENTITY_THRESHOLD:
                idx = np.argmax(sims)
                existing_id = self.known_ids[idx]
                if str(existing_id) != str(user_id):
                    return {"success": False, "message": f"Face too similar to existing user ID {existing_id} ({max_sim:.2f})"}

        # Save Centroid
        database_manager.save_embedding(user_id, centroid, source='centroid')

        # Update User Face Status in DB
        database_manager.update_user_face_status(user_id, 'Enrolled')

        warning_msg = ""
        if len(samples) < face_config.MIN_SAMPLES_FOR_CENTROID:
            warning_msg = f" (Warning: Using {len(samples)} samples, recommended {face_config.MIN_SAMPLES_FOR_CENTROID})"

        # Rebuild Index
        self._rebuild_index()

        return {"success": True, "message": "Enrollment finalized successfully." + warning_msg}


    def process_video_frame(self, subject, marked_by_id):
        """
        Generator for video streaming + recognition.
        OPTIMIZED: Frame Skipping (1/3), Cooldowns (60s), Dual Thresholds.
        """
        log_debug(f"Video Feed Requested for subject: {subject} by: {marked_by_id}")
        
        # Try different camera indices and backends
        camera = None
        for idx in [0, 1, 2]:
            # Default backend first (usually faster/more reliable for index 0)
            camera = cv2.VideoCapture(idx)
            if camera.isOpened():
                break
                
            # Try DSHOW if default fails
            camera = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if camera.isOpened():
                break

        if not camera or not camera.isOpened():
            log_debug("FATAL: No camera could be opened.")
            yield (b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera Error: No device found\r\n')
            return

        # Warm up
        for _ in range(5):
             camera.read()

        log_debug("Starting frame processing loop.")
        frame_count = 0
        skip_frames = 2  # Process 1 out of every (skip_frames + 1) frames
        
        last_faces = [] # Cache for skipped frames
        
        # Performance Counters
        start_time = time.time()
        fps_counter = 0

        # Thresholds
        THRESH_STRONG = 0.60
        THRESH_WEAK = 0.40
        COOLDOWN_SECONDS = 60

        try:
            while True:
                success, frame = camera.read()
                if not success:
                    break

                frame_count += 1
                fps_counter += 1
                
                # Log FPS every second
                if time.time() - start_time > 1.0:
                    # log_debug(f"FPS: {fps_counter}")
                    fps_counter = 0
                    start_time = time.time()

                # --- 1. Frame Skipping Logic ---
                if frame_count % (skip_frames + 1) != 0:
                     # Reuse last detections (visual tracking)
                     for (bbox, name, color) in last_faces:
                          cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                          cv2.putText(frame, name, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                     
                     ret, buffer = cv2.imencode('.jpg', frame)
                     if ret:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                     continue

                # --- 2. Detection (Resize 320px) ---
                faces = []
                try:
                    h, w = frame.shape[:2]
                    target_w = 320
                    scale = target_w / w
                    target_h = int(h * scale)
                    small_frame = cv2.resize(frame, (target_w, target_h))
                    
                    faces = self.app.get(small_frame)
                    
                    # Scale back
                    for face in faces:
                        face.bbox *= (1.0 / scale)
                        if face.kps is not None:
                            face.kps *= (1.0 / scale)
                except Exception:
                     pass


                current_visuals = [] # Store for next frames

                # --- 3. Process Faces ---
                for face in faces:
                    bbox = face.bbox.astype(int)
                    
                    # A. Quality Check
                    is_good, q_msg = self.check_face_quality(face, frame.shape)
                    
                    # Extra Blur Check
                    face_crop = frame[max(0,bbox[1]):min(h,bbox[3]), max(0,bbox[0]):min(w,bbox[2])]
                    if face_crop.size > 0:
                        is_blurry, blur_val = self.is_image_blurry(face_crop)
                        if is_blurry:
                             is_good = False
                             q_msg = f"Blurry ({int(blur_val)})"

                    if not is_good:
                        color = (0, 0, 255) # Red
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        current_visuals.append((bbox, "Low Quality", color))
                        continue

                    # --- MANDATORY STANDARDIZED PREPROCESSING ---
                    # Align and normalize the face crop before extraction
                    processed_face = self.preprocess_face(frame, face, target_size=(224, 224))
                    v_img_uint8 = (processed_face * 255).astype(np.uint8)
                    v_faces = self.app.get(v_img_uint8)
                    
                    if v_faces:
                        v_faces = sorted(v_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                        face = v_faces[0]

                    # B. Recognition
                    emb = normalize(face.embedding.reshape(1, -1), norm='l2').flatten()
                    
                    # Use unified match_face with centralized thresholds
                    user_id, score, is_relaxed = self.match_face(emb)
                    
                    name = "Unknown"
                    color = (0, 0, 255) # Default Red for Unknown

                    # --- GENDER PREDICTION (INTEGRATED) ---
                    gender_label = ""
                    try:
                        # Extract gender directly from the analyzed face object
                        # 1 = Male, 0 = Female in buffalo_l model
                        gender_val = getattr(face, 'gender', -1)
                        if gender_val == 1:
                            gender = "Male"
                        elif gender_val == 0:
                            gender = "Female"
                        else:
                            gender = "Unknown"
                        
                        gender_label = f" [{gender}]"
                    except Exception as ge:
                        logging.warning(f"Gender extraction failed: {ge}")
                    # --------------------------------------
                    
                    if user_id:
                        user_info = database_manager.get_user_info(user_id)
                        if user_info:
                             name = user_info['full_name']
                             color = (0, 255, 0) if not is_relaxed else (0, 255, 255)
                             
                             self.latest_recognition = {
                                 "user_id": user_id,
                                 "name": name,
                                 "confidence": score,
                                 "timestamp": time.time(),
                                 "subject": subject,
                                 "gender": gender
                             }
                             
                             last_mark_time = self.recognition_cooldowns.get(user_id, 0)
                             if time.time() - last_mark_time < COOLDOWN_SECONDS:
                                  name = f"{name} (Marked)"
                             else:
                                  self.recognition_cooldowns[user_id] = time.time()
                                  self.recognition_queue.append((user_id, name, score, time.time(), subject))
                                  if len(self.recognition_queue) > self.MAX_QUEUE_SIZE:
                                      self.recognition_queue.pop(0)

                             # Add gender to name for display
                             name += gender_label


                    else:
                        # Append gender to Unknown if recognized as face but identity unknown
                        name = f"Unknown{gender_label}"

                    # --- LIVENESS DETECTION (WEBSITE) ---
                    is_live, liveness_msg = self.liveness_validator.validate_frame(
                        frame, face, self._liveness_history, self._spoof_counter
                    )
                    
                    # Store current frame for next temporal check
                    self._liveness_history.append({'image': frame.copy(), 'face': face})
                    if len(self._liveness_history) > self._MAX_LIVENESS_HISTORY:
                        self._liveness_history.pop(0)
                        
                    if not is_live:
                        color = (0, 165, 255) # Orange for suspicious
                        # User requested identifying the person even if spoofed
                        # We extract the clean name by stripping metadata
                        base_name = name.split(" [")[0] if " [" in name else name
                        base_name = base_name.replace(" (Marked)", "")
                        
                        if "Unknown" not in base_name:
                            name = f"SPOOF ({base_name}): {liveness_msg}"
                        else:
                            name = f"SPOOF: {liveness_msg}"
                        # Invalidate recent recognition if spoofing detected
                        if hasattr(self, 'latest_recognition') and self.latest_recognition and self.latest_recognition.get('user_id') == user_id:
                             self.latest_recognition['liveness_failed'] = True
                             self.latest_recognition['liveness_msg'] = liveness_msg
                    # ------------------------------------

                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, name, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    # Add Liveness Status Overlay
                    status_color = (0, 255, 0) if is_live else (0, 0, 255)
                    cv2.putText(frame, f"Liveness: {liveness_msg}", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                    current_visuals.append((bbox, name, color))

                last_faces = current_visuals

                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                     yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        finally:
            log_debug(f"Releasing camera after {frame_count} frames.")
            camera.release()

    def generate_enrollment_frames(self, user_id):
        # Similar to process_video_frame but just for preview
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera = cv2.VideoCapture(0)
        
        while True:
            success, frame = camera.read()
            if not success:
                break

            # Just detection visualization
            faces = self.app.get(frame)
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[2], bbox[3]), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        camera.release()



    def _preprocess_image(self, img):
        """Applies CLAHE, Denoising, and sharpening."""
        try:
            # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            # Sharpening removed as it causes artifacts in low-res recognition
            
            return enhanced
        except:
            return img.copy()

    def _nms_boxes(self, faces, iou_thresh=0.4):
        """Non-Maximum Suppression for face objects."""
        if not faces:
            return []
            
        # Extract bboxes and scores
        bboxes = np.array([f.bbox for f in faces])
        scores = np.array([f.det_score for f in faces])
        
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(faces[i]) 
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]
            
        return keep

    def _detect_faces_tiled(self, img):
        """
        Robust detection:
        1. Preprocess
        2. Detect on Original (Using High-Res Model)
        3. Detect on Tiles (3x3 grid) - Optimized for speed
        4. Merge with NMS
        """
        all_faces = []

        # Single engine (app) handles both normal and high-res requests
        detector = self.app
        
        # A. Preprocessing
        img_input = self._preprocess_image(img)
        h, w = img_input.shape[:2]

        # B. Multi-Scale: Original
        logging.info("Detection: Running on full image (1x)...")
        faces_1x = detector.get(img_input)
        
        # FILTER LOW CONFIDENCE (Reduces False Positives)
        # Using centralized threshold for tiled/group photos
        min_conf = face_config.MIN_DETECTION_CONFIDENCE_TILED
        
        for f in faces_1x:
            if f.det_score >= min_conf:
                all_faces.append(f)
        
        # C. Multi-Scale: 2x Upscale - REDUCED/REMOVED for Speed & False Positive Reduction
        # High-res class photos don't need 2x upscaling, which often introduces noise/artifacts (hands as faces).
        # We skip this step to satisfy "Very Slow" complaint.

        # D. Tiled Detection (3x3 Grid - Optimized from 4x4)
        # 16 tiles -> 9 tiles (Nearly 2x faster tiling phase)
        gc.collect() # Force cleanup before tiling
        logging.info("Detection: Running tiled detection (3x3)...")
        grid_rows = 3
        grid_cols = 3
        gw = w // grid_cols
        gh = h // grid_rows
        
        for r in range(grid_rows):
            for c in range(grid_cols):
                base_x1 = c * gw
                base_y1 = r * gh
                base_x2 = (c + 1) * gw if c < grid_cols - 1 else w
                base_y2 = (r + 1) * gh if r < grid_rows - 1 else h
                
                tile_idx = (r * grid_cols) + c + 1
                logging.info(f"Detection: Processing tile {tile_idx}/{grid_rows * grid_cols}...")
                
                # 25% overlap padding
                pad_w_x = int((base_x2 - base_x1) * 0.25)
                pad_w_y = int((base_y2 - base_y1) * 0.25)
                
                xx1 = max(0, base_x1 - pad_w_x)
                yy1 = max(0, base_y1 - pad_w_y)
                xx2 = min(w, base_x2 + pad_w_x)
                yy2 = min(h, base_y2 + pad_w_y)
                
                tile = img_input[yy1:yy2, xx1:xx2]
                if tile.size == 0: continue
                
                try:
                    tile_faces = detector.get(tile)
                    for f in tile_faces:
                        # Relaxed threshold for tiles
                        if f.det_score < (min_conf - 0.05): # 0.30 for tiles
                            continue
                            
                        f.bbox[0] += xx1
                        f.bbox[1] += yy1
                        f.bbox[2] += xx1
                        f.bbox[3] += yy1
                        if f.kps is not None:
                             f.kps[:, 0] += xx1
                             f.kps[:, 1] += yy1
                        all_faces.append(f)
                except Exception as e:
                    logging.warning(f"Tile detection failed: {e}")
                finally:
                    # Clear tile memory immediately
                    del tile
                    if 'tile_faces' in locals(): del tile_faces
                    gc.collect()

        # E. NMS
        final_faces = self._nms_boxes(all_faces, iou_thresh=0.45) # Slightly tighter NMS
        logging.info(f"Detection: Total candidates {len(all_faces)} -> Final NMS {len(final_faces)}")
        return final_faces

    def _compute_quality_score(self, face, img_shape):
        """
        Computes a normalized quality score (0.0 - 1.0)
        Based on: Detection Score + Face Size (Resolution)
        """
        h, w = img_shape[:2]
        
        # 1. Detection Confidence (0.0 - 1.0)
        det_score = min(face.det_score, 0.9) # Cap at 0.9 to align with normalization
        
        # 2. Face Resolution (relative to 100x100 ideal)
        bbox = face.bbox
        fw = bbox[2] - bbox[0]
        fh = bbox[3] - bbox[1]
        face_area = fw * fh
        ideal_area = 100 * 100
        
        res_score = min(face_area / ideal_area, 1.0)
        
        # Weighted average (60% det, 40% res)
        quality = (det_score * 0.6) + (res_score * 0.4)
        return quality

    def process_class_photo_attendance(self, photo_path, subject, hour, department, year, marked_by_id, batch_year=None, semester=None, mark_absent=True, period_no=None, method='class_photo'):
        if not self.initialized:
            self.initialize()
        logging.info("DEBUG: Entering process_class_photo_attendance (New Version)")

        # 1. Load Image with Enhanced Error Handling
        # Normalize path to handle spaces and special characters
        photo_path = os.path.normpath(photo_path)
        
        if not os.path.exists(photo_path):
            raise Exception(f"Failed to load image: File does not exist at path: {photo_path}")
        
        # Try loading with cv2 using imdecode (handles paths with spaces better on Windows)
        img = None
        try:
            with open(photo_path, 'rb') as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                logging.info(f"Successfully loaded image using cv2.imdecode: {photo_path}")
        except Exception as decode_error:
            logging.warning(f"cv2.imdecode failed: {decode_error}")
        
        if img is None:
            # Fallback to PIL
            try:
                from PIL import Image
                pil_img = Image.open(photo_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                logging.info(f"Successfully loaded image using PIL fallback: {photo_path}")
            except Exception as pil_error:
                raise Exception(f"Failed to load image with all methods. Path: {photo_path}, PIL Error: {str(pil_error)}")

        h, w = img.shape[:2]
        # Optimization: Max 4000px width (Increased to capture back-row detail)
        max_w = 4000
        scale = 1.0
        if w > max_w:
            scale = max_w / w
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))

        # 2. LOAD SYSTEM INDEX (Ensure latest embeddings are available)
        if not self.known_embeddings:
            self._rebuild_index()

        # Build metadata map for all registered students (needed for results)
        all_registered = database_manager.get_all_registered_users_with_faces() 
        student_data = {s['id']: {'info': {'id': s['id'], 'full_name': s['full_name'], 'roll_number': s.get('roll_number', 'N/A')}} for s in all_registered}
        

        if not self.known_embeddings:
             return {"success": False, "message": "No face embeddings found in system.", "total_faces": 0}

        # 3. Fetch Targeted Class List (to distinguish Present vs Others)
        class_students = database_manager.get_students_by_class(department, year, batch_year=batch_year, semester=semester)
        class_uids = {s['id'] for s in class_students}
        logging.info(f"Class Photo: Ready with index of {len(self.known_embeddings)} embeddings. Class size: {len(class_uids)}")

        # 4. Detect Faces (Tiled)
        faces = self._detect_faces_tiled(img)
        logging.info(f"Class Photo: Detected {len(faces)} faces.")

        if len(faces) == 0:
            # If skipping absent marking, we effectively return nothing happened for this photo
            absent_list = [s for s in class_students] if mark_absent else []
            if mark_absent:
                 # Mark all absent if single photo and nothing found
                 pass # Logic below handles list return, strictly marking is done below? 
                 # Wait, previous code marked absent returning strictly.
                 # Let's align.
                 pass

            return {
                "success": True, 
                "message": "No faces detected.",
                "total_faces": 0,
                "present_students": [],
                "absent_students": absent_list, 
                "unknown_faces": []
            }

        matched_results = []
        unknown_faces = []
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        proof_base_dir = os.path.join(ATTENDANCE_PROOFS_DIR, "class_crops", timestamp_str)
        os.makedirs(proof_base_dir, exist_ok=True)

        # ADAPTIVE THRESHOLDS (Relaxed for Group Photos as requested)
        # 0.45 Distance ~ 0.89 Sim? No, standard ArcFace Sim is usually 0.4-0.5.
        # We will use the ultra-lenient but safe thresholds.
        # Using centralized thresholds from face_config
        

        # 5. Process Faces in two passes to ensure strict Match vs Unknown separation
        processing_results = []
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(img.shape[1], bbox[2]), min(img.shape[0], bbox[3])
            if x2 <= x1 or y2 <= y1: continue
            
            # --- STRICT QUALITY FILTER (Fix for False Positives) ---
            # 1. Size Check (Relaxed to 15px to catch back-row students)
            fw, fh = (x2 - x1), (y2 - y1)
            if fw < 15 or fh < 15: continue
            
            # 2. Confidence Check (Double Verification)
            if face.det_score < 0.45: continue
            
            # 3. Blur/Noise Check (Relaxed to 40 to allow slightly blurry faces)
            face_crop = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if variance < 25: # Only filter extremely blurry blobs
                continue
                
            # --- MANDATORY STANDARDIZED PREPROCESSING (Enrollment Sync) ---
            # Smart Skip: If face is too small, re-extraction (upscaling) destroys quality.
            # Use original high-res embedding for small faces (< 50px)
            if fw < 50 or fh < 50:
                 # Use raw crop from original image (no alignment/resize) for proof/display
                 if y2 > y1 and x2 > x1:
                    face_img_uint8 = img[y1:y2, x1:x2].copy()
                 else:
                    face_img_uint8 = np.zeros((100, 100, 3), dtype=np.uint8) # Safe fallback

                 rec_faces = []
            else:
                 processed_face_img = self.preprocess_face(img, face, target_size=(224, 224))
                 face_img_uint8 = (processed_face_img * 255).astype(np.uint8)
            
                 # Temporary crop for re-extraction
                 rec_faces = self.app.get(face_img_uint8)
                 if not rec_faces:
                     enhanced_crop = self.enhance_image_clahe(face_img_uint8)
                     rec_faces = self.app.get(enhanced_crop)
            
            if rec_faces:
                rec_faces = sorted(rec_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                face = rec_faces[0]

            # Recognition Scoring
            feat = normalize(face.embedding.reshape(1, -1), norm='l2').flatten()
            
            # --- GLOBAL MATCHING (STRICT SCOPE REMOVED) ---
            # We now match globally to differentiate between "Unknown" and "Other Class"
            user_id, score, is_relaxed = self.match_face(feat, allowed_user_ids=None)
            
            processing_results.append({
                "face_idx": i,
                "bbox": bbox,
                "user_id": user_id,
                "score": score,
                "is_relaxed": is_relaxed,
                "face_obj": face,
                "crop_img": face_img_uint8
            })

        # --- SEPARATION LOGIC (STRICT) ---
        matched_results_candidates = []
        matched_bboxes = []
        raw_unknown_candidates = []
        other_class_students_list = []

        # Pass 1: Identify ALL identities found in the photo
        for res in processing_results:
            if res['user_id']:
                matched_results_candidates.append(res)
                matched_bboxes.append(res['bbox'])

        # Pass 2: Filter Unknowns (Must NOT overlap with any matched face)
        used_student_ids = set()
        for res in processing_results:
            if res['user_id']: continue # Already matched
            
            # Check for overlap with any matched face
            is_matched_elsewhere = False
            for m_bbox in matched_bboxes:
                # Custom IoU check
                ix1 = max(res['bbox'][0], m_bbox[0])
                iy1 = max(res['bbox'][1], m_bbox[1])
                ix2 = min(res['bbox'][2], m_bbox[2])
                iy2 = min(res['bbox'][3], m_bbox[3])
                
                iw = max(0, ix2 - ix1 + 1)
                ih = max(0, iy2 - iy1 + 1)
                inter = iw * ih
                
                area1 = (res['bbox'][2] - res['bbox'][0] + 1) * (res['bbox'][3] - res['bbox'][1] + 1)
                area2 = (m_bbox[2] - m_bbox[0] + 1) * (m_bbox[3] - m_bbox[1] + 1)
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0
                
                if iou > 0.45:
                    is_matched_elsewhere = True
                    break
            
            if not is_matched_elsewhere:
                # Truly Unknown -> Save crop
                crop_filename = f"unknown_{res['face_idx']}_{int(time.time()*1000)}.jpg"
                crop_path = os.path.join(proof_base_dir, crop_filename)
                cv2.imwrite(crop_path, res['crop_img'])
                raw_unknown_candidates.append(crop_path)

        # Final Attendance Resolution
        matched_results_candidates.sort(key=lambda x: x['score'], reverse=True)
        present_students_list = []
        
        for m in matched_results_candidates:
            uid = m['user_id']
            if uid not in used_student_ids:
                if uid in class_uids:
                    # Case 1: Present (Valid Match in Target Class)
                    crop_filename = f"match_{uid}_{int(time.time()*1000)}.jpg"
                    crop_path = os.path.join(proof_base_dir, crop_filename)
                    cv2.imwrite(crop_path, m['crop_img'])
                    
                    s_data = student_data.get(uid)
                    s_info = s_data['info'] if s_data else {"full_name": f"Student {uid}", "roll_number": "N/A"}
                    roll_no = s_info['roll_number']

                    logging.info(f"DEBUG: Marking PRESENT for student Roll: {roll_no} (ID: {uid})")

                     # Auto-fix: If found, they ARE enrolled.
                    if s_data.get('face_status') == 'Not Enrolled':
                        logging.info(f"AUTO-FIX: Updating status to Enrolled for {roll_no}")
                        database_manager.update_user_face_status(uid, 'Enrolled')

                    try:
                        database_manager.mark_attendance_for_user(
                            uid, subject, marked_by_id,
                            status='Present',
                            confidence=float(m['score']),
                            proof_path=f"{timestamp_str}/{crop_filename}",
                            method=method,
                            period_no=period_no
                        )
                    except Exception as ae:
                        logging.error(f"Error marking present student {roll_no}: {ae}")
                    
                    present_students_list.append({
                        "id": uid,
                        "full_name": s_info['full_name'],
                        "roll_number": roll_no,
                        "confidence": float(m['score']),
                        "proof_path": f"/{timestamp_str}/{crop_filename}"
                    })
                    used_student_ids.add(uid)
                else:
                    # Case 2: Other Class (Valid Match but OUT OF SCOPE)
                    s_data = student_data.get(uid)
                    s_info = s_data['info'] if s_data else {"full_name": f"Student {uid}", "roll_number": "N/A"}
                    logging.warning(f"SCOPE MISMATCH: Student {s_info['full_name']} ({uid}) found but not in class list.")
                    
                    other_class_students_list.append({
                        "id": uid,
                        "full_name": s_info['full_name'], 
                        "roll_number": s_info.get('roll_number', 'N/A'),
                        "confidence": float(m['score']),
                        "status": "Other Class"
                    })
                    used_student_ids.add(uid)

        # Final Lists
        absent_students_list = [s for s in class_students if s['id'] not in used_student_ids]
        
        # --- CONDITIONAL ABSENTEE MARKING ---
        if mark_absent:
            for s in absent_students_list:
                try:
                    roll = s.get('roll_number', 'N/A')
                    logging.info(f"DEBUG: Marking ABSENT for student Roll: {roll} (ID: {s['id']})")
                    database_manager.mark_attendance_for_user(s['id'], subject, marked_by_id, status='Absent', method=method, period_no=period_no)
                except Exception as ae:
                    logging.error(f"Error marking absent student {s['id']}: {ae}")
        else:
             logging.info("DEBUG: Skipping absent marking (Multi-file batch mode)")

 
        unknown_faces_relative = [f"{timestamp_str}/{os.path.basename(p)}" for p in raw_unknown_candidates]
 
        # Fix: Total Faces should only count those that passed quality filters
        # processed_count = len(processing_results) 
        # Actually, let's use the sum of categorized faces to be exact 100% of the time
        final_total = len(present_students_list) + len(other_class_students_list) + len(unknown_faces_relative)
        logging.info(f"DEBUG: Final Resolution - Present: {len(present_students_list)}, Other: {len(other_class_students_list)}, Unknown: {len(unknown_faces_relative)}, Absent: {len(absent_students_list)}")

        return {
            "success": True,
            "total_faces": final_total, # Corrected from len(faces)
            "matched_count": len(present_students_list),
            "other_count": len(other_class_students_list),
            "unknown_count": len(unknown_faces_relative),
            "present_students": present_students_list,
            "other_students": other_class_students_list, # New Return Field
            "absent_students": absent_students_list,
            "unknown_faces": unknown_faces_relative
        }

    def process_client_frame(self, frame, subject, marked_by_id, spoof_counter=None):
        """
        Stateless handler for a single frame from client camera.
        Returns recognition and liveness results.
        """
        try:
            # 1. Detection (Resize 320px for speed)
            h, w = frame.shape[:2]
            target_w = 320
            scale = target_w / w
            target_h = int(h * scale)
            small_frame = cv2.resize(frame, (target_w, target_h))
            
            faces = self.app.get(small_frame)
            
            if not faces:
                return {"success": False, "message": "No face detected"}

            # Scale back and process largest face
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            face = faces[0]
            face.bbox *= (1.0 / scale)
            if face.kps is not None:
                face.kps *= (1.0 / scale)

            # 2. Liveness Check
            is_live, liveness_msg = self.liveness_validator.validate_frame(frame, face, spoof_counter=spoof_counter)
            
            # 3. Gender extraction (from InsightFace genderage model — available on any detected face)
            gender = "Unknown"
            try:
                gender_val = getattr(face, 'gender', -1)
                if gender_val == 1:
                    gender = "Male"
                elif gender_val == 0:
                    gender = "Female"
            except Exception:
                pass

            # 4. Recognition
            user_id = None
            name = "Unknown"
            score = 0.0
            
            if is_live:
                # Standardized preprocessing
                processed_face = self.preprocess_face(frame, face, target_size=(224, 224))
                v_img_uint8 = (processed_face * 255).astype(np.uint8)
                v_faces = self.app.get(v_img_uint8)
                
                if v_faces:
                    v_faces = sorted(v_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                    raw_emb = v_faces[0].embedding.reshape(1, -1)
                    norm = np.linalg.norm(raw_emb)
                    emb = (raw_emb / norm).flatten() if norm != 0 else None
                    
                    if emb is not None:
                        user_id, score, _ = self.match_face(emb)
                        if user_id:
                            # Robust Name Retrieval
                            log_debug(f"Match Found: user_id={user_id}, score={score:.4f}")
                            info = database_manager.get_user_info(user_id)
                            if info and info.get('full_name'):
                                name = info['full_name']
                                log_debug(f"Identity Resolved: {name}")
                            else:
                                name = f"Student {user_id}"
                                log_debug(f"Identity Resolution Warning: Missing name for ID {user_id}")
                        else:
                            log_debug(f"Recognition Ambiguous: No high-confidence match found (top score: {score:.4f})")

            return {
                "success": True,
                "is_live": is_live,
                "liveness_msg": liveness_msg,
                "user_id": user_id,
                "name": name,
                "score": float(score),
                "bbox": [int(x) for x in face.bbox],
                "gender": gender
            }

        except Exception as e:
            logging.error(f"Error in process_client_frame: {e}")
            return {"success": False, "message": str(e)}

# Bridge functions to maintain backward compatibility with main.py calls


def initialize_system():
    _system.initialize()


def get_student_directory_path(user_id):
    if not _system.initialized:
        _system.initialize()
    return _system.get_student_folder(user_id)


def capture_and_save_face(user_id, image_data):
    if not _system.initialized:
        _system.initialize()
    return _system.add_enrollment_sample(user_id, image_data)


def get_enrollment_status(user_id):
    samples = database_manager.get_user_samples(user_id)
    return {"embedding_count": len(samples)}


def finalize_enrollment_logic(user_id):
    # This might need to be exposed if main.py calls logic directly
    if not _system.initialized:
        _system.initialize()
    return _system.finalize_enrollment(user_id)


def generate_frames(subject, marked_by_id):
    if not _system.initialized:
        _system.initialize()
    return _system.process_video_frame(subject, marked_by_id)


def generate_enrollment_frames(user_id):
    if not _system.initialized:
        _system.initialize()
    return _system.generate_enrollment_frames(user_id)


def get_latest_recognition():
    if not _system.initialized:
        _system.initialize()
    return getattr(_system, 'latest_recognition', None)


def resume_scanning():
    pass


def finish_scanning():
    pass


def enroll_face_from_upload(user_id, file_obj):
    # Process uploaded file using the new augmentation logic
    if not _system.initialized:
        _system.initialize()
    return _system.process_uploaded_photo(user_id, file_obj)

def process_uploaded_photo(user_id, file_obj):
    if not _system.initialized:
        _system.initialize()
    return _system.process_uploaded_photo(user_id, file_obj)


def update_student_face_encodings(user_id):
    """
    Triggers centroid calculation and index rebuild for a student.
    Matches the call signature used in main.py.
    """
    if not _system.initialized:
        _system.initialize()
    return _system.finalize_enrollment(user_id)



# --- Singleton Instance & Bridge Functions for Existing Main.py Calls ---


_system = FaceRecognitionSystem()

# Bridge functions to maintain backward compatibility with main.py calls


def initialize_system():
    _system.initialize()


def get_student_directory_path(user_id):
    if not _system.initialized:
        _system.initialize()
    return _system.get_student_folder(user_id)

def get_student_folder(user_id):
    """Alias for main.py compatibility"""
    return get_student_directory_path(user_id)


def create_face_directory(user_id):
    path = get_student_directory_path(user_id)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def capture_and_save_face(user_id, image_data):
    if not _system.initialized:
        _system.initialize()
    return _system.add_enrollment_sample(user_id, image_data)


def get_enrollment_status(user_id):
    samples = database_manager.get_user_samples(user_id)
    return {"embedding_count": len(samples)}



# Validated Wrappers
def enroll_face_from_upload(user_id, file_obj):
    if not _system.initialized:
        _system.initialize()
    return _system.process_uploaded_photo(user_id, file_obj)

def process_class_photo_attendance(photo_path, subject, hour, department, year, marked_by_id, batch_year=None, semester=None, mark_absent=True, period_no=None, method='class_photo'):
    if not _system.initialized:
        _system.initialize()
    return _system.process_class_photo_attendance(photo_path, subject, hour, department, year, marked_by_id, batch_year=batch_year, semester=semester, mark_absent=mark_absent, period_no=period_no, method=method)


def update_student_face_encodings(user_id):
    if not _system.initialized:
        _system.initialize()
    return _system.update_student_face_encodings(user_id)

def get_student_face_status(user_id):
    return _system.get_student_face_status(user_id)

def finalize_enrollment_logic(user_id):
    if not _system.initialized:
        _system.initialize()
    return _system.finalize_enrollment(user_id)

def generate_frames(subject, marked_by_id):
    if not _system.initialized:
        _system.initialize()
    return _system.process_video_frame(subject, marked_by_id)

def generate_enrollment_frames(user_id):
    if not _system.initialized:
        _system.initialize()
    return _system.generate_enrollment_frames(user_id)


def get_latest_recognition():
    if not _system.initialized:
        _system.initialize()
    return getattr(_system, 'latest_recognition', None)

def resume_scanning():
    pass

def process_client_frame(frame, subject, marked_by_id, spoof_counter=None):
    if not _system.initialized:
        _system.initialize()
    return _system.process_client_frame(frame, subject, marked_by_id, spoof_counter=spoof_counter)

def is_system_ready():
    """Checks if the biometric system is fully initialized and operational."""
    return _system.initialized

def finish_scanning():
    pass

