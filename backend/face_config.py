import os

# --- HARD LOCK CONFIGURATION ---

# Model Settings
MODEL_NAME = "buffalo_l"
# INSIGHTFACE_HOME must be set before importing insightface app
PROJECT_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_BACKEND_DIR, "insightface_models")
os.environ["INSIGHTFACE_HOME"] = MODELS_DIR

# Quality Filters (Golden Standards)
# Quality Filters (Golden Standards)
MIN_FACE_SIZE = (15, 15) # Pixels
MIN_DETECTION_CONFIDENCE = 0.65 # Tightened
MIN_DETECTION_CONFIDENCE_TILED = 0.40 # Tightened

MAX_YAW = 25.0  # Degrees
MAX_PITCH = 20.0 # Degrees

# Matching Standards (Single Source of Truth)
SIMILARITY_METRIC = "cosine"
# Strict: Definite match (Permissive: 0.45 allows typical group faces)
# 0.45 is the sweet spot for group attendance.
MATCH_THRESHOLD_STRICT = 0.38 
# Relaxed: Needs margin check (0.38 catches angled/partial faces)
MATCH_THRESHOLD_RELAXED = 0.30
# Adaptive: Safe lower bound (0.32 is absolute minimum for ArcFace)
MATCH_THRESHOLD_ADAPTIVE = 0.20
# Margin for deduplication/relaxed matches
MATCH_MARGIN = 0.01
# Gap required for Dominance Check (Reduced from 0.01 to 0.005 for better group photo handling)
MATCH_DOMINANCE_GAP = 0.005

DUPLICATE_IDENTITY_THRESHOLD = 0.30 

# Enrollment
MIN_SAMPLES_FOR_CENTROID = 3 
MAX_SAMPLES_FOR_CENTROID = 10

# Embedding
EMBEDDING_DIM = 512
NORMALIZATION = "L2"
