import cv2
import numpy as np
import random
import uuid

class AntiSpoofingValidator:
    def __init__(self):
        self.challenges = [
            {'id': '1', 'text': 'Turn head left'},
            {'id': '2', 'text': 'Turn head right'},
            {'id': '3', 'text': 'Blink'},
            {'id': '4', 'text': 'Smile'}
        ]

    def generate_challenge(self):
        """Generates a random liveness challenge."""
        challenge = random.choice(self.challenges)
        return {
            'id': str(uuid.uuid4()),
            'challenge_type': challenge['id'],
            'text': challenge['text']
        }

    def detect_moire(self, img):
        """
        Detects moiré patterns (indicating a screen replay) using FFT analysis.
        Returns True if moiré is detected.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
            
            h, w = gray.shape
            cy, cx = h // 2, w // 2
            # Mask out the low frequencies (center of FFT)
            cv2.circle(magnitude_spectrum, (cx, cy), 60, 0, -1)
            high_freq_energy = np.mean(magnitude_spectrum)
            
            # Increased threshold from 50 to 80 to accommodate lower quality cameras
            # that naturally produce noise mimicking Moiré patterns.
            return high_freq_energy > 80
        except Exception:
            return False

    def validate_frame(self, frame, face=None, history=None, spoof_counter=None):
        """
        Validates a single frame for liveness (Moiré, Static Photo, Glare).
        Returns a tuple: (is_live: bool, liveness_msg: str)
        """
        if spoof_counter is None:
            spoof_counter = [0]
            
        # 1. Moiré (Screen) Detection
        if self.detect_moire(frame):
            spoof_counter[0] += 1
            if spoof_counter[0] >= 3:
                return False, "Screen Detected (Moiré)"
            return True, "Analyzing..."

        # 2. Static Photo Detection (Temporal Variance)
        if history is not None and len(history) >= 5:
            pixel_differences = []
            for i in range(len(history) - 1):
                try:
                    img1 = cv2.cvtColor(history[i]['image'], cv2.COLOR_BGR2GRAY)
                    img2 = cv2.cvtColor(history[i+1]['image'], cv2.COLOR_BGR2GRAY)
                    
                    img1 = cv2.resize(img1, (64, 64))
                    img2 = cv2.resize(img2, (64, 64))
                    
                    mse = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
                    pixel_differences.append(mse)
                except Exception:
                    pass
            
            if pixel_differences:
                mean_diff = np.mean(pixel_differences)
                if mean_diff < 0.1:
                    spoof_counter[0] += 1
                    if spoof_counter[0] >= 3:
                        return False, "Static Photo Detected"
                    return True, "Analyzing..."

        # 3. Specular Highlights (Glare)
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 3:
                spoof_counter[0] += 1
                if spoof_counter[0] >= 3:
                    return False, "Screen Glare Detected"
                return True, "Analyzing..."
        except Exception:
            pass

        spoof_counter[0] = max(0, spoof_counter[0] - 1)
        return True, "Live"
