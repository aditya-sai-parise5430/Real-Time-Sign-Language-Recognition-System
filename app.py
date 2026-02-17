"""
SIGN LANGUAGE RECOGNITION SYSTEM
Premium Yellow UI – Final Stable Version
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
from pathlib import Path
from tensorflow.keras.models import model_from_json  # type: ignore

# ==========================================================
# COLOR PALETTE (YELLOW PREMIUM)
# ==========================================================
COLOR_BG        = (18, 18, 18)       # Dark background
COLOR_PRIMARY   = (0, 215, 255)      # Yellow / Gold
COLOR_SUCCESS   = (40, 200, 120)     # Emerald
COLOR_ACCENT    = (255, 255, 255)    # WHITE (readable text)
COLOR_DIM       = (120, 120, 120)    # Only for borders
COLOR_TEXT      = (245, 245, 245)    # Main text

# ==========================================================
# MEDIAPIPE
# ==========================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ==========================================================
# LOAD MODEL
# ==========================================================
model = None
pairs = [
    ("model33k.json", "newmodel33k.h5"),
    ("model(0.35).json", "newmodel(0.35).h5"),
    ("model(0.2).json", "newmodel(0.2).h5"),
]

for j, h in pairs:
    if Path(j).exists() and Path(h).exists():
        with open(j) as f:
            model = model_from_json(f.read())
        model.load_weights(h)
        print(f"Loaded model: {j} + {h}")
        break

if model is None:
    raise RuntimeError("❌ No valid model found")

# ==========================================================
# PARAMETERS
# ==========================================================
actions = np.array([chr(i) for i in range(65, 91)])
sequence_length = 30
threshold = 0.85
consistency_len = 12

# ==========================================================
# CAMERA
# ==========================================================
def open_camera():
    for i in range(4):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return cap
            cap.release()
    raise RuntimeError("Camera not found")

cap = open_camera()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==========================================================
# HANDS
# ==========================================================
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ==========================================================
# STATE
# ==========================================================
sequence = []
predictions = []
confidence_history = deque(maxlen=60)

current_letter = ""
current_confidence = 0.0
recognized_word = ""
last_letter = ""
last_time = 0

# ==========================================================
# UI HELPERS
# ==========================================================
def panel(img, x, y, w, h, color, a=0.25):
    overlay = img.copy()
    cv2.rectangle(overlay, (x,y), (x+w,y+h), color, -1)
    cv2.addWeighted(overlay, a, img, 1-a, 0, img)
    cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)

def confidence_bar(img, x, y, w, h, conf):
    cv2.rectangle(img, (x,y), (x+w,y+h), COLOR_DIM, -1)
    fill = int(w * conf)
    cv2.rectangle(img, (x,y), (x+fill,y+h), COLOR_SUCCESS, -1)
    cv2.rectangle(img, (x,y), (x+w,y+h), COLOR_PRIMARY, 2)
    cv2.putText(img, f"{int(conf*100)}%",
                (x+w+10, y+h-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2, cv2.LINE_AA)

def confidence_wave(img, x, y, w, h, data):
    if len(data) < 2:
        return
    panel(img, x, y, w, h, COLOR_PRIMARY, 0.15)
    maxv = max(data)
    pts = []
    for i, v in enumerate(data):
        px = x + int(i / len(data) * w)
        py = y + h - int((v / maxv) * h)
        pts.append((px, py))
    for i in range(len(pts)-1):
        cv2.line(img, pts[i], pts[i+1], COLOR_SUCCESS, 2, cv2.LINE_AA)

# ==========================================================
# WINDOW
# ==========================================================
cv2.namedWindow("ASL", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("ASL", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("🚀 ASL SYSTEM STARTED — Press Q to Exit")

# ==========================================================
# MAIN LOOP
# ==========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    win = cv2.getWindowImageRect("ASL")
    frame = cv2.resize(frame, (win[2], win[3]))

    h, w, _ = frame.shape
    sx, sy = w/1280, h/720
    Sx = lambda x: int(x*sx)
    Sy = lambda y: int(y*sy)

    overlay = np.full_like(frame, COLOR_BG)
    cv2.addWeighted(frame, 0.55, overlay, 0.45, 0, frame)

    # ================= ROI (1cm bigger each side) =================
    extra = 76
    rw = int(w*0.35) + extra
    rh = int(h*0.5) + extra
    rx = (w-rw)//2
    ry = (h-rh)//2
    roi = frame[ry:ry+rh, rx:rx+rw]

    result = hands.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    keypoints = np.zeros(63)
    detected = False

    if result.multi_hand_landmarks:
        detected = True
        hand = result.multi_hand_landmarks[0]
        keypoints = np.array([[lm.x,lm.y,lm.z] for lm in hand.landmark]).flatten()
        mp_drawing.draw_landmarks(roi, hand, mp_hands.HAND_CONNECTIONS)

    sequence.append(keypoints)
    sequence = sequence[-sequence_length:]

    if len(sequence) == sequence_length:
        res = model.predict(np.expand_dims(sequence,0), verbose=0)[0]
        idx = int(np.argmax(res))
        predictions.append(idx)
        predictions = predictions[-consistency_len:]
        conf = float(np.max(res))
        confidence_history.append(conf)

        if len(set(predictions)) == 1 and conf >= threshold:
            current_letter = actions[idx]
            current_confidence = conf
            last_time = time.time()

    cv2.rectangle(frame, (rx,ry), (rx+rw,ry+rh),
                  COLOR_SUCCESS if detected else COLOR_PRIMARY, 3)

    # ================= TOP BAR =================
    panel(frame, Sx(20), Sy(20), w-Sx(40), Sy(60), COLOR_PRIMARY)
    cv2.putText(frame, "SIGN LANGUAGE RECOGNITION",
                (Sx(40),Sy(55)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9*sy,
                COLOR_TEXT, 2, cv2.LINE_AA)

    cv2.putText(frame, "Press Q to Exit",
                (w-Sx(200),Sy(55)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                COLOR_TEXT, 2, cv2.LINE_AA)

    # ================= LEFT PANEL =================
    panel(frame, Sx(20), Sy(100), Sx(320), Sy(280), COLOR_PRIMARY)
    cv2.putText(frame, "DETECTED SIGN",
                (Sx(40),Sy(130)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2)

    if current_letter and time.time()-last_time < 2:
        cv2.putText(frame, current_letter,
                    (Sx(140),Sy(260)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,
                    COLOR_PRIMARY, 8, cv2.LINE_AA)

    confidence_bar(frame, Sx(40), Sy(330), Sx(240), Sy(25), current_confidence)

    # ================= CONFIDENCE GRAPH =================
    # -------- CONFIDENCE HISTORY GRAPH --------
    cv2.putText(
        frame,
        "CONFIDENCE HISTORY",
        (Sx(30), h - Sy(190)),   # title just above graph
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_TEXT,
        2,
        cv2.LINE_AA
    )

    confidence_wave(
        frame,
        Sx(20),
        h - Sy(180),
        Sx(320),
        Sy(120),
        list(confidence_history)
    )


    # ================= WORD BUILDER =================
    panel(frame, w-Sx(360), Sy(100), Sx(340), Sy(180), COLOR_PRIMARY)
    cv2.putText(frame, "WORD BUILDER",
                (w-Sx(330),Sy(130)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2)

    cv2.putText(frame, recognized_word or "...",
                (w-Sx(330),Sy(190)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                COLOR_PRIMARY, 4, cv2.LINE_AA)

    cv2.putText(frame,
        "[SPACE] Add   [BACK] Delete   [ENTER] Clear",
        (w-Sx(330),Sy(230)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

    cv2.imshow("ASL", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break
    elif key == ord(' ') and current_letter and current_letter != last_letter:
        recognized_word += current_letter
        last_letter = current_letter
    elif key == 8:
        recognized_word = recognized_word[:-1]
    elif key == 13:
        recognized_word = ""
        last_letter = ""

cap.release()
hands.close()
cv2.destroyAllWindows()
print("✅ System closed")
