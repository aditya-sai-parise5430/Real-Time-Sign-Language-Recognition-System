# test_camera_full.py
import cv2
import time

def try_camera(idx, backend=None):
    if backend is None:
        cap = cv2.VideoCapture(idx)
        name = f"index {idx} (default backend)"
    else:
        cap = cv2.VideoCapture(idx, backend)
        name = f"index {idx} (backend {backend})"
    opened = cap.isOpened()
    ok, _ = False, None
    if opened:
        ok, frame = cap.read()
    print(f"{name} -> opened: {opened}, read_ok: {ok}")
    if opened:
        cap.release()
    time.sleep(0.2)

if __name__ == "__main__":
    # Try indices 0..5 with default backend and DirectShow (Windows)
    backends = [None, cv2.CAP_DSHOW]
    for b in backends:
        for i in range(0, 6):   # tries 0..5
            try_camera(i, b)
