import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import cv2
from PIL import Image, ImageTk
try:
    from fer import FER
except ImportError:
    from fer.fer import FER
import threading
import pickle
import numpy as _np
# Try to import facenet-pytorch and torchvision transforms; allow fallback
try:
    from facenet_pytorch import InceptionResnetV1
    import torchvision.transforms as transforms
    _FACENET_AVAILABLE = True
except Exception:
    InceptionResnetV1 = None
    transforms = None
    _FACENET_AVAILABLE = False

import os

class SecurityManager:
    """Inline security manager using facenet-pytorch embeddings.

    If facenet-pytorch isn't available, recognition methods will return None.
    """
    def __init__(self, authorized_dir=None, emb_file="embeddings.pkl", device=None):
        self.authorized_dir = authorized_dir or os.path.join(os.path.dirname(__file__), "authorized")
        os.makedirs(self.authorized_dir, exist_ok=True)
        self.emb_file = os.path.join(self.authorized_dir, emb_file)
        self.embeddings = {}
        self.model = None
        self.transform = None
        if _FACENET_AVAILABLE and InceptionResnetV1 is not None:
            try:
                self.model = InceptionResnetV1(pretrained='vggface2').eval()
                self.transform = transforms.Compose([
                    transforms.Resize((160,160)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                ])
            except Exception:
                self.model = None
                self.transform = None
        self._load_embeddings()

    def _load_embeddings(self):
        if os.path.exists(self.emb_file):
            try:
                with open(self.emb_file, "rb") as f:
                    self.embeddings = pickle.load(f)
            except Exception:
                self.embeddings = {}

    def _save_embeddings(self):
        try:
            with open(self.emb_file, "wb") as f:
                pickle.dump(self.embeddings, f)
        except Exception:
            pass

    def _pil_from_bgr(self, arr):
        if arr is None:
            return None
        import cv2
        from PIL import Image as _PILImage
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return _PILImage.fromarray(rgb)

    def _get_embedding(self, face_bgr):
        # Prefer using torchvision transforms + PIL (when available), but
        # fall back to manual OpenCV->numpy->torch conversion on failure.
        try:
            if self.model is None:
                return None
            # Try torchvision path first
            if self.transform is not None:
                try:
                    img = self._pil_from_bgr(face_bgr)
                    if img is None:
                        return None
                    x = self.transform(img).unsqueeze(0)
                    import torch
                    with torch.no_grad():
                        e = self.model(x).cpu().numpy()[0]
                    e = _np.asarray(e)
                    e = e / (_np.linalg.norm(e) + 1e-10)
                    return e
                except Exception:
                    # fallback to manual path
                    pass

            # Manual fallback: resize with OpenCV, convert to float tensor
            try:
                import cv2
                import torch
                arr = face_bgr
                if arr is None:
                    return None
                # ensure color and size
                arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                arr_rgb = cv2.resize(arr_rgb, (160, 160), interpolation=cv2.INTER_CUBIC)
                arr_f = arr_rgb.astype(_np.float32) / 255.0
                # normalize to [-1,1]
                arr_f = (arr_f - 0.5) / 0.5
                tensor = torch.from_numpy(arr_f).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    e = self.model(tensor).cpu().numpy()[0]
                e = _np.asarray(e)
                e = e / (_np.linalg.norm(e) + 1e-10)
                return e
            except Exception:
                return None
        except Exception:
            return None

    def enroll_from_folder(self, name, folder_path):
        from PIL import Image
        import cv2
        embs = []
        report = {
            'folder': folder_path,
            'total_files': 0,
            'files_opened': 0,
            'embeddings_generated': 0,
            'failed_files': []
        }
        for fn in os.listdir(folder_path):
            p = os.path.join(folder_path, fn)
            if not os.path.isfile(p):
                continue
            report['total_files'] += 1
            try:
                # Use PIL to robustly open many image formats (RGBA, P, GIF, TIFF, etc.)
                with Image.open(p) as im:
                    # If animated (multiple frames), take the first frame
                    try:
                        im.seek(0)
                    except Exception:
                        pass
                    # Convert any mode to RGB (handles L, RGBA, P, CMYK)
                    im = im.convert("RGB")
                    arr = _np.asarray(im)
                    # Convert RGB (PIL) -> BGR (OpenCV) expected by _get_embedding
                    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                report['files_opened'] += 1
            except Exception as ex:
                report['failed_files'].append((fn, str(ex)))
                continue
            emb = self._get_embedding(img)
            if emb is not None:
                embs.append(emb)
        report['embeddings_generated'] = len(embs)
        self.last_enroll_report = report
        if not embs:
            return False
        # store multiple embeddings: keep list per person
        try:
            cur = self.embeddings.get(name, [])
            if not isinstance(cur, list):
                cur = [cur]
            cur.extend(embs)
            self.embeddings[name] = cur
            self._save_embeddings()
            return True
        except Exception:
            return False

    def load_authorized(self):
        for name in os.listdir(self.authorized_dir):
            p = os.path.join(self.authorized_dir, name)
            if os.path.isdir(p) and name not in self.embeddings:
                try:
                    self.enroll_from_folder(name, p)
                except Exception:
                    pass

    def recognize(self, face_bgr, threshold=0.6):
        if not self.embeddings or self.model is None:
            return None, 0.0
        emb = self._get_embedding(face_bgr)
        if emb is None:
            return None, 0.0
        best_name = None
        best_sim = -1.0
        for name, ref in self.embeddings.items():
            # support list of embeddings per person
            if isinstance(ref, list) or isinstance(ref, tuple):
                for r in ref:
                    sim = float(_np.dot(emb, r) / (_np.linalg.norm(emb) * (_np.linalg.norm(r) + 1e-10)))
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name
            else:
                r = ref
                sim = float(_np.dot(emb, r) / (_np.linalg.norm(emb) * (_np.linalg.norm(r) + 1e-10)))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim
import os
from datetime import datetime
import numpy as np
import csv
import time
import math
from concurrent.futures import ThreadPoolExecutor

# Initialize the emotion detection model
emotion_detector_haar = FER(mtcnn=False)
emotion_detector_mtcnn = FER(mtcnn=True)

# Initialize security manager (face recognition)
secmgr = None
if SecurityManager is not None:
    try:
        secmgr = SecurityManager()
        secmgr.load_authorized()
    except Exception:
        secmgr = None

def trigger_alarm(duration=2.0, freq=1000):
    """Play a short alarm beep in background (Windows winsound if available)."""
    def _alarm():
        end = time.time() + duration
        try:
            import winsound
            while time.time() < end:
                winsound.Beep(freq, 300)
                time.sleep(0.1)
        except Exception:
            # Fallback: rapid console output so there's at least some indication
            try:
                import sys
                sys.stdout.write('\a')
                sys.stdout.flush()
            except Exception:
                pass
    threading.Thread(target=_alarm, daemon=True).start()

def log_imposter_event(saved_path, score):
    try:
        logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_emotions")
        os.makedirs(logdir, exist_ok=True)
        logfile = os.path.join(logdir, "alerts.log")
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} IMPROPER_ACCESS {os.path.basename(saved_path)} score={score:.3f}\n")
    except Exception:
        pass

# Global variables for camera feed and current frame
cap = None
current_frame = None
frame_counter = 0
current_emotion = None
current_emotion_score = 0
emotion_history = []
last_annotated_bgr = None
after_job_id = None

class EmotionSmoother:
    def __init__(self, alpha=0.6, keys=None):
        self.alpha = alpha
        self.keys = keys or ["happy", "sad", "angry", "surprise"]
        self.smoothed = {}
    def update(self, probs):
        for k in self.keys:
            v = probs.get(k, 0.0)
            prev = self.smoothed.get(k, v)
            self.smoothed[k] = self.alpha * v + (1 - self.alpha) * prev
        if not self.smoothed:
            return None, 0.0
        dom = max(self.smoothed, key=self.smoothed.get)
        return dom, float(self.smoothed[dom])

smoother = EmotionSmoother()
executor = ThreadPoolExecutor(max_workers=1)
detection_fut = None
last_frame_ts = None
fps = 0.0

EMOJI_MAP = {
    "happy": "🙂",
    "sad": "😢",
    "angry": "😠",
    "surprise": "😮",
    "neutral": "😐",
    "fear": "😨",
    "disgust": "🤢"
}

def get_emoji_for_emotion(emo):
    return EMOJI_MAP.get(emo, "🙂")

class Track:
    def __init__(self, tid, box):
        self.tid = tid
        self.box = box
        self.center = (box[0] + box[2]//2, box[1] + box[3]//2)
        self.smoother = EmotionSmoother(alpha=0.6, keys=list(EMOJI_MAP.keys()))
        self.last_emotion = None
        self.last_score = 0.0
        # recognition state
        self.recognized_name = None
        self.recognized_score = 0.0
        # count consecutive frames seen as unknown
        self.unknown_count = 0
        self.last_seen_ts = time.time()
        self.alerted = False

tracks = {}
next_track_id = 1

# Security tuning parameters (can be changed from Security UI)
security_threshold_var = None
security_unknown_limit_var = None

def open_camera():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open the camera.")
        return

    # Set a lower resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    update_frame()

# Function to perform emotion detection in a single-worker executor
def threaded_emotion_detection(frame):
    global tracks, next_track_id
    img_bgr = prepare_for_detection(frame)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = detect_emotions_with_fallback(img_rgb)
    allowed = {"happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"}
    img_annotated = img_bgr.copy()
    aggregate = {k: 0.0 for k in allowed}
    if results:
        detections = []
        for r in results:
            x, y, w, h = r["box"]
            emotions = r["emotions"]
            filtered = {k: emotions.get(k, 0.0) for k in allowed}
            detections.append(((x, y, w, h), filtered))
        for box, probs in detections:
            cx = box[0] + box[2]//2
            cy = box[1] + box[3]//2
            tid_match = None
            best_d = 9999
            for tid, tr in tracks.items():
                d = (tr.center[0]-cx)**2 + (tr.center[1]-cy)**2
                if d < best_d and d < 80**2:
                    best_d = d
                    tid_match = tid
            if tid_match is None:
                tid = next_track_id
                next_track_id += 1
                tracks[tid] = Track(tid, box)
                tid_match = tid
            tr = tracks[tid_match]
            tr.box = box
            tr.center = (cx, cy)
            dom, score = tr.smoother.update(probs)
            tr.last_emotion, tr.last_score = dom, score
            for k, v in probs.items():
                aggregate[k] += v
            x, y, w, h = box
            # recognition + robust imposter handling
            rect_col = (0, 255, 0)
            label = f"ID{tid_match}:{dom}:{score:.2f}"
            # read security params before recognition
            thresh, unknown_limit = get_security_params()
            try:
                if secmgr is not None:
                    face_roi = img_bgr[y:y+h, x:x+w].copy()
                    recognized_name, recognized_score = secmgr.recognize(face_roi, threshold=thresh)
                else:
                    recognized_name, recognized_score = None, 0.0
            except Exception:
                recognized_name, recognized_score = None, 0.0

            thresh, unknown_limit = get_security_params()
            if recognized_name:
                tr.recognized_name = recognized_name
                tr.recognized_score = recognized_score
                tr.unknown_count = 0
                tr.alerted = False
                label = f"{recognized_name}:{recognized_score:.2f}"
                rect_col = (0, 200, 0)
            else:
                # Increase unknown count and check threshold for imposter
                tr.unknown_count = (tr.unknown_count or 0) + 1
                tr.last_seen_ts = time.time()
                # recognized_score may hold best_sim even when name is None
                if tr.unknown_count >= max(1, unknown_limit):
                    # mark as imposter once
                    if not getattr(tr, 'alerted', False):
                        tr.alerted = True
                        rect_col = (0, 0, 255)
                        label = f"IMPOSTER ({recognized_score:.2f})"
                        # save snapshot and log
                        try:
                            def _save_and_alarm_imp():
                                try:
                                    savedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_emotions")
                                    os.makedirs(savedir, exist_ok=True)
                                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    fname = f"imposter_ID{tid_match}_{ts}.jpg"
                                    fp = os.path.join(savedir, fname)
                                    cv2.imwrite(fp, img_bgr)
                                    log_imposter_event(fp, recognized_score)
                                except Exception:
                                    pass
                                trigger_alarm()
                            threading.Thread(target=_save_and_alarm_imp, daemon=True).start()
                        except Exception:
                            pass
                else:
                    label = f"Unknown:{recognized_score:.2f}"
                    rect_col = (0, 180, 180)

            cv2.rectangle(img_annotated, (x, y), (x + w, y + h), rect_col, 2)
            tx = x + w//2 - 75
            ty = max(0, y - 24)
            cv2.rectangle(img_annotated, (tx, ty), (tx + 150, ty + 22), (0, 0, 0), -1)
            cv2.putText(img_annotated, label, (tx + 6, ty + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            draw_emoji(img_annotated, x, y, w, h, dom)
    agg_dom = None
    agg_score = 0.0
    if any(aggregate.values()):
        agg_dom = max(aggregate, key=aggregate.get)
        agg_score = float(aggregate[agg_dom]/max(1, len(results) or 1))
    root.after(0, apply_detection_result, img_annotated if results else frame, agg_dom, agg_score)

# Apply detection result on Tk main thread
def apply_detection_result(img_annotated, sm_dom, sm_score):
    global last_annotated_bgr, current_emotion, current_emotion_score, emotion_history
    last_annotated_bgr = img_annotated
    if sm_dom:
        current_emotion = sm_dom
        current_emotion_score = sm_score
        emotion_history.append((datetime.now().strftime("%H:%M:%S"), sm_dom, float(f"{sm_score:.2f}")))
        lbl_result.config(text=f"Detected: {sm_dom.capitalize()} {get_emoji_for_emotion(sm_dom)} ({sm_score:.2f})", fg="blue")
    else:
        current_emotion = None
        current_emotion_score = 0
        lbl_result.config(text="No emotion detected", fg="red")

# Function to update the camera feed and perform emotion detection
def update_frame():
    global current_frame, cap, frame_counter, last_annotated_bgr, after_job_id, detection_fut, last_frame_ts, fps
    if not cap or not cap.isOpened():
        lbl_result.config(text="Camera stopped", fg="orange")
        return
    ret, frame = cap.read()
    now = time.time()
    if ret:
        frame_counter += 1
        if last_frame_ts:
            dt = now - last_frame_ts
            if dt > 0:
                fps = (0.9 * fps + 0.1 * (1.0 / dt)) if fps else (1.0 / dt)
        last_frame_ts = now
        current_frame = frame
        display_src = last_annotated_bgr if last_annotated_bgr is not None else frame
        display_bgr = apply_display_filter(display_src)
        img_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
        if detection_fut is None or detection_fut.done():
            detection_fut = executor.submit(threaded_emotion_detection, frame.copy())
        if 'lbl_status' in globals():
            lbl_status.config(text=f"Frames: {frame_counter}  FPS: {fps:.1f}")
    after_job_id = lbl_video.after(50, update_frame)

# Function to detect emotions in a frame
def detect_emotion(frame):
    emotion_data = emotion_detector_haar.top_emotion(frame)
    return emotion_data


def prepare_for_detection(img_bgr):
    # Normalize channels
    if len(img_bgr.shape) == 2:
        # Grayscale → apply CLAHE for contrast, then convert to BGR
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_bgr)
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 1:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    # Boost contrast for color images using CLAHE on L channel
    if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Upscale if the image is small (helps face detection)
    h, w = img_bgr.shape[:2]
    min_side = min(h, w)
    if min_side < 160:
        scale = 160.0 / min_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return img_bgr

def apply_display_filter(img_bgr):
    f = current_filter
    img = img_bgr.copy()
    if f == 'None':
        return img
    elif f == 'Grayscale':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif f == 'Sepia':
        kernel = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
        img = np.clip(img.dot(kernel.T), 0, 255).astype(np.uint8)
    elif f == 'Blur':
        img = cv2.GaussianBlur(img, (9,9), 0)
    elif f == 'Sketch':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21,21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        img = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    elif f == 'Brightness+':
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=30)
    elif f == 'Contrast+':
        img = cv2.convertScaleAbs(img, alpha=1.3, beta=0)
    elif f == 'Beauty':
        sm = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        s = cv2.add(s, 10)
        v = cv2.add(v, 10)
        hsv = cv2.merge((h,s,v))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif f == 'Cartoon':
        color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        med = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(med, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(color, edges_col)
    elif f == 'Vignette':
        rows, cols = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/2)
        kernel_y = cv2.getGaussianKernel(rows, rows/2)
        mask = kernel_y * kernel_x.T
        mask = mask / mask.max()
        img = (img * mask[..., None]).astype(np.uint8)
    elif f == 'Pixelate':
        h, w = img.shape[:2]
        scale = max(1, min(h, w)//32)
        small = cv2.resize(img, (max(1,w//scale), max(1,h//scale)), interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    elif f == 'VHS':
        h, w = img.shape[:2]
        scan = np.zeros_like(img)
        scan[::2, :, :] = 20
        shifted = np.roll(img, 2, axis=1)
        img = cv2.addWeighted(img, 0.8, shifted, 0.2, 0)
        img = cv2.add(img, scan)
    elif f == 'Warm':
        b,g,r = cv2.split(img)
        r = cv2.add(r, 20)
        b = cv2.subtract(b, 10)
        img = cv2.merge((b,g,r))
    elif f == 'Cool':
        b,g,r = cv2.split(img)
        b = cv2.add(b, 20)
        r = cv2.subtract(r, 10)
        img = cv2.merge((b,g,r))
    elif f == 'Invert':
        img = cv2.bitwise_not(img)
    elif f == 'Mirror':
        img = cv2.flip(img, 1)
    return img

def detect_emotions_with_fallback(img_rgb):
    # Try MTCNN first (better on some images), then Haar
    try:
        results = emotion_detector_mtcnn.detect_emotions(img_rgb)
    except Exception:
        results = None

    if not results:
        try:
            results = emotion_detector_haar.detect_emotions(img_rgb)
        except Exception:
            results = None

    return results or []

def annotate_emotions(img_bgr):
    img_bgr = prepare_for_detection(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = detect_emotions_with_fallback(img_rgb)
    if not results:
        return img_bgr, None, 0.0

    allowed = {"happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"}
    img_annotated = img_bgr.copy()
    best_probs = None
    top_score = 0.0

    for r in results:
        x, y, w, h = r["box"]
        emotions = r["emotions"]

        # Prefer target emotions, fallback to any available
        filtered = {k: v for k, v in emotions.items() if k in allowed}
        target = filtered if filtered else emotions

        local_emotion = max(target, key=target.get)
        local_score = target[local_emotion]

        cv2.rectangle(img_annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{local_emotion}:{local_score:.2f}"
        bx, by = x, max(0, y - 20)
        cv2.rectangle(img_annotated, (bx, by), (bx + 150, by + 22), (0, 0, 0), -1)
        cv2.putText(img_annotated, label, (bx + 6, by + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        draw_emoji(img_annotated, x, y, w, h, local_emotion)
        if local_score > top_score:
            dominant = local_emotion
            top_score = local_score

    return img_annotated, dominant, top_score

def update_trend_canvas():
    if not hasattr(trend_canvas, "winfo_exists") or not trend_canvas.winfo_exists():
        return
    trend_canvas.delete("all")
    max_items = 60
    data = emotion_history[-max_items:]
    width = 600
    height = 120
    if len(data) < 2:
        if data:
            trend_canvas.create_text(10, 10, anchor="nw", text=f"{data[-1][1].capitalize()} {data[-1][2]:.2f}")
        trend_canvas.after(1000, update_trend_canvas)
        return
    scores = [s for _, _, s in data]
    xs = [int(i * width / max_items) for i in range(len(data))]
    ma = []
    w = 5
    for i in range(len(scores)):
        start = max(0, i - w + 1)
        ma.append(sum(scores[start:i+1]) / (i - start + 1))
    for i in range(len(ma) - 1):
        y0 = height - int(ma[i] * height)
        y1 = height - int(ma[i+1] * height)
        trend_canvas.create_line(xs[i], y0, xs[i+1], y1, fill="#666", width=2)
    last_emo = data[-1][1]
    color = {"happy": "#4CAF50", "sad": "#2196F3", "angry": "#f44336", "surprise": "#FFA500"}.get(last_emo, "#9E9E9E")
    trend_canvas.create_text(10, 10, anchor="nw", text=f"{last_emo.capitalize()} {data[-1][2]:.2f}", fill=color)
    trend_canvas.after(1000, update_trend_canvas)

def export_session_csv():
    if not emotion_history:
        messagebox.showerror("Error", "No session data to export.")
        return
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_emotions")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "emotion", "score"])
        writer.writerows(emotion_history)
    messagebox.showinfo("Session Exported", f"Saved: {filename}\nLocation: {save_dir}")

def reset_session():
    global emotion_history, last_annotated_bgr
    emotion_history = []
    last_annotated_bgr = None
    lbl_result.config(text="Session reset", fg="orange")
    trend_canvas.delete("all")


def draw_emoji(img, x, y, w, h, emotion):
    size = max(28, min(64, int(min(w, h) * 0.35)))
    phase = time.time()*2.0 + (x+y)*0.01
    s = 1.0 + 0.08*math.sin(phase)
    r = int(size*s)//2
    cx = x + w - r
    cy = y + r + int(3*math.sin(phase*1.5))
    color_map = {"happy": (255, 215, 0), "sad": (70, 130, 180), "angry": (220, 20, 60), "surprise": (255, 140, 0)}
    col = color_map.get(emotion, (180, 180, 180))
    x0 = max(0, cx - r)
    y0 = max(0, cy - r)
    x1 = min(img.shape[1], cx + r)
    y1 = min(img.shape[0], cy + r)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    rcx = cx - x0
    rcy = cy - y0
    overlay = roi.copy()
    cv2.circle(overlay, (rcx, rcy), min(r, overlay.shape[1]//2, overlay.shape[0]//2), col, -1)
    cv2.addWeighted(overlay, 0.7, roi, 0.3, 0, roi)
    hl = roi.copy()
    cv2.circle(hl, (int(rcx - 0.35*r), int(rcy - 0.35*r)), max(2, r//5), (255, 255, 255), -1)
    cv2.addWeighted(hl, 0.25, roi, 0.75, 0, roi)
    sh = roi.copy()
    cv2.circle(sh, (rcx, int(rcy + 0.6*r)), max(2, r//3), (0, 0, 0), -1)
    cv2.addWeighted(sh, 0.12, roi, 0.88, 0, roi)
    eye_offset = int(r*0.6)
    eye_r = max(2, r//10)
    evy = rcy - int(0.5*eye_offset) + int(1*math.sin(phase*3))
    cv2.circle(roi, (rcx - int(0.6*eye_offset), evy), eye_r, (0, 0, 0), -1)
    cv2.circle(roi, (rcx + int(0.6*eye_offset), evy), eye_r, (0, 0, 0), -1)
    if emotion == "happy":
        cv2.ellipse(roi, (rcx, rcy + int(0.2*eye_offset)), (int(r*0.35), int(r*0.18)), 0, 0, 180, (0, 0, 0), 2)
    elif emotion == "sad":
        cv2.ellipse(roi, (rcx, rcy + int(0.2*eye_offset)), (int(r*0.35), int(r*0.18)), 0, 180, 360, (0, 0, 0), 2)
    elif emotion == "angry":
        cv2.line(roi, (rcx - int(eye_offset*0.7), evy - 6), (rcx - int(eye_offset*0.3), evy - 12), (0, 0, 0), 2)
        cv2.line(roi, (rcx + int(eye_offset*0.7), evy - 6), (rcx + int(eye_offset*0.3), evy - 12), (0, 0, 0), 2)
        cv2.line(roi, (rcx - int(r*0.25), rcy + int(0.2*eye_offset)), (rcx + int(r*0.25), rcy + int(0.2*eye_offset)), (0, 0, 0), 2)
    elif emotion == "surprise":
        cv2.circle(roi, (rcx, rcy + int(0.2*eye_offset)), max(2, r//6), (0, 0, 0), 2)
    else:
        cv2.line(roi, (rcx - int(r*0.25), rcy + int(0.2*eye_offset)), (rcx + int(r*0.25), rcy + int(0.2*eye_offset)), (0, 0, 0), 2)


def init_soft_gradient():
    global bg_canvas
    bg_canvas = tk.Canvas(root, highlightthickness=0, bd=0)
    bg_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
    tk.Misc.lower(bg_canvas)
    draw_soft_gradient()
    root.bind("<Configure>", lambda e: draw_soft_gradient())


def draw_soft_gradient():
    width = root.winfo_width()
    height = root.winfo_height()
    bg_canvas.delete("all")
    top = "#f5f7fb"
    bottom = "#e9edf5"
    bands = 60
    for i in range(bands):
        y0 = int(i * height / bands)
        y1 = int((i + 1) * height / bands)
        t = i / max(1, bands - 1)
        rt = int(int(top[1:3], 16) * (1 - t) + int(bottom[1:3], 16) * t)
        gt = int(int(top[3:5], 16) * (1 - t) + int(bottom[3:5], 16) * t)
        bt = int(int(top[5:7], 16) * (1 - t) + int(bottom[5:7], 16) * t)
        color = f"#{rt:02x}{gt:02x}{bt:02x}"
        bg_canvas.create_rectangle(0, y0, width, y1, fill=color, outline="")

# Function to save the current frame with emotion label
def save_image():
    global current_frame, current_emotion, current_emotion_score
    
    if current_frame is None:
        messagebox.showerror("Error", "No image to save. Please start the camera first.")
        return
    
    # Create a directory to save images if it doesn't exist
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_emotions")
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate a filename with timestamp and emotion
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    emotion_text = current_emotion if current_emotion else "unknown"
    filename = f"{emotion_text}_{timestamp}.jpg"
    filepath = os.path.join(save_dir, filename)
    
    # Add emotion text to the image
    img_with_text = current_frame.copy()
    if current_emotion:
        text = f"{current_emotion.capitalize()}: {current_emotion_score:.2f}"
        cv2.putText(img_with_text, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save the image
    cv2.imwrite(filepath, img_with_text)
    
    messagebox.showinfo("Image Saved", f"Image saved as {filename}\nEmotion: {emotion_text.capitalize()}\nLocation: {save_dir}")

# Function to browse and select an image for emotion detection
def browse_image():
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    
    if file_path:
        process_image(file_path)

# Function to process the selected image and detect emotions
def process_image(file_path):
    global current_emotion, current_emotion_score
    
    try:
        # Read the image robustly (handles grayscale/RGBA/RGB)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("Error", "Failed to load the image.")
            return

        # Detect and annotate emotions (handles all channel types inside)
        img_annotated, dominant_emotion, emotion_score = annotate_emotions(img)

        # Convert to RGB for Tkinter preview safely
        if img_annotated.ndim == 2:
            img_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(img_rgb)

        # Resize image if it's too large for UI
        max_width = 600
        if img_pil.width > max_width:
            ratio = max_width / img_pil.width
            new_height = int(img_pil.height * ratio)
            img_pil = img_pil.resize((max_width, new_height), Image.LANCZOS)

        # Show annotated preview
        imgtk = ImageTk.PhotoImage(image=img_pil)
        lbl_scanned_image.imgtk = imgtk
        lbl_scanned_image.configure(image=imgtk)

        # Update status
        if dominant_emotion:
            current_emotion = dominant_emotion
            current_emotion_score = emotion_score
            lbl_scan_result.config(
                text=f"Detected: {dominant_emotion.capitalize()} ({emotion_score:.2f})",
                fg="blue",
            )
        else:
            current_emotion = None
            current_emotion_score = 0
            lbl_scan_result.config(text="No emotion detected", fg="red")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Function to save the processed image with emotion label
def save_scanned_image():
    if not hasattr(lbl_scanned_image, "imgtk") or lbl_scanned_image.imgtk is None:
        messagebox.showerror("Error", "No image to save. Please scan an image first.")
        return

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_emotions")
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    emotion_text = current_emotion if current_emotion else "unknown"
    filename = f"scanned_{emotion_text}_{timestamp}.jpg"
    filepath = os.path.join(save_dir, filename)

    # Convert PhotoImage to PIL Image, then to OpenCV with robust channel handling
    img_pil = ImageTk.getimage(lbl_scanned_image.imgtk)
    arr = np.array(img_pil)

    if arr.ndim == 2:
        # Grayscale
        img_cv = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        # RGBA
        img_cv = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        # RGB
        img_cv = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    if current_emotion:
        text = f"{current_emotion.capitalize()}: {current_emotion_score:.2f}"
        cv2.putText(img_cv, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite(filepath, img_cv)

    display_emotion = emotion_text.capitalize() if emotion_text != "unknown" else "Unknown"
    messagebox.showinfo("Image Saved", f"Image saved as {filename}\nEmotion: {display_emotion}\nLocation: {save_dir}")

# Function to close the camera and cleanup
def close_camera():
    global cap, after_job_id, last_annotated_bgr
    if cap:
        cap.release()
        cap = None
    if after_job_id:
        try:
            lbl_video.after_cancel(after_job_id)
        except Exception:
            pass
        after_job_id = None
    last_annotated_bgr = None
    lbl_video.config(image="")
    lbl_result.config(text="Camera Closed", fg="green")

# Function to display the emotion detection app after successful login
def open_emotion_detection_app():
    # Hide the login window
    login_frame.pack_forget()
    # Show the main menu
    show_main_menu()

def hide_all_frames():
    """Hide all top-level app frames so only the target frame is visible."""
    for f in [login_frame, main_menu_frame, camera_frame, image_scan_frame, security_frame]:
        try:
            f.pack_forget()
        except Exception:
            pass

# Function to show the main menu
def show_main_menu():
    hide_all_frames()
    main_menu_frame.pack(fill="both", expand=True)

# Function to show the camera section
def show_camera_section():
    hide_all_frames()
    camera_frame.pack(fill="both", expand=True)

# Function to show the image scanning section
def show_image_scan_section():
    hide_all_frames()
    image_scan_frame.pack(fill="both", expand=True)

def show_security_section():
    hide_all_frames()
    security_frame.pack(fill="both", expand=True)

def enroll_from_folder_ui():
    if secmgr is None:
        messagebox.showerror("Error", "Security manager not available (missing dependencies).")
        return
    name = simpledialog.askstring("Enroll Person", "Enter person name:")
    if not name:
        return
    folder = filedialog.askdirectory(title=f"Select images folder for {name}")
    if not folder:
        return
    try:
        ok = secmgr.enroll_from_folder(name, folder)
        # show detailed report if available
        rpt = getattr(secmgr, 'last_enroll_report', None)
        if ok:
            try:
                if 'enrolled_label' in globals():
                    enrolled_label.config(text=f"Enrolled: {len(secmgr.embeddings)}")
            except Exception:
                pass
            if rpt is not None:
                messagebox.showinfo("Enrolled", f"Person '{name}' enrolled.\nFiles: {rpt['total_files']}\nOpened: {rpt['files_opened']}\nEmbeddings: {rpt['embeddings_generated']}")
            else:
                messagebox.showinfo("Enrolled", f"Person '{name}' enrolled from {folder}.")
        else:
            if rpt is not None:
                fail_msg = f"No embeddings created. Files scanned: {rpt['total_files']}. Opened: {rpt['files_opened']}."
                if rpt['failed_files']:
                    sample = rpt['failed_files'][:5]
                    fail_msg += "\nFailed files: " + ", ".join([n for n,_ in sample])
                messagebox.showerror("Failed", fail_msg)
            else:
                messagebox.showerror("Failed", "No valid images found or embedding failed.")
    except Exception as e:
        messagebox.showerror("Error", f"Enrollment failed: {e}")

def load_authorized_ui():
    if secmgr is None:
        messagebox.showerror("Error", "Security manager not available.")
        return
    secmgr.load_authorized()
    # update enrolled label if available
    try:
        if 'enrolled_label' in globals():
            enrolled_label.config(text=f"Enrolled: {len(secmgr.embeddings)}")
    except Exception:
        pass
    messagebox.showinfo("Loaded", "Authorized folders scanned and embeddings loaded.")

def list_authorized_ui():
    if secmgr is None:
        messagebox.showerror("Error", "Security manager not available.")
        return
    names = list(secmgr.embeddings.keys())
    if not names:
        messagebox.showinfo("Authorized", "No authorized people enrolled.")
        return
    messagebox.showinfo("Authorized", "\n".join(names))

def remove_authorized_ui():
    if secmgr is None:
        messagebox.showerror("Error", "Security manager not available.")
        return
    name = simpledialog.askstring("Remove Person", "Enter person name to remove:")
    if not name:
        return
    if name in secmgr.embeddings:
        del secmgr.embeddings[name]
        secmgr._save_embeddings()
        messagebox.showinfo("Removed", f"Removed '{name}' from authorized list.")
    else:
        messagebox.showerror("Not found", f"'{name}' is not enrolled.")

def test_current_frame_ui():
    if secmgr is None:
        messagebox.showerror("Error", "Security manager not available.")
        return
    if current_frame is None:
        messagebox.showerror("Error", "No current frame. Start camera first.")
        return
    # prefer current tracked faces
    if tracks:
        tid, tr = next(iter(tracks.items()))
        x, y, w, h = tr.box
        try:
            face = current_frame[y:y+h, x:x+w].copy()
        except Exception:
            messagebox.showerror("Error", "Failed to crop face from frame.")
            return
        name, score = secmgr.recognize(face)
        if name:
            messagebox.showinfo("Recognized", f"{name} ({score:.2f})")
        else:
            messagebox.showinfo("Unknown", f"No match (best score {score:.2f})")
    else:
        messagebox.showerror("Error", "No faces tracked yet. Ensure someone is in frame.")

def get_security_params():
    try:
        thresh = float(security_threshold_var.get()) if security_threshold_var is not None else 0.6
    except Exception:
        thresh = 0.6
    try:
        limit = int(security_unknown_limit_var.get()) if security_unknown_limit_var is not None else 5
    except Exception:
        limit = 5
    return thresh, limit

def enroll_from_camera_ui():
    if secmgr is None:
        messagebox.showerror("Error", "Security manager not available (missing dependencies).")
        return
    name = simpledialog.askstring("Enroll From Camera", "Enter person name:")
    if not name:
        return
    samples = simpledialog.askinteger("Samples", "Number of samples to capture:", initialvalue=10, minvalue=3, maxvalue=50)
    if not samples:
        return
    # Ensure camera is open and visible for user feedback
    try:
        open_camera()
        show_camera_section()
    except Exception:
        pass
    # Give a short moment for frames to start arriving
    time.sleep(0.2)
    # start background capture
    threading.Thread(target=_enroll_from_camera_thread, args=(name, samples), daemon=True).start()

def _enroll_from_camera_thread(name, samples):
    # Try to use live `current_frame` if camera already running, else open temporary capture
    use_temp_cap = False
    cap_local = None
    try:
        if 'cap' in globals() and cap is not None and cap.isOpened():
            cap_local = None
        else:
            cap_local = cv2.VideoCapture(0)
            use_temp_cap = True
    except Exception:
        cap_local = None
        use_temp_cap = False

    # quick check: ensure embedding model exists
    if getattr(secmgr, 'model', None) is None:
        root.after(0, lambda: messagebox.showerror("Error", "Face embedding model not available. Ensure torch/facenet-pytorch are installed and compatible."))
        return

    collected = 0
    embeddings = []
    savedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "authorized", name)
    os.makedirs(savedir, exist_ok=True)
    failures = []
    attempts = 0
    max_attempts = samples * 10
    while collected < samples and attempts < max_attempts:
        attempts += 1
        # get frame
        frame = None
        if cap_local is not None and use_temp_cap:
            ret, f = cap_local.read()
            if not ret:
                time.sleep(0.2)
                continue
            frame = f
        else:
            # use shared current_frame
            frame = globals().get('current_frame', None)
            if frame is None:
                time.sleep(0.2)
                continue

        try:
            if frame is None:
                time.sleep(0.1)
                continue
            img_bgr = prepare_for_detection(frame.copy())
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = detect_emotions_with_fallback(img_rgb)
            if not results:
                time.sleep(0.2)
                continue
            # pick largest face
            best = max(results, key=lambda r: r['box'][2]*r['box'][3])
            x, y, w, h = best['box']
            # expand box slightly
            pad = int(0.2 * max(w, h))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(img_bgr.shape[1], x + w + pad)
            y1 = min(img_bgr.shape[0], y + h + pad)
            face = img_bgr[y0:y1, x0:x1].copy()
            # compute embedding
            emb = None
            try:
                emb = secmgr._get_embedding(face)
            except Exception:
                emb = None
            if emb is not None:
                embeddings.append(emb)
                # save sample image
                try:
                    fn = os.path.join(savedir, f"sample_{collected+1}.jpg")
                    cv2.imwrite(fn, face)
                except Exception:
                    pass
                collected += 1
                # update UI
                try:
                    root.after(0, lambda c=collected, s=samples: lbl_sec_status.config(text=f"Captured {c}/{s} samples"))
                except Exception:
                    pass
            else:
                failures.append('no_emb')
            time.sleep(0.4)
        except Exception as e:
            failures.append(str(e))
            time.sleep(0.2)

    # close temp cap if used
    try:
        if cap_local is not None and use_temp_cap:
            cap_local.release()
    except Exception:
        pass

    # save embeddings to manager
    added = 0
    if embeddings:
        try:
            cur = secmgr.embeddings.get(name, [])
            if not isinstance(cur, list):
                cur = [cur]
            cur.extend(embeddings)
            secmgr.embeddings[name] = cur
            secmgr._save_embeddings()
            added = len(embeddings)
        except Exception:
            added = 0

    # update enrolled label
    try:
        root.after(0, lambda: enrolled_label.config(text=f"Enrolled: {len(secmgr.embeddings)}"))
    except Exception:
        pass

    # final dialog
    if added > 0:
        root.after(0, lambda: messagebox.showinfo("Enrolled", f"Captured {added} embeddings for {name}."))
    else:
        msg = f"No embeddings captured for {name}. Attempts: {attempts}."
        if failures:
            msg += " Check camera or image quality."
        root.after(0, lambda: messagebox.showerror("Failed", msg))

def open_security_camera():
    """Open the main camera and switch to camera view so security detection runs."""
    # Start camera (if not running) and navigate to camera section
    try:
        open_camera()
    except Exception:
        pass
    show_camera_section()

# Function to handle user login
def login():
    username = entry_username.get()
    password = entry_password.get()

    # Simple hardcoded check for demonstration purposes
    if username == "Amlan" and password == "amlan123":
        messagebox.showinfo("Login Success", "Welcome!")
        open_emotion_detection_app()
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

# Function to go back to the login page
def go_back_to_login():
    # Hide all frames
    hide_all_frames()
    
    # Close camera if open
    close_camera()
    
    # Show the login frame again
    login_frame.pack()

# Function to go back to the main menu
def go_back_to_main_menu():
    # Hide current frame and show main menu
    try:
        close_camera()
    except Exception:
        pass
    hide_all_frames()
    show_main_menu()

# Keyboard and quick control helpers
def toggle_camera():
    if cap and hasattr(cap, "isOpened") and cap.isOpened():
        close_camera()
    else:
        open_camera()

def init_global_shortcuts():
    root.bind("<Escape>", lambda e: go_back_to_main_menu())
    root.bind("<space>", lambda e: toggle_camera())
    root.bind("s", lambda e: save_image())
    root.bind("e", lambda e: export_session_csv())
    root.bind("r", lambda e: reset_session())

current_filter = 'None'

def on_filter_change(*args):
    global current_filter
    current_filter = filter_var.get()

# Main application window
root = tk.Tk()
root.title("Advanced Emotion Detection App")
root.geometry("1000x700")
root.configure(bg='#f2f2f2')
init_soft_gradient()
init_global_shortcuts()

# Import numpy for image processing
import numpy as np

# Login page UI
login_frame = tk.Frame(root, bg='#f2f2f2')

lbl_login_title = tk.Label(login_frame, text="User Login", font=("Arial", 24), bg='#f2f2f2', fg='#333')
lbl_login_title.pack(pady=20)

lbl_username = tk.Label(login_frame, text="Username:", font=("Arial", 14), bg='#f2f2f2', fg='#333')
lbl_username.pack(pady=10)
entry_username = tk.Entry(login_frame, font=("Arial", 14))
entry_username.insert(0, "Amlan")  # Pre-fill for convenience
entry_username.pack(pady=10)

lbl_password = tk.Label(login_frame, text="Password:", font=("Arial", 14), bg='#f2f2f2', fg='#333')
lbl_password.pack(pady=10)
entry_password = tk.Entry(login_frame, show="*", font=("Arial", 14))
entry_password.insert(0, "amlan123")  # Pre-fill for convenience
entry_password.pack(pady=10)

btn_login = tk.Button(login_frame, text="Login", font=("Arial", 14), command=login, bg='#4CAF50', fg='white')
btn_login.pack(pady=20)

# Main menu UI (after login)
main_menu_frame = tk.Frame(root, bg='#f2f2f2')

lbl_main_title = tk.Label(main_menu_frame, text="Emotion Detection App", font=("Arial", 24), bg='#f2f2f2', fg='#333')
lbl_main_title.pack(pady=20)

lbl_main_description = tk.Label(main_menu_frame, text="Choose an option below:", font=("Arial", 16), bg='#f2f2f2', fg='#666')
lbl_main_description.pack(pady=20)

btn_camera_section = tk.Button(main_menu_frame, text="Real-time Camera Detection", font=("Arial", 14), 
                               command=show_camera_section, bg='#4CAF50', fg='white', width=25, height=2)
btn_camera_section.pack(pady=15)

btn_image_scan_section = tk.Button(main_menu_frame, text="Scan Image for Emotions", font=("Arial", 14), 
                                  command=show_image_scan_section, bg='#2196F3', fg='white', width=25, height=2)
btn_image_scan_section.pack(pady=15)

btn_security_section = tk.Button(main_menu_frame, text="Security Management", font=("Arial", 14),
                                 command=show_security_section, bg='#607D8B', fg='white', width=25, height=2)
btn_security_section.pack(pady=15)

btn_main_logout = tk.Button(main_menu_frame, text="Logout", font=("Arial", 14), 
                           command=go_back_to_login, bg='#f44336', fg='white', width=15)
btn_main_logout.pack(pady=30)

# Camera section UI
camera_frame = tk.Frame(root, bg='#f2f2f2')

lbl_title = tk.Label(camera_frame, text="Real-time Emotion Detection", font=("Arial", 24), bg='#f2f2f2', fg='#333')
lbl_title.pack(pady=20)

lbl_description = tk.Label(camera_frame, text="This app detects various emotions like happy, sad, angry, etc.\n"
                                      "Simply look into the camera and see your mood analyzed in real-time.",
                                      font=("Arial", 14), bg='#f2f2f2', fg='#666', justify="center")
lbl_description.pack(pady=10)

# Video display area
lbl_video = tk.Label(camera_frame, bg='#ddd')
lbl_video.pack(pady=20)

# Result area
lbl_result = tk.Label(camera_frame, text="Press 'Start Camera' to begin", font=("Arial", 16), bg='#f2f2f2', fg='blue')
lbl_result.pack(pady=10)

trend_canvas = tk.Canvas(camera_frame, width=600, height=120, bg='#fff', highlightthickness=1, highlightbackground='#ddd')
trend_canvas.pack(pady=10)
update_trend_canvas()
status_frame = tk.Frame(camera_frame, bg='#f2f2f2')
status_frame.pack(pady=5)
lbl_status = tk.Label(status_frame, text='Ready', font=('Arial', 12), bg='#f2f2f2', fg='#666')
lbl_status.pack()

camera_buttons_frame = tk.Frame(camera_frame, bg='#f2f2f2')
camera_buttons_frame.pack(pady=10)

btn_open_camera = tk.Button(camera_buttons_frame, text="Start Camera", font=("Arial", 14), command=open_camera, bg='#4CAF50', fg='white')
btn_open_camera.pack(side="left", padx=10)

btn_save_image = tk.Button(camera_buttons_frame, text="Save Image", font=("Arial", 14), command=save_image, bg='#2196F3', fg='white')
btn_save_image.pack(side="left", padx=10)

btn_export_session = tk.Button(camera_buttons_frame, text="Export Session", font=("Arial", 14), command=export_session_csv, bg='#9C27B0', fg='white')
btn_export_session.pack(side="left", padx=10)

btn_reset_session = tk.Button(camera_buttons_frame, text="Reset Session", font=("Arial", 14), command=reset_session, bg='#795548', fg='white')
btn_reset_session.pack(side="left", padx=10)

filter_var = tk.StringVar(value='None')
filter_options = ['None','Beauty','Cartoon','Vignette','Pixelate','VHS','Warm','Cool','Grayscale','Sepia','Sketch','Blur','Brightness+','Contrast+','Invert','Mirror']
opt_filter = tk.OptionMenu(camera_buttons_frame, filter_var, *filter_options)
opt_filter.config(font=("Arial", 12))
opt_filter.pack(side="left", padx=10)
filter_var.trace_add('write', on_filter_change)

btn_camera_back_inline = tk.Button(camera_buttons_frame, text="Back to Menu", font=("Arial", 14), command=go_back_to_main_menu, bg='#FFA500', fg='white')
btn_camera_back_inline.pack(side="left", padx=10)

btn_camera_back = tk.Button(camera_frame, text="Back to Menu", font=("Arial", 14), command=go_back_to_main_menu, bg='#FFA500', fg='white')
btn_camera_back.pack(pady=20)

# Image scanning section UI
image_scan_frame = tk.Frame(root, bg='#f2f2f2')

lbl_scan_title = tk.Label(image_scan_frame, text="Image Emotion Scanner", font=("Arial", 24), bg='#f2f2f2', fg='#333')
lbl_scan_title.pack(pady=20)

lbl_scan_description = tk.Label(image_scan_frame, text="Upload an image to detect emotions in faces.\n"
                                         "The app will analyze the image and show the detected emotions.",
                                         font=("Arial", 14), bg='#f2f2f2', fg='#666', justify="center")
lbl_scan_description.pack(pady=10)

# Image display area
lbl_scanned_image = tk.Label(image_scan_frame, bg='#ddd', width=80, height=20)
lbl_scanned_image.pack(pady=20)

# Result area for scanned image
lbl_scan_result = tk.Label(image_scan_frame, text="Upload an image to begin", font=("Arial", 16), bg='#f2f2f2', fg='blue')
lbl_scan_result.pack(pady=10)

# Control buttons container for image scanning
scan_buttons_frame = tk.Frame(image_scan_frame, bg='#f2f2f2')
scan_buttons_frame.pack(pady=10)

# Control buttons for image scanning
btn_browse_image = tk.Button(scan_buttons_frame, text="Browse Image", font=("Arial", 14), command=browse_image, bg='#4CAF50', fg='white')
btn_browse_image.pack(side="left", padx=10)

btn_save_scanned_image = tk.Button(scan_buttons_frame, text="Save Result", font=("Arial", 14), command=save_scanned_image, bg='#2196F3', fg='white')
btn_save_scanned_image.pack(side="left", padx=10)

# Back button to go to the main menu from image scanning section
btn_scan_back = tk.Button(image_scan_frame, text="Back to Menu", font=("Arial", 14), command=go_back_to_main_menu, bg='#FFA500', fg='white')
btn_scan_back.pack(pady=20)

# Security management UI
security_frame = tk.Frame(root, bg='#f2f2f2')

lbl_sec_title = tk.Label(security_frame, text="Security Management", font=("Arial", 24), bg='#f2f2f2', fg='#333')
lbl_sec_title.pack(pady=20)

lbl_sec_description = tk.Label(security_frame, text="Manage authorized people and test recognition.", font=("Arial", 14), bg='#f2f2f2', fg='#666')
lbl_sec_description.pack(pady=10)

sec_buttons_frame = tk.Frame(security_frame, bg='#f2f2f2')
sec_buttons_frame.pack(pady=20)

btn_enroll_folder = tk.Button(sec_buttons_frame, text="Enroll From Folder", font=("Arial", 12), command=enroll_from_folder_ui, bg='#4CAF50', fg='white')
btn_enroll_folder.pack(side='left', padx=8)

btn_enroll_camera = tk.Button(sec_buttons_frame, text="Enroll From Camera", font=("Arial", 12), command=lambda: enroll_from_camera_ui(), bg='#3EA055', fg='white')
btn_enroll_camera.pack(side='left', padx=8)

btn_load_authorized = tk.Button(sec_buttons_frame, text="Load Authorized", font=("Arial", 12), command=load_authorized_ui, bg='#2196F3', fg='white')
btn_load_authorized.pack(side='left', padx=8)

btn_list_authorized = tk.Button(sec_buttons_frame, text="List Authorized", font=("Arial", 12), command=list_authorized_ui, bg='#9C27B0', fg='white')
btn_list_authorized.pack(side='left', padx=8)

btn_remove_authorized = tk.Button(sec_buttons_frame, text="Remove Authorized", font=("Arial", 12), command=remove_authorized_ui, bg='#f44336', fg='white')
btn_remove_authorized.pack(side='left', padx=8)

btn_test_frame = tk.Button(security_frame, text="Test Current Frame", font=("Arial", 14), command=test_current_frame_ui, bg='#795548', fg='white')
btn_test_frame.pack(pady=10)

btn_open_sec_camera = tk.Button(security_frame, text="Open Camera (Security)", font=("Arial", 14), command=open_security_camera, bg='#4CAF50', fg='white')
btn_open_sec_camera.pack(pady=6)

# Security settings
settings_frame = tk.Frame(security_frame, bg='#f2f2f2')
settings_frame.pack(pady=6)

# Threshold for recognition (0..1)
security_threshold_var = tk.DoubleVar(value=0.6)
lbl_thresh = tk.Label(settings_frame, text="Recognition threshold:", bg='#f2f2f2')
lbl_thresh.pack(side='left', padx=(0,6))
spin_thresh = tk.Spinbox(settings_frame, from_=0.1, to=0.99, increment=0.01, textvariable=security_threshold_var, width=6)
spin_thresh.pack(side='left', padx=(0,12))

# Unknown frames until alarm
security_unknown_limit_var = tk.IntVar(value=5)
lbl_limit = tk.Label(settings_frame, text="Unknown frames to alert:", bg='#f2f2f2')
lbl_limit.pack(side='left', padx=(0,6))
spin_limit = tk.Spinbox(settings_frame, from_=1, to=30, textvariable=security_unknown_limit_var, width=4)
spin_limit.pack(side='left', padx=(0,12))

# Enrolled count
enrolled_label = tk.Label(security_frame, text="Enrolled: 0", bg='#f2f2f2', fg='#333')
enrolled_label.pack(pady=(6,0))

lbl_sec_status = tk.Label(security_frame, text="", bg='#f2f2f2', fg='#333')
lbl_sec_status.pack(pady=(4,0))


btn_sec_back = tk.Button(security_frame, text="Back to Menu", font=("Arial", 14), command=go_back_to_main_menu, bg='#FFA500', fg='white')
btn_sec_back.pack(pady=20)

# Initially show the login frame
login_frame.pack()

# Start the main Tkinter event loop
root.mainloop()
