import os
import pickle
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

# facenet_pytorch imports
from facenet_pytorch import InceptionResnetV1, MTCNN

# optional OS-specific buzzer / tts
try:
    import winsound  # Windows-only buzzer
except Exception:
    winsound = None

# optional text-to-speech (best-effort)
try:
    import pyttsx3  # type: ignore
except Exception:
    pyttsx3 = None


class SecurityManager:
    """Face-recognition security manager.

    Features:
    - Stores one mean embedding per authorized person (pickle file under authorized/)
    - Enroll from a folder of images, a single image, or via camera capture
    - Live camera monitoring: recognizes faces, announces "OK" for authorized,
      and signals (beep + "Imposter") for unknown people.
    """

    def __init__(self, authorized_dir=None, emb_file="embeddings.pkl", device=None):
        self.authorized_dir = authorized_dir or os.path.join(os.path.dirname(__file__), "authorized")
        os.makedirs(self.authorized_dir, exist_ok=True)
        self.emb_file = os.path.join(self.authorized_dir, emb_file)

        # choose device
        import torch
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # models
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # transform for embeddings (model expects 160x160 normalized)
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.embeddings = {}  # name -> np.array (L2-normalized)
        self._load_embeddings()

        # optional TTS engine (lazy-init)
        self._tts_engine = None

    def _init_tts(self):
        if self._tts_engine is not None:
            return
        if pyttsx3 is None:
            self._tts_engine = None
            return
        try:
            self._tts_engine = pyttsx3.init()
        except Exception:
            self._tts_engine = None

    def _say(self, text):
        """Announce text (tts if available) or print as fallback."""
        self._init_tts()
        if self._tts_engine:
            try:
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
                return
            except Exception:
                pass
        # fallback
        print(text)

    def _buzz_imposter(self, freq=1500, duration_ms=600):
        """Try a buzzer sound on Windows; otherwise print a warning."""
        if winsound is not None:
            try:
                winsound.Beep(freq, duration_ms)
                return
            except Exception:
                pass
        # fallback
        print("[BUZZER] IMPOSTER!")

    def _load_embeddings(self):
        if os.path.exists(self.emb_file):
            try:
                with open(self.emb_file, "rb") as f:
                    self.embeddings = pickle.load(f)
            except Exception:
                self.embeddings = {}

    def _save_embeddings(self):
        with open(self.emb_file, "wb") as f:
            pickle.dump(self.embeddings, f)

    def _pil_from_bgr(self, arr):
        """Convert OpenCV BGR ndarray to PIL RGB Image."""
        if arr is None:
            return None
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _get_embedding_from_pil(self, pil_img):
        """Return L2-normalized embedding for a PIL Image (face crop)."""
        if pil_img is None:
            return None
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        import torch
        with torch.no_grad():
            e = self.model(x).cpu().numpy()[0]
        e = e / (np.linalg.norm(e) + 1e-10)
        return e

    def _get_embedding(self, face_bgr_or_pil):
        """Accept either BGR ndarray (OpenCV) or PIL Image and return embedding."""
        if face_bgr_or_pil is None:
            return None
        if isinstance(face_bgr_or_pil, Image.Image):
            return self._get_embedding_from_pil(face_bgr_or_pil)
        # assume OpenCV BGR ndarray
        pil = self._pil_from_bgr(face_bgr_or_pil)
        return self._get_embedding_from_pil(pil)

    def enroll_from_folder(self, name, folder_path):
        """Enroll a person using all images in a folder; computes mean embedding."""
        embs = []
        for fn in os.listdir(folder_path):
            if not fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            p = os.path.join(folder_path, fn)
            img = cv2.imread(p)
            if img is None:
                continue
            # try to detect/crop face(s) using MTCNN; use the first face per image
            pil = self._pil_from_bgr(img)
            crops = self.mtcnn(pil, return_prob=False)
            # mtcnn(...) returns tensor or list of tensors when keep_all=True
            tensors = crops
            if tensors is None:
                # fallback: use whole image
                emb = self._get_embedding(img)
                if emb is not None:
                    embs.append(emb)
                continue
            # handle both single tensor and list
            if isinstance(tensors, list):
                for t in tensors:
                    pil_crop = transforms.ToPILImage()(t.cpu())
                    emb = self._get_embedding_from_pil(pil_crop)
                    if emb is not None:
                        embs.append(emb)
                        break
            else:
                pil_crop = transforms.ToPILImage()(tensors.cpu())
                emb = self._get_embedding_from_pil(pil_crop)
                if emb is not None:
                    embs.append(emb)

        if not embs:
            return False
        mean = np.mean(np.stack(embs, axis=0), axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-10)
        self.embeddings[name] = mean
        self._save_embeddings()
        return True

    def enroll_from_image(self, name, image_path):
        """Enroll a person using a single image path (will attempt to detect a face)."""
        img = cv2.imread(image_path)
        if img is None:
            return False
        pil = self._pil_from_bgr(img)
        tensors = self.mtcnn(pil, return_prob=False)
        if tensors is None:
            # use whole image as fallback
            emb = self._get_embedding(img)
            if emb is None:
                return False
            self.embeddings[name] = emb
            self._save_embeddings()
            return True

        # use first detected face
        if isinstance(tensors, list):
            t = tensors[0]
        else:
            t = tensors
        pil_crop = transforms.ToPILImage()(t.cpu())
        emb = self._get_embedding_from_pil(pil_crop)
        if emb is None:
            return False
        self.embeddings[name] = emb
        self._save_embeddings()
        return True

    def register_via_camera(self, name, camera_index=0, samples=5, timeout=30):
        """Capture a number of face samples from camera and enroll the person.

        - name: label for the person
        - samples: number of successful face captures to average
        - timeout: total seconds to wait before giving up
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return False
        collected = []
        import time
        start = time.time()
        while len(collected) < samples and (time.time() - start) < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
            pil = self._pil_from_bgr(frame)
            tensors = self.mtcnn(pil, return_prob=False)
            if tensors is None:
                cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Register - press q to cancel", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            # take first detected face
            if isinstance(tensors, list):
                t = tensors[0]
            else:
                t = tensors
            pil_crop = transforms.ToPILImage()(t.cpu())
            emb = self._get_embedding_from_pil(pil_crop)
            if emb is not None:
                collected.append(emb)
                cv2.putText(frame, f"Captured {len(collected)}/{samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Register - press q to cancel", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        if not collected:
            return False
        mean = np.mean(np.stack(collected, axis=0), axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-10)
        self.embeddings[name] = mean
        self._save_embeddings()
        return True

    def load_authorized(self):
        # convenience: attempt to enroll any subfolders under authorized/
        for name in os.listdir(self.authorized_dir):
            p = os.path.join(self.authorized_dir, name)
            if os.path.isdir(p) and name not in self.embeddings:
                try:
                    self.enroll_from_folder(name, p)
                except Exception:
                    pass

    def recognize(self, face_bgr, threshold=0.6):
        """Return (name, score) if matched else (None, best_score).

        Uses cosine similarity; higher is better.
        """
        if not self.embeddings:
            return None, 0.0
        emb = self._get_embedding(face_bgr)
        if emb is None:
            return None, 0.0
        best_name = None
        best_sim = -1.0
        for name, ref in self.embeddings.items():
            sim = float(np.dot(emb, ref) / (np.linalg.norm(emb) * (np.linalg.norm(ref) + 1e-10)))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim

    def monitor_camera(self, camera_index=0, threshold=0.6, show=True):
        """Open camera and continuously recognize faces.

        Behavior:
        - If a face matches an authorized person (sim >= threshold): overlay "OK: <name>" and announce OK.
        - If no match: overlay "IMPOSTER" and trigger buzzer + announce.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Unable to open camera")
            return
        recent_announced = {}  # name -> last announced timestamp to avoid repeated announcements
        import time
        ANNOUNCE_COOLDOWN = 4.0  # seconds

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            pil = self._pil_from_bgr(frame)
            # detect faces and get cropped tensors
            tensors = self.mtcnn(pil, return_prob=False)
            if tensors is not None:
                faces = []
                if isinstance(tensors, list):
                    for t in tensors:
                        faces.append(transforms.ToPILImage()(t.cpu()))
                else:
                    faces.append(transforms.ToPILImage()(tensors.cpu()))

                # For each detected face, recognize
                for idx, face_pil in enumerate(faces):
                    emb = self._get_embedding_from_pil(face_pil)
                    if emb is None:
                        continue
                    name, score = self._match_embedding(emb, threshold=threshold)
                    # draw rectangle & text: attempt to find approximate bbox by running mtcnn.detect
                    boxes, _ = self.mtcnn.detect(pil)
                    if boxes is not None and idx < len(boxes):
                        box = boxes[idx].astype(int)
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if name else (0, 0, 255), 2)
                        label = f"OK: {name} ({score:.2f})" if name else f"IMPOSTER ({score:.2f})"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if name else (0, 0, 255), 2)

                    # announce and buzzer logic with cooldown
                    now = time.time()
                    if name:
                        last = recent_announced.get(name, 0)
                        if now - last > ANNOUNCE_COOLDOWN:
                            self._say(f"OK. {name} recognized.")
                            recent_announced[name] = now
                    else:
                        # use 'unknown' key to prevent continuous beeping
                        last = recent_announced.get("unknown", 0)
                        if now - last > ANNOUNCE_COOLDOWN:
                            self._say("Imposter detected")
                            self._buzz_imposter()
                            recent_announced["unknown"] = now

            if show:
                cv2.imshow("Security Monitor - press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def _match_embedding(self, emb, threshold=0.6):
        """Internal: match an embedding against stored ones, return (name, sim) or (None, best_sim)."""
        if not self.embeddings:
            return None, 0.0
        best_name = None
        best_sim = -1.0
        for name, ref in self.embeddings.items():
            sim = float(np.dot(emb, ref) / (np.linalg.norm(emb) * (np.linalg.norm(ref) + 1e-10)))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim
