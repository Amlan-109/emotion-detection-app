import os
import pickle
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms

class SecurityManager:
    """Simple face-recognition manager using facenet-pytorch embeddings.

    Stores one mean embedding per authorized person in a pickle file under
    the `authorized` folder.
    """
    def __init__(self, authorized_dir=None, emb_file="embeddings.pkl", device=None):
        self.authorized_dir = authorized_dir or os.path.join(os.path.dirname(__file__), "authorized")
        os.makedirs(self.authorized_dir, exist_ok=True)
        self.emb_file = os.path.join(self.authorized_dir, emb_file)
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.transform = transforms.Compose([
            transforms.Resize((160,160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        self.embeddings = {}  # name -> np.array
        self._load_embeddings()

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
        # arr expected BGR (OpenCV). Convert to RGB PIL Image
        if arr is None:
            return None
        from PIL import Image
        import cv2
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _get_embedding(self, face_bgr):
        img = self._pil_from_bgr(face_bgr)
        if img is None:
            return None
        x = self.transform(img).unsqueeze(0)
        with __import__('torch').no_grad():
            e = self.model(x).cpu().numpy()[0]
        # L2-normalize
        e = e / (np.linalg.norm(e) + 1e-10)
        return e

    def enroll_from_folder(self, name, folder_path):
        """Enroll a person using all images in a folder; computes mean embedding."""
        import cv2
        embs = []
        for fn in os.listdir(folder_path):
            if not fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            p = os.path.join(folder_path, fn)
            img = cv2.imread(p)
            if img is None:
                continue
            emb = self._get_embedding(img)
            if emb is not None:
                embs.append(emb)
        if not embs:
            return False
        mean = np.mean(np.stack(embs, axis=0), axis=0)
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
