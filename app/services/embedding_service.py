import numpy as np
import os
import pickle
import logging
import cv2
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

# Max dimension before resizing — keeps memory low and helps tall portrait crops
MAX_INPUT_DIM = 640


def _resize_for_detection(image_bgr):
    """Downscale image if either dimension exceeds MAX_INPUT_DIM."""
    h, w = image_bgr.shape[:2]
    if max(h, w) <= MAX_INPUT_DIM:
        return image_bgr
    scale = MAX_INPUT_DIM / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


class EmbeddingService:
    def __init__(self):
        # Single shared model instance — avoids loading two copies into memory
        self._model = None
        self._current_thresh = 0.5  # track active threshold to avoid redundant sets

    def _get_model(self, relaxed=False):
        """
        Lazy-load a single FaceAnalysis instance (buffalo_sc).
        buffalo_sc uses ~80MB vs buffalo_l ~300MB — safe for 512MB free-tier hosts.
        Toggles det_thresh between standard (0.5) and relaxed (0.3) as needed.
        """
        if self._model is None:
            try:
                self._model = FaceAnalysis(
                    name='buffalo_sc',
                    providers=['CPUExecutionProvider']
                )
                # 320x320 input is sufficient for buffalo_sc and halves memory vs 640
                self._model.prepare(ctx_id=0, det_size=(320, 320))
                logger.info("[EMBED] InsightFace buffalo_sc model loaded successfully")
            except Exception as e:
                logger.error(f"InsightFace load error: {e}")
                self._model = None

        if self._model is not None:
            target_thresh = 0.3 if relaxed else 0.5
            if target_thresh != self._current_thresh:
                self._model.det_model.det_thresh = target_thresh
                self._current_thresh = target_thresh

        return self._model

    def extract_embedding(self, image_bgr):
        """
        Extract 512-d face embedding from a BGR image.
        - Resizes tall/large images before detection
        - Falls back to relaxed detector for sunglasses / partial occlusion
        - Returns (embedding, bbox) or (None, None)
        """
        # Resize if image is very large or has extreme aspect ratio
        image_bgr = _resize_for_detection(image_bgr)

        # Attempt 1: standard detection threshold
        faces = self._detect_faces(image_bgr, relaxed=False)

        # Attempt 2: relaxed threshold (handles sunglasses / occlusion)
        if not faces:
            logger.info("[EMBED] Standard detection found no faces — trying relaxed threshold")
            faces = self._detect_faces(image_bgr, relaxed=True)

        # Attempt 3: try with image flipped horizontally (helps some angles)
        if not faces:
            logger.info("[EMBED] Trying horizontally flipped image")
            flipped = cv2.flip(image_bgr, 1)
            faces = self._detect_faces(flipped, relaxed=True)

        if not faces:
            logger.warning("[EMBED] No face found after all attempts")
            return None, None

        # Pick the largest face by bounding box area
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Guard: InsightFace can return a face object with None embedding
        # when landmarks are heavily occluded (e.g. mirrored sunglasses)
        if face.embedding is None:
            logger.warning("[EMBED] Face detected but embedding is None (likely heavy occlusion)")
            return None, None

        return face.embedding, face.bbox

    def _detect_faces(self, image_bgr, relaxed=False):
        """Run InsightFace detection and return list of face objects with valid embeddings."""
        try:
            model = self._get_model(relaxed=relaxed)
            if model is None:
                return []
            faces = model.get(image_bgr)
            # Filter out any faces where embedding extraction failed
            return [f for f in faces if f.embedding is not None]
        except Exception as e:
            logger.error(f"[EMBED] Detection error (relaxed={relaxed}): {e}")
            return []

    def cosine_similarity(self, emb1, emb2):
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-10)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-10)
        return float(np.dot(emb1, emb2))

    def save_embeddings(self, driver_id, embeddings, embedding_path):
        embedding_dir = os.path.dirname(embedding_path)
        if embedding_dir:
            os.makedirs(embedding_dir, exist_ok=True)
        data = {
            'driver_id': driver_id,
            'embeddings': embeddings,
            'mean_embedding': np.mean(embeddings, axis=0)
        }
        with open(embedding_path, 'wb') as f:
            pickle.dump(data, f)

    def load_embeddings(self, embedding_path):
        if not os.path.exists(embedding_path):
            return None
        with open(embedding_path, 'rb') as f:
            return pickle.load(f)

    def load_all_embeddings(self, embeddings_folder):
        """Load all driver embeddings from disk."""
        all_embeddings = []
        if not os.path.exists(embeddings_folder):
            return all_embeddings
        for fname in os.listdir(embeddings_folder):
            if fname.endswith('.pkl'):
                path = os.path.join(embeddings_folder, fname)
                data = self.load_embeddings(path)
                if data:
                    all_embeddings.append(data)
        return all_embeddings

    def identify_face(self, query_embedding, all_embeddings, threshold=0.4):
        """Match query embedding against all enrolled drivers."""
        best_match = None
        best_score = -1
        for data in all_embeddings:
            mean_emb = data['mean_embedding']
            score = self.cosine_similarity(query_embedding, mean_emb)
            if score > best_score:
                best_score = score
                best_match = data['driver_id']
        if best_score >= threshold:
            return best_match, best_score
        return None, best_score


embedding_service = EmbeddingService()
