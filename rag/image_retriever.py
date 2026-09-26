"""
Handcrafted (no torch, no downloaded weights) image encoder + a RandomForest
scan-type ("modality") classifier trained on your own case images. Kept
deliberately lightweight so IMAGE_ENCODER_MODE never has a dimension
mismatch between build time and request time.
"""
import numpy as np
from PIL import Image
import joblib
from sklearn.ensemble import RandomForestClassifier

import config

_THUMB_SIZE = (64, 64)


def _load_gray(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L").resize(_THUMB_SIZE)
    return np.asarray(img, dtype=np.float32) / 255.0


def _load_rgb(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize(_THUMB_SIZE)
    return np.asarray(img, dtype=np.float32) / 255.0


def encode_image(image_path: str) -> np.ndarray:
    """512-dim handcrafted descriptor: grayscale intensity histogram +
    per-channel color histograms + simple Sobel-edge histogram + flattened
    low-res thumbnail. Deterministic, fast, no pretrained weights."""
    gray = _load_gray(image_path)
    rgb = _load_rgb(image_path)

    gray_hist, _ = np.histogram(gray, bins=64, range=(0, 1))
    r_hist, _ = np.histogram(rgb[:, :, 0], bins=32, range=(0, 1))
    g_hist, _ = np.histogram(rgb[:, :, 1], bins=32, range=(0, 1))
    b_hist, _ = np.histogram(rgb[:, :, 2], bins=32, range=(0, 1))

    gy, gx = np.gradient(gray)
    edge_mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_hist, _ = np.histogram(edge_mag, bins=32, range=(0, edge_mag.max() + 1e-6))

    thumb_small = np.asarray(
        Image.fromarray((gray * 255).astype(np.uint8)).resize((16, 16)), dtype=np.float32
    ).flatten() / 255.0  # 256 dims

    vec = np.concatenate([
        gray_hist.astype(np.float32),
        r_hist.astype(np.float32), g_hist.astype(np.float32), b_hist.astype(np.float32),
        edge_hist.astype(np.float32),
        thumb_small,
    ])


    vec = vec / (np.linalg.norm(vec) + 1e-8)

    if vec.shape[0] < config.IMAGE_EMBED_DIM:
        vec = np.pad(vec, (0, config.IMAGE_EMBED_DIM - vec.shape[0]))
    else:
        vec = vec[: config.IMAGE_EMBED_DIM]
    return vec.astype(np.float32)


def train_modality_classifier(image_paths: list, modalities: list) -> RandomForestClassifier:
    X = np.stack([encode_image(p) for p in image_paths])
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X, modalities)
    joblib.dump(clf, config.MODALITY_CLASSIFIER_PATH)
    return clf


def load_modality_classifier() -> RandomForestClassifier:
    return joblib.load(config.MODALITY_CLASSIFIER_PATH)


def predict_modality(image_path: str) -> str:
    clf = load_modality_classifier()
    vec = encode_image(image_path).reshape(1, -1)
    return clf.predict(vec)[0]
