from __future__ import annotations
import numpy as np
import cv2 as cv

def _ensure_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Bemenet: BGR vagy szürkeárnyalatos kép.
    Kimenet: 8 bites, 1 csatornás szürkeárnyalatos kép.
    """

    if img is None:
        raise ValueError("A bemeneti kép nem létezik!")

    if img.ndim == 2:
        result = img
    elif img.ndim == 3:
        result = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Ismeretlen kép dimenzió: {img.shape}")

    if result.dtype != np.uint8:
        gmin, gmax = float(result.min()), float(result.max())
        if gmax > gmin:
            result = ((result - gmin) / (gmax - gmin) * 255.0).astype(np.uint8)
        else:
            result = np.zeros_like(result, dtype=np.uint8)

    return result

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Szürkeárnyalatos konverzió."""
    print("\t\t1. Szürkeárnyalatos konverzió")
    
    return _ensure_grayscale(img)
