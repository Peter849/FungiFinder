import cv2
import numpy as np

def apply_gaussian_threshold(img: np.ndarray) -> np.ndarray:
    """
    Bináris maszkolás megvalósítása Otsu küszöbölési módszer alkalmazásával szürkeárnyalatos képen.
    """
    print("\t\t1. Otsu küszöbölés")

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    binary_mask = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 2
    )

    return binary_mask
