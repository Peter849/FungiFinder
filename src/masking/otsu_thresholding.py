import cv2
import numpy as np

def apply_otsu_threshold(img: np.ndarray) -> np.ndarray:
    """
    Bináris maszkolás megvalósítása Otsu küszöbölési módszer alkalmazásával szürkeárnyalatos képen.
    """
    print("\t\t1. Otsu küszöbölés")

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, binary_mask = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Megjegyzés: általában fehér objektum és fekete háttér színt szokás használni a Watershed algoritmusnál.
    binary_mask = cv2.bitwise_not(binary_mask)

    return binary_mask
