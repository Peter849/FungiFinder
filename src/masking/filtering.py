import cv2
import numpy as np

def filter_by_shape_and_size(
    binary_mask: np.ndarray,
    min_area: int = 150,
    circularity_thresh: float = 0.3
) -> np.ndarray:
    """
    A bináris maszk szűrése a gombák alakjának és méretének jellemzői alapján.
    Csak a kellően nagy és közel kör alakú objektumok megtartása a cél.

    Paraméterek:
        binary_mask: Bináris maszk (0/255)
        min_area: Minimum terület (pixelekben) – ennél kisebb foltok törlődnek
        circularity_thresh: Körkörösség küszöb (0–1), ahol 1.0 a tökéletes kör
    """

    print("\t\t2. Bináris maszk szűrése méret és alak alapján")

    # Másolat és kontúrkeresés
    mask = binary_mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_mask = np.zeros_like(mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        # Csak közel kör alakú objektumok megtartása
        if circularity >= circularity_thresh:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    return filtered_mask
