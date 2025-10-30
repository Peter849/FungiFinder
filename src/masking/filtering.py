import cv2
import numpy as np

def filter_by_shape_and_sizes(
    binary_mask: np.ndarray,
    min_area: int = 80,             # kisebb minimum, hogy a kisebb gombák is megmaradjanak
    circularity_thresh: float = 0.2,  # lazább körkörösség
) -> np.ndarray:
    """
    Finomított alak- és méretszűrés a gombák megtartására.
    Kevésbé szigorú circularity és morfológiai zárás hozzáadva.
    """

    print("\t\t2. Finomított bináris maszk szűrés")

    # Kontúrkeresés
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(binary_mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        if circularity >= circularity_thresh:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    return filtered_mask
