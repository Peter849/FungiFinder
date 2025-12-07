import cv2
import numpy as np


def filter_by_circularity(labels: np.ndarray, min_circularity: float) -> np.ndarray:
    """
    Szűri a szegmentált régiókat (címkéket) körkörösség alapján.
    Eltávolítja a nem eléggé kör alakú objektumokat.

    A körkörösség képlete: (4 * pi * terület) / (kerület^2)
    Egy tökéletes kör értéke 1.0.

    Paraméterek:
        labels: A szűrendő címkézett kép.
        min_circularity: A minimális körkörösségi érték (0 és 1 között).

    Visszatérési érték:
        A szűrt címkék képe, ahol a nem megfelelő régiók el lettek távolítva.
        A megmaradt régiók megtartják az eredeti címkéjüket.
    """
    print("\t\t2. Körkörösség vizsgálata")
    filtered_labels = np.copy(labels)
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label <= 1:
            continue

        component_mask = (labels == label).astype(np.uint8)

        area = np.sum(component_mask)
        if area == 0:
            continue

        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        perimeter = cv2.arcLength(contours[0], True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter**2)

        if circularity < min_circularity:
            filtered_labels[labels == label] = 0

    return filtered_labels