import cv2
import numpy as np

def filter_by_size(labels: np.ndarray, min_area: int) -> np.ndarray:
    """
    Szűri a szegmentált régiókat (címkéket) méret alapján.
    Eltávolítja a kis objektumokat.

    Paraméterek:
        labels: A watershed algoritmus által generált címkézett kép.
        min_area: A minimális terület (pixelek száma), ami alatt egy régió zajnak minősül.

    Kimenet:
        A szűrt címkék képe, ahol a nem megfelelő régiók el lettek távolítva
        A megmaradt régiók megtartják az eredeti címkéjüket.
    """
    print("\t\t1. Kis régiók kiszűrése méret alapján")
    filtered_labels = np.copy(labels)
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label <= 1: 
            continue

        component_mask = labels == label

        area = np.sum(component_mask)

        if area < min_area:
            filtered_labels[component_mask] = 0

    return filtered_labels
