import numpy as np
import cv2 as cv

def colorize_labels(gray_image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    A watershed címkekép (labels) színezett vizualizálása.

    - Minden detektált objektum régiója (label > 1) más színt kap.
    - A watershed vonalak (label == -1) piros színűek.
    - A háttér halványan megjelenik az eredeti képből.
    """

    base = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)
    colorized = np.zeros_like(base)

    num_labels = labels.max()
    rng = np.random.default_rng(42) 

    colors = rng.integers(0, 255, size=(num_labels + 1, 3), dtype=np.uint8)

    for label in range(2, num_labels + 1): 
        colorized[labels == label] = colors[label]

    colorized[labels == -1] = (0, 0, 255)

    blended = cv.addWeighted(base, 0.4, colorized, 0.6, 0)

    return blended
