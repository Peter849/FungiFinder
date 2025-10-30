import numpy as np
import cv2 as cv

def colorize_labels(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    A watershed címkekép (labels) színezett vizualizálása.

    - Minden detektált objektum (label > 1) véletlen színt kap.
    - A watershed határvonalak (label == -1) pirosak.
    - Az eredeti kép halványan látszik a háttérben.
    """

    # 3 csatornás alap kép (ha grayscale volt)
    if len(image.shape) == 2:
        base = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    h, w = labels.shape
    colorized = np.zeros((h, w, 3), dtype=np.uint8)

    num_labels = labels.max()
    rng = np.random.default_rng(42)
    colors = rng.integers(0, 255, size=(num_labels + 1, 3), dtype=np.uint8)

    for label in range(2, num_labels + 1):
        mask = labels == label
        colorized[mask] = colors[label]  # most helyes: mask 2D, color 3 elemű vektor → broadcasting OK

    # watershed határok pirossal
    colorized[labels == -1] = (0, 0, 255)

    # keverés az eredeti képpel
    blended = cv.addWeighted(base, 0.4, colorized, 0.6, 0)
    return blended