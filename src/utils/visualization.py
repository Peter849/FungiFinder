import numpy as np
import cv2 as cv
from typing import Optional, List, Tuple


def colorize_labels(
    image: np.ndarray, labels: np.ndarray, colors: Optional[List[Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    A watershed címkekép (labels) színezett vizualizálása.

    - Minden detektált objektum (label > 1) egyedi színt kap.
    - Ha a 'colors' paletta meg van adva, azt használja, egyébként véletlen színeket generál.
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

    # A paletta használata vagy generálása
    num_labels = labels.max()
    if colors is not None:
        # A megadott paletta használata
        palette = np.array(colors, dtype=np.uint8)
    else:
        # Fallback: véletlen paletta generálása, ha nincs megadva
        rng = np.random.default_rng(42)
        palette = rng.integers(50, 255, size=(num_labels + 1, 3), dtype=np.uint8)
        if num_labels >= 0: palette[0] = [0, 0, 0]
        if num_labels >= 1: palette[1] = [0, 0, 0]

    # Címkék színezése a paletta alapján
    for label in np.unique(labels):
        # Csak a valós objektum címkéket (>=2) színezzük, és csak ha van hozzájuk szín a palettán
        if label >= 2 and label < len(palette):
            mask = labels == label
            colorized[mask] = palette[label]

    # watershed határok pirossal
    colorized[labels == -1] = (0, 0, 255)

    # keverés az eredeti képpel
    blended = cv.addWeighted(base, 0.4, colorized, 0.6, 0)
    return blended