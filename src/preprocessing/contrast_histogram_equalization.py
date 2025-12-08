from typing import Tuple
import numpy as np
import cv2 as cv

def enhance_contrast_clahe(
    clip_limit,
    tile_grid_size,
    img: np.ndarray,
) -> np.ndarray:
    """
    Lokális hisztogram-kiegyenlítés CLAHE algoritmussal; a gyenge megvilágítású vagy árnyékos részek részleteinek kiemelése.
    Paraméterek:
        - clip_limit (float): a kontraszt-korlátozás értéke (a magasabb értékek erősebb kontrasztot eredményeznek).
        - tile_grid_size: a rácsméret meghatározza a képet felosztó csempék számát (minél kisebb a csempe, annál részletesebb a kiegyenlítés).
    """
    print("\t\t3. Kontrasztjavítás hisztogram kiegyenlítéssel (CLAHE)")
    
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)