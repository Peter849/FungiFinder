import numpy as np
import cv2 as cv

def denoise_median(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Zajcsökkentés medián szűrővel; minden pixel értékét a környező pixelek mediánjával helyettesíti.
    Paraméterek:
        - ksize: A medián kernel mérete (páratlan pozitív egész szám).
    """
    print("\t\t2. Zajcsökkentés medián szűrővel")
    
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError("A medián kernelméretnek páratlannak és >= 3-nak kell lennie.")

    return cv.medianBlur(gray, ksize)
