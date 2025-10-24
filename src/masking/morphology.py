import cv2
import numpy as np

def clean_mask_with_morphology(binary_mask: np.ndarray, kernel_size: int = 5, iterations: int = 3) -> np.ndarray:
    """
    Maszk tisztítása morfológiai műveletekkel (opening -> closing).
    Paraméterek:
        binary_mask (np.ndarray): Bináris bemeneti kép.
        kernel_size (int): A morfológiai műveletekhez használt kernel mérete (minél nagyobb, annál simább).
        iterations (int): A morfológiai műveletek ismétléseinek száma (minél több, annál simább).
    Kimenet: Finomított bináris maszk
    """

    print("\t\t2. Morfológiai tisztítás")

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return cleaned_mask