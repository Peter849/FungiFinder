import cv2 as cv
import numpy as np

def compute_distance_transform(cleaned_mask: np.ndarray, distance_type=cv.DIST_L2, mask_size=cv.DIST_MASK_PRECISE) -> np.ndarray:
    """
    A bináris maszkban minden pixelhez megadja a legközelebbi háttérpont távolságát.
    A nagyobb értékek a régiók közepét (szegmentálandó objektumok magját) jelzik.

    Paraméterek:
        cleaned_mask: Bináris maszk (0 háttér, 255 előtér)
        distance_type: Azon távolságmetrika, amellyel a program a távolságot méri a háttértől:
            - Manhattan-távolság (cv.DIST_L1)
            - Euklideszi távolság (cv.DIST_L2) - ajánlott választás a feladathoz
            stb.
        mask_size: Meghatározza meg, mekkora ablakot használ a távolság-számítás lokális approximációjához.
                   Nagyobb -> lassabb, de simább és pontosabb
                   Kisebb -> gyorsabb, de zajosabb és pontatlanabb
                   -> ajánlott: 3x3 vagy 5x5 vagy cv.DIST_MASK_PRECISE
    """

    print("\t\t1. Távolságtranszformáció kiszámítása")


    mask_uint8 = cleaned_mask.astype(np.uint8)

    # Távolságtérkép számítása
    distance = cv.distanceTransform(mask_uint8, distance_type, mask_size)

    # Normalizálás 0–1 közé a további számításokhoz
    distance_map = cv.normalize(distance, None, 0, 1.0, cv.NORM_MINMAX)

    # Vizuális mentéshez 0–255 közé skálázott uint8 verzió
    distance_visualization = cv.normalize(distance, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    return distance_map, distance_visualization