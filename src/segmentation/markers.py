import cv2 as cv
import numpy as np

def create_markers(distance_map: np.ndarray,
                   cleaned_mask: np.ndarray,
                   fg_threshold: float = 0.1,
                   dilate_iters: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A távolságtérkép alapján megjelöljük a biztos előteret (szegmentálandó objektum(ok)).
    Dilatációval pedig megjelöljük a biztos hátteret (pl. termelőföld).
    A kettő különbsége lesz az úgynevezett ismeretlen régió (előtér és háttér közötti határvonal)

    Paraméterek:
        distance_map: A távolságtranszformáció eredménye (0–1)
        cleaned_mask: Bináris maszk (0 háttér, 255 előtér)
        fg_threshold: Az a távolságérték, ami felett előtérnek tekintjük a pixelt (0 és 1 közötti érték)
        dilate_iters: Hányszor dilatáljunk a háttérképzéshez.
    """
    
    print("\t\t2. Markers - biztos előtér és háttér megjelölése")

    # Biztos előtér távolságküszöböléssel
    threshold = fg_threshold * distance_map.max()
    sure_fg = (distance_map > threshold).astype(np.uint8)

    # Biztos háttér (dilatált bináris maszkal)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv.dilate(cleaned_mask, kernel, iterations=dilate_iters)
    sure_bg = (sure_bg > 0).astype(np.uint8)

    # Ismeretlen régió
    unknown_region = cv.subtract(sure_bg, sure_fg)

    # Komponenscímkézés – az előtér régiókhoz külön azonosítókat rendelve
    num_labels, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown_region == 1] = 0

    vis = np.zeros((*markers.shape, 3), dtype=np.uint8)
    vis[markers == 0] = (0, 0, 255)      # ismeretlen (piros)
    vis[markers == 1] = (0, 0, 0)        # háttér (fekete)
    vis[markers > 1] = (0, 255, 0)       # előtér (zöld)
    cv.imshow("Markers", vis)
    cv.waitKey(0)

    return sure_fg, sure_bg, markers
