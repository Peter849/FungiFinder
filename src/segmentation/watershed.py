import numpy as np
import cv2 as cv
import heapq


def run_watershed(preprocessed_gray_image: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """
    A képet domborzati térképként kezeli: a világos részek magaslatok (hegyek),
    a sötétebb részek völgyek. A marker régiókból "vizet eresztünk" lefelé a
    gradiens mentén, és a különböző források találkozásánál gát (határvonal)
    alakul ki.

    Paraméterek:
        preprocessed_gray_image:Szürkeárnyalatos, előfeldolgozott kép (uint8).
        markers: A korábban létrehozott marker kép, amelyen
            - 0 : ismeretlen régió
            - 1 : háttér
            - 2..N : különböző objektumok magpontjai/középpontjai

    Kimenetel:
        labels: A végső, szegmentált címkekép.
    """

    print("\t\t3. Watershed algoritmus futtatása")

    height, width = preprocessed_gray_image.shape

    # LUT: 0–255 közötti normalizált intenzitás értékek (~ gyors előre kiszámolt magasság értékek)
    LUT = np.linspace(0, 255, 256, dtype=np.float32)
    gray = LUT[preprocessed_gray_image]

    labels = markers.copy().astype(np.int32)

    # Szomszéd irányok (előre fix táblázatként mentve a memóriába) ---
    neighbor_offsets = [(-1, 0), (1, 0),
                        (0, -1), (0, 1),
                        (-1, -1), (-1, 1),
                        (1, -1), (1, 1)]

    # 256 "vödör" (egy minden lehetséges intenzitásértékhez)
    # minden vödör egy lista, amiben az adott szintű pixelek koordinátái vannak
    buckets = [[] for _ in range(256)]

    # Markerek körüli ismeretlen szomszédok bekerülnek a bucketekbe
    for y in range(height):
        for x in range(width):
            if labels[y, x] > 1:
                for dy, dx in neighbor_offsets:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < height and 0 <= xx < width and labels[yy, xx] == 0:
                        intensity = int(gray[yy, xx])
                        buckets[intensity].append((yy, xx))

    for level in range(256):
        while buckets[level]:
            y, x = buckets[level].pop()

            if labels[y, x] != 0:
                continue

            # Szomszédos címkék összegyűjtése
            neighbor_labels = set()
            for dy, dx in neighbor_offsets:
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width:
                    lab = labels[yy, xx]
                    if lab > 1:
                        neighbor_labels.add(lab)

            # Címkézés
            if len(neighbor_labels) == 0:
                # Még nem érkezett el a "víz" ide
                continue
            elif len(neighbor_labels) == 1:
                # Új címke elmentése
                labels[y, x] = neighbor_labels.pop()
            else:
                labels[y, x] = -1  # határvonal

            #  Új szomszédok hozzáadása
            for dy, dx in neighbor_offsets:
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width and labels[yy, xx] == 0:
                    intensity = int(gray[yy, xx])
                    buckets[intensity].append((yy, xx))

    return labels