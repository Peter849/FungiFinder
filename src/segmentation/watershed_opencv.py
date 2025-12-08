import cv2
import numpy as np

def run_watershed(preprocessed_gray_image: np.ndarray, markers: np.ndarray) -> np.ndarray:

    print("\t\t3. Watershed algoritmus futtatása a beépített OpenCV függvény segítségével")

    if len(preprocessed_gray_image.shape) == 2:
        img_color = cv2.cvtColor(preprocessed_gray_image, cv2.COLOR_GRAY2BGR)
    else:
        img_color = preprocessed_gray_image

    markers_int32 = markers.astype(np.int32)

    # A Watershed algoritmus futtatása OpenCV-vel
    cv2.watershed(img_color, markers_int32)

    return markers_int32
