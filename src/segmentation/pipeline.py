from typing import Dict, Any

import numpy as np

from .distance_transform import compute_distance_transform
from .markers import create_markers
from .watershed import run_watershed
from utils.image_loader import save_pipeline_results
from utils.visualization import colorize_labels

def watershed_pipeline(
    preprocessed_image,
    cleaned_mask: Dict[str, Any],
    save_intermediates_results: bool = True,
) -> Dict[str, Any]:
    """
    III. Gomba szegmentáció Watershed algoritmussal
    ---------------------------------------------------------------
    3.1. Távolság-transzformáció
    3.2. Markerek képzése
    3.3. Watershed implementációja
    """

    print("\tIII. Szegmentáció – Watershed algoritmus")

    # --- 3.1 Távolságtranszformáció ---
    distance_map, distance_visualization = compute_distance_transform(cleaned_mask)

    # --- 3.2 Markerek képzése ---
    sure_fg, sure_bg, markers = create_markers(distance_map, cleaned_mask)

    # --- 3.3 Saját Watershed futtatása ---
    watershed_labels = run_watershed(preprocessed_image, markers)

    # --- Színezett eredmény a vizualizációhoz ---
    watershed_colorized = colorize_labels(preprocessed_image, watershed_labels)

    # --- Mentés ---
    if save_intermediates_results:
        save_pipeline_results(
            {
                "7_distance_visualization": distance_visualization,
                "8_sure_fg": sure_fg * 255,
                "9_sure_bg": sure_bg * 255,
                "10_markers": markers,
                "11_watershed_labels": (watershed_labels > 1).astype(np.uint8) * 255,
                "12_watershed_colorized": watershed_colorized,
            },
            stage="3_segmentation",
        )

    results: Dict[str, Any] = {
        "distance_map": distance_map,
        "sure_fg": sure_fg,
        "sure_bg": sure_bg,
        "markers": markers,
        "watershed_labels": watershed_labels,
        "watershed_colorized": watershed_colorized
    }

    return results
