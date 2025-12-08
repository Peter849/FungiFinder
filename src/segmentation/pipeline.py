from typing import Dict, Any

import numpy as np

from .distance_transform import compute_distance_transform
from .markers import create_markers
from .watershed_opencv import run_watershed
from utils.image_loader import save_pipeline_results
from utils.visualization import colorize_watershed_lines

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

    # --- Mentés ---
    if save_intermediates_results:
        save_pipeline_results(
            {
                "6_distance_visualization": distance_visualization,
                "7_sure_fg": sure_fg * 255,
                "8_sure_bg": sure_bg * 255,
                "9_markers": markers,
                "10_watershed_labels": (watershed_labels > 1).astype(np.uint8) * 255,
                "11_watershed_lines_colorized": colorize_watershed_lines(
                    (watershed_labels > 1).astype(np.uint8) * 255,
                    watershed_labels,
                ),
            },
            stage="3_segmentation",
        )

    results: Dict[str, Any] = {
        "distance_map": distance_map,
        "sure_fg": sure_fg,
        "sure_bg": sure_bg,
        "markers": markers,
        "watershed_labels": watershed_labels,
    }

    return results
