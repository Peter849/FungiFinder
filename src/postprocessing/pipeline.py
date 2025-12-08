from typing import Any, Dict, List, Tuple

import numpy as np

from utils import image_loader
from utils.visualization import colorize_labels
from .region_filter import filter_by_size
from .shape_filter import filter_by_circularity


def _generate_color_palette(num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Generál egy determinisztikus színpalettát a címkékhez.
    A 0 és 1 indexű címkék (háttér) feketék lesznek.
    """
    # A seed biztosítja, hogy a színek minden futtatáskor ugyanazok legyenek.
    np.random.seed(0)
    palette = np.random.randint(50, 255, size=(num_colors, 3), dtype=np.uint8)

    # A 0 és 1 indexű címkék (háttér) legyenek feketék
    if num_colors > 0:
        palette[0] = [0, 0, 0]
    if num_colors > 1:
        palette[1] = [0, 0, 0]

    return [tuple(map(int, color)) for color in palette]

def postprocessing_pipeline(
    original_image: np.ndarray,
    labels: np.ndarray,
    min_area: int,
    min_circularity: float,
    save_intermediates_results: bool = True,
) -> Dict[str, Any]:
    """
    Utófeldolgozási pipeline:
      4.1. Kis régiók elvetése
      4.2. Körkörösség vizsgálata
    """
    print("\tIV. Utófeldolgozás")

    # Színpaletta generálása az összes eredeti objektumhoz.
    # Így a színek konzisztensek maradnak a szűrési lépések között.
    max_label = np.max(labels)
    colors = _generate_color_palette(max_label + 1)

    # 4.1. Szűrés méret alapján
    size_filtered_labels = filter_by_size(labels=labels, min_area=min_area)

    # 4.2. Szűrés körkörösség alapján
    final_shape_filtered_labels = filter_by_circularity(labels=size_filtered_labels, min_circularity=min_circularity)

    results: Dict[str, Any] = {
        "size_filtered_labels": size_filtered_labels,
        "final_shape_filtered_labels": final_shape_filtered_labels,
    }

    if save_intermediates_results:
        image_loader.save_pipeline_results(
            {
                "12_watershed_original_result": colorize_labels(original_image, labels, colors=colors),
                "13_watershed_size_filtered_result": colorize_labels(original_image, size_filtered_labels, colors=colors),
                "14_watershed_shape_filtered_result": colorize_labels(original_image, final_shape_filtered_labels, colors=colors),
            },
            stage="4_postprocessing"
        )

    return results
