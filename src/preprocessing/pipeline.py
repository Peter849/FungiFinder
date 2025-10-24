from pathlib import Path
from typing import Tuple, Dict, Any

from utils import image_loader
from .grayscale import to_grayscale
from .denoising_median_filter import denoise_median
from .contrast_histogram_equalization import enhance_contrast_clahe

def preprocessing_pipeline(
    original_image,
    median_ksize: int = 3,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: Tuple[int, int] = (8, 8),
    save_intermediates_results: bool = True,
) -> Dict[str, Any]:
    """
    Teljes előfeldolgozás:
      1) BGR/GRAY -> GRAY
      2) Medián szűrés
      3) Kontrasztjavíítás hisztogram kiegyenlítéssel (CLAHE)

    Kimenet: dict az egyes lépések kimeneteivel.
    """
    print("\tI. Előfeldolgozás")

    # 1. Szürkeárnyalatos konverzió
    gray = to_grayscale(original_image)

    # 2. Zajcsökkentés medián szűrővel
    denoised = denoise_median(gray, ksize=median_ksize)

    # 3. Kontrasztjavítás hisztogram kiegyenlítéssel
    contrast_enhanced = enhance_contrast_clahe(
        denoised, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid
    )

    # Kimenet: a pipeline egyes lépéseinek eredményei
    results: Dict[str, Any] = {
        "gray": gray,
        "denoised": denoised,
        "contrast_enhanced": contrast_enhanced,
    }

    if save_intermediates_results:
        image_loader.save_pipeline_results(
            {
                "1_gray": gray,
                "2_median_k{median_ksize}": denoised,
                "3_clahe_clip{clahe_clip_limit}_grid{clahe_tile_grid[0]}x{clahe_tile_grid[1]}": contrast_enhanced,
            },
            stage="preprocessed",
        )

    return results