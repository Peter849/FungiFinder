from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import cv2 as cv

from utils import image_loader
from .grayscale import to_grayscale
from .denoising_median_filter import denoise_median
from .contrast_histogram_equalization import enhance_contrast_clahe

def preprocess_pipeline(
    image_path: str | Path,
    out_dir: Optional[str | Path] = None,
    *,
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

    bgr = image_loader.read_image(image_path)

    if bgr is None:
        raise ValueError(f"Sikertelen beolvasás: {image_path}")

    # 1. Szürkeárnyalatos konverzió
    gray = to_grayscale(bgr)

    # 2. Zajcsökkentés medián szűrővel
    denoised = denoise_median(gray, ksize=median_ksize)

    # 3. Kontrasztjavítás hisztogram kiegyenlítéssel
    enhanced = enhance_contrast_clahe(
        denoised, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid
    )

    results: Dict[str, Any] = {
        "input_bgr": bgr,
        "gray": gray,
        "denoised": denoised,
        "enhanced": enhanced,
    }

    if save_intermediates_results:
        out_dir = Path(out_dir) if out_dir is not None else image_path.parent.parent / "preprocessed"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem

        cv.imwrite(str(out_dir / f"{stem}_1_gray.png"), gray)
        cv.imwrite(str(out_dir / f"{stem}_2_denoised_m{median_ksize}.png"), denoised)
        cv.imwrite(
            str(out_dir / f"{stem}_3_enhanced_cl{clahe_clip_limit}_tg{clahe_tile_grid[0]}x{clahe_tile_grid[1]}.png"),
            enhanced,
        )
        results["out_dir"] = str(out_dir)
        
    return results
