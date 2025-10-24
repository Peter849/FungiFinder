import numpy as np
import cv2 as cv
from typing import Any, Dict
from pathlib import Path

def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """
    Relatív vagy abszolút útvonal megadása.
    Ha a megadott útvonal abszolút, akkor azt visszaadja, ellenkező esetben a bázis útvonalhoz fűzi.
    """
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    
    relative_distance_from_project_root = 2

    if base is None:
        base = Path(__file__).resolve().parents[relative_distance_from_project_root]
    else:
        base = Path(base).resolve()

    return (base / path).resolve()

def read_image(full_path: str | Path) -> np.ndarray:
    """
    Képfájl beolvasása OpenCV-vel.
    """
    image_path = Path(full_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Nem található a megadott képfájl az alábbi helyen!: {image_path}")

    image = cv.imread(str(image_path), cv.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Hibás képformátum vagy sikertelen beolvasás: {image_path}")

    return image

def save_pipeline_results(
    images: Dict[str, Any],
    stage: str = "results",
) -> Path:
    """
    A pipeline lépéseinek képkimeneteit a projekt data/<stage> mappájába menti a paraméterekkel elnevezve.
    """

    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "data" / stage
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, image in images.items():
        if hasattr(image, "shape"):  # np.ndarray ellenőrzés
            filename = f"{name}.png"
            cv.imwrite(str(out_dir / filename), image)

    return out_dir