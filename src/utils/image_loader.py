from pathlib import Path
import cv2 as cv
import numpy as np

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