from typing import Any, Dict

from utils import image_loader
from .otsu_thresholding import apply_otsu_threshold
from .filtering import filter_by_shape_and_size
from .morphology import clean_mask_with_morphology

def masking_pipeline(
        preprocessed_image,
        save_intermediates_results: bool = True,
        kernel_size: int = 3,
        iterations: int = 2,
    ) -> Dict[str, Any]:
    """
    Teljes maszkolási:
      1. Otsu küszöbölés
      2. Morfológiai tisztítás (opening, closing)
    """

    print("\tII. Maszkolás")

    # 1. Otsu thresholding
    binary_mask = apply_otsu_threshold(preprocessed_image)

    # 2. Morfológiai javítás
    cleaned_mask = clean_mask_with_morphology(binary_mask, kernel_size=kernel_size, iterations=iterations)

    # 3. Nem körszerű, nem elég nagy részek kiszűrése
    filtered_binary_mask = filter_by_shape_and_size(cleaned_mask)

    results: Dict[str, Any] = {
        "binary_mask": binary_mask,
        "cleaned_mask": cleaned_mask,
        "filtered_binary_mask": filtered_binary_mask,
    }

    if save_intermediates_results:
        image_loader.save_pipeline_results(
            {
                "4_otsu_thresholding": binary_mask,
                "5_morphology_k{kernel_size}_it{iterations}": cleaned_mask,
                "6_filtered_binary_mask": filtered_binary_mask,
            },
            stage="2_masking",
        )

    return results
