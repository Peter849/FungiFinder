from typing import Any, Dict

from utils import image_loader
from .otsu_thresholding import apply_otsu_threshold
from .morphology import clean_mask_with_morphology

def masking_pipeline(
        preprocessed_image,
        kernel_size,
        iterations,
        save_intermediates_results: bool = True,
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

    results: Dict[str, Any] = {
        "binary_mask": binary_mask,
        "cleaned_mask": cleaned_mask,
    }

    if save_intermediates_results:
        image_loader.save_pipeline_results(
            {
                "4_otsu_thresholding": binary_mask,
                "5_morphology_k{kernel_size}_it{iterations}": cleaned_mask,
            },
            stage="2_masking",
        )

    return results
