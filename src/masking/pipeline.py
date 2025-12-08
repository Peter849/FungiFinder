from typing import Any, Dict

from utils import image_loader
from .gaussian_thresholding import apply_gaussian_threshold
from .morphology import clean_mask_with_morphology

def masking_pipeline(
        preprocessed_image,
        kernel_size_open,
        kernel_size_close,
        iterations_open,
        iterations_close,
        save_intermediates_results: bool = True,
    ) -> Dict[str, Any]:
    
    """
    Teljes maszkolási:
      1. Gauss-alapú adaptív küszöbölés
      2. Morfológiai tisztítás (opening, closing)
    """

    print("\tII. Maszkolás")

    # 1. Adaptive Gaussian Thresholding
    binary_mask = apply_gaussian_threshold(preprocessed_image)

    # 2. Morfológiai javítás
    cleaned_mask = clean_mask_with_morphology(
        binary_mask, 
        kernel_size_open = kernel_size_open,
        kernel_size_close = kernel_size_close,
        iterations_open = iterations_open,
        iterations_close = iterations_close,
    )


    results: Dict[str, Any] = {
        "binary_mask": binary_mask,
        "cleaned_mask": cleaned_mask,
    }

    if save_intermediates_results:
        image_loader.save_pipeline_results(
            {
                "4_gaussian_thresholding": binary_mask,
                "5_morphology": cleaned_mask,
            },
            stage="2_masking",
        )

    return results
