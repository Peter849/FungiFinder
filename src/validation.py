import numpy as np
from typing import Optional

def validate_results(
    watershed_labels: np.ndarray,
    size_filtered_labels: np.ndarray,
    final_labels: np.ndarray,
    manual_count: Optional[int],
    tolerance: int,
):
    print("\n\n=== Validáció - az algoritmus eredményeinek ellenőrzése ===")

    watershed_count = len(np.unique(watershed_labels[watershed_labels > 1]))
    size_filtered_count = len(np.unique(size_filtered_labels[size_filtered_labels > 1]))
    final_count = len(np.unique(final_labels[final_labels > 1]))

    print("\n--- Szűrési Statisztikák ---")
    print(f"\tWatershed algoritmus alapján szegmentált objektumok: {watershed_count}")
    print(f"\tA watershed által szegmentált objektumokon végzett MÉRET alapján történő szűrés után megmaradt objektumok száma: {size_filtered_count} (eltávolítva: {initial_count - size_filtered_count})")
    print(f"\tA watershed által szegmentált objektumokon végzett ALAK alapján történő szűrés után megmaradt objektumok száma:: {final_count} (eltávolítva: {size_filtered_count - final_count})")
    print(f"\t  -> Összesen kiszűrt objektum: {watershed_count - final_count}")

    if manual_count is not None:
        print("\n--- Manuális Összehasonlítás ---")
        difference = abs(final_count - manual_count)
        print(f"\tManuális gombaszám (referencia): {manual_count}")
        print(f"\tAlgoritmus által detektált gombaszám: {final_count}")
        print(f"\tEltérés: {difference} (megengedett: {tolerance})")
