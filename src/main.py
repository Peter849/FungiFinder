import argparse

from utils.image_loader import resolve_path, read_image, save_pipeline_results
from preprocessing.pipeline import preprocessing_pipeline
from masking.pipeline import masking_pipeline
from segmentation.pipeline import watershed_pipeline
from postprocessing.pipeline import postprocessing_pipeline
from validation import validate_results

def main():

    parser = argparse.ArgumentParser(
        description="Teljes képfeldolgozási pipeline (előfeldolgozás + később szegmentálás)"
    )

    # A bemeneti kép (kötelezően megadandó)
    parser.add_argument("image", type=str, help="Bemeneti kép elérési útja")
    
    # Előfeldolgozási paraméterek
    parser.add_argument("--median-ksize", type=int, default=5, help="Medián szűrő kernelméret")
    parser.add_argument("--clahe-clip", type=float, default=5.0, help="CLAHE clip limit (minél nagyobb, annál erősebb kontraszt)")
    parser.add_argument("--clahe-grid", type=str, default="3x3", help="CLAHE tile grid méret pl. 4x4 (minél kisebb, annál erősebb kontraszt)")
    
    # Maszkolási paraméterek
    parser.add_argument("--kernel-size-open", type=int, default=3, help="Nyitás morfológiai művelet kernelmérete")
    parser.add_argument("--kernel-size-close", type=int, default=2, help="Zárás morfológiai művelet kernelmérete")
    parser.add_argument("--iterations-open", type=int, default=3, help="Nyitás morfológiai művelet iterációszáma")
    parser.add_argument("--iterations-close", type=int, default=2, help="Zárás morfológiai művelet iterációszáma")

    # Utófeldolgozási paraméterek
    parser.add_argument("--min-area", type=int, default=300, help="Minimális gombaterület a szűréshez")
    parser.add_argument("--min-circularity", type=float, default=0.5, help="Minimális körkörösség a gombaszűréshez")
    
    # Validáció paraméterek
    parser.add_argument("--manual-count", type=int, help="Manuálisan megszámolt gombák száma a validációhoz")
    parser.add_argument("--tolerance", type=int, default=0, help="A Watershed által detektált gombák számának megengedett eltérése a manuálisan megszámolthoz képest")
    
    # Egyéb paraméterek
    parser.add_argument("--no-save", action="store_true", help="Ne mentsen köztes fájlokat")

    args = parser.parse_args()

    print("==== Képfeldolgozási Pipeline ====")

    bgr = read_image(resolve_path(args.image))

    if bgr is None:
        raise ValueError(f"Az eredeti kép sikertelen beolvasása!: {args.image}")

    preprocessed_results = preprocessing_pipeline(
        original_image=bgr,
        median_ksize=args.median_ksize,
        clahe_clip_limit=args.clahe_clip,
        clahe_tile_grid=tuple(map(int, args.clahe_grid.lower().split("x"))),
        save_intermediates_results=not args.no_save,
    )

    preprocessed_image = preprocessed_results["contrast_enhanced"]

    mask_results = masking_pipeline(
        kernel_size_open=args.kernel_size_open,
        kernel_size_close=args.kernel_size_close,
        iterations_open=args.iterations_open,
        iterations_close=args.iterations_close,
        preprocessed_image=preprocessed_image,
        save_intermediates_results=not args.no_save,
    )

    cleaned_mask = mask_results["cleaned_mask"]

    segmentation_results = watershed_pipeline(
        preprocessed_image=preprocessed_image,
        cleaned_mask=cleaned_mask,
        save_intermediates_results=not args.no_save,
    )

    watershed_labels = segmentation_results["watershed_labels"]

    postprocessed_results = postprocessing_pipeline(
        original_image=bgr,
        labels=watershed_labels,
        min_area=args.min_area,
        min_circularity=args.min_circularity,
        save_intermediates_results=not args.no_save,
    )
    
    print("=== Pipeline sikeresen lefutott ===")    

    validate_results(
        watershed_labels=watershed_labels,
        size_filtered_labels=postprocessed_results["size_filtered_labels"],
        final_labels=postprocessed_results["final_shape_filtered_labels"],
        manual_count=args.manual_count,
        tolerance=args.tolerance,
    )

if __name__ == "__main__":
    main()
