import argparse

from utils.image_loader import resolve_path
from preprocessing.pipeline import preprocess_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Teljes képfeldolgozási pipeline (előfeldolgozás + később szegmentálás)"
    )

    parser.add_argument("image", type=str, help="Bemeneti kép elérési útja")
    parser.add_argument("--median-ksize", type=int, default=3, help="Medián szűrő kernelméret")
    parser.add_argument("--clahe-clip", type=float, default=3.0, help="CLAHE clip limit (minél nagyobb, annál erősebb kontraszt)")
    parser.add_argument("--clahe-grid", type=str, default="4x4", help="CLAHE tile grid méret pl. 4x4 (minél kisebb, annál erősebb kontraszt)")
    parser.add_argument("--no-save", action="store_true", help="Ne mentsen köztes fájlokat")

    args = parser.parse_args()

    print("==== Képfeldolgozási Pipeline ====")

    preprocessed_result = preprocess_pipeline(
        image_path=resolve_path(args.image),
        median_ksize=args.median_ksize,
        clahe_clip_limit=args.clahe_clip,
        clahe_tile_grid=tuple(map(int, args.clahe_grid.lower().split("x"))),
        save_intermediates_results=not args.no_save,
    )

    print("=== Pipeline sikeresen lefutott ===")


if __name__ == "__main__":
    main()
