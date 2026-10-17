import argparse
from pathlib import Path

from src import data_prep
from experiments.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the full real-data project, including Ridge.")
    parser.add_argument("--raw", type=Path, default=Path(data_prep.DATA_PATH))
    parser.add_argument("--output", type=Path, default=Path("results/final"))
    parser.add_argument("--skip-cv", action="store_true", help="Diagnostic run without five-fold uncertainty.")
    parser.add_argument("--skip-svm-search", action="store_true", help="Reuse recorded SVM winners for a shorter diagnostic run.")
    parser.add_argument("--resume", action="store_true", help="Resume an incomplete run using a verified completed SVM search.")
    parser.add_argument("--quality-study", action="store_true", help="Compare improved models on validation only; does not evaluate test data.")
    parser.add_argument("--price-study", action="store_true", help="Compare price targets and report errors by price band on validation.")
    parser.add_argument("--clean", type=Path, default=Path(data_prep.CLEAN_PATH))
    args = parser.parse_args()
    if args.price_study:
        if args.quality_study or args.resume or args.skip_cv or args.skip_svm_search:
            parser.error("--price-study cannot be combined with other experiment controls")
        from experiments.price_quality import run_price_study
        run_price_study(args.clean, args.output)
        return
    if args.quality_study:
        if args.resume or args.skip_cv or args.skip_svm_search:
            parser.error("--quality-study cannot be combined with full-pipeline controls")
        from experiments.model_quality import run_quality_study
        run_quality_study(args.raw, args.output)
        return
    run_pipeline(args.raw, args.output, include_cv=not args.skip_cv, search_svm=not args.skip_svm_search, resume=args.resume)


if __name__ == "__main__":
    main()
