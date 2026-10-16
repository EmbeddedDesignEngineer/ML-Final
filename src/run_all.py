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
    args = parser.parse_args()
    run_pipeline(args.raw, args.output, include_cv=not args.skip_cv, search_svm=not args.skip_svm_search, resume=args.resume)


if __name__ == "__main__":
    main()
