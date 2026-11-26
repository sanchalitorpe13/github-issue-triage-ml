"""
Command-Line Interface (CLI) for Training, Evaluating & Benchmarking Models.
Executes reproducible Phase 3 training and artifact generation.
"""
import argparse
import sys
from pathlib import Path

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from config.settings import settings
from src.triage_ml.models.trainer import ModelTrainer
from src.triage_ml.models.comparison import ModelComparator
from src.triage_ml.models.evaluator import ModelEvaluator
from src.triage_ml.utils.logger import get_logger

logger = get_logger("train_cli")


def run_training_pipeline(
    data_path: Path,
    test_size: float = settings.TEST_SPLIT_SIZE,
    random_state: int = settings.RANDOM_STATE
) -> None:
    """Runs end-to-end model training, evaluation, comparison, and serialization."""
    logger.info("=" * 70)
    logger.info("STARTING PHASE 3: ML MODEL TRAINING & BENCHMARKING")
    logger.info("=" * 70)

    if not data_path.exists():
        logger.error(f"Dataset not found at: {data_path}")
        sys.exit(1)

    logger.info(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # 1. Train all baseline models
    trainer = ModelTrainer(test_size=test_size, random_state=random_state)
    train_output = trainer.train_and_evaluate_all(df)
    results = train_output["results"]
    vectorizer = train_output["vectorizer"]

    # 2. Build structured comparison table
    comparison_df = ModelComparator.build_comparison_dataframe(results)
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON BENCHMARK TABLE")
    logger.info("=" * 70)
    print(comparison_df.to_string(index=False))

    # 3. Select best model based on Macro F1
    best_name, best_model, best_metrics = ModelComparator.select_best_model(results, selection_metric="macro_f1")
    logger.info("\n" + ModelEvaluator.format_metrics_table(best_metrics, model_name=best_name))

    # 4. Save production artifacts
    dataset_info = {
        "dataset_path": str(data_path),
        "total_rows": len(df),
        "train_samples": len(train_output["X_train"]),
        "test_samples": len(train_output["X_test"]),
        "classes": best_metrics.get("classes", [])
    }

    pipeline_path, meta_path = ModelComparator.save_production_artifacts(
        best_model_name=best_name,
        best_model=best_model,
        vectorizer=vectorizer,
        metrics=best_metrics,
        comparison_df=comparison_df,
        dataset_info=dataset_info
    )

    logger.info("=" * 70)
    logger.info("PHASE 3 TRAINING & BENCHMARKING COMPLETE")
    logger.info(f"  Artifact: {pipeline_path}")
    logger.info(f"  Metadata: {meta_path}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate GitHub Issue Triage ML models.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(settings.PROCESSED_DATA_PATH / "multi_repo_processed.csv"),
        help="Path to processed dataset CSV"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=settings.TEST_SPLIT_SIZE,
        help="Fraction of data for test split (default: 0.20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.RANDOM_STATE,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()
    run_training_pipeline(
        data_path=Path(args.data),
        test_size=args.test_size,
        random_state=args.seed
    )


if __name__ == "__main__":
    main()
