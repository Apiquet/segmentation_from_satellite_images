"""Script to evaluate a model."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from config import Config
from metrics.metrics_utils import fill_history_dict, get_metrics_results
from viewers.viewers_utils import update_compute_and_save_viewers

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(training_dir: Path, training_config: Config) -> None:
    """Evaluate model at a training directory.

    Args:
        training_dir (Path): training dir where to find best_model.pth, an "evaluation" directory will be created
        training_config (Config): config of the training
    """
    training_config.model.load_state_dict(torch.load(training_dir / "best_model.pth"))
    evaluation_dir = training_dir / "evaluation" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with torch.no_grad():
        training_config.model.train(False)

        # evaluate model on train and val set
        train_metrics_results = get_metrics_results(model=training_config.model, dataloader=training_config.train_dl, loss_func=training_config.loss_func, metrics=training_config.metrics)
        val_metrics_results = get_metrics_results(model=training_config.model, dataloader=training_config.val_dl, loss_func=training_config.loss_func, metrics=training_config.metrics)

        # start filling metrics history
        history: dict[str, Any] = {}
        fill_history_dict(history, train_metrics_results, val_metrics_results)

        evaluation_dir.mkdir(exist_ok=True, parents=True)
        with open(evaluation_dir / "metrics_history.json", "w") as json_file:
            json.dump(history, json_file, indent=4)
        update_compute_and_save_viewers(viewers_dir=evaluation_dir / "viewers", training_config=training_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a training.")
    parser.add_argument(
        "-training_dir",
        "-t",
        type=Path,
        help="Training directory to store results.",
        required=True,
    )
    parser.add_argument(
        "-db_path",
        "-d",
        type=Path,
        help="Path to the database containings at least a labels folder.",
        required=True,
    )
    args = parser.parse_args()
    training_config = Config(db_path=args.db_path)
    evaluate_model(training_dir=args.training_dir, training_config=training_config)
