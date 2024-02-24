"""Script to train a model."""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from shutil import copy
from typing import Any, Callable

import torch
import torch.nn as nn
from torchmetrics import Metric
from tqdm.auto import tqdm

from config import Config
from metrics.metrics_utils import compute_and_reset_metrics, fill_history_dict, get_metrics_results, save_metrics
from viewers.viewers_utils import update_compute_and_save_viewers

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_on_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dl: torch.utils.data.DataLoader,
    loss_func: Callable,
    metrics: list[Metric],
    epoch: int,
    epochs_count: int,
) -> dict[str, float]:
    """Train function for an epoch.

    Args:
        model (nn.Module): model to be trained
        optimizer (torch.optim.Optimizer): optimizer to be used for training
        train_dl (torch.utils.data.DataLoader): dataloader to be used for training
        loss_func (Callable): loss function to be used for training
        metrics (list[Metric]): list of torch metrics to compute
        epoch (int): current epoch
        epochs_count (int): total number of epochs to do

    Returns:
        dict[str, float]: metrics results 'metric_name': metric value
    """
    model.train(True)
    totalTrainLoss = torch.tensor(0.0).to(DEVICE)
    progress_bar = tqdm(train_dl)

    # loop over the training set
    for batch_idx, (x, y) in enumerate(train_dl, start=1):
        # perform a forward pass and calculate the training loss
        pred = model(x)
        train_loss = loss_func(pred, y)
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        totalTrainLoss += train_loss
        # update all metrics
        _ = [metric(pred, y) for metric in metrics]

        # update the progress bar
        progress_bar.set_description(f"Epoch {epoch}/{epochs_count}")
        progress_bar.set_postfix(
            train_loss=round(totalTrainLoss.cpu().detach().item() / batch_idx, 2),
        )
        progress_bar.update()

    # get results from metrics and loss
    metrics_results = compute_and_reset_metrics(metrics)
    metrics_results[str(loss_func)] = totalTrainLoss.item() / len(train_dl)
    return metrics_results


def run_training(training_dir: Path, db_path: Config, gee_project_name: str) -> None:
    """Train a model from configuration.

    Args:
        training_dir (Path): path to store the training results
        db_path (Path): path to the MiniFrance dataset containing at least labels folder
        gee_project_name (str): name of the gee project to use (if sat images to download)
    """
    start_time = datetime.now()
    training_dir = training_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    config_dir = training_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    copy(Path(__file__).resolve().parent / "config.py", config_dir / "config.py")

    training_config = Config(db_path=db_path, gee_project_name=gee_project_name)

    with torch.no_grad():
        training_config.model.train(False)

        # evaluate model on train and val set
        train_metrics_results = get_metrics_results(model=training_config.model, dataloader=training_config.train_dl, loss_func=training_config.loss_func, metrics=training_config.metrics)
        val_metrics_results = get_metrics_results(model=training_config.model, dataloader=training_config.val_dl, loss_func=training_config.loss_func, metrics=training_config.metrics)

    # start filling metrics history
    history: dict[str, Any] = {}
    fill_history_dict(history, train_metrics_results, val_metrics_results)

    # save initial model
    torch.save(training_config.model.state_dict(), training_dir / "best_model.pth")

    # creating viz at each epoch takes time and increases storage requirements
    epochs_count_without_improvement = 0
    for epoch in range(training_config.epochs_count):
        # train model
        train_metrics_results = train_on_epoch(
            model=training_config.model,
            optimizer=training_config.optimizer,
            train_dl=training_config.train_dl,
            loss_func=training_config.loss_func,
            metrics=training_config.metrics,
            epoch=epoch + 1,
            epochs_count=training_config.epochs_count,
        )

        # evaluate model
        with torch.no_grad():
            val_metrics_results = get_metrics_results(
                model=training_config.model,
                dataloader=training_config.val_dl,
                loss_func=training_config.loss_func,
                metrics=training_config.metrics,
            )

        # update our training history
        fill_history_dict(history, train_metrics_results, val_metrics_results)

        # save model if its val loss is the better
        if history[str(training_config.loss_func)]["val"][-1] < min(history[str(training_config.loss_func)]["val"][:-1]):
            torch.save(training_config.model.state_dict(), training_dir / "best_model.pth")
            epochs_count_without_improvement = 0
        else:
            epochs_count_without_improvement += 1
            if epochs_count_without_improvement > training_config.max_epochs_count_without_improvement:
                logging.info(f"Max epochs count without improvement was reached at epoch {epoch}")
                break

        for metric in training_config.metrics:
            metric.reset()

    save_metrics(history, folder_path=training_dir / "metrics")
    training_config.model.load_state_dict(torch.load(training_dir / "best_model.pth"))
    update_compute_and_save_viewers(viewers_dir=training_dir / "viewers", training_config=training_config)

    logging.info(f"Training time: {datetime.now() - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a training.")
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
    parser.add_argument(
        "-gee_project_name",
        "-n",
        type=str,
        default=None,
        help="Google Earth Engine project name.",
        required=False,
    )
    args = parser.parse_args()
    run_training(training_dir=args.training_dir, db_path=args.db_path, gee_project_name=args.gee_project_name)
