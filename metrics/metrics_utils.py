"""Useful functions for metrics."""
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric
from tqdm.auto import tqdm


def save_metrics(history: dict[str, dict[str, list[float]]], folder_path: Path) -> None:
    """Save one png file per metric.

    Args:
        history (dict[str, dict[str, list[float]]]): history with one key per metric then a dict containing train and val keys with list of values per epoch
        folder_path (Path): path to save png files
    """
    folder_path.mkdir(exist_ok=True, parents=True)
    with open(folder_path / "metrics_history.json", "w") as json_file:
        json.dump(history, json_file, indent=4)

    for metric_name, train_val_results in history.items():
        fig, ax = plt.subplots()
        ax.plot(range(len(train_val_results["train"])), np.c_[train_val_results["train"], train_val_results["val"]], label=["train", "val"])
        ax.set(xlabel="Epochs", ylabel="Values", title=metric_name)
        ax.legend()
        ax.grid()
        fig.savefig(folder_path / f"{metric_name}.png")


def compute_and_reset_metrics(metrics: list[Metric]) -> dict[str, float]:
    """Compute and reset all the metrics and return a dictionary with the results.

    Args:
        metrics: list of torch metrics to compute

    Returns:
        results containing the averaged loss value and the averaged metrics values
        in a dict with key=metric name, value=metric value
    """
    metrics_results: dict[str, float] = {}
    for metric in metrics:
        metrics_results[str(metric)] = metric.compute().item()
        metric.reset()
    return metrics_results


def get_metrics_results(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_func: Callable,
    metrics: list[Metric],
) -> dict[str, float]:
    """Get loss and metrics values.

    Args:
        model (nn.Module): model to be evaluated
        dataloader (torch.utils.data.DataLoader): dataloader to be used for evaluation
        loss_func (Callable): loss function to be used for evaluation
        metrics (list[Metric]): list of torch metrics to compute

    Returns:
        dict[str, float]: results containing the averaged loss value and the averaged metrics values
        in a dict with key=metric name, value=metric value
    """
    model.train(False)
    totalLoss = 0.0
    progress_bar = tqdm(dataloader)
    db_size = len(dataloader)

    for batch_idx, (x, y) in tqdm(enumerate(dataloader), desc="Evaluate metrics"):
        preds = model(x)
        totalLoss += loss_func(preds, y).cpu().detach().item()
        # update all metrics
        _ = [metric(preds, y) for metric in metrics]

        # update the progress bar
        progress_bar.set_postfix(loss=round(totalLoss / (batch_idx + 1), 2))
        progress_bar.update()

    # save results and reset the metrics classes
    metrics_results = compute_and_reset_metrics(metrics)
    metrics_results[str(loss_func)] = totalLoss / db_size
    return metrics_results


def fill_history_dict(history: dict[str, Any], train_metrics_results: dict[str, float], val_metrics_results: dict[str, float]):
    """Fill history dict inplace.

    One key per metric, value is a dict with train val keys and the list of values per epoch.

    Args:
        history (dict[str, Any]): history to fill in
        train_metrics_results (dict[str, float]): train results with metric name as key and its value for the current epoch
        val_metrics_results (dict[str, float]): val results with metric name as key and its value for the current epoch
    """
    for (train_k, train_v), (val_k, val_v) in zip(
        train_metrics_results.items(),
        val_metrics_results.items(),
    ):
        if train_k not in history.keys():
            history[train_k] = {"train": [], "val": []}
        if val_k not in history.keys():
            history[val_k] = {"train": [], "val": []}
        history[train_k]["train"].append(train_v)
        history[val_k]["val"].append(val_v)
