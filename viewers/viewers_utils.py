"""Useful functions for visualization."""
from pathlib import Path

from torch import no_grad
from tqdm import tqdm

from config import Config


def update_compute_and_save_viewers(viewers_dir: Path, training_config: Config):
    """Infer model to get visualizations saved into the viewers_dir.

    Args:
        viewers_dir (Path): directory to save one figure per viewer
        training_config (Config): config to get val dataloader, viewers and model
    """
    with no_grad():
        training_config.model.train(False)
        for features, labels in tqdm(training_config.val_dl, desc="Generate final visualizations"):
            preds = training_config.model(features)
            _ = [viewer.update(preds=preds, labels=labels) for viewer in training_config.viewers]

        viewers_dir.mkdir(exist_ok=True, parents=True)
        for viewer in training_config.viewers:
            viewer.compute().savefig(viewers_dir / f"{viewer}.png")
