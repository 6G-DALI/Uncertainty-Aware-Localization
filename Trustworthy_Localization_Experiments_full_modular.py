"""
Unified script for uncertainty-aware CSI localization experiments.

Supports:
- Methods: Adaptive Split Conformal Prediction (Adaptive SCP, scale-based) and
  one-sided two-stage CQR on top of a frozen ADN backbone.
- Evaluation data: pooled dynamic scenarios (0–5) or a single dynamic scenario.
- CQR head: train from scratch or load an existing checkpoint.
- Metrics: overall uncertainty/SLA coverage metrics on the full test set,
  plus optional single-sample evaluation with a per-sample uncertainty plot.

This script assumes that `dataset.NomadicLocalizationDataModule` and
`models.AttentionDenseNet` are available.
"""

import math
import time
from pathlib import Path
from typing import List, Tuple, Optional

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from dataset import NomadicLocalizationDataModule
from models import AttentionDenseNet

# ============================================================================
# User-configurable section
# ============================================================================

# Method: "ASCP" (adaptive SCP with scale model) or "CQR" (two-stage, one-sided CQR)
METHOD: str = "CQR"  # "ASCP" or "CQR"

# Evaluation mode: "pooled" (all dynamic scenarios 0-5) or "single" (one scenario)
EVAL_MODE: str = "pooled"  # "pooled" or "single"

# If EVAL_MODE == "single", specify the scenario ID (0-5)
SINGLE_SCENARIO_ID: int = 0

# Backbone checkpoint handling
USE_EXISTING_BACKBONE_CHECKPOINT: bool = True
BACKBONE_CHECKPOINT_PATH: str = r"./mlartifacts/851097306110734214/c5ce26cc459c457fb37c42b721f203d6/artifacts/checkpoints/best_model.ckpt"  # random-attenuation best

# random attenuation augmentation checkpoint model (best)
#BACKBONE_CHECKPOINT_PATH = r".\mlartifacts\851097306110734214\c5ce26cc459c457fb37c42b721f203d6\artifacts\checkpoints\best_model.ckpt"

# vanilla augmentation checkpoint model
#BACKBONE_CHECKPOINT_PATH = r".\mlartifacts\851097306110734214\8a31fb1083504eb580e14437b0002d5c\artifacts\checkpoints\best_model.ckpt"

# no augmentation checkpoint model
#BACKBONE_CHECKPOINT_PATH = r".\mlartifacts\851097306110734214\4b7c1e2582a04239ab3fbfe0957605c3\artifacts\checkpoints\best_model.ckpt"

# CQR head checkpoint handling (only used when METHOD == "CQR")
USE_EXISTING_CQR_HEAD: bool = True
#CQR_HEAD_CHECKPOINT_PATH: Optional[str] = r'./833503865933581751/1f258e2818b14bee9d1333e0b7bc5cf1/checkpoints/best_cqr_head.ckpt'  
CQR_HEAD_CHECKPOINT_PATH: Optional[str] = r'./483259487935076753/fa4d15510f974de586300cc61026b010/checkpoints/best_cqr_head.ckpt'
# set path if USE_EXISTING_CQR_HEAD is True

# SLA levels
SLA_LEVELS: List[float] = [0.90, 0.95, 0.99]
ALPHAS: List[float] = [1.0 - x for x in SLA_LEVELS]

# Data and MLflow config
SEED: int = 42
DATA_DIR: str = "data/nomadic_dataset/ULA_lab_LoS"
TRACKING_URI: str = "http://127.0.0.1:8080" # mlflow server --host 127.0.0.1 --port 8080
EXPERIMENT_NAME: str = "ADN-Conformal-Unified"
RUN_NAME: str = "unified_conformal_experiments"

# Scenario and dataset parameters
# Backbone training config (static scenario, ADN training)
TRAIN_ADN_IF_NEEDED: bool = False
ADN_MAX_EPOCHS: int = 200
ADN_AUGMENT_METHOD: str = "random_attenuation"  # "random_attenuation", "vanilla", "no_augmentation"

SCENARIO_IDS: List[int] = [0, 1, 2, 3, 4, 5]
NUM_USERS: int = 4
NUM_SAMPLES: int = 240
BATCH_SIZE: int = 32
NUM_WORKERS: int = 0

# Split fractions for head/scale training, conformal calibration, and testing
HEAD_TRAIN_FRACTION: float = 0.25
CALIBRATION_FRACTION: float = 0.45
TEST_FRACTION: float = 0.30

# Training hyperparameters
MAX_EPOCHS_SCALE: int = 60
SCALE_LR: float = 1e-3
SCALE_WEIGHT_DECAY: float = 1e-5

MAX_EPOCHS_CQR_HEAD: int = 100
CQR_HEAD_LR: float = 1e-3
CQR_HEAD_WEIGHT_DECAY: float = 1e-5

# Output directory
OUTPUT_DIR: Path = Path("output/unified_conformal_experiments")

# Optional single-sample evaluation
ENABLE_SINGLE_SAMPLE_EVAL: bool = True
SINGLE_SAMPLE_INDEX: int = 0  # index within the test split
SINGLE_SAMPLE_SLA: float = 0.95  # SLA level for which to draw the uncertainty circle

# ============================================================================
# Utility functions
# ============================================================================


def euclidean_np(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Euclidean distance between 2D predictions and targets in millimeters."""
    return np.sqrt(((y_pred - y_true) ** 2).sum(axis=1))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Standard split-conformal quantile: ceil((n+1)(1-alpha))/n order statistic."""
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration scores are empty.")
    rank = math.ceil((n + 1) * (1 - alpha))
    rank = min(max(rank, 1), n)
    return float(np.sort(scores)[rank - 1])


def write_fig(fig: go.Figure, path_base: Path, width: int = 800, height: int = 800) -> None:
    path_base = path_base.with_suffix("")  # ensure we control suffixes
    html_path = path_base.with_suffix(".html")
    png_path = path_base.with_suffix(".png")
    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path), width=width, height=height, scale=2)
    except Exception:
        # Static image export requires kaleido; ignore if unavailable
        pass


# ============================================================================
# Dataset helpers
# ============================================================================


def build_scenario_dataset(data_dir: str, scenario_id: int) -> torch.utils.data.Dataset:
    """Build dataset for a single dynamic scenario."""
    sample_ids = [
        scenario_id * 10000 + user * 1000 + s
        for user in range(NUM_USERS)
        for s in range(NUM_SAMPLES)
    ]
    dm = NomadicLocalizationDataModule(
        data_dir=data_dir,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        num_users=NUM_USERS,
        num_samples=NUM_SAMPLES,
        mode="test_only",
        sample_ids=sample_ids,
    )
    dm.setup()
    return dm.test_dataset


def build_pooled_dataset(data_dir: str, scenario_ids: List[int]) -> ConcatDataset:
    """Concat datasets from multiple scenarios."""
    datasets = [build_scenario_dataset(data_dir, sid) for sid in scenario_ids]
    return ConcatDataset(datasets)


def split_indices(n_total: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into head-train / calibration / test subsets."""
    if abs((HEAD_TRAIN_FRACTION + CALIBRATION_FRACTION + TEST_FRACTION) - 1.0) > 1e-9:
        raise ValueError("HEAD_TRAIN_FRACTION + CALIBRATION_FRACTION + TEST_FRACTION must equal 1.0")
    indices = np.arange(n_total)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_head = int(round(n_total * HEAD_TRAIN_FRACTION))
    n_cal = int(round(n_total * CALIBRATION_FRACTION))
    head_indices = indices[:n_head]
    cal_indices = indices[n_head:n_head + n_cal]
    test_indices = indices[n_head + n_cal:]
    return head_indices, cal_indices, test_indices


# ============================================================================
# Models: Adaptive SCP scale model and CQR head
# ============================================================================


class FrozenBackboneScaleModel(L.LightningModule):
    """Learns a local scale sigma_hat(x) for radial error, using backbone features.

    This is the adaptive part of adaptive SCP: residuals are normalized by
    sigma_hat(x), and conformal quantiles are computed on the normalized errors.
    """

    def __init__(self, backbone: nn.Module, lr: float = SCALE_LR, weight_decay: float = SCALE_WEIGHT_DECAY):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        in_channels = 512
        h1, h2 = in_channels // 2, in_channels // 4
        self.fc1 = nn.Linear(in_channels, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.scale_head = nn.Linear(h2, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.backbone.denseBlock1(x)
            x = [
                self.backbone.subcarrier_attentions[i](x[:, :, i, :]).unsqueeze(2)
                for i in range(self.backbone.num_antennas)
            ]
            x = torch.cat(x, dim=2)
            x = self.backbone.ap1(x)
            x = self.backbone.denseBlock2(x)
            x = self.backbone.ap2(x)
            x = self.backbone.denseBlock3(x)
            x = self.backbone.denseBlock4(x)
            x = self.backbone.ap3(x)
            x = self.backbone.flat1(x.transpose(2, 3))
            x = self.backbone.antenna_attention(x)
            x = self.backbone.flat2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        h = F.relu(self.fc1(feats))
        h = F.relu(self.fc2(h))
        sigma = F.softplus(self.scale_head(h)).squeeze(-1)
        sigma = torch.clamp(sigma, min=1e-3)
        return sigma

    def _step(self, batch, stage: str):
        x, y = batch
        with torch.no_grad():
            center = self.backbone(x)
            radial_error = torch.sqrt(torch.sum((center - y) ** 2, dim=1) + 1e-8)
        sigma_hat = self(x)
        loss = F.l1_loss(sigma_hat, radial_error)
        self.log(f"{stage}_loss", loss, sync_dist=True)
        self.log(f"{stage}_mean_error", radial_error.mean(), sync_dist=True)
        self.log(f"{stage}_mean_sigma", sigma_hat.mean(), sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}


class OneSidedMultiSLAQuantileHead(nn.Module):
    """Predicts one upper-quantile radius r_hi(x) per SLA level.

    Monotonicity across SLA levels is enforced via cumulative max.
    """

    def __init__(self, in_channels: int = 512, num_levels: int = 3):
        super().__init__()
        h1, h2, h3 = in_channels // 2, in_channels // 4, in_channels // 8
        self.fc1 = nn.Linear(in_channels, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.rhi_head = nn.Linear(h3, num_levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        rhi = F.softplus(self.rhi_head(x))
        # enforce monotonic increase across SLA levels (e.g., 99% radius >= 95% >= 90%)
        rhi = torch.cummax(rhi, dim=1)[0]
        return rhi


class FrozenBackboneOneSidedCQR(L.LightningModule):
    """One-sided multi-SLA CQR head on top of a frozen ADN backbone.

    The backbone provides point predictions; the head predicts upper radii r_hi(x)
    for each SLA level, trained via pinball loss.
    """

    def __init__(self, backbone: nn.Module, sla_levels: Optional[List[float]] = None,
                 lr: float = CQR_HEAD_LR, weight_decay: float = CQR_HEAD_WEIGHT_DECAY):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.sla_levels = sla_levels or SLA_LEVELS
        self.alphas = [1.0 - x for x in self.sla_levels]
        self.lr = lr
        self.weight_decay = weight_decay
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.quantile_head = OneSidedMultiSLAQuantileHead(512, len(self.sla_levels))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.backbone.denseBlock1(x)
            x = [
                self.backbone.subcarrier_attentions[i](x[:, :, i, :]).unsqueeze(2)
                for i in range(self.backbone.num_antennas)
            ]
            x = torch.cat(x, dim=2)
            x = self.backbone.ap1(x)
            x = self.backbone.denseBlock2(x)
            x = self.backbone.ap2(x)
            x = self.backbone.denseBlock3(x)
            x = self.backbone.denseBlock4(x)
            x = self.backbone.ap3(x)
            x = self.backbone.flat1(x.transpose(2, 3))
            x = self.backbone.antenna_attention(x)
            x = self.backbone.flat2(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            center = self.backbone(x)
        feats = self.extract_features(x)
        rhi = self.quantile_head(feats)
        return center, rhi

    def _loss_components(self, batch):
        x, y = batch
        center, rhi = self(x)
        radial_error = torch.sqrt(torch.sum((center - y) ** 2, dim=1) + 1e-8)
        quantile_losses, coverages, radii = [], [], []
        for i, alpha in enumerate(self.alphas):
            q_hi = 1.0 - alpha
            # one-sided pinball loss at upper quantile
            err = radial_error - rhi[:, i]
            loss_i = torch.mean(torch.maximum(q_hi * err, (q_hi - 1.0) * err))
            quantile_losses.append(loss_i)
            coverages.append((radial_error <= rhi[:, i]).float().mean())
            radii.append(rhi[:, i].mean())
        loss = torch.stack(quantile_losses).mean()
        return loss, center, rhi, radial_error, torch.stack(coverages), torch.stack(radii)

    def training_step(self, batch, batch_idx):
        loss, _, _, radial_error, coverages, radii = self._loss_components(batch)
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_mean_radial_error", radial_error.mean(), sync_dist=True)
        for i, lvl in enumerate(self.sla_levels):
            name = str(lvl).replace(".", "p")
            self.log(f"train_raw_cov_{name}", coverages[i], sync_dist=True)
            self.log(f"train_rhi_{name}", radii[i], sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _, radial_error, coverages, radii = self._loss_components(batch)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_mean_radial_error", radial_error.mean(), sync_dist=True)
        for i, lvl in enumerate(self.sla_levels):
            name = str(lvl).replace(".", "p")
            self.log(f"val_raw_cov_{name}", coverages[i], sync_dist=True)
            self.log(f"val_rhi_{name}", radii[i], sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}


# ============================================================================
# Prediction helpers
# ============================================================================


def predict_center_and_sigma(scale_model: FrozenBackboneScaleModel, loader: DataLoader,
                             device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run backbone and scale model to get center predictions and sigma_hat."""
    scale_model.eval()
    scale_model.to(device)
    centers, sigmas, targets = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            center = scale_model.backbone(x)
            sigma_hat = scale_model(x)
            centers.append(center.detach().cpu().numpy())
            sigmas.append(sigma_hat.detach().cpu().numpy().reshape(-1))
            targets.append(y.detach().cpu().numpy())
    return np.vstack(centers), np.concatenate(sigmas), np.vstack(targets)


def predict_center_and_rhi(cqr_model: FrozenBackboneOneSidedCQR, loader: DataLoader,
                           device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run backbone+CQR head to get center and base radii r_hi(x) per SLA level."""
    cqr_model.eval()
    cqr_model.to(device)
    centers, rhi_list, targets = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            center, rhi = cqr_model(x)
            centers.append(center.detach().cpu().numpy())
            rhi_list.append(rhi.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
    return np.vstack(centers), np.vstack(rhi_list), np.vstack(targets)


# ============================================================================
# Evaluation routines for Adaptive SCP and CQR
# ============================================================================


def run_adaptive_scp(backbone: AttentionDenseNet, dataset, device: torch.device,
                     output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train scale model on head split, calibrate on cal split, evaluate on test split."""
    n_total = len(dataset)
    head_idx, cal_idx, test_idx = split_indices(n_total, SEED)
    head_dataset = Subset(dataset, head_idx)
    cal_dataset = Subset(dataset, cal_idx)
    test_dataset = Subset(dataset, test_idx)

    head_loader = DataLoader(head_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    cal_loader = DataLoader(cal_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

    scale_model = FrozenBackboneScaleModel(backbone=backbone)

    logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        run_name=f"{RUN_NAME}-ASCP",
        tracking_uri=TRACKING_URI,
        log_model=False,
        synchronous=True,
    )

    trainer = Trainer(
        max_epochs=MAX_EPOCHS_SCALE,
        logger=logger,
        log_every_n_steps=10,
        accelerator="auto",
        devices=1,
    )

    t0 = time.time()
    trainer.fit(scale_model, train_dataloaders=head_loader, val_dataloaders=cal_loader)
    runtime_seconds = time.time() - t0

    # Move to device and evaluate
    scale_model = scale_model.to(device)
    scale_model.backbone.to(device)
    scale_model.eval()

    cal_center, sigma_cal, cal_true = predict_center_and_sigma(scale_model, cal_loader, device)
    test_center, sigma_test, test_true = predict_center_and_sigma(scale_model, test_loader, device)

    cal_errors = euclidean_np(cal_center, cal_true)
    test_errors = euclidean_np(test_center, test_true)

    metrics_rows, prediction_rows = [], []
    for sla, alpha in zip(SLA_LEVELS, ALPHAS):
        scores = cal_errors / sigma_cal
        qhat = conformal_quantile(scores, alpha)
        final_radius = sigma_test * qhat
        covered = test_errors <= final_radius

        row = {
            "method": "AdaptiveSCP-Scale",
            "target_assurance": float(sla),
            "alpha": float(alpha),
            "calibration_size": int(len(cal_errors)),
            "test_size": int(len(test_errors)),
            "qhat_mm": float(qhat),
            "calibration_mean_error_mm": float(np.mean(cal_errors)),
            "calibration_median_error_mm": float(np.median(cal_errors)),
            "test_mean_error_mm": float(np.mean(test_errors)),
            "test_median_error_mm": float(np.median(test_errors)),
            "test_p90_error_mm": float(np.quantile(test_errors, 0.90)),
            "test_p95_error_mm": float(np.quantile(test_errors, 0.95)),
            "true_coverage": float(np.mean(covered)),
            "mean_interval_radius_mm": float(np.mean(final_radius)),
            "median_interval_radius_mm": float(np.median(final_radius)),
        }
        metrics_rows.append(row)

        sla_name = str(sla).replace(".", "p")
        for j in range(len(test_errors)):
            prediction_rows.append({
                "method": "AdaptiveSCP-Scale",
                "target_assurance": float(sla),
                "sla_name": sla_name,
                "split": "test",
                "pred_x": float(test_center[j, 0]),
                "pred_y": float(test_center[j, 1]),
                "true_x": float(test_true[j, 0]),
                "true_y": float(test_true[j, 1]),
                "euclidean_error_mm": float(test_errors[j]),
                "base_radius_mm": float(sigma_test[j]),
                "qhat_mm": float(qhat),
                "final_radius_mm": float(final_radius[j]),
                "covered": float(covered[j]),
            })

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["target_assurance"]).reset_index(drop=True)
    predictions_df = pd.DataFrame(prediction_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "adaptive_scp_metrics.csv"
    preds_csv = output_dir / "adaptive_scp_predictions.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    predictions_df.to_csv(preds_csv, index=False)

    # Simple coverage and radius plots
    figs_dir = output_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    fig_cov = go.Figure()
    fig_cov.add_bar(x=metrics_df["target_assurance"], y=metrics_df["true_coverage"], name="empirical_coverage")
    fig_cov.add_scatter(x=metrics_df["target_assurance"], y=metrics_df["target_assurance"],
                        mode="lines", name="ideal", line=dict(color="black", dash="dot"))
    fig_cov.update_layout(title="Adaptive SCP: empirical vs target coverage",
                          xaxis_title="target_assurance", yaxis_title="coverage")
    write_fig(fig_cov, figs_dir / "adaptive_scp_coverage_vs_target")

    fig_rad = go.Figure()
    fig_rad.add_bar(x=metrics_df["target_assurance"], y=metrics_df["mean_interval_radius_mm"], name="mean_radius")
    fig_rad.update_layout(title="Adaptive SCP: mean final radius vs target assurance",
                          xaxis_title="target_assurance", yaxis_title="radius (mm)")
    write_fig(fig_rad, figs_dir / "adaptive_scp_mean_radius_vs_target")

    # Log to MLflow using the same run as logger
    run_id = logger.run_id
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("method", "AdaptiveSCP-Scale")
        mlflow.log_metric("scale_train_runtime_seconds", float(runtime_seconds))
        mlflow.log_metric("pooled_total_size", int(len(dataset)))
        mlflow.log_metric("head_train_size", int(len(head_dataset)))
        mlflow.log_metric("calibration_size", int(len(cal_dataset)))
        mlflow.log_metric("test_size", int(len(test_dataset)))
        for _, row in metrics_df.iterrows():
            prefix = f"adaptivescp_scale_{str(row['target_assurance']).replace('.', 'p')}"
            for col in metrics_df.columns:
                if col in {"method", "target_assurance"}:
                    continue
                value = row[col]
                if pd.notna(value) and np.isscalar(value):
                    try:
                        mlflow.log_metric(f"{prefix}_{col}", float(value))
                    except Exception:
                        pass
        mlflow.log_artifact(str(metrics_csv))
        mlflow.log_artifact(str(preds_csv))
        for fig_path in figs_dir.rglob("*"):
            if fig_path.is_file():
                mlflow.log_artifact(str(fig_path), artifact_path="figures_adaptive_scp")

    print("Adaptive SCP metrics:\n", metrics_df.to_string(index=False))
    return metrics_df, predictions_df


def run_cqr(backbone: AttentionDenseNet, dataset, device: torch.device,
            output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train or load CQR head, calibrate, and evaluate on test split.

    Returns metrics_df, predictions_df, test_centers, test_rhi, test_true, test_errors.
    """
    n_total = len(dataset)
    head_idx, cal_idx, test_idx = split_indices(n_total, SEED)
    head_dataset = Subset(dataset, head_idx)
    cal_dataset = Subset(dataset, cal_idx)
    test_dataset = Subset(dataset, test_idx)

    head_loader = DataLoader(head_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    cal_loader = DataLoader(cal_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

    logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        run_name=f"{RUN_NAME}-CQR",
        tracking_uri=TRACKING_URI,
        log_model=False,
        synchronous=True,
    )

    if USE_EXISTING_CQR_HEAD and CQR_HEAD_CHECKPOINT_PATH:
        cqr_model = FrozenBackboneOneSidedCQR.load_from_checkpoint(
            CQR_HEAD_CHECKPOINT_PATH, backbone=backbone
        )
        cqr_model.to(device)
        print(f"Loaded existing CQR head from {CQR_HEAD_CHECKPOINT_PATH}")
        head_runtime_seconds = 0.0
    else:
        cqr_model = FrozenBackboneOneSidedCQR(backbone=backbone)
        ckpt_cb = ModelCheckpoint(
            monitor="val_loss", mode="min", filename="best_cqr_head", save_top_k=1
        )
        trainer = Trainer(
            max_epochs=MAX_EPOCHS_CQR_HEAD,
            callbacks=[
                ckpt_cb,
                LearningRateMonitor(),
                EarlyStopping(monitor="val_loss", mode="min", patience=15),
            ],
            logger=logger,
            log_every_n_steps=10,
            accelerator="auto",
            devices=1,
        )
        t0 = time.time()
        trainer.fit(cqr_model, train_dataloaders=head_loader, val_dataloaders=cal_loader)
        head_runtime_seconds = time.time() - t0
        best_head_ckpt = ckpt_cb.best_model_path
        if best_head_ckpt:
            cqr_model = FrozenBackboneOneSidedCQR.load_from_checkpoint(
                best_head_ckpt, backbone=backbone
            )
            print(f"Best CQR head checkpoint: {best_head_ckpt}")
        cqr_model.to(device)

    # Evaluate backbone + CQR head on cal and test splits
    cal_center, cal_rhi, cal_true = predict_center_and_rhi(cqr_model, cal_loader, device)
    test_center, test_rhi, test_true = predict_center_and_rhi(cqr_model, test_loader, device)

    cal_errors = euclidean_np(cal_center, cal_true)
    test_errors = euclidean_np(test_center, test_true)

    metrics_rows, prediction_rows = [], []

    for i, (sla, alpha) in enumerate(zip(SLA_LEVELS, ALPHAS)):
        # One-sided nonconformity: e - r_hi(x)
        cal_scores = cal_errors - cal_rhi[:, i]
        qhat = conformal_quantile(cal_scores, alpha)
        final_radius = test_rhi[:, i] + qhat
        covered = test_errors <= final_radius

        row = {
            "method": "CQR-OneSided",
            "target_assurance": float(sla),
            "alpha": float(alpha),
            "calibration_size": int(len(cal_errors)),
            "test_size": int(len(test_errors)),
            "qhat_mm": float(qhat),
            "calibration_mean_error_mm": float(np.mean(cal_errors)),
            "calibration_median_error_mm": float(np.median(cal_errors)),
            "test_mean_error_mm": float(np.mean(test_errors)),
            "test_median_error_mm": float(np.median(test_errors)),
            "test_p90_error_mm": float(np.quantile(test_errors, 0.90)),
            "test_p95_error_mm": float(np.quantile(test_errors, 0.95)),
            "true_coverage": float(np.mean(covered)),
            "mean_base_radius_mm": float(np.mean(test_rhi[:, i])),
            "mean_final_radius_mm": float(np.mean(final_radius)),
        }
        metrics_rows.append(row)

        sla_name = str(sla).replace(".", "p")
        for j in range(len(test_errors)):
            prediction_rows.append({
                "method": "CQR-OneSided",
                "target_assurance": float(sla),
                "sla_name": sla_name,
                "split": "test",
                "pred_x": float(test_center[j, 0]),
                "pred_y": float(test_center[j, 1]),
                "true_x": float(test_true[j, 0]),
                "true_y": float(test_true[j, 1]),
                "euclidean_error_mm": float(test_errors[j]),
                "base_radius_mm": float(test_rhi[j, i]),
                "qhat_mm": float(qhat),
                "final_radius_mm": float(final_radius[j]),
                "covered": float(covered[j]),
            })

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["target_assurance"]).reset_index(drop=True)
    predictions_df = pd.DataFrame(prediction_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "cqr_metrics.csv"
    preds_csv = output_dir / "cqr_predictions.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    predictions_df.to_csv(preds_csv, index=False)

    figs_dir = output_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    fig_cov = go.Figure()
    fig_cov.add_bar(x=metrics_df["target_assurance"], y=metrics_df["true_coverage"], name="empirical_coverage")
    fig_cov.add_scatter(x=metrics_df["target_assurance"], y=metrics_df["target_assurance"],
                        mode="lines", name="ideal", line=dict(color="black", dash="dot"))
    fig_cov.update_layout(title="CQR: empirical vs target coverage",
                          xaxis_title="target_assurance", yaxis_title="coverage")
    write_fig(fig_cov, figs_dir / "cqr_coverage_vs_target")

    fig_rad = go.Figure()
    fig_rad.add_bar(x=metrics_df["target_assurance"], y=metrics_df["mean_final_radius_mm"], name="mean_final_radius")
    fig_rad.update_layout(title="CQR: mean final radius vs target assurance",
                          xaxis_title="target_assurance", yaxis_title="radius (mm)")
    write_fig(fig_rad, figs_dir / "cqr_mean_radius_vs_target")

    # Log to MLflow
    run_id = logger.run_id
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("method", "CQR-OneSided")
        mlflow.log_metric("head_train_runtime_seconds", float(head_runtime_seconds))
        mlflow.log_metric("pooled_total_size", int(len(dataset)))
        mlflow.log_metric("head_train_size", int(len(head_dataset)))
        mlflow.log_metric("calibration_size", int(len(cal_dataset)))
        mlflow.log_metric("test_size", int(len(test_dataset)))
        for _, row in metrics_df.iterrows():
            prefix = f"cqr_{str(row['target_assurance']).replace('.', 'p')}"
            for col in metrics_df.columns:
                if col in {"method", "target_assurance"}:
                    continue
                value = row[col]
                if pd.notna(value) and np.isscalar(value):
                    try:
                        mlflow.log_metric(f"{prefix}_{col}", float(value))
                    except Exception:
                        pass
        mlflow.log_artifact(str(metrics_csv))
        mlflow.log_artifact(str(preds_csv))
        for fig_path in figs_dir.rglob("*"):
            if fig_path.is_file():
                mlflow.log_artifact(str(fig_path), artifact_path="figures_cqr")

    print("CQR metrics:\n", metrics_df.to_string(index=False))
    return metrics_df, predictions_df, test_center, test_rhi, test_true, test_errors


# ============================================================================
# Single-sample uncertainty visualization
# ============================================================================


def plot_single_sample_uncertainty(
    sample_idx: int,
    sla_level: float,
    sla_levels: List[float],
    test_centers: np.ndarray,
    test_rhi: np.ndarray,
    test_true: np.ndarray,
    test_errors: np.ndarray,
    method_name: str,
    output_dir: Path,
) -> Path:
    """Create and save a plot showing the prediction disk and ground truth for one sample.

    For CQR, `test_rhi` should be an array of shape [n_test, n_sla_levels] containing
    either the final conformal radii or the base radii depending on how it is passed
    by the caller. The caller is responsible for ensuring that `test_rhi` represents
    the final prediction radius for the selected SLA.
    """
    if sample_idx < 0 or sample_idx >= len(test_centers):
        raise IndexError("SINGLE_SAMPLE_INDEX out of range for the test split.")

    # Find SLA index
    if sla_level not in sla_levels:
        raise ValueError(f"SINGLE_SAMPLE_SLA={sla_level} not in SLA_LEVELS={sla_levels}")
    sla_idx = sla_levels.index(sla_level)

    center = test_centers[sample_idx]
    true = test_true[sample_idx]
    radius = test_rhi[sample_idx, sla_idx]
    error = test_errors[sample_idx]
    covered = error <= radius

    # Parametric circle
    t = np.linspace(0, 2 * np.pi, 200)
    circle_x = center[0] + radius * np.cos(t)
    circle_y = center[1] + radius * np.sin(t)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=circle_x,
        y=circle_y,
        mode="lines",
        name=f"Prediction disk (radius={radius:.1f} mm)",
        line=dict(color="royalblue"),
    ))
    fig.add_trace(go.Scatter(
        x=[center[0]],
        y=[center[1]],
        mode="markers",
        name="Predicted location",
        marker=dict(color="orange", size=10, symbol="x"),
    ))
    fig.add_trace(go.Scatter(
        x=[true[0]],
        y=[true[1]],
        mode="markers",
        name="True location",
        marker=dict(color="green", size=10),
    ))

    fig.update_layout(
        title=(
            f"{method_name}: Single-sample uncertainty (index={sample_idx}, "
            f"SLA={sla_level:.2f}, covered={covered})"
        ),
        xaxis_title="x (mm)",
        yaxis_title="y (mm)",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(x=0.02, y=0.98),
    )

    figs_dir = output_dir / "figures_single_samples"
    figs_dir.mkdir(parents=True, exist_ok=True)
    base = figs_dir / f"single_sample_{method_name}_idx{sample_idx}_sla{str(sla_level).replace('.', 'p')}"
    write_fig(fig, base)
    return base.with_suffix(".html")


# ============================================================================
# Main entry point
# ============================================================================


def main():
    seed_everything(SEED, workers=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    if abs((HEAD_TRAIN_FRACTION + CALIBRATION_FRACTION + TEST_FRACTION) - 1.0) > 1e-9:
        raise ValueError("Split fractions HEAD_TRAIN_FRACTION + CALIBRATION_FRACTION + TEST_FRACTION must sum to 1.0")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load or train backbone
    if USE_EXISTING_BACKBONE_CHECKPOINT:
        backbone = AttentionDenseNet.load_from_checkpoint(BACKBONE_CHECKPOINT_PATH)
        print(f"Loaded backbone checkpoint from {BACKBONE_CHECKPOINT_PATH}")
    else:
        if not TRAIN_ADN_IF_NEEDED:
            raise RuntimeError("USE_EXISTING_BACKBONE_CHECKPOINT is False but TRAIN_ADN_IF_NEEDED is also False. "
                               "Set TRAIN_ADN_IF_NEEDED=True or provide a backbone checkpoint.")

        print(f"Training AttentionDenseNet backbone from scratch with augment_method={ADN_AUGMENT_METHOD}...")
        mlflow.set_tracking_uri(TRACKING_URI)
        adn_logger = MLFlowLogger(
            experiment_name=EXPERIMENT_NAME,
            run_name=f"{RUN_NAME}-ADN-train-{ADN_AUGMENT_METHOD}",
            tracking_uri=TRACKING_URI,
            log_model=False,
            synchronous=True,
        )

        datamodule = NomadicLocalizationDataModule(
            data_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            num_users=NUM_USERS,
            num_samples=NUM_SAMPLES,
            mode="train_static",
            augment_method=ADN_AUGMENT_METHOD,
        )

        backbone = AttentionDenseNet()

        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=20)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="best_adn_backbone",
            save_top_k=1,
        )
        lr_logger = LearningRateMonitor()

        trainer = Trainer(
            max_epochs=ADN_MAX_EPOCHS,
            callbacks=[early_stopping, lr_logger, checkpoint_callback],
            logger=adn_logger,
            log_every_n_steps=10,
            accelerator="auto",
            devices=1,
        )

        start = time.time()
        trainer.fit(backbone, datamodule=datamodule)
        train_runtime_seconds = time.time() - start

        best_ckpt = checkpoint_callback.best_model_path
        if not best_ckpt:
            raise RuntimeError("No backbone checkpoint was saved by ModelCheckpoint.")

        print(f"Best ADN backbone checkpoint: {best_ckpt}")
        backbone = AttentionDenseNet.load_from_checkpoint(best_ckpt)

        with mlflow.start_run(run_id=adn_logger.run_id):
            mlflow.log_param("adn_augment_method", ADN_AUGMENT_METHOD)
            mlflow.log_param("adn_max_epochs", ADN_MAX_EPOCHS)
            mlflow.log_metric("adn_train_runtime_seconds", train_runtime_seconds)
            mlflow.log_artifact(best_ckpt, artifact_path="adn_backbone_checkpoints")

    backbone.eval()
    backbone.to(device)

    # Build dataset depending on evaluation mode
    if EVAL_MODE.lower() == "pooled":
        dataset = build_pooled_dataset(DATA_DIR, SCENARIO_IDS)
        eval_tag = "pooled_scenarios"
    elif EVAL_MODE.lower() == "single":
        dataset = build_scenario_dataset(DATA_DIR, SINGLE_SCENARIO_ID)
        eval_tag = f"single_scenario_{SINGLE_SCENARIO_ID}"
    else:
        raise ValueError("EVAL_MODE must be 'pooled' or 'single'")

    print(f"Evaluation mode: {eval_tag}, total samples = {len(dataset)}")

    # Run selected method
    if METHOD.upper() == "ASCP":
        metrics_df, preds_df = run_adaptive_scp(backbone, dataset, device, OUTPUT_DIR / eval_tag / "ASCP")

        if ENABLE_SINGLE_SAMPLE_EVAL:
            # For ASCP, the final radius is symmetric; to reuse the plotting helper,
            # treat sigma * qhat per SLA as radius in an [n_test, n_sla] matrix.
            # Recompute on the test split from stored predictions for consistency.
            # Group predictions by SLA and rebuild arrays.
            single_dir = OUTPUT_DIR / eval_tag / "ASCP"
            for sla in SLA_LEVELS:
                sla_name = str(sla).replace(".", "p")
                sub = preds_df[preds_df["target_assurance"] == sla].copy().reset_index(drop=True)
                if sub.empty:
                    continue
                # All sub rows are for all test samples; pivot by test index order
                n_test = sub.shape[0]
                # Build arrays
                test_centers = sub[["pred_x", "pred_y"]].values
                test_true = sub[["true_x", "true_y"]].values
                test_errors = sub["euclidean_error_mm"].values
                # For ASCP, radius is 'final_radius_mm'
                # We build [n_test, 1] and pass with SLA_LEVELS=[sla]
                test_rhi = sub[["final_radius_mm"]].values
                html_path = plot_single_sample_uncertainty(
                    SINGLE_SAMPLE_INDEX,
                    sla,
                    [sla],
                    test_centers,
                    test_rhi,
                    test_true,
                    test_errors,
                    method_name="AdaptiveSCP-Scale",
                    output_dir=single_dir,
                )
                with mlflow.start_run(run_name=f"single_sample_ASCP_{eval_tag}", nested=True):
                    mlflow.log_param("method", "AdaptiveSCP-Scale")
                    mlflow.log_param("eval_mode", eval_tag)
                    mlflow.log_param("sample_index", SINGLE_SAMPLE_INDEX)
                    mlflow.log_param("sla_level", sla)
                    mlflow.log_artifact(str(html_path), artifact_path="single_sample_plots")
                # only generate for first SLA for simplicity
                break

    elif METHOD.upper() == "CQR":
        metrics_df, preds_df, test_centers, test_rhi_base, test_true, test_errors = run_cqr(
            backbone, dataset, device, OUTPUT_DIR / eval_tag / "CQR"
        )

        # For plotting single sample, we need final conformal radius per SLA.
        # Reconstruct per-SLA qhat from metrics_df and combine with base radii.
        qhat_map = {row["target_assurance"]: row["qhat_mm"] for _, row in metrics_df.iterrows()}
        final_radii = np.zeros_like(test_rhi_base)
        for i, sla in enumerate(SLA_LEVELS):
            q = qhat_map.get(sla, 0.0)
            final_radii[:, i] = test_rhi_base[:, i] + q

        if ENABLE_SINGLE_SAMPLE_EVAL:
            single_dir = OUTPUT_DIR / eval_tag / "CQR"
            html_path = plot_single_sample_uncertainty(
                SINGLE_SAMPLE_INDEX,
                SINGLE_SAMPLE_SLA,
                SLA_LEVELS,
                test_centers,
                final_radii,
                test_true,
                test_errors,
                method_name="CQR-OneSided",
                output_dir=single_dir,
            )
            with mlflow.start_run(run_name=f"single_sample_CQR_{eval_tag}", nested=True):
                mlflow.log_param("method", "CQR-OneSided")
                mlflow.log_param("eval_mode", eval_tag)
                mlflow.log_param("sample_index", SINGLE_SAMPLE_INDEX)
                mlflow.log_param("sla_level", SINGLE_SAMPLE_SLA)
                mlflow.log_artifact(str(html_path), artifact_path="single_sample_plots")

    else:
        raise ValueError("METHOD must be 'ASCP' or 'CQR'")


if __name__ == "__main__":
    main()
