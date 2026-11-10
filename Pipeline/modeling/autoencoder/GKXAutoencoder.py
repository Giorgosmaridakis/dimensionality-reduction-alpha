"""
GKX conditional autoencoder implementation.

Unlike a standard feature autoencoder, this model prices realized returns with
conditional betas and latent factors. NeuralFeatureExtractor wraps it and
converts the learned betas/factor forecast into leakage-safe expected returns.
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Sampler, TensorDataset
from typing import Optional

# Guard against OpenMP segfaults on macOS ARM (Apple Silicon).
if torch.backends.mps.is_available() or (
    not torch.cuda.is_available()
    and hasattr(os, "uname")
    and os.uname().machine == "arm64"
):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)


def _to_float_tensor(values) -> torch.Tensor:
    """Convert numpy, pandas, or tensor inputs to a float32 torch tensor."""
    if isinstance(values, torch.Tensor):
        return values.detach().clone().float()
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    return torch.as_tensor(np.asarray(values), dtype=torch.float32)


def _to_long_tensor(values) -> torch.Tensor:
    """Convert numpy, pandas, or tensor inputs to an int64 torch tensor."""
    if isinstance(values, torch.Tensor):
        return values.detach().clone().long()
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    return torch.as_tensor(np.asarray(values), dtype=torch.long)


class DateBatchSampler(Sampler):
    """Yield one cross-sectional batch of row indices per date code."""

    def __init__(self, date_codes):
        date_codes = torch.as_tensor(date_codes, dtype=torch.long).view(-1)
        self.batches = [
            torch.where(date_codes == date_code)[0].tolist()
            for date_code in torch.unique(date_codes, sorted=True)
        ]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def compute_managed_portfolios(
    characteristics,
    realized_returns,
    ridge: float = 0.0,
) -> torch.Tensor:
    """
    Compute characteristic-managed portfolio returns.

    For each cross-section, the GKX factor network receives managed portfolio
    returns rather than raw firm characteristics:

        x_t = (Z_{t-1}' Z_{t-1})^{-1} Z_{t-1}' r_t

    A constant column is always prepended to `Z`, adding the equal-weighted
    market portfolio component used by GKX. This is explicit rather than
    inferred from all-one feature values, because a real preprocessed feature
    can also be all ones in a small cross-section.

    `torch.linalg.pinv` is used instead of a direct inverse for numerical
    stability, so the helper also works when characteristics are collinear.
    """
    z = _to_float_tensor(characteristics)
    r = _to_float_tensor(realized_returns).view(-1, 1)

    if z.ndim != 2:
        raise ValueError("characteristics must be a 2D array/tensor with shape [n_assets, input_dim].")
    if r.shape[0] != z.shape[0]:
        raise ValueError("realized_returns must have the same number of rows as characteristics.")

    constant = torch.ones((z.shape[0], 1), dtype=z.dtype, device=z.device)
    z = torch.cat([constant, z], dim=1)

    gram = z.T.matmul(z)
    if ridge > 0:
        gram = gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    managed = torch.linalg.pinv(gram).matmul(z.T).matmul(r)
    return managed.view(-1)


def expand_managed_portfolio(managed_portfolio, n_rows: int) -> torch.Tensor:
    """Repeat one cross-sectional managed portfolio vector for each stock row."""
    x_t = _to_float_tensor(managed_portfolio).view(1, -1)
    return x_t.repeat(n_rows, 1)


class GKXConditionalAutoencoder(nn.Module):
    """
    Conditional autoencoder asset-pricing model from Gu, Kelly & Xiu.

    This is different from a standard autoencoder because it reconstructs or
    prices realized stock returns, not characteristics:

        r_{i,t} = beta_{i,t-1}' f_t + u_{i,t}

    `beta_net` learns conditional factor exposures from characteristics.
    `factor_net` learns latent factors from characteristic-managed portfolio
    returns. The dot product beta'f reconstructs/prices realized stock returns.
    """

    BETA_HIDDEN_DIMS = [64, 48, 32]

    def __init__(
        self,
        input_dim: int = 82,
        latent_dim: int = 8,
        managed_portfolio_dim: Optional[int] = None,
        hidden_dims=None,
    ):
        super().__init__()

        # Default thesis architecture for the beta network:
        # characteristics -> 64 -> 48 -> 32 -> K conditional betas.
        if hidden_dims is None:
            hidden_dims = self.BETA_HIDDEN_DIMS

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.managed_portfolio_dim = managed_portfolio_dim or (input_dim + 1)
        self.hidden_dims = list(hidden_dims)

        # beta_net learns conditional factor exposures beta_{i,t-1}
        # from firm characteristics z_{i,t-1}.
        beta_layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            beta_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        # Linear output, no activation, as required for conditional betas.
        beta_layers.append(nn.Linear(prev_dim, latent_dim))
        self.beta_net = nn.Sequential(*beta_layers)

        # The empirical GKX factor network is a single linear layer so each
        # latent factor remains interpretable as a portfolio of managed returns.
        self.factor_net = nn.Linear(self.managed_portfolio_dim, latent_dim)

    def forward(self, characteristics: torch.Tensor, managed_portfolios: torch.Tensor):
        """
        Parameters
        ----------
        characteristics:
            [batch_size, input_dim] firm characteristics z_{i,t-1}.
        managed_portfolios:
            [batch_size, managed_portfolio_dim] managed portfolio returns x_t.

        Returns
        -------
        pred_return:
            [batch_size] beta_{i,t-1}' f_t predicted excess returns.
        beta:
            [batch_size, K] conditional factor exposures.
        factor:
            [batch_size, K] latent factors.
        """
        beta = self.beta_net(characteristics)
        factor = self.factor_net(managed_portfolios)
        pred_return = torch.sum(beta * factor, dim=1)
        return pred_return, beta, factor

    def encode(self, characteristics: torch.Tensor) -> torch.Tensor:
        """Return conditional betas as the learned characteristic representation."""
        return self.beta_net(characteristics)


def build_optimizer(
    model: GKXConditionalAutoencoder,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


criterion = nn.MSELoss()


def train_autoencoder_with_early_stopping(
    model: GKXConditionalAutoencoder,
    X_train,
    X_managed_train,
    y_train,
    X_val,
    X_managed_val,
    y_val,
    date_codes_train=None,
    date_codes_val=None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    l1_lambda: float = 0.0,
    batch_size: int = 64,
    max_epochs: int = 200,
    patience: int = 10,
    lr_scheduler_patience: int = 5,
    lr_scheduler_factor: float = 0.5,
    random_seed: int = 42,
    verbose: bool = True,
    progress_bar=None,
):
    """Train the GKX conditional autoencoder with MSE pricing loss."""
    if date_codes_train is None:
        raise ValueError(
            "date_codes_train is required for GKX conditional autoencoder training; "
            "batches must contain one date/cross-section at a time."
        )
    if date_codes_val is None:
        raise ValueError(
            "date_codes_val is required for GKX conditional autoencoder validation; "
            "batches must contain one date/cross-section at a time."
        )

    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Training GKXConditionalAutoencoder on {device} | date-grouped cross-sectional batches")
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
    )

    train_char = _to_float_tensor(X_train)
    train_managed = _to_float_tensor(X_managed_train)
    train_y = _to_float_tensor(y_train).view(-1)
    train_dates = _to_long_tensor(date_codes_train).view(-1)
    val_char = _to_float_tensor(X_val)
    val_managed = _to_float_tensor(X_managed_val)
    val_y = _to_float_tensor(y_val).view(-1)
    val_dates = _to_long_tensor(date_codes_val).view(-1)

    if train_char.shape[0] != train_managed.shape[0] or train_char.shape[0] != train_y.shape[0]:
        raise ValueError("X_train, X_managed_train, and y_train must have the same number of rows.")
    if val_char.shape[0] != val_managed.shape[0] or val_char.shape[0] != val_y.shape[0]:
        raise ValueError("X_val, X_managed_val, and y_val must have the same number of rows.")
    if train_dates.shape[0] != train_char.shape[0]:
        raise ValueError("date_codes_train must have the same number of rows as X_train.")
    if val_dates.shape[0] != val_char.shape[0]:
        raise ValueError("date_codes_val must have the same number of rows as X_val.")
    if train_managed.shape[1] != model.managed_portfolio_dim:
        raise ValueError(
            "X_managed_train has managed portfolio dimension "
            f"{train_managed.shape[1]}, but model expects {model.managed_portfolio_dim}."
        )
    if val_managed.shape[1] != model.managed_portfolio_dim:
        raise ValueError(
            "X_managed_val has managed portfolio dimension "
            f"{val_managed.shape[1]}, but model expects {model.managed_portfolio_dim}."
        )

    train_dataset = TensorDataset(train_char, train_managed, train_y, train_dates)
    val_dataset = TensorDataset(val_char, val_managed, val_y, val_dates)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=DateBatchSampler(train_dates),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=DateBatchSampler(val_dates),
    )

    best_val_loss = float("inf")
    best_val_rank_ic = -float("inf")
    epochs_no_improve = 0
    best_model_state = None
    history = {
        "epoch": [],
        "train_loss": [],
        "train_mse_loss": [],
        "val_loss": [],
        "val_mse_loss": [],
        "val_rank_ic": [],
        "val_rank_icir": [],
        "val_forecast_rank_ic": [],
        "val_forecast_rank_icir": [],
        "val_pricing_rank_ic": [],
        "val_pricing_rank_icir": [],
        "lr": [],
        "next_lr": [],
        "best_val_loss": [],
        "best_val_rank_ic": [],
        "is_best_epoch": [],
        "epochs_no_improve": [],
        "early_stopped": [],
        "train_samples": [],
        "val_samples": [],
        "latent_dim": [],
        "learning_rate": [],
        "weight_decay": [],
        "l1_lambda": [],
        "batch_size": [],
        "max_epochs": [],
        "patience": [],
        "random_seed": [],
        "shuffle": [],
        "date_grouped_batches": [],
        "train_batches": [],
        "val_batches": [],
        "device": [],
    }
    train_steps_per_epoch = len(train_loader)
    last_val_error = "pending"
    last_val_rank_ic = "pending"

    for epoch in range(max_epochs):
        if progress_bar is not None:
            progress_bar.reset(total=train_steps_per_epoch)
            progress_bar.set_description(f"GKX-CA epoch {epoch + 1}/{max_epochs}")

        model.train()
        train_mse_loss = 0.0

        for batch_char, batch_managed, batch_y, batch_dates in train_loader:
            batch_char = batch_char.to(device)
            batch_managed = batch_managed.to(device)
            batch_y = batch_y.to(device)
            batch_dates = batch_dates.to(device)
            if torch.unique(batch_dates).numel() != 1:
                raise RuntimeError("Internal error: GKX training batch contains more than one date code.")

            # Use one x_t per date so every stock in the cross-section shares f_t.
            batch_managed = batch_managed[0:1].repeat(batch_char.size(0), 1)

            optimizer.zero_grad()
            pred_return, _, _ = model(batch_char, batch_managed)
            mse_loss = loss_fn(pred_return, batch_y)
            loss = mse_loss
            if l1_lambda > 0:
                l1_penalty = sum(param.abs().sum() for param in model.parameters())
                loss = loss + l1_lambda * l1_penalty
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_mse_loss += mse_loss.item() * batch_char.size(0)

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "epoch": epoch + 1,
                    "train_error": f"{mse_loss.item():.4g}",
                    "val_error": last_val_error,
                    "val_rank_ic": last_val_rank_ic,
                }, refresh=False)

        train_mse_loss /= len(train_loader.dataset)
        train_loss = train_mse_loss

        model.eval()
        train_factors = []
        with torch.no_grad():
            for _, batch_managed, _, batch_dates in train_loader:
                batch_managed = batch_managed.to(device)
                batch_dates = batch_dates.to(device)
                if torch.unique(batch_dates).numel() != 1:
                    raise RuntimeError("Internal error: GKX training batch contains more than one date code.")
                train_factors.append(model.factor_net(batch_managed[0:1]))
        factor_forecast = torch.cat(train_factors, dim=0).mean(dim=0)

        val_mse_loss = 0.0
        forecast_daily_ics = []
        pricing_daily_ics = []
        with torch.no_grad():
            for batch_char, batch_managed, batch_y, batch_dates in val_loader:
                batch_char = batch_char.to(device)
                batch_managed = batch_managed.to(device)
                batch_y = batch_y.to(device)
                batch_dates = batch_dates.to(device)
                if torch.unique(batch_dates).numel() != 1:
                    raise RuntimeError("Internal error: GKX validation batch contains more than one date code.")
                batch_managed = batch_managed[0:1].repeat(batch_char.size(0), 1)
                pred_return, beta, _ = model(batch_char, batch_managed)
                mse_loss = loss_fn(pred_return, batch_y)
                val_mse_loss += mse_loss.item() * batch_char.size(0)

                forecast_return = beta.matmul(factor_forecast)
                forecast_np = forecast_return.detach().cpu().numpy()
                pricing_np = pred_return.detach().cpu().numpy()
                actual_np = batch_y.detach().cpu().numpy()
                if (
                    forecast_np.shape[0] >= 2
                    and np.std(forecast_np) > 1e-8
                    and np.std(actual_np) > 1e-8
                ):
                    ic = spearmanr(forecast_np, actual_np).correlation
                    if np.isfinite(ic):
                        forecast_daily_ics.append(float(ic))
                if (
                    pricing_np.shape[0] >= 2
                    and np.std(pricing_np) > 1e-8
                    and np.std(actual_np) > 1e-8
                ):
                    pricing_ic = spearmanr(pricing_np, actual_np).correlation
                    if np.isfinite(pricing_ic):
                        pricing_daily_ics.append(float(pricing_ic))

        val_mse_loss /= len(val_loader.dataset)
        val_loss = val_mse_loss
        if forecast_daily_ics:
            daily_ics_array = np.asarray(forecast_daily_ics, dtype=np.float64)
            val_rank_ic = float(np.mean(daily_ics_array))
            val_rank_icir = float(val_rank_ic / (np.std(daily_ics_array) + 1e-8))
        else:
            val_rank_ic = -float("inf")
            val_rank_icir = 0.0
        if pricing_daily_ics:
            pricing_ics_array = np.asarray(pricing_daily_ics, dtype=np.float64)
            val_pricing_rank_ic = float(np.mean(pricing_ics_array))
            val_pricing_rank_icir = float(val_pricing_rank_ic / (np.std(pricing_ics_array) + 1e-8))
        else:
            val_pricing_rank_ic = -float("inf")
            val_pricing_rank_icir = 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_rank_ic if np.isfinite(val_rank_ic) else -1.0)
        next_lr = optimizer.param_groups[0]["lr"]

        if progress_bar is not None:
            last_val_error = f"{val_mse_loss:.4g}"
            last_val_rank_ic = f"{val_rank_ic:.4g}"
            progress_bar.set_postfix({
                "epoch": epoch + 1,
                "train_error": f"{train_mse_loss:.4g}",
                "val_error": last_val_error,
                "val_rank_ic": last_val_rank_ic,
            }, refresh=True)

        is_best_epoch = (best_model_state is None) or (val_rank_ic > best_val_rank_ic)
        if is_best_epoch:
            best_val_rank_ic = val_rank_ic
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy({
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            })
        else:
            epochs_no_improve += 1

        early_stopped = epochs_no_improve >= patience
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_mse_loss"].append(train_mse_loss)
        history["val_loss"].append(val_loss)
        history["val_mse_loss"].append(val_mse_loss)
        history["val_rank_ic"].append(val_rank_ic)
        history["val_rank_icir"].append(val_rank_icir)
        history["val_forecast_rank_ic"].append(val_rank_ic)
        history["val_forecast_rank_icir"].append(val_rank_icir)
        history["val_pricing_rank_ic"].append(val_pricing_rank_ic)
        history["val_pricing_rank_icir"].append(val_pricing_rank_icir)
        history["lr"].append(current_lr)
        history["next_lr"].append(next_lr)
        history["best_val_loss"].append(best_val_loss)
        history["best_val_rank_ic"].append(best_val_rank_ic)
        history["is_best_epoch"].append(is_best_epoch)
        history["epochs_no_improve"].append(epochs_no_improve)
        history["early_stopped"].append(early_stopped)
        history["train_samples"].append(len(train_loader.dataset))
        history["val_samples"].append(len(val_loader.dataset))
        history["latent_dim"].append(model.latent_dim)
        history["learning_rate"].append(learning_rate)
        history["weight_decay"].append(weight_decay)
        history["l1_lambda"].append(l1_lambda)
        history["batch_size"].append(batch_size)
        history["max_epochs"].append(max_epochs)
        history["patience"].append(patience)
        history["random_seed"].append(random_seed)
        history["shuffle"].append(False)
        history["date_grouped_batches"].append(True)
        history["train_batches"].append(len(train_loader))
        history["val_batches"].append(len(val_loader))
        history["device"].append(str(device))

        if early_stopped:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best val Forecast Rank IC: {best_val_rank_ic:.4f} "
                    f"(MSE: {best_val_loss:.6f})"
                )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.to("cpu")
    return model, history


# Backward-compatible import alias for existing pipeline code. The underlying
# model is now the GKX conditional autoencoder, not a reconstruction autoencoder.
GKXAutoencoder = GKXConditionalAutoencoder
