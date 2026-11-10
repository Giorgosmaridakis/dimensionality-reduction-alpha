"""
Active FactorVAE implementation used by features/neural_extractors.py.

Training uses realized returns through the posterior path, while validation and
test signals use the prior path so expected returns can be produced without
same-date realized returns.
"""

import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr

# Guard against OpenMP segfaults on macOS ARM (Apple Silicon).
# PyTorch's GRU backward pass can crash when multiple OMP threads contend.
if torch.backends.mps.is_available() or (
    not torch.cuda.is_available()
    and hasattr(os, "uname")
    and os.uname().machine == "arm64"
):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)



class FactorFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.linear = nn.Linear(input_dim, input_dim)
        self.activation = nn.LeakyReLU()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear(x_seq))
        stock_latent, _ = self.gru(x)
        return stock_latent[:, -1, :]


class FactorEncoder(nn.Module):
    def __init__(self, num_factors: int, num_portfolios: int, hidden_size: int):
        super().__init__()
        self.num_factors = num_factors
        self.num_portfolios = num_portfolios
        self.portfolio_layer = nn.Linear(hidden_size, num_portfolios)
        self.linear_mu = nn.Linear(num_portfolios, num_factors)
        self.linear_sigma = nn.Linear(num_portfolios, num_factors)
        self.softplus = nn.Softplus()

    def forward(self, stock_latent: torch.Tensor, returns: torch.Tensor):
        if returns.dim() == 1:
            returns = returns.unsqueeze(1)

        weights = torch.softmax(self.portfolio_layer(stock_latent), dim=0)
        portfolio_return = weights.transpose(1, 0).matmul(returns).transpose(1, 0)
        factor_mu = self.linear_mu(portfolio_return)
        factor_sigma = self.softplus(self.linear_sigma(portfolio_return)) + 1e-6
        return factor_mu.squeeze(0), factor_sigma.squeeze(0)


class AlphaLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, stock_latent: torch.Tensor):
        h = self.activation(self.linear(stock_latent))
        alpha_mu = self.mu_layer(h)
        alpha_sigma = self.softplus(self.sigma_layer(h)) + 1e-6
        return alpha_mu, alpha_sigma


class BetaLayer(nn.Module):
    def __init__(self, hidden_size: int, num_factors: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_factors)

    def forward(self, stock_latent: torch.Tensor):
        return self.linear(stock_latent)


class FactorDecoder(nn.Module):
    def __init__(self, hidden_size: int, num_factors: int):
        super().__init__()
        self.alpha_layer = AlphaLayer(hidden_size)
        self.beta_layer = BetaLayer(hidden_size, num_factors)

    @staticmethod
    def reparameterize(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def expected_return(self, stock_latent: torch.Tensor, factor_mu: torch.Tensor):
        alpha_mu, alpha_sigma = self.alpha_layer(stock_latent)
        beta = self.beta_layer(stock_latent)
        expected = alpha_mu + beta.matmul(factor_mu.view(-1, 1))
        return expected, alpha_mu, alpha_sigma, beta

    def forward(self, stock_latent: torch.Tensor, factor_mu: torch.Tensor, factor_sigma: torch.Tensor):
        mu, alpha_mu, alpha_sigma, beta = self.expected_return(stock_latent, factor_mu)
        sigma = torch.sqrt(alpha_sigma.pow(2) + beta.pow(2).matmul(factor_sigma.pow(2).view(-1, 1)) + 1e-6)
        reconstruction = self.reparameterize(mu, sigma) if self.training else mu
        return reconstruction, mu, sigma, alpha_mu, alpha_sigma, beta


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size))
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, stock_latent: torch.Tensor):
        keys = self.key_layer(stock_latent)
        values = self.value_layer(stock_latent)

        q = self.query.view(1, -1)
        scores = F.cosine_similarity(q, keys, dim=1)
        scores = torch.relu(scores)

        weights = scores / (scores.sum() + 1e-8)
        return weights.matmul(values)


class FactorPredictor(nn.Module):
    def __init__(self, hidden_size: int, num_factors: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_factors = num_factors
        self.attention_layers = nn.ModuleList([AttentionLayer(hidden_size) for _ in range(num_factors)])
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, stock_latent: torch.Tensor):
        factor_contexts = [attention(stock_latent) for attention in self.attention_layers]
        h = torch.stack(factor_contexts, dim=0)
        h = self.activation(self.linear(h))
        pred_mu = self.mu_layer(h).view(-1)
        pred_sigma = self.softplus(self.sigma_layer(h)).view(-1) + 1e-6
        return pred_mu, pred_sigma


class FactorVAEModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 32,
        num_factors: int = 8,
        num_portfolios: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_factors = num_factors
        self.num_portfolios = num_portfolios

        self.feature_extractor = FactorFeatureExtractor(input_dim=input_dim, hidden_size=hidden_size)
        self.factor_encoder = FactorEncoder(
            num_factors=num_factors,
            num_portfolios=num_portfolios,
            hidden_size=hidden_size,
        )
        self.factor_decoder = FactorDecoder(hidden_size=hidden_size, num_factors=num_factors)
        self.factor_predictor = FactorPredictor(hidden_size=hidden_size, num_factors=num_factors)

    @staticmethod
    def kl_divergence(mu_q, sigma_q, mu_p, sigma_p):
        return (
            torch.log(sigma_p / sigma_q)
            + (sigma_q.pow(2) + (mu_q - mu_p).pow(2)) / (2 * sigma_p.pow(2))
            - 0.5
        ).sum()

    def forward(self, x_seq: torch.Tensor, returns: torch.Tensor, kl_weight: float = 1e-3):
        stock_latent = self.feature_extractor(x_seq)
        posterior_mu, posterior_sigma = self.factor_encoder(stock_latent, returns)
        prior_mu, prior_sigma = self.factor_predictor(stock_latent)
        reconstruction, expected_return, sigma, alpha_mu, alpha_sigma, beta = self.factor_decoder(
            stock_latent,
            posterior_mu,
            posterior_sigma,
        )
        reconstruction_loss = 0.5 * (
            ((returns - expected_return) / sigma).pow(2)
            + 2 * torch.log(sigma)
        ).mean()
        #kl_loss = self.kl_divergence(posterior_mu, posterior_sigma, prior_mu, prior_sigma)
        #total_loss = reconstruction_loss + kl_loss
        kl_loss = self.kl_divergence(
        posterior_mu,
        posterior_sigma,
        prior_mu,
        prior_sigma,
        )

        kl_loss = kl_loss / returns.shape[0]

        total_loss = reconstruction_loss + kl_weight * kl_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "reconstruction": reconstruction,
            "expected_return": expected_return,
            "stock_latent": stock_latent,
            "posterior_mu": posterior_mu,
            "posterior_sigma": posterior_sigma,
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "alpha_mu": alpha_mu,
            "alpha_sigma": alpha_sigma,
            "beta": beta,
        }

    def encode_prior_features(self, x_seq: torch.Tensor):
        stock_latent = self.feature_extractor(x_seq)
        prior_mu, prior_sigma = self.factor_predictor(stock_latent)
        expected_return, alpha_mu, alpha_sigma, beta = self.factor_decoder.expected_return(stock_latent, prior_mu)
        return {
            "stock_latent": stock_latent,
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "alpha_mu": alpha_mu,
            "alpha_sigma": alpha_sigma,
            "beta": beta,
            "expected_return": expected_return,
        }


def train_factor_vae_with_early_stopping(
    model: FactorVAEModel,
    X_seq,
    y_full,
    train_groups,
    val_groups,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    max_epochs: int = 200,
    patience: int = 10,
    random_seed: int = 42,
    verbose: bool = True,
    progress_bar=None,
):
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Training FactorVAE on {device} | date-batch mode")

    model.to(device)

    all_x = torch.as_tensor(X_seq, dtype=torch.float32)
    all_y = torch.as_tensor(y_full, dtype=torch.float32).view(-1, 1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    train_samples = sum(len(g) for g in train_groups)
    val_samples = sum(len(g) for g in val_groups)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_reconstruction_loss": [],
        "train_kl_loss": [],
        "val_loss": [],
        "val_reconstruction_loss": [],
        "val_kl_loss": [],
        "val_rank_ic": [],
        "val_rank_icir": [],
        "lr": [],
        "next_lr": [],
        "best_val_loss": [],
        "is_best_epoch": [],
        "epochs_no_improve": [],
        "early_stopped": [],
        "train_samples": [],
        "val_samples": [],
        "hidden_size": [],
        "num_factors": [],
        "num_portfolios": [],
        "learning_rate": [],
        "weight_decay": [],
        "max_epochs": [],
        "patience": [],
        "random_seed": [],
        "device": [],
    }
    
    last_val_error = "pending"
    best_val_rank_ic = -float("inf")

    for epoch in range(max_epochs):
        anneal_epochs = 50
        base_kl_weight = 1e-3
        current_kl_weight = base_kl_weight * min(1.0, epoch / anneal_epochs)
        
        if progress_bar is not None:
            progress_bar.reset(total=len(train_groups))
            progress_bar.set_description(
                f"FactorVAE epoch {epoch + 1}/{max_epochs}"
            )

        model.train()

        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        train_n = 0

        rng = np.random.default_rng(random_seed + epoch)
        shuffled_train_groups = list(train_groups)
        rng.shuffle(shuffled_train_groups)

        for group_idx in shuffled_train_groups:
            batch_x = all_x[group_idx].to(device)
            batch_y = all_y[group_idx].to(device)

            optimizer.zero_grad()

            outputs = model(batch_x, batch_y, kl_weight=current_kl_weight)
            outputs["loss"].backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            n = batch_x.size(0)

            train_loss += outputs["loss"].item() * n
            train_recon += outputs["reconstruction_loss"].item() * n
            train_kl += outputs["kl_loss"].item() * n
            train_n += n

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "epoch": epoch + 1,
                    "train_error": f"{outputs['loss'].item():.4g}",
                    "val_error": last_val_error,
                }, refresh=False)

        train_loss /= max(train_n, 1)
        train_recon /= max(train_n, 1)
        train_kl /= max(train_n, 1)

        model.eval()

        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        val_n = 0
        
        daily_ics = []

        with torch.no_grad():
            for group_idx in val_groups:
                batch_x = all_x[group_idx].to(device)
                batch_y = all_y[group_idx].to(device)

                prior_outputs = model.encode_prior_features(batch_x)
                pred = prior_outputs["expected_return"]

                val_pred_loss = F.mse_loss(pred, batch_y)

                n = batch_x.size(0)

                val_loss += val_pred_loss.item() * n
                val_recon += val_pred_loss.item() * n
                val_kl += 0.0
                val_n += n
                
                pred_np = pred.detach().cpu().numpy().ravel()
                true_np = batch_y.detach().cpu().numpy().ravel()

                if np.std(pred_np) < 1e-12 or np.std(true_np) < 1e-12:
                    continue

                ic = spearmanr(pred_np, true_np).correlation

                if np.isfinite(ic):
                    daily_ics.append(ic)

        val_loss /= max(val_n, 1)
        val_recon /= max(val_n, 1)
        val_kl /= max(val_n, 1)
        
        val_rank_ic = float(np.mean(daily_ics)) if daily_ics else 0.0
        val_rank_icir = float(np.mean(daily_ics) / (np.std(daily_ics) + 1e-8)) if daily_ics else 0.0

        if progress_bar is not None:
            last_val_error = f"IC:{val_rank_ic:.4f} L:{val_loss:.4g}"
            progress_bar.set_postfix({
                "epoch": epoch + 1,
                "train_error": f"{train_loss:.4g}",
                "val_error": last_val_error,
                "kl_wt": f"{current_kl_weight:.3f}",
            }, refresh=True)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        next_lr = optimizer.param_groups[0]["lr"]

        is_best_epoch = val_rank_ic > best_val_rank_ic

        if is_best_epoch:
            best_val_rank_ic = val_rank_ic
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        early_stopped = epochs_no_improve >= patience

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_reconstruction_loss"].append(train_recon)
        history["train_kl_loss"].append(train_kl)
        history["val_loss"].append(val_loss)
        history["val_reconstruction_loss"].append(val_recon)
        history["val_kl_loss"].append(val_kl)
        history["val_rank_ic"].append(val_rank_ic)
        history["val_rank_icir"].append(val_rank_icir)
        history["lr"].append(current_lr)
        history["next_lr"].append(next_lr)
        history["best_val_loss"].append(best_val_loss)
        history["is_best_epoch"].append(is_best_epoch)
        history["epochs_no_improve"].append(epochs_no_improve)
        history["early_stopped"].append(early_stopped)
        history["train_samples"].append(train_samples)
        history["val_samples"].append(val_samples)
        history["hidden_size"].append(model.hidden_size)
        history["num_factors"].append(model.num_factors)
        history["num_portfolios"].append(model.num_portfolios)
        history["learning_rate"].append(learning_rate)
        history["weight_decay"].append(weight_decay)
        history["max_epochs"].append(max_epochs)
        history["patience"].append(patience)
        history["random_seed"].append(random_seed)
        history["device"].append(str(device))

        if early_stopped:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best val Rank IC: {best_val_rank_ic:.4f} (MSE: {best_val_loss:.6f})"
                )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.to("cpu")

    return model, history
