import copy
import logging
import math

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class DLM(RegressorMixin, BaseEstimator):
    n_features_in_: int
    feature_names_in_: np.ndarray
    net_: nn.Module
    device_: torch.device
    history_: dict[str, list]
    best_epoch_: int
    stopped_epoch_: int

    def __init__(
        self,
        *,
        layer_sizes: tuple = (64, 32, 16),
        dropout_values: tuple = (0.4, 0.4, 0.4, 0.0),
        output_size: int = 1,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        warmup_epochs: int = 10,
        plateau_epochs: int = 10,
        min_lr_ratio: float = 0.05,
        device: str = 'cpu',
        early_stopping: bool = False,
        tol: float = 0.001,
        validation_fraction: float = 0.2,
        n_iter_no_change: int = 25,
        verbose: int = 0,
        random_state: int = 42,
    ):
        self.layer_sizes = layer_sizes
        self.dropout_values = dropout_values
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.plateau_epochs = plateau_epochs
        self.min_lr_ratio = min_lr_ratio
        self.device = device
        self.early_stopping = early_stopping
        self.tol = tol
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        self.random_state = random_state


    def _build_net(self) -> None:
        layers = []
        layers.append(nn.Dropout(self.dropout_values[0]))
        previous_size = self.n_features_in_
        for current_size, dropout_value in zip(self.layer_sizes, self.dropout_values[1:]):
            layers.extend([
                nn.Linear(previous_size, current_size),
                nn.BatchNorm1d(current_size),
                nn.Hardswish(),
                nn.Dropout(dropout_value),
            ])
            previous_size = current_size
        layers.append(nn.Linear(previous_size, self.output_size))
        self.net_ = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, 'net_'):
            raise RuntimeError('net is not built, call fit or load first')
        return self.net_(x)


    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DLM':
        """
        Train with Adam and a warmup / plateau / cosine decay / low-plateau lr schedule.
        With early_stopping, validation_fraction of X is held out internally and the weights
        of the best epoch by val loss are restored at the end. The schedule assumes all epochs
        run, so its decay phase may never be reached with early stopping.
        """
        X, y = validate_data(self, X, y, dtype=np.float32, y_numeric=True, ensure_min_samples=2)
        if not np.isfinite(y).all():
            raise ValueError('y must contain only finite values')
        if len(self.dropout_values) != len(self.layer_sizes) + 1:
            raise ValueError('dropout_values must contain one value per hidden layer plus one input value')
        if self.early_stopping and not 0 < self.validation_fraction < 1:
            raise ValueError('validation_fraction must be in (0, 1)')
        if self.warmup_epochs + 2 * self.plateau_epochs >= self.epochs:
            raise ValueError('warmup_epochs and plateau_epochs are too large')
        self.device_ = torch.device(self.device)
        torch.manual_seed(self.random_state)

        self._build_net()
        self.net_.to(self.device_)
        optimizer = Adam(self.net_.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self._lr_multiplier)
        loss_fn = nn.MSELoss()

        if self.early_stopping:
            X, X_val, y, y_val = train_test_split(
                X, y,
                test_size=self.validation_fraction,
                random_state=self.random_state,
            )
            X_val_t, y_val_t = self._to_tensors(X_val, y_val)
        else:
            X_val_t, y_val_t = None, None

        X_train_t, y_train_t = self._to_tensors(X, y)
        if self.batch_size < 2:
            raise ValueError('batch_size must be at least 2')
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(self.random_state)
        )
        if len(train_loader) == 0:
            raise ValueError('Training data must contain at least batch_size rows')

        self.history_ = {'epoch': [], 'train_loss': [], 'train_metric': [], 'val_loss': [], 'val_metric': [], 'lr': []}
        best_loss = np.inf
        best_epoch = 0
        best_state = copy.deepcopy(self.net_.state_dict())
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self.history_['lr'].append(optimizer.param_groups[0]['lr'])
            self._train_epoch(train_loader, optimizer=optimizer, loss_fn=loss_fn)
            scheduler.step()

            train_loss, train_metric = self._evaluate(X_train_t, y_train_t, loss_fn=loss_fn)
            self.history_['epoch'].append(epoch + 1)
            self.history_['train_loss'].append(train_loss)
            self.history_['train_metric'].append(train_metric)
            if self._should_log(epoch):
                logging.getLogger(__name__).info(
                    f'Epoch {epoch + 1}/{self.epochs}'
                    f'\t\tTrain loss: {train_loss:.4f}\tTrain metric: {train_metric:.4f}'
                )

            if X_val_t is None or y_val_t is None:
                continue

            val_loss, val_metric = self._evaluate(X_val_t, y_val_t, loss_fn=loss_fn)
            self.history_['val_loss'].append(val_loss)
            self.history_['val_metric'].append(val_metric)
            if self._should_log(epoch):
                logging.getLogger(__name__).info(f'\t\t\tVal loss: {val_loss:.4f}\tVal metric: {val_metric:.4f}')

            if val_loss < best_loss - self.tol:
                best_loss = val_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.net_.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.n_iter_no_change:
                break

        self.stopped_epoch_ = epoch + 1
        if self.early_stopping:
            self.net_.load_state_dict(best_state)
            self.best_epoch_ = best_epoch
        else:
            self.best_epoch_ = self.stopped_epoch_
        return self


    def get_training_artifacts(self) -> dict:
        """Return a copy of the training history and the best/stopped epochs."""
        check_is_fitted(self, 'stopped_epoch_')
        return {
            'history': copy.deepcopy(self.history_),
            'best_epoch': self.best_epoch_,
            'stopped_epoch': self.stopped_epoch_,
        }


    def _lr_multiplier(self, epoch: int) -> float:
        """
        Factor for the base learning rate at a given epoch.
        Phases: linear warmup -> plateau at 1 -> cosine decay -> plateau at min_lr_ratio;
        both plateaus last plateau_epochs.
        """
        if epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs
        decay_start = self.warmup_epochs + self.plateau_epochs
        decay_end = self.epochs - self.plateau_epochs
        if epoch < decay_start:
            return 1.0
        if epoch >= decay_end:
            return self.min_lr_ratio
        progress = (epoch - decay_start) / (decay_end - decay_start)
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


    def _to_tensors(self, X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        return X_t, y_t


    def _should_log(self, epoch: int) -> bool:
        return bool(self.verbose and epoch % self.verbose == 0) or epoch + 1 == self.epochs


    def _train_epoch(
        self, train_loader: DataLoader, *, optimizer: torch.optim.Optimizer, loss_fn: nn.Module
    ) -> None:
        self.net_.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device_)
            batch_y = batch_y.to(self.device_)

            optimizer.zero_grad()
            loss = loss_fn(self.forward(batch_X), batch_y)
            loss.backward()
            optimizer.step()


    @torch.no_grad()
    def _predict(self, X_t: torch.Tensor) -> torch.Tensor:
        self.net_.eval()
        outputs = [
            self.forward(X_t[start:start + self.batch_size].to(self.device_)).cpu()
            for start in range(0, X_t.shape[0], self.batch_size)
        ]
        return torch.cat(outputs).ravel()


    def _evaluate(self, X_t: torch.Tensor, y_t: torch.Tensor, *, loss_fn: nn.Module) -> tuple[float, float]:
        y_pred = self._predict(X_t)
        y_true = y_t.ravel()
        loss = float(loss_fn(y_pred, y_true))
        metric = float(root_mean_squared_error(y_true.numpy(), y_pred.numpy()))
        return loss, metric


    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'stopped_epoch_')
        X = validate_data(self, X, reset=False, dtype=np.float32)
        return self._predict(torch.tensor(X, dtype=torch.float32)).numpy()


    def save(self, *, path: str) -> None:
        check_is_fitted(self, 'stopped_epoch_')
        torch.save({
            'params': self.get_params(deep=False),
            'n_features_in': self.n_features_in_,
            'feature_names_in': self.feature_names_in_.tolist() if hasattr(self, 'feature_names_in_') else None,
            'state_dict': {name: value.detach().cpu() for name, value in self.net_.state_dict().items()},
            'training_artifacts': self.get_training_artifacts(),
        }, path)


    @classmethod
    def load(cls, *, path: str, device: str | None = None) -> 'DLM':
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(**checkpoint['params'])
        if device is not None:
            model.set_params(device=device)
        model.device_ = torch.device(model.device)
        model.n_features_in_ = checkpoint['n_features_in']
        if checkpoint['feature_names_in'] is not None:
            model.feature_names_in_ = np.asarray(checkpoint['feature_names_in'], dtype=object)
        model._build_net()
        model.net_.load_state_dict(checkpoint['state_dict'])
        model.net_.to(model.device_)
        model.net_.eval()
        artifacts = checkpoint['training_artifacts']
        model.history_ = artifacts['history']
        model.best_epoch_ = artifacts['best_epoch']
        model.stopped_epoch_ = artifacts['stopped_epoch']
        return model
