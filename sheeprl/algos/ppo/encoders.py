from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from sheeprl.models.models import MLP, NatureCNN


class CNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features_dim: int,
        screen_size: int,
        keys: Sequence[str],
    ) -> None:
        super().__init__()
        self.keys = keys
        self.input_dim = (in_channels, screen_size, screen_size)
        self.output_dim = features_dim
        self.model = NatureCNN(in_channels=in_channels, features_dim=features_dim, screen_size=screen_size)

    def forward(self, obs: Dict[str, Tensor]) -> Tensor:
        x = torch.cat([obs[k] for k in self.keys], dim=-3)
        return self.model(x)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        features_dim: int | None,
        keys: Sequence[str],
        dense_units: int = 64,
        mlp_layers: int = 2,
        dense_act: nn.Module = nn.ReLU,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.input_dim = input_dim
        self.output_dim = features_dim if features_dim else dense_units
        if mlp_layers == 0:
            self.model = nn.Identity()
            self.output_dim = input_dim
        else:
            self.model = MLP(
                input_dim,
                features_dim,
                [dense_units] * mlp_layers,
                activation=dense_act,
                norm_layer=[nn.LayerNorm for _ in range(mlp_layers)] if layer_norm else None,
                norm_args=[{"normalized_shape": dense_units} for _ in range(mlp_layers)] if layer_norm else None,
            )

    def forward(self, obs: Dict[str, Tensor]) -> Tensor:
        x = torch.cat([obs[k] for k in self.keys], dim=-1)
        return self.model(x)