import torch.nn as nn
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from ..components.positional_encoding import PositionalEncoding
from .field import Field


class FieldMLP(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up an MLP for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/mlp.yaml):

        - positional_encoding_octaves: The number of octaves in the positional encoding.
          If this parameter is None, do not positionally encode the input.
        - num_hidden_layers: The number of hidden linear layers.
        - d_hidden: The dimensionality of the hidden layers.

        Don't forget to add ReLU between your linear layers!
        """

        super().__init__(cfg, d_coordinate, d_out)

        layers = []

        # Input layer
        if cfg.positional_encoding_octaves is None:
            # No positional encoding
            # Layer 1 is Linear, d_coordinate -> d_hidden
            layers.append(nn.Linear(d_coordinate, cfg.d_hidden))
            layers.append(nn.ReLU())
        else:
            # Layer 1a is positional encoding, from d_coordinate -> pe.d_out==d_pe
            # Layer 1b is Linear, d_pe -> d_hidden
            pe=PositionalEncoding(cfg.positional_encoding_octaves)
            d_pe=pe.d_out(d_coordinate)
            layers.append(pe)
            layers.append(nn.Linear(d_pe, cfg.d_hidden))
            layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(cfg.num_hidden_layers):
            layers.append(nn.Linear(cfg.d_hidden, cfg.d_hidden))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(cfg.d_hidden, d_out))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the MLP at the specified coordinates."""

        return self.mlp(coordinates)
