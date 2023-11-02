import torch
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from ..components.positional_encoding import PositionalEncoding
from ..field.field_grid import FieldGrid
from ..field.field_mlp import FieldMLP
from .field import Field


class FieldGroundPlan(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a neural ground plan. You should reuse the following components:

        - FieldGrid from  src/field/field_grid.py
        - FieldMLP from src/field/field_mlp.py
        - PositionalEncoding from src/components/positional_encoding.py

        Your ground plan only has to handle the 3D case.
        """
        super().__init__(cfg, d_coordinate, d_out)
        assert d_coordinate == 3
        self.field_grid = FieldGrid(cfg, d_coordinate-1, d_out)  # X and Y, hence d_coordinate-1

        # Check if positional encoding is needed
        if cfg.positional_encoding_octaves is not None:
            self.positional_encoding = PositionalEncoding(cfg.positional_encoding_octaves)
            d_input_mlp = d_out + self.positional_encoding.d_out(1)
        else:
            self.positional_encoding = None
            d_input_mlp = d_out + 1  # grid output + Z coordinate

        # Create the MLP with the adjusted input dimension
        self.field_mlp = FieldMLP(cfg, d_input_mlp, d_out)

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the ground plan at the specified coordinates. You should:

        - Sample the grid using the X and Y coordinates.
        - Positionally encode the Z coordinates.
        - Concatenate the grid's outputs with the corresponding encoded Z values, then
          feed the result through the MLP.
        """

        # Extract X, Y, and Z
        x, y, z = torch.split(coordinates, 1, dim=-1)

        # Sample grid using X and Y
        grid_output = self.field_grid(torch.cat([x, y], dim=-1))

        # Positionally encode Z if encoding is available
        if self.positional_encoding is not None:
            z_encoded = self.positional_encoding(z)
        else:
            z_encoded = z

        # Concatenate the grid's outputs with encoded Z (or just Z) and feed through MLP
        mlp_input = torch.cat([grid_output, z_encoded], dim=-1)
        mlp_output = self.field_mlp(mlp_input)

        return mlp_output
