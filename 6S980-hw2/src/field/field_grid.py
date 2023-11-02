import torch
import torch.nn.functional as F
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from .field import Field


class FieldGrid(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a grid for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/grid.yaml):

        - side_length: the side length in each dimension

        Your architecture only needs to support 2D and 3D grids.
        """
        super().__init__(cfg, d_coordinate, d_out)
        assert d_coordinate in (2, 3)
        side_length = cfg.side_length

        # Initialize the grid tensor based on dimensionality
        if d_coordinate == 2:
            self.grid = torch.randn((side_length, side_length, d_out))
        elif d_coordinate == 3:
            self.grid = torch.randn((side_length, side_length, side_length, d_out))

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Use torch.nn.functional.grid_sample to bilinearly sample from the image grid.
        Remember that your implementation must support either 2D and 3D queries,
        depending on what d_coordinate was during initialization.
        """

        # We need to normalize the coordinates to the range [-1, 1] to use grid_sample
        normalized_coordinates = (coordinates * 2) - 1

        # The grid_sample expects a batch dimension for grid, add it
        grid_batched = self.grid.unsqueeze(0) 

        if self.d_coordinate == 2:
            # Change the coordinate shape to (B, H, W, 2) for 2D
            normalized_coordinates = normalized_coordinates \
                                        .view(*normalized_coordinates \
                                        .shape[:-1], 1, 1, 2)
        elif self.d_coordinate == 3:
            # Change the coordinate shape to (B, D, H, W, 3) for 3D
            normalized_coordinates = normalized_coordinates \
                                        .view(*normalized_coordinates \
                                        .shape[:-1], 1, 1, 1, 3)

        sampled_values = \
            F.grid_sample(grid_batched, normalized_coordinates, align_corners=True)

        return sampled_values.squeeze()
