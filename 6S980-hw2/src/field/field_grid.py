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
        self.side_length=side_length
        # Initialize the grid tensor based on dimensionality
        inp=[]
        if d_coordinate == 2:
            inp = torch.randn((1, d_out, side_length, side_length))
        elif d_coordinate == 3:
            inp = torch.randn((1, d_out, side_length, side_length, side_length))
        self.input = torch.nn.Parameter(inp)
        self.d_coordinate=d_coordinate

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Use torch.nn.functional.grid_sample to bilinearly sample from the image grid.
        Remember that your implementation must support either 2D and 3D queries,
        depending on what d_coordinate was during initialization.

        """
        batch_size = coordinates.size(0)  # Obtain the batch size from the 'coordinates' tensor

        # We need to normalize the coordinates to the range [-1, 1] to use grid_sample
        normalized_coordinates = coordinates*2 - 1

        inp=[]
        if self.d_coordinate == 2:
            inp = self.input.expand(batch_size, self.input.size(1), self.input.size(2), self.input.size(3))
            normalized_coordinates = normalized_coordinates.unsqueeze(1).unsqueeze(2)
        elif self.d_coordinate == 3:
            inp = self.input.expand(batch_size, self.input.size(1), self.input.size(2), self.input.size(3), self.input.size(4))
            normalized_coordinates = normalized_coordinates.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        sampled_values = \
            F.grid_sample(inp,normalized_coordinates,align_corners=True)

        sampled_values = sampled_values.squeeze(-1).squeeze(-1)

        return sampled_values
