from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from .field import Field
from .field_grid import FieldGrid
from .field_mlp import FieldMLP


class FieldHybridGrid(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a hybrid grid-mlp neural field. You should reuse FieldGrid from
        src/field/field_grid.py and FieldMLP from src/field/field_mlp.py in your
        implementation.

        Hint: Since you're reusing existing components, you only need to add one line
        each to __init__ and forward!
        """
        super().__init__(cfg, d_coordinate, d_out)
        self.field_grid = FieldGrid(cfg, d_coordinate, d_out)
        self.field_mlp = FieldMLP(cfg, d_coordinate, d_out)

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        grid_output = self.field_grid(coordinates)
        mlp_output = self.field_mlp(coordinates)
        return mlp_output+grid_output
