import torch
import torchvision.transforms as T
from jaxtyping import Float
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor

from .field_dataset import FieldDataset


class FieldDatasetImage(FieldDataset):
    def __init__(self, cfg: DictConfig) -> None:
        """Load the image in cfg.path into memory here."""

        super().__init__(cfg)
        # Load the image and convert to torch tensor
        image = Image.open(cfg.path).convert("RGB")
        self.image = T.ToTensor()(image).unsqueeze(0)  # Add batch dimension
        self.image_size = self.image.shape[-2:]  # H, W

    def query(
        self,
        coordinates: Float[Tensor, "batch d_coordinate"],
    ) -> Float[Tensor, "batch d_out"]:
        """Sample the image at the specified coordinates and return the corresponding
        colors. Remember that the coordinates will be in the range [0, 1].

        You may find the grid_sample function from torch.nn.functional helpful here.
        Pay special attention to grid_sample's expected input range for the grid
        parameter.
        """

        # Map coordinates from [0, 1] to [-1, 1]
        coordinates = 2.0 * coordinates - 1.0
        
        # Get the batch size
        batch_size = coordinates.shape[0]

        # Reshape coordinates to (N, H_out, W_out, 2). Since we're querying individual coordinates, H_out and W_out are both 1.
        grid = coordinates.view(batch_size, 1, 1, 2)

        # Replicate the image tensor to match the batch size of the grid
        image_batch = self.image.repeat(batch_size, 1, 1, 1)
        
        # Use grid_sample to sample the image
        sampled_colors = torch.nn.functional.grid_sample(image_batch, grid)
        
        # Remove singleton dimensions
        sampled_colors = sampled_colors.squeeze(2).squeeze(2)
        
        return sampled_colors

    @property
    def d_coordinate(self) -> int:
        return 2

    @property
    def d_out(self) -> int:
        return 3

    @property
    def grid_size(self) -> tuple[int, ...]:
        """Return a grid size that corresponds to the image's shape."""

        return self.image_size
