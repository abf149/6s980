from jaxtyping import Float
from torch import Tensor, arange, ones, round

from .geometry import homogenize_points, project, transform_world2cam


def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """

    # Homogenize the vertices
    vertices_homogeneous = homogenize_points(vertices)

    # Transform the vertices into camera space
    vertices_cam = transform_world2cam(
        vertices_homogeneous.unsqueeze(0), extrinsics.unsqueeze(1)
    )

    # Project the transformed vertices onto the image plane
    projected_points = project(vertices_cam, intrinsics.unsqueeze(1))

    # Create a white canvas
    batch_size = extrinsics.shape[0]
    canvas = ones((batch_size, resolution[1], resolution[0]), device=vertices.device)

    # Convert projected points to pixel indices and set the corresponding pixel to black
    x_indices = (
        round(projected_points[..., 0] * (resolution[0] - 1))
        .long()
        .clamp(0, resolution[0] - 1)
    )
    y_indices = (
        round(projected_points[..., 1] * (resolution[1] - 1))
        .long()
        .clamp(0, resolution[1] - 1)
    )

    # Create a batch index tensor and expand it to match the shape of x_indices and
    # y_indices
    batch_indices = (
        arange(batch_size, device=vertices.device).unsqueeze(1).expand_as(x_indices)
    )

    # Flatten the tensors for 1D indexing
    x_indices_flat = x_indices.flatten()
    y_indices_flat = y_indices.flatten()
    batch_indices_flat = batch_indices.flatten()

    # Use advanced indexing to set the canvas to zero for every batch and every
    # projected vertex
    canvas[batch_indices_flat, y_indices_flat, x_indices_flat] = 0

    return canvas
