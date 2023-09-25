from jaxtyping import Float
from torch import (
    Tensor,
    ones_like,
    float32,
    sum as torch_sum,
    inverse,
    cat,
    ones,
    zeros,
    matmul,
)

import torch

# from .geometry import homogenize_points, project, transform_world2cam


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional points into (n+1)-dimensional homogeneous points."""

    # Homogenous points require fourth coordinate == 1 to allow displacement
    return cat(
        [points, ones(*points.shape[:-1], 1, dtype=points.dtype, device=points.device)],
        dim=-1,
    )


def homogenize_vectors(
    points: Float[Tensor, "#batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional vectors into (n+1)-dimensional homogeneous vectors."""

    # Homogenous vectors require fourth coordinate == 0 to avoid displacement
    return cat(
        [
            points,
            zeros(*points.shape[:-1], 1, dtype=points.dtype, device=points.device),
        ],
        dim=-1,
    )


def transform_rigid(
    xyz: Float[Tensor, "*#batch 4"],
    transform: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Apply a rigid-body transform to homogeneous points or vectors."""

    return matmul(transform, xyz.unsqueeze(-1)).squeeze(-1)


def transform_world2cam(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous
    3D camera coordinates.
    """

    return transform_rigid(xyz, inverse(cam2world))


def transform_cam2world(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D camera coordinates to homogeneous
    3D world coordinates.
    """

    return transform_rigid(xyz, cam2world)


def project(
    xyz: Float[Tensor, "*#batch 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 2"]:
    """Project homogenized 3D points in camera coordinates to pixel coordinates."""

    proj = matmul(intrinsics, xyz[..., :3].unsqueeze(-1)).squeeze(-1)
    return proj[..., :2] / proj[..., 2:3]


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
    vertices_cam = transform_world2cam(vertices_homogeneous, extrinsics)

    # Project the transformed vertices onto the image plane
    projected_points = project(vertices_cam, intrinsics)

    # Create a white canvas
    batch_size = extrinsics.shape[0]
    canvas = ones_like(
        (batch_size, resolution[1], resolution[0]), device=vertices.device
    )

    # Convert projected points to pixel indices and set the corresponding pixel to black
    x_indices = round(projected_points[..., 0]).long().clamp(0, resolution[0] - 1)
    y_indices = round(projected_points[..., 1]).long().clamp(0, resolution[1] - 1)

    for i in range(batch_size):
        canvas[i, y_indices[i], x_indices[i]] = 0

    return canvas
