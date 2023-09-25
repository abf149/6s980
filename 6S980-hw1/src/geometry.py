from jaxtyping import Float
from torch import Tensor, cat, inverse, matmul, ones, zeros


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
    points: Float[Tensor, "*batch dim"],
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
