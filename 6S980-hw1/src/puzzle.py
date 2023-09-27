import json
from pathlib import Path
from typing import Literal, TypedDict

import torch
import torchvision.transforms as transforms
from jaxtyping import Float
from PIL import Image
from torch import Tensor


class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]


def load_dataset(path: Path) -> PuzzleDataset:
    """Load the dataset into the required format."""

    with open(path / "metadata.json", "r") as f:
        data = json.load(f)

    extrinsics = torch.tensor(data["extrinsics"])
    intrinsics = torch.tensor(data["intrinsics"])

    # Convert images to grayscale tensors
    to_tensor = transforms.ToTensor()
    images = [
        to_tensor(Image.open(path / "images" / f"{i:02d}.png").convert("L"))
        for i in range(32)
    ]

    return {"extrinsics": extrinsics, "intrinsics": intrinsics, "images": images}


def convert_dataset(dataset: PuzzleDataset) -> PuzzleDataset:
    """Convert the dataset into OpenCV-style camera-to-world format. As a reminder, this
    format has the following specification:

    - The camera look vector is +Z.
    - The camera up vector is -Y.
    - The camera right vector is +X.
    - The extrinsics are in camera-to-world format, meaning that they transform points
      in camera space to points in world space.

    """

    look_idx = 0
    up_idx = 1
    right_idx = 2

    c2w_or_w2c = dataset["extrinsics"]

    RT_or_R = c2w_or_w2c[:, :3, :3]  # bx3x3
    t_or_mRt = c2w_or_w2c[:, :3, 3]  # bx3

    world_look_if_c2w = RT_or_R[:, :, look_idx]  # bx3
    t_if_c2w = t_or_mRt  # bx3

    is_c2w_plus_x_look = torch.all(
        torch.matmul(world_look_if_c2w.unsqueeze(1), t_if_c2w.unsqueeze(2)) < 0
    ).item()

    assert is_c2w_plus_x_look  # c2w matrix with +x look direction

    c2w = c2w_or_w2c

    world_look_pos_x = world_look_if_c2w

    RT = RT_or_R

    up = RT[:, :, up_idx]  # up vector
    up_y = up[:, 1]  # y component of up

    # Assert up_y has consistent sign per problem description
    is_up_idx_correct = torch.all(up_y > 0).item()
    is_neg_up_idx_correct = torch.all(up_y < 0).item()
    assert is_up_idx_correct or is_neg_up_idx_correct
    assert is_neg_up_idx_correct

    # If consistent sign is negative (assert confirms it is), up is -y
    up = -up

    # X and Z are consistent between dataset frame and OpenCV frame.
    # Use cross-product to establish consistent handedness

    # - Compare the sign of OpenCV -Y to the sign of X cross Z
    #   opencv_neg_y_dot_x_cross_z_is_pos indicates handedness
    opencv_neg_y_dot_x_cross_z_is_pos = (
        torch.cross(torch.tensor([1, 0, 0]), torch.tensor([0, 0, 1]))
        .dot(torch.tensor([0, -1, 0]))
        .item()
        > 0
    )

    # Assert OpenCV handedness corresponds to cross-product handedness
    assert opencv_neg_y_dot_x_cross_z_is_pos

    # Now get dataset handedness
    #   From above description of OpenCV:
    #    - The camera look vector is +Z.
    #    - The camera up vector is -Y.
    #    - The camera right vector is +X.
    #
    #   And X cross Z = -Y implies that right cross look = up
    #
    right_ambig = RT[:, :, right_idx]
    dataset_handedness_matches_right_cross_look = torch.all(
        (torch.cross(right_ambig, world_look_pos_x, dim=1) * up).sum(dim=1) > 0
    ).item()

    # Right vector is +Z
    assert dataset_handedness_matches_right_cross_look

    # So at this point we have shown that for the dataset,
    # look: +X; want to map to +Z for OpenCV
    # up: -Y; want to keep as-is
    # right: +Z; want to map to +X for OpenCV
    #
    # We can define a transformation against c2w to swap +X and +Z:
    right_transform = (
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        .unsqueeze(0)
        .expand_as(c2w)
    )

    c2w_opencv = torch.matmul(c2w, right_transform)

    dataset["extrinsics"] = c2w_opencv

    return dataset


def quiz_question_1() -> Literal["w2c", "c2w"]:
    """In what format was your puzzle dataset?"""

    # w2c because multiplying by these extrinsics yields points
    # which are "behind the camera", when we know that for c2w the look
    # direction is toward the origin.
    return "c2w"


def quiz_question_2() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera look vector?"""

    return "+x"


def quiz_question_3() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera up vector?"""

    return "-y"


def quiz_question_4() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera right vector?"""

    return "-z"


def explanation_of_problem_solving_process() -> str:
    """Please return a string (a few sentences) to describe how you solved the puzzle.
    We'll only grade you on whether you provide a descriptive answer, not on how you
    solved the puzzle (brute force, deduction, etc.).
    """

    res = r"""
    I solved part 2 on an entirely mathematical basis without visualizing the projected
    images until the very end. I did have to apply trial-and-error in order to determine
    the mapping from up/right/look to X/Y/Z, however (1) my success criterion for when
    I had found the correct mapping was entirely mathematical, and (2) for each of
    up/right/look I was able to determine the sign of the mapping. I determined that
    the provided extrinsics were c2w by decomposing the extrinsics into R^T and t
    (or potentially R and -Rt, although this did not turn out to be the case.) I started
    by assuming that the extrinsics are c2w, in which case I had isolated R^T and t; I
    then tested my hypothesis. Each column of R^T is the projection of a different 
    camera-coordinate axis onto world coordinates. I used trial-and-error to find that
    there is exactly one column of R^T which has a nonzero projection onto t, and that
    projection was negative - which indicates that the extrinsic matrix is a valid c2w
    because the look direction is antiparallel to t, the vector emanating from the
    origin. The 0th R^T column was a match and had a negative projection, thus the look
    direction is +X. Next, I found y by taking advantage of the prior knowledge that
    all camera coordinate systems are upright: I used trial and error to determine that
    over all extrinsic matrices in the batch, the y-component of the 1st R^T had a
    consistent (negative) sign. Since we are given that the extrinsic matrix up
    direction has a positive dot-product onto world-up which is y, then camera-frame
    up must be -y so that the y-component will be positive. Finally, I exploited 
    process of elimination, the idea of handedness, and the cross-product operator to 
    determine the right direction. By process of elimination right must lie parallel
    or anti-parallel to Z. What remains is to determine the sign, using the cross-
    product operator. OpenCV is defined such that 
    the cross-product of right times look == up. In order to find right, I started by 
    naively assuming that  If I found that for all dataset batches, the cross-product of
    right and look had a negative projection onto up, then I would say that right == -Z.
    However, the project of the cross-product onto up turned out to be positive, so
    right == +Z. Thus overall we can see that the organization of the extrinsic is as
    a c2w matrix with a look direction of +X, an up direction -Y, and a right direction
    of +Z. Now, we are given that in OpenCV look and right are swapped, i.e. they map
    to +Z and +X respectively. I implemented a transformation matrix which implements
    this swap. The c2w extrinsic is right-multiplied by the transformation to get a
    canonical OpenCV extrinsic, which is the return. And I can see that I successfully 
    reproduce the images in my puzzle.
    """

    return res
