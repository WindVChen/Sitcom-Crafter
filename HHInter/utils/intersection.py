import torch
import math
from typing import NewType

Tensor = NewType('Tensor', torch.Tensor)


def pcl_pcl_pairwise_distance(
        x: torch.Tensor,
        y: torch.Tensor,
        use_cuda: bool = True,
        squared: bool = False
):
    """
    Calculate the pairse distance between two point clouds.
    """

    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()

    # dtype = torch.cuda.LongTensor if \
    #     use_cuda else torch.LongTensor

    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))

    diag_ind_x = torch.arange(0, num_points_x).type(torch.LongTensor).to(x.device)
    diag_ind_y = torch.arange(0, num_points_y).type(torch.LongTensor).to(x.device)
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = rx.transpose(2, 1) + ry - 2 * zz

    if not squared:
        P = torch.clamp(P, min=1e-12)  # make sure we dont get nans
        P = torch.sqrt(P)

    return P


def solid_angles(
        points: Tensor,
        triangles: Tensor,
        thresh: float = 1e-8
) -> Tensor:
    ''' Compute solid angle between the input points and triangles
        Follows the method described in:
        The Solid Angle of a Plane Triangle
        A. VAN OOSTEROM AND J. STRACKEE
        IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING,
        VOL. BME-30, NO. 2, FEBRUARY 1983
        Parameters
        -----------
            points: BxQx3
                Tensor of input query points
            triangles: BxFx3x3
                Target triangles
            thresh: float
                float threshold
        Returns
        -------
            solid_angles: BxQxF
                A tensor containing the solid angle between all query points
                and input triangles
    '''
    # Center the triangles on the query points. Size should be BxQxFx3x3
    centered_tris = triangles[:, None] - points[:, :, None, None]

    # BxQxFx3
    norms = torch.norm(centered_tris, dim=-1)

    # Should be BxQxFx3
    cross_prod = torch.cross(
        centered_tris[:, :, :, 1], centered_tris[:, :, :, 2], dim=-1)
    # Should be BxQxF
    numerator = (centered_tris[:, :, :, 0] * cross_prod).sum(dim=-1)
    del cross_prod

    dot01 = (centered_tris[:, :, :, 0] * centered_tris[:, :, :, 1]).sum(dim=-1)
    dot12 = (centered_tris[:, :, :, 1] * centered_tris[:, :, :, 2]).sum(dim=-1)
    dot02 = (centered_tris[:, :, :, 0] * centered_tris[:, :, :, 2]).sum(dim=-1)
    del centered_tris

    denominator = (
            norms.prod(dim=-1) +
            dot01 * norms[:, :, :, 2] +
            dot02 * norms[:, :, :, 1] +
            dot12 * norms[:, :, :, 0]
    )
    del dot01, dot12, dot02, norms

    # Should be BxQ
    solid_angle = torch.atan2(numerator, denominator)
    del numerator, denominator

    torch.cuda.empty_cache()

    return 2 * solid_angle


def winding_numbers(
        points: Tensor,
        triangles: Tensor,
        thresh: float = 1e-8
) -> Tensor:
    ''' Uses winding_numbers to compute inside/outside
        Robust inside-outside segmentation using generalized winding numbers
        Alec Jacobson,
        Ladislav Kavan,
        Olga Sorkine-Hornung
        Fast Winding Numbers for Soups and Clouds SIGGRAPH 2018
        Gavin Barill
        NEIL G. Dickson
        Ryan Schmidt
        David I.W. Levin
        and Alec Jacobson
        Parameters
        -----------
            points: BxQx3
                Tensor of input query points
            triangles: BxFx3x3
                Target triangles
            thresh: float
                float threshold
        Returns
        -------
            winding_numbers: BxQ
                A tensor containing the Generalized winding numbers
    '''
    # The generalized winding number is the sum of solid angles of the point
    # with respect to all triangles.
    # 4 pi here is because solid angle is 4 pi, different from the 2D plane angle.
    return 1 / (4 * math.pi) * solid_angles(
        points, triangles, thresh=thresh).sum(dim=-1)
