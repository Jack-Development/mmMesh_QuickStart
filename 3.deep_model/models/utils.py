import torch


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def point_ball_set(nsample, xyz, new_xyz):
    """
    Input:
        nsample: number of points to sample
        xyz: all points, [B, N, 3]
        new_xyz: anchor points [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = (
        torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    )
    sqrdists = square_distance(new_xyz, xyz)
    _, sort_idx = torch.sort(sqrdists)
    sort_idx = sort_idx[:, :, :nsample]
    batch_idx = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view((B, 1, 1))
        .repeat((1, S, nsample))
    )
    centroids_idx = (
        torch.arange(S, dtype=torch.long)
        .to(device)
        .view((1, S, 1))
        .repeat((B, 1, nsample))
    )
    return group_idx[batch_idx, centroids_idx, sort_idx]


def AnchorInit(
    x_min=-0.3,
    x_max=0.3,
    x_interval=0.3,
    y_min=-0.3,
    y_max=0.3,
    y_interval=0.3,
    z_min=-0.3,
    z_max=2.1,
    z_interval=0.3,
):  # [z_size, y_size, x_size, npoint] => [9,3,3,3]
    """
    Input:
        x,y,z min, max and sample interval
    Return:
        centroids: sampled controids [z_size, y_size, x_size, npoint] => [9,3,3,3]
    """
    x_size = round((x_max - x_min) / x_interval) + 1
    y_size = round((y_max - y_min) / y_interval) + 1
    z_size = round((z_max - z_min) / z_interval) + 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    centroids = torch.zeros((z_size, y_size, x_size, 3), dtype=torch.float32).to(device)
    for z_no in range(z_size):
        for y_no in range(y_size):
            for x_no in range(x_size):
                lx = x_min + x_no * x_interval
                ly = y_min + y_no * y_interval
                lz = z_min + z_no * z_interval
                centroids[z_no, y_no, x_no, 0] = lx
                centroids[z_no, y_no, x_no, 1] = ly
                centroids[z_no, y_no, x_no, 2] = lz
    return centroids


def AnchorGrouping(anchors, nsample, xyz, points):
    """
    Input:
        anchors: [B, 9*3*3, 3], npoint=9*3*3
        nsample: number of points to sample
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    _, S, _ = anchors.shape
    idx = point_ball_set(nsample, xyz, anchors)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_anchors = anchors.view(B, S, 1, C).repeat(1, 1, nsample, 1)
    grouped_xyz_norm = grouped_xyz - grouped_anchors  # anchors.view(B, S, 1, C)

    grouped_points = index_points(points, idx)
    new_points = torch.cat(
        [grouped_anchors, grouped_xyz_norm, grouped_points], dim=-1
    )  # [B, npoint, nsample, C+C+D]
    return new_points
