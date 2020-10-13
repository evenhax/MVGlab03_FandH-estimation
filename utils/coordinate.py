# author:nannan
# contact: zhaozhaoran@bupt.edu.cn
# datetime:2020/9/13 7:40 下午
# software: PyCharm

import numpy as np


def homoco_pts_2_euco_pts(pts):
    """
    Homogeneous coordinate to Euclidean coordinates
    :param pts: (np.ndarray): Homogeneous coordinate
    :return: (np.ndarray): Euclidean coordinates
    """
    if len(pts.shape) == 1:
        pts = pts.reshape(1, -1)
    res = pts / pts[:, -1, None]
    return res[:, :-1].squeeze()


def euco_pts_2_homoco_pts(pts):
    """
    Euclidean coordinate to Homogeneous coordinates
    :param pts: (np.ndarray): Euclidean coordinate
    :return: (np.ndarray): Homogeneous coordinates
    """
    if len(pts.shape) == 1:
        pts = pts.reshape(1, -1)
    one = np.ones(pts.shape[0])
    res = np.c_[pts, one]
    return res.squeeze()

def normalize(pts, T=None):
    """
    normalize points
    :param pts: (np.ndarray): points to be normalized
    :param T: (np.ndarray): IS None means we need to computer T
    :return: (np.ndarray, np.ndarray): normalized points and T
    """
    if T is None:
        u = np.mean(pts, 0)
        d = np.sum(np.sqrt(np.sum(np.power(pts, 2), 1)))
        T = np.array([
            [np.sqrt(2) / d, 0, -(np.sqrt(2) / d * u[0])],
            [0, np.sqrt(2) / d, -(np.sqrt(2) / d * u[1])],
            [0, 0, 1]
        ])
    return homoco_pts_2_euco_pts(np.matmul(T, euco_pts_2_homoco_pts(pts).T).T), T


def warpPerspective(source_img, H):
    """
    Perspective transformation.

    Args:
        source_img: Input image.
        H: Perspective matrix.

    Returns:
        Output image after projective mapping H.
    """
    h, w = source_img.shape
    target_img = np.zeros_like(source_img)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    points = np.hstack((x, y, np.ones(x.shape))).T
    mapped_points = np.dot(H, points)
    mapx, mapy, mapw = mapped_points[0, :], mapped_points[1, :], mapped_points[2, :]
    mapx = np.int32(mapx / mapw)
    mapy = np.int32(mapy / mapw)
    valid_indices = np.where((mapx >= 0) & (mapy >= 0) & (mapx < w) & (mapy < h))[0]
    mapx = mapx[valid_indices]
    mapy = mapy[valid_indices]
    y = y[valid_indices].flatten()
    x = x[valid_indices].flatten()
    target_img[mapy, mapx] = source_img[y, x]
    return target_img
