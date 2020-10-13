# author:nannan
# contact: zhaozhaoran@bupt.edu.cn
# datetime:2020/9/13 7:39 下午
# software: PyCharm

import os
import random

import cv2
import numpy as np
from .feature_process import FeatureProcess, get_matches, get_match_point
from .coordinate import homoco_pts_2_euco_pts, normalize
from .feature_process import FeatureProcess
from .photo_exif_info import PhotoExifInfo



class CC:
    """
    Get connected component
    """

    def __init__(self, edges):
        """
        init
        :param edges: (list[list[int]]): adjacency list
        """
        self.edges = edges
        self.tracks = [-1] * len(edges)
        self.cnt = 0

    def run(self):
        """
        find each nodes' CC
        :return: (list): CC
        """
        for i in range(len(self.edges)):
            if self.tracks[i] == -1:
                self._dfs(i)
                self.cnt += 1
        return self.tracks

    def _dfs(self, i):
        """
        dfs for compute CC
        :param i: (int): node index
        :return: None
        """
        self.tracks[i] = self.cnt
        for j in self.edges[i]:
            if self.tracks[j] == -1:
                self._dfs(j)


def build_M(K, R, T):
    """
    Build projection from K,R,T
    :param K: (np.ndarray): intrinsic of camera
    :param R: (np.ndarray): rotation of camera
    :param T: (np.ndarray): translation of camera
    :return: (np.ndarray): projection
    """
    external = np.append(R, T, axis=1)
    M = K.dot(external)
    return M


def triangulate_pts(proj1, proj2, pts1T, pts2T):
    """
    Triangulate points to 3D points
    :param proj1: (np.ndarray): projection metrix for img1
    :param proj2: (np.ndarray): projection metrix for img2
    :param pts1T: (np.ndarray): transpose of points in img1
    :param pts2T: (np.ndarray): transpose of points in img2
    :return: (np.ndarray): transpose of 3D points
    """
    pts1 = pts1T.T
    pts2 = pts2T.T
    pts3d = np.zeros((pts1.shape[0], 4))
    for i in range(pts1.shape[0]):
        A = np.zeros((4, 4))
        A[0, :] = pts1[i, 0] * proj1[2, :] - proj1[0, :]
        A[1, :] = pts1[i, 1] * proj1[2, :] - proj1[1, :]
        A[2, :] = pts2[i, 0] * proj2[2, :] - proj2[0, :]
        A[3, :] = pts2[i, 1] * proj2[2, :] - proj2[1, :]
        U, sigma, VT = np.linalg.svd(A)
        pts3d[i, :] = VT[-1, :]
    return pts3d.T


def reconstruct(K1, R1, T1, K2, R2, T2, pts1, pts2):
    """
    reconstruct 3d cloud
    :param K1: (np.ndarray): intrinsic matrix of camera1
    :param R1: (np.ndarray): rotation matrix of camera1
    :param T1: (np.ndarray): position matrix of camera1
    :param K2: (np.ndarray): intrinsic matrix of camera2
    :param R2: (np.ndarray): rotation matrix of camera2
    :param T2: (np.ndarray): position matrix of camera2
    :param pts1: (np.ndarray): inner points of image1
    :param pts2: (np.ndarray): inner points of image2
    :return: (np.ndarray): 3d cloud reconstructed by two images
    """
    projection1 = build_M(K1, R1, T1)
    projection2 = build_M(K2, R2, T2)
    pts3d = triangulate_pts(projection1, projection2, pts1.T, pts2.T).T
    return homoco_pts_2_euco_pts(pts3d)


def build_F_pair_match(feats):
    """
    Build F, H, pair and match
    :param feats: (list[dict]): feat of imgs
    :return: (np.ndarray, np.ndarray, dict, dict): F, H, pair of imgs, match of pairs
    """


    pair = dict()
    match = dict()

    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            print(i, j)
            matches = get_matches(
                feats[i]['des'], feats[j]['des'])
            pts1, pts2 = get_match_point(
                feats[i]['kpt'], feats[j]['kpt'], matches)
            assert pts1.shape == pts2.shape
            # Need 8 points to estimate models
            if pts1.shape[0] < 8:
                continue

            F_single = estimate_fundamental(pts1, pts2)

            if pts1.shape[0] < 8:
                continue

            pair.update({(i, j): {'pts1': pts1, 'pts2': pts2}})
            match.update({(i, j): {'match': matches}})


    return F_single,pair, match


def build_img_info(img_root):
    """
    Get info(img,feat,K) from img
    :param img_root: (str): images root
    :return: (list[np.ndarray], list[dict], list[np.ndarray]): info from img
    """
    imgs = []
    feats = []
    K = []
    for i, name in enumerate(os.listdir(img_root)):
        if '.jpg' in name or '.JPG' in name:
            path = os.path.join(img_root, name)
            img = cv2.imread(path)
            imgs.append(img)
            feature_process = FeatureProcess(img)
            kpt, des = feature_process.extract_features()
            photo_info = PhotoExifInfo(path)
            photo_info.get_tags()
            K.append(photo_info.get_intrinsic_matrix())
            A = photo_info.get_area()
            D = photo_info.get_diam()
            feats.append({'kpt': kpt, 'des': des, 'A': A, 'D': D})
    return imgs, feats, K


def estimate_fundamental(pts1, pts2, num_sample=8):
    n = pts1.shape[0]
    pts_index = range(n)
    sample_index = random.sample(pts_index, num_sample)
    p1 = pts1[sample_index, :]
    p2 = pts2[sample_index, :]
    n = len(sample_index)
    p1_norm, T1 = normalize(p1, None)
    p2_norm, T2 = normalize(p2, None)
    w = np.zeros((n, 9))
    for i in range(n):
        w[i, 0] = p1_norm[i, 0] * p2_norm[i, 0]
        w[i, 1] = p1_norm[i, 1] * p2_norm[i, 0]
        w[i, 2] = p2_norm[i, 0]
        w[i, 3] = p1_norm[i, 0] * p2_norm[i, 1]
        w[i, 4] = p1_norm[i, 1] * p2_norm[i, 1]
        w[i, 5] = p2_norm[i, 1]
        w[i, 6] = p1_norm[i, 0]
        w[i, 7] = p1_norm[i, 1]
        w[i, 8] = 1

    U, sigma, VT = np.linalg.svd(w)
    f = VT[-1, :].reshape(3, 3)
    U, sigma, VT = np.linalg.svd(f)
    sigma[2] = 0
    f = U.dot(np.diag(sigma)).dot(VT)
    f = T2.T.dot(f).dot(T1)
    return f


def estimate_homo(pts1, pts2, num_sample=4):
    n = pts1.shape[0]
    pts_index = range(n)
    sample_index = random.sample(pts_index, num_sample)
    p1 = pts1[sample_index, :]
    p2 = pts2[sample_index, :]
    n = len(sample_index)
    w = np.zeros((n * 2, 9))
    for i in range(n):
        w[2 * i, 0] = p1[i, 0]
        w[2 * i, 1] = p1[i, 1]
        w[2 * i, 2] = 1
        w[2 * i, 3] = 0
        w[2 * i, 4] = 0
        w[2 * i, 5] = 0
        w[2 * i, 6] = -p1[i, 0] * p2[i, 0]
        w[2 * i, 7] = -p1[i, 1] * p2[i, 0]
        w[2 * i, 8] = -p2[i, 0]
        w[2 * i + 1, 0] = 0
        w[2 * i + 1, 1] = 0
        w[2 * i + 1, 2] = 0
        w[2 * i + 1, 3] = p1[i, 0]
        w[2 * i + 1, 4] = p1[i, 1]
        w[2 * i + 1, 5] = 1
        w[2 * i + 1, 6] = -p1[i, 0] * p2[i, 1]
        w[2 * i + 1, 7] = -p1[i, 1] * p2[i, 1]
        w[2 * i + 1, 8] = -p2[i, 1]
    U, sigma, VT = np.linalg.svd(w)
    h = VT[-1, :].reshape(3, 3)
    return h

