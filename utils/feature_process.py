# author:nannan
# contact: zhaozhaoran@bupt.edu.cn
# datetime:2020/9/13 7:44 下午
# software: PyCharm

import cv2
import numpy as np


class FeatureProcess:
    """
    Simple SIFT feature process
    """

    def __init__(self, image):
        """
        Init
        :param image: (np.ndarray): Image in RGB
        """
        self.image = image
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.keypoints = None
        self.descriptors = None

    def extract_features(self):
        """
        Extract SIFT features in image.
        :return: (list, np.ndarray): keypoints, descriptors in image
        """
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(self.gray, None)

        if len(keypoints) <= 20:
            return None, None
        else:
            self.keypoints = keypoints
            self.descriptors = descriptors
            return keypoints, descriptors


def get_matches(des_query, des_train):
    """
    Match features between query and train
    :param des_query: (np.ndarray): query descriptors
    :param des_train: (np.ndarray): train descriptors
    :return: (list[cv2.DMatch]): Match info
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_query, des_train, k=2)

    good = []
    for m, m_ in matches:
        # Ratio is 0.6 ,which to remain enough features
        if m.distance < 0.6 * m_.distance:
            good.append(m)
    return good


def get_match_point(p, p_, matches):
    """
    Find matched keypoints
    :param p: (list[cv2.KeyPoint]): query keypoints
    :param p_: (list[cv2.KeyPoint]): train keypoints
    :param matches: (list[cv2.DMatch]): match info between query and train
    :return: (np.ndarray, np.ndarray): matched keypoints between query and train
    """
    points_query = np.asarray([p[m.queryIdx].pt for m in matches])
    points_train = np.asarray([p_[m.trainIdx].pt for m in matches])
    return points_query, points_train
