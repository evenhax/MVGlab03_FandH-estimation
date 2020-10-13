import exifread
import numpy as np


class PhotoExifInfo:
    """
    Extract photo exif info
    """

    def __init__(self, photo_path):
        """
        init
        :param photo_path: (str): photo path
        """
        self.photo_path = photo_path
        self.focal_length = None
        self.image_width = None
        self.image_length = None
        self.sensor_pixel_size = None

    def get_tags(self):
        """
        Get tags with interested info
        :return: None
        """
        image_content = open(self.photo_path, 'rb')
        tags = exifread.process_file(image_content)
        self.focal_length = float(
            tags['EXIF FocalLength'].values[0].num) / float(tags['EXIF FocalLength'].values[0].den)
        self.image_width = float(tags['EXIF ExifImageWidth'].values[0])
        self.image_length = float(tags['EXIF ExifImageLength'].values[0])
        self.sensor_pixel_size = tags['MakerNote SensorPixelSize']

    def get_intrinsic_matrix(self):
        """
        Get intrinsic matrix of photo's camera
        :return: (np.ndarray): intrinsic matrix K
        """
        K = np.zeros([3, 3])
        dx = self.sensor_pixel_size.values[0].num / self.sensor_pixel_size.values[0].den / self.image_width
        dy = self.sensor_pixel_size.values[1].num / self.sensor_pixel_size.values[1].den / self.image_length
        fu = self.focal_length / dx
        fv = self.focal_length / dy
        u0 = self.image_width / 2
        v0 = self.image_length / 2
        K[0][0] = fu
        K[1][1] = fv
        K[0][2] = u0
        K[1][2] = v0
        K[2][2] = 1
        return K

    def get_area(self):
        """
        Get area of photo
        :return: (int): area
        """
        return int(self.image_width * self.image_length)

    def get_diam(self):
        """
        Get diam of photo
        :return: (int): diam
        """
        return int(max(self.image_width, self.image_length))
