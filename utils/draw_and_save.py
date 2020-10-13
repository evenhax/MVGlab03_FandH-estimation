from mayavi import mlab
import numpy as np
import open3d as o3d

@mlab.show
def draw_v1(cloud3d):
    """
    Draw cloud3d using mayavi
    :param cloud3d: (np.ndarray): 3d points
    :return: None
    """
    for point in cloud3d:
        mlab.points3d(point[0], point[1], point[2], mode='point', name='dinosaur')
    mlab.show()


def save_3d(mypts, filename):
    mypts = np.array(mypts).reshape(-1, 3)
    np.savetxt(filename, mypts, fmt='%f %f %f ')
    ply_header = '''ply\nformat ascii 1.0\nelement vertex %(vert_num)d\nproperty float x\nproperty float y\nproperty float z\nend_header\n'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(mypts)))
        f.write(old)

def show3d(filename,window_name):
    pcd = o3d.io.read_point_cloud(filename)
    o3d.visualization.draw_geometries([pcd],window_name)