# author:nannan
# contact: zhaozhaoran@bupt.edu.cn
# datetime:2020/9/13 7:18 下午
# software: PyCharm

import numpy as np
import utils

def my_run():

    img_root = './data/'
    imgs, feats, K = utils.build_img_info(img_root)
    F, pair, match = utils.build_F_pair_match(feats)
    print("The Fundamental Matrix is:")
    print(F)





if __name__=='__main__':
    my_run()