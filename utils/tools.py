
import numpy as np
import pandas as pd
import os
import cv2

def caluate_homography_matrix(src_pts,dst_pts):
    """计算单应性矩阵"""
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

if __name__ == "__main__":
    #物理坐标点
    src_pts = [(124.14,0.2),(103.83,0.2),(114.64,55),(109.19,60.6),(10.1,0.28)]
    #旁轴相机像素点
    dst_pts_rgb = [(699,411),(635,411),(676,259),(640,243),(371,413)]
    #旁轴红外像素点
    dst_pts_ir = [(355,324),(292,323),(321,181),(311,167),(105,322)]
    H_rgb = caluate_homography_matrix(src_pts,dst_pts_rgb)
    H_ir = caluate_homography_matrix(src_pts,dst_pts_ir)
    print("H_rgb:",H_rgb)
    print("H_ir:",H_ir)
    '''
    H_rgb: [[-1.57739488e+00 -3.01652536e+00  4.95179045e+02]
            [-1.89268649e+00 -2.49586238e+00  4.11121812e+02]
            [-4.60507661e-03 -4.59075942e-03  1.00000000e+00]]
    H_ir: [[-1.46859995e+01 -8.81963039e+00  1.02122124e+03]
            [-8.58883405e+00  8.90290676e-02  3.32716381e+02]
            [-2.62457270e-02 -2.84697457e-02  1.00000000e+00]]
    '''