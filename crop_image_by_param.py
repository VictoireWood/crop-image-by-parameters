yaw = 3.300000+2.200000     # 偏航角，用度做单位
pitch = -19.799999+6.200000+180    # 俯仰角
roll = -20.400000+180    # 滚转角
h = 199.000000      # 相机到地面高度，用米做单位
# 内参矩阵，都以像素为单位  8.0 mm
f_x = 1200
f_y = 1200
c_x = 2000
c_y = 1500

# 底片像素宽、高，单位为像素
H = 3000
W = 4000

# 已知遥感地图的四角UTM坐标以及遥感地图大小
# UTM坐标系下的点
# e_1 = 500
# n_1 = 500
# e_2 = 1500
# n_2 = 1500
# e = 500
# n = 500
# 地图四个角点的经纬度坐标，1为左上，2为右下
lat_1 = 36.607321780842
lon_1 = 120.430745854908
lat_2 = 36.587287108611
lon_2 = 120.456556510736
# 相机中心的垂点的经纬度坐标
# 36 deg 35' 28.05" N, 120 deg 27' 17.95" E
lat = 36.5911250000
lon = 120.4549861111

# 原始地图路径
map_path = r"E:\GeoVINS\Datasets\0521test\L20\0521test.tif"
# 切割后地图路径
save_path = r"E:\GeoVINS\Datasets\0521test\L20\0521test@cuttest.tif"

from math import sin, cos, radians
import utm
import numpy as np
import cv2
# 计算UTM坐标
e_1, n_1,_ ,_ = utm.from_latlon(latitude=lat_1, longitude=lon_1)
e_2, n_2, _, _ = utm.from_latlon(latitude=lat_2, longitude=lon_2)
e, n, _, _ = utm.from_latlon(latitude=lat, longitude=lon)

map_raw = cv2.imread(map_path, cv2.IMREAD_UNCHANGED)
map_H = map_raw.shape[0]
map_W = map_raw.shape[1]
UTM = np.array([e, n])
def rotate_yaw_pitch_roll(yaw, pitch, roll):
    # 求三维旋转矩阵
    R_yaw = np.array([
        [cos(yaw), -sin(yaw), 0],
        [sin(yaw), cos(yaw), 0],
        [0, 0, 1]
        ])
    R_pitch = np.array([
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)]
        ])
    R_roll = np.array([
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)]
        ])
    # 先绕z轴旋转yaw，再绕y轴旋转pitch，最后绕x轴旋转roll
    R = R_yaw @ R_pitch @ R_roll
    return R

def rotate_point(point, utm, R: np.ndarray): # point是指在成像平面上的点的坐标
    u_p = point[0]
    v_p = point[1]
    e = utm[0]
    n = utm[1]
    # 将成像平面上的点化为UTM坐标点
    X = (u_p - c_x) / f_x
    Y = (v_p - c_y) / f_y
    d = R @ np.array([X, Y, 1])
    d_x = d[0]
    d_y = d[1]
    d_z = d[2]
    # 将UTM坐标点化为成像平面上的点
    s = -h / d_z
    E = e + s * d_x
    N = n + s * d_y
    UTM = np.array([E, N])
    return UTM

vertex = np.array([[0, 0],
                  [W, 0],
                  [W, H],
                  [0, H]])
# 左上角：(0,0)
# 右上角：(W,0)
# 右下角：(W,H)
# 左下角：(0,H)

UTM_vertex = np.zeros((4, 2))

R = rotate_yaw_pitch_roll(radians(yaw), radians(pitch), radians(roll))

for i in range(4):
    p = vertex[i]
    # 计算UTM坐标
    UTM_vertex_tmp = rotate_point(p, UTM, R)
    UTM_vertex[i] = UTM_vertex_tmp

# 已知4个点的UTM坐标，求在遥感图像上对应的像素坐标点
def get_pixel_coordinate(UTM_vertex):
    pixel_coordinate = np.zeros((4, 2))
    for i in range(4):
        e = UTM_vertex[i][0]
        n = UTM_vertex[i][1]
        # 计算像素坐标
        u = (e - e_1) / (e_2 - e_1) * map_W
        if u < 0:
            u = 0
        if u > map_W:
            u = map_W
        v = (n - n_1) / (n_2 - n_1) * map_H
        if v < 0:
            v = 0
        if v > map_H:
            v = map_H
        pixel_coordinate[i] = np.array([u, v], dtype=np.int32)
        # print(pixel_coordinate)
    return pixel_coordinate

def get_cut_image(pixel_coordinate):
    # 根据像素坐标裁剪图像
    # 构造四边形顶点数组，注意数据类型必须是整数
    # pts = np.array([[x0, y0],
    #                 [x1, y1],
    #                 [x2, y2],
    #                 [x3, y3]], dtype=np.int32)
    
    pts = np.array(pixel_coordinate, dtype=np.int32)


    # 创建与图像尺寸相同的单通道黑色遮罩
    mask = np.zeros((map_H, map_W), dtype=np.uint8)

    # 在遮罩上填充四边形区域，像素值 255 表示白色区域（保留）
    cv2.fillPoly(mask, [pts], 255)

    # 利用遮罩将原图像四边形外的区域置为黑色
    masked_img = cv2.bitwise_and(map_raw, map_raw, mask=mask)

    # 计算四边形的最小外接矩形，返回的 (x, y) 为长方形左上角坐标，w, h 分别为宽和高
    x, y, w, h = cv2.boundingRect(pts)

    # 裁剪出该最小外接矩形区域
    cropped = masked_img[y:y+h, x:x+w]

    # 保存裁剪后的图像
    cv2.imwrite(save_path, cropped)

    print("图像处理完成，新图像已保存！")


map_cropped = get_cut_image(get_pixel_coordinate(UTM_vertex))






    





