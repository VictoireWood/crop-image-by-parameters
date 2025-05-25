import numpy as np
import cv2
from PIL import Image
import utm
from math import sin, cos, radians

# ============================
# ① 参数设置（请根据实际情况填写）
# ============================

# 相机内参（单位：像素）
# 内参矩阵，都以像素为单位  8.0 mm
f_x = 1200
f_y = 1200
c_x = 2000
c_y = 1500

# 相机离地高度（单位：米）
h = 199.0       # 示例：100 米

# 相机姿态（弧度制）
pitch = 0.0     # 示例：轻微俯仰（根据实际情况确定正负）
yaw   = 0.0     # 示例：
roll  = 0.0     # 示例：


lat_1 = 36.607321780842
lon_1 = 120.430745854908
lat_2 = 36.587287108611
lon_2 = 120.456556510736
# 相机中心的垂点的经纬度坐标
# 36 deg 35' 28.05" N, 120 deg 27' 17.95" E
lat = 36.5911250000
lon = 120.4549861111

e_1, n_1,_ ,_ = utm.from_latlon(latitude=lat_1, longitude=lon_1)
e_2, n_2, _, _ = utm.from_latlon(latitude=lat_2, longitude=lon_2)
e, n, _, _ = utm.from_latlon(latitude=lat, longitude=lon)


map_path = r"E:\GeoVINS\Datasets\0521test\L20\0521test.tif"
# 切割后地图路径
save_path = r"E:\GeoVINS\Datasets\0521test\L20\0521test@cuttest.png"
warp_path = r"E:\GeoVINS\Datasets\0521test\L20\0521test@warptest.png"

# ============================
# ② 遥感地图读取与地理配准信息
# ============================

# 使用 PIL 读入地图
map_img_pil = Image.open(map_path)
# 转换为 numpy 数组（RGB 顺序）
map_img = np.array(map_img_pil)

# 地图的地理配准信息
# 地图左上角在 UTM 坐标下的位置
map_east = 499000.0   # 示例数值
map_north = 4601000.0  # 示例数值
# 每个像素代表的实际大小（米/像素）
resolution = 0.5      # 示例：0.5 米/像素

resolution = 


# ============================
# ③ 构造相机内参矩阵
# ============================
K = np.array([
    [f_x,   0, c_x],
    [  0, f_y, c_y],
    [  0,   0,   1]
])

# ============================
# ④ 构造旋转矩阵
# ============================

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

R = rotate_yaw_pitch_roll(radians(yaw), radians(pitch), radians(roll))

def get_rotation_matrix(roll, pitch, yaw):
    """
    构造旋转矩阵，旋转顺序：先绕 x（roll），再绕 y（pitch），最后绕 z（yaw）。
    注：根据具体应用，旋转顺序和定义可能需要调整。
    """
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    
    R_y = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [         0,           0, 1]
    ])
    
    return R_z @ R_y @ R_x

R = get_rotation_matrix(roll, pitch, yaw)

# ============================
# ⑤ 构造地面（z=0）到相机图像的单应变换
# ============================

# 相机在 UTM 坐标中的位置
C = np.array([e, n, h])
# 平移向量 t = -R · C
t = - R @ C

# 构造： H_ground2img = K · [r1  r2  t]
# 其中 r1 和 r2 为 R 的第一列和第二列
H_ground2img = K @ np.hstack((R[:, :2], t.reshape(3, 1)))
# 此矩阵满足：对于任意地面上点 [X, Y, 1]ᵀ（UTM 坐标中的齐次表示），
# [u, v, 1]ᵀ ~ H_ground2img · [X, Y, 1]ᵀ

# ============================
# ⑥ 构造地图像素到 UTM 坐标的转换矩阵
# ============================
# 假设地图像素坐标 (u_map, v_map, 1) 与 UTM 坐标转换满足：
#   UTM_x = map_east + resolution * u_map
#   UTM_y = map_north - resolution * v_map
T_map_to_utm = np.array([
    [resolution,          0, map_east],
    [         0, -resolution, map_north],
    [         0,          0,       1 ]
])

# ============================
# ⑦ 复合变换：从遥感地图像素到相机图像
# ============================
H_map2img = H_ground2img @ T_map_to_utm
# cv2.warpPerspective 要求提供“从输出图像像素到输入图像像素”的映射，
# 故这里取 H_map2img 的逆：
H_warp = np.linalg.inv(H_map2img)

# ============================
# ⑧ 定义输出（模拟相机）图像尺寸并生成图像
# ============================
# 输出尺寸可依据实际相机传感器分辨率设定（此处示例取宽=2*c_x， 高=2*c_y）
W_sim = int(2 * c_x)
H_sim = int(2 * c_y)

# 使用 warpPerspective 将遥感地图转换为模拟相机图像
simulated_img = cv2.warpPerspective(map_img, H_warp, (W_sim, H_sim))

# 保存生成的模拟图像
output_path = "simulated_camera_view.jpg"
cv2.imwrite(output_path, simulated_img)
print(f"模拟的相机拍摄场景已保存为 '{output_path}'.")
