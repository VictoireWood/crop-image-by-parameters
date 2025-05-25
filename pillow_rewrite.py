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

yaw = 0     # 偏航角，用度做单位
pitch = 0    # 俯仰角
roll = 0    # 滚转角


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



import numpy as np
import cv2
from PIL import Image

# ==============================
# ② 遥感地图及其地理配准信息
# ==============================

# 遥感地图文件路径（请替换成实际路径）
map_path = "path/to/your/map_image.jpg"
# 用 PIL 读取地图（注意：返回RGB图像，shape 为 (H, W, 3)）
map_img_pil = Image.open(map_path)
map_img = np.array(map_img_pil)
# 获取地图图像尺寸
H_map, W_map = map_img.shape[:2]

# 构造从地图像素坐标 [u_map, v_map, 1] 到 UTM 坐标 [X, Y, 1] 的变换矩阵
T_map2utm = np.array([
    [(e_2 - e_1) / W_map,           0, e_1],
    [          0,       (n_2 - n_1) / H_map, n_1],
    [          0,                0,   1]
])
# 说明：对于地图像素 (u_map, v_map)，有
#   X_utm = e1 + (e2 - e1) * (u_map / W_map)
#   Y_utm = n1 + (n2 - n1) * (v_map / H_map)

# ==============================
# ③ 构造相机内参矩阵
# ==============================

K = np.array([
    [f_x,   0, c_x],
    [  0, f_y, c_y],
    [  0,   0,   1]
])

# ==============================
# ④ 构造相机旋转矩阵
# ==============================
# 注：为了让 pitch = yaw = roll = 0 时相机正垂直俯视，
# 我们先构造下视基准旋转矩阵 R0，使得将世界中 (X, Y, 0) 点（相对于相机垂直下方）映射到相机时获得正深度。
# 定义 R0 为：
R0 = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])
# 接下来构造附加旋转矩阵 R_add，根据惯常 Tait–Bryan 角（旋转顺序 z-y-x），要求当各角为0时 R_add 为单位矩阵。
def rotation_matrix_from_euler(roll, pitch, yaw):
    """
    构造旋转矩阵（旋转顺序：先绕 x 轴（roll），再绕 y 轴（pitch），最后绕 z 轴（yaw））。
    输入角度单位为弧度。
    """
    R_x = np.array([
        [1,             0,              0],
        [0,  np.cos(roll), -np.sin(roll)],
        [0,  np.sin(roll),  np.cos(roll)]
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

# 将角度从度转换为弧度
roll_rad  = np.deg2rad(roll)
pitch_rad = np.deg2rad(pitch)
yaw_rad   = np.deg2rad(yaw)

R_add = rotation_matrix_from_euler(roll_rad, pitch_rad, yaw_rad)
# 整体相机旋转矩阵 R
R = R_add @ R0

# ==============================
# ⑤ 构造地面（UTM 坐标平面，z=0）到相机图像的单应变换
# ==============================
# 相机中心在UTM下的坐标为 C = [e, n, h]ᵀ
C = np.array([e, n, h])
# 平移向量 t = -R · C
t = - R @ C

# 提取 R 的前两列作为 r₁ 和 r₂
# 构造 3×3 单应矩阵：H_ground = K · [r1, r2, t]
H_ground = K @ np.hstack((R[:, 0:2], t.reshape(3, 1)))
# 此矩阵满足：对于 UTM 地面点 [X, Y, 1]ᵀ（Z=0），图像齐次坐标 ~ H_ground · [X, Y, 1]ᵀ

# ==============================
# ⑥ 复合变换：从地图像素到相机图像
# ==============================
# 先将地图像素映射到 UTM 坐标，再投影到图像上：
H_total = H_ground @ T_map2utm
# cv2.warpPerspective 要求提供的是从输出图像像素到源像素的映射矩阵，
# 因此取 H_total 的逆：
H_warp = np.linalg.inv(H_total)

# ==============================
# ⑦ 用遥感地图生成模拟相机图像
# ==============================
# 定义模拟相机图像的尺寸。此处可按需要设置；
# 比如，我们可以采用与相机传感器相同的尺寸（例如 2*c_x × 2*c_y）
sim_width  = int(2 * c_x)
sim_height = int(2 * c_y)

# 使用 cv2.warpPerspective 对遥感地图进行透视变换，生成模拟相机视图
simulated_img = cv2.warpPerspective(map_img, H_warp, (sim_width, sim_height))

# 保存结果
output_path = "simulated_camera_view.jpg"
cv2.imwrite(output_path, simulated_img)
print(f"模拟的相机拍摄场景已保存为 '{output_path}'.")
