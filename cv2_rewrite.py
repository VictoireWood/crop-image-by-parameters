yaw_deg = 0     # 偏航角，用度做单位
pitch_deg = 0    # 俯仰角
roll_deg = 0    # 滚转角

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
# 地图四个角点的经纬度坐标，1为左上，2为右下
lat_1 = 36.607321780842
lon_1 = 120.430745854908
lat_2 = 36.587287108611
lon_2 = 120.456556510736
# 相机中心的垂点的经纬度坐标
# 36 deg 35' 28.05" N, 120 deg 27' 17.95" E
lat = 36.5911250000
lon = 120.4549861111
import utm

e_1, n_1,_ ,_ = utm.from_latlon(latitude=lat_1, longitude=lon_1)
e_2, n_2, _, _ = utm.from_latlon(latitude=lat_2, longitude=lon_2)
e, n, _, _ = utm.from_latlon(latitude=lat, longitude=lon)

# 原始地图路径
map_path = r"E:\GeoVINS\Datasets\0521test\L20\0521test.tif"
# 切割后地图路径
save_path = r"E:\GeoVINS\Datasets\0521test\L20\0521test@cuttest.png"
warp_path = r"E:\GeoVINS\Datasets\0521test\L20\0521test@warptest.png"
import numpy as np
import cv2

# ==============================
# ① 相机参数设置（请替换成实际数值）
# ==============================

# 转换为弧度
pitch = np.deg2rad(pitch_deg)
yaw   = np.deg2rad(yaw_deg)
roll  = np.deg2rad(roll_deg)

# ==============================
# ② 遥感地图及其地理配准信息
# ==============================

# 遥感地图文件路径（请替换为实际路径）
# 使用 cv2.imread 读入地图（默认 BGR 格式）
map_img = cv2.imread(map_path)
if map_img is None:
    raise ValueError("读取地图图像失败，请检查路径！")

# 地图图像的尺寸（单位：像素）
H_map, W_map = map_img.shape[:2]   # H_map：高度, W_map：宽度

# 构造地图像素到 UTM 坐标的变换矩阵 T_map2utm
# 对于地图像素 (u, v)：
#   X_utm = e1 + (e2 - e1) * (u / W_map)
#   Y_utm = n1 + (n2 - n1) * (v / H_map)
T_map2utm = np.array([
    [(e_2 - e_1) / W_map, 0, e_1],
    [0, (n_2 - n_1) / H_map, n_1],
    [0, 0, 1]
])

# ==============================
# ③ 构造相机内参矩阵
# ==============================
K = np.array([
    [f_x,  0,   c_x],
    [ 0,  f_y,  c_y],
    [ 0,   0,    1 ]
])

# ==============================
# ④ 构造相机旋转矩阵
# ==============================
# 为使当 pitch=yaw=roll=0 时相机正垂直俯视，我们先定义下视基准 R0：
R0 = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

# 根据 Euler 角（旋转顺序：先绕 x（roll），再绕 y（pitch），最后绕 z（yaw））构造附加旋转 R_add
def rotation_from_euler(roll, pitch, yaw):
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return R_z @ R_y @ R_x

R_add = rotation_from_euler(roll, pitch, yaw)
# 整体相机旋转矩阵 R （注意旋转顺序与定义需与系统一致）
R = R_add @ R0

# ==============================
# ⑤ 构造从地面（UTM 坐标平面，z=0）到相机图像的单应变换
# ==============================
# 相机中心在 UTM 中：C = [e, n, h]ᵀ
C = np.array([e, n, h])
# 平移向量： t = -R · C
t = - R @ C

# 对于地面上 (X, Y, 0) 的点，其齐次表示为 [X, Y, 1]ᵀ，
# 投影到图像： [u, v, 1]ᵀ ~ K · [r₁, r₂, t] · [X, Y, 1]ᵀ，
# 其中 r₁ 和 r₂ 分别为 R 的第一、二列
H_ground = K @ np.hstack((R[:, :2], t.reshape(3, 1)))

# ==============================
# ⑥ 复合变换：从地图像素到相机图像
# ==============================
# 先将地图像素转换到 UTM 坐标，再由 H_ground 投影到图像上：
H_total = H_ground @ T_map2utm

# 注意：cv2.warpPerspective 需要的是从输出图像像素到输入（地图）像素的映射，
# 故这里使用 H_total 的逆：
H_warp = np.linalg.inv(H_total)

# ==============================
# ⑦ 生成模拟相机图像
# ==============================
# 设定模拟相机图像的尺寸；例如可取 2*c_x × 2*c_y（可根据实际需要修改）
simulated_width  = int(2 * c_x)
simulated_height = int(2 * c_y)

# 使用 warpPerspective 将遥感地图转换为模拟相机视图
simulated_view = cv2.warpPerspective(map_img, M, (simulated_width, simulated_height))

# 保存模拟结果
# output_path = "simulated_camera_view.jpg"
cv2.imwrite(warp_path, simulated_view)
print(f"模拟的相机拍摄场景已保存为 '{warp_path}'.")
