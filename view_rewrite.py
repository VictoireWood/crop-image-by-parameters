import cv2
import numpy as np

# ==============================
# ① 相机参数设置（请按实际情况替换数值）
# ==============================
# 相机内参（单位：像素）
f_x = 1200.0      # x方向焦距
f_y = 1200.0      # y方向焦距
c_x = 640.0       # 主点 x 坐标（假设图像宽度的一半）
c_y = 360.0       # 主点 y 坐标（假设图像高度的一半）

# 相机距离地面的高度（单位：米）
h = 100.0

# 相机姿态（角度单位，注意当 pitch=yaw=roll=0 时，相机正垂直俯视地面）
pitch_deg = 10.0  # 示例：10度
yaw_deg   = 5.0   # 示例：5度
roll_deg  = 0.0   # 示例：0度

# 将角度转换为弧度
pitch = np.deg2rad(pitch_deg)
yaw   = np.deg2rad(yaw_deg)
roll  = np.deg2rad(roll_deg)

# 相机垂足的UTM坐标（单位：米）
e = 500000.0    
n = 4600000.0   

# ==============================
# ② 地图信息与地理配准
# ==============================
# 遥感地图文件路径（请替换为实际路径）
map_path = "path/to/your/map_image.jpg"
# 使用 cv2.imread 读取地图（注意：默认 BGR 顺序）
map_img = cv2.imread(map_path)
if map_img is None:
    raise ValueError("读取地图图像失败，请检查路径！")

# 地图图像的尺寸（单位：像素）
H_map, W_map = map_img.shape[:2]

# 地图配准信息：
# 左上顶点UTM坐标 (e₁, n₁) 和右下顶点UTM 坐标 (e₂, n₂)
e1 = 499000.0    # 地图左上顶点 easting
n1 = 4601000.0   # 地图左上顶点 northing
e2 = 501000.0    # 地图右下顶点 easting
n2 = 4600000.0   # 地图右下顶点 northing

# ==============================
# ③ 计算相机可视区域在地面上的四个角点（UTM坐标）
# ==============================
# 为确保当 pitch=yaw=roll=0 时相机正垂直向下，需要构造“下视基准”旋转：
R0 = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])
# 根据 Euler角（旋转顺序：先绕 x（roll），再绕 y（pitch），最后绕 z（yaw））构造附加旋转矩阵：
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
# 整体旋转矩阵
R = R_add @ R0

# 假设相机成像分辨率未知，此处可根据内参信息估计传感器尺寸，通常令图像尺寸为 2*c_x×2*c_y
width_cam = int(2 * c_x)
height_cam = int(2 * c_y)

# 定义相机图像四个角点（像素坐标）
cam_corners = np.array([
    [0, 0],                             # 左上角
    [width_cam - 1, 0],                 # 右上角
    [width_cam - 1, height_cam - 1],      # 右下角
    [0, height_cam - 1]                 # 左下角
], dtype=np.float32)

# 投影步骤：
# ① 将像素坐标转换到归一化相机坐标： x = (u - c_x) / f_x, y = (v - c_y) / f_y, 构成向量 [x, y, 1]
# ② 在相机坐标下，该向量乘以旋转矩阵 R 得到世界坐标下的方向 d
# ③ 相机中心在世界中的位置 C = [e, n, h]ᵀ，射线方程为: P = C + s·d
# ④ 求 s 使得 P_z = 0 ，即 s = -h / d_z
ground_pts_utm = []
C = np.array([e, n, h])
for corner in cam_corners:
    u, v = corner
    x = (u - c_x) / f_x
    y = (v - c_y) / f_y
    ray_cam = np.array([x, y, 1.0])
    # 方向向量（世界坐标）
    d = R @ ray_cam
    # 求交比例系数 s（确保 d[2] 非0）
    s = -h / d[2]
    # 地面交点 (E, N) 在UTM中的坐标
    E = e + s * d[0]
    N = n + s * d[1]
    ground_pts_utm.append([E, N])
ground_pts_utm = np.array(ground_pts_utm, dtype=np.float32)

# ==============================
# ④ UTM 坐标 转 到 地图图像像素坐标
# ==============================
# 假设地图图像坐标系统的 u 轴（横向）与 UTM easting 对应，
# 垂直方向 v 轴与 UTM northing 以“上”为正，但图像中 v 由上到下递增，因此：
#   u_map = ((E - e1) / (e2 - e1)) * W_map
#   v_map = ((n1 - N) / (n1 - n2)) * H_map
map_poly = []
for pt in ground_pts_utm:
    E, N = pt
    u_map = ((E - e1) / (e2 - e1)) * W_map
    v_map = ((n1 - N) / (n1 - n2)) * H_map
    map_poly.append([u_map, v_map])
map_poly = np.array(map_poly, dtype=np.int32)

# ==============================
# ⑤ 绘制掩模：在遥感地图上用凸四边形（可视区域）框出，并将外部区域变为黑色
# ==============================
# 创建一个与地图同大小的单通道掩模（初始全黑）
mask = np.zeros((H_map, W_map), dtype=np.uint8)
# 在掩模上填充多边形区域（值设为255）
cv2.fillPoly(mask, [map_poly], 255)
# 将掩模应用到地图图像上（对彩色图，每个通道均应用）
map_visible = cv2.bitwise_and(map_img, map_img, mask=mask)

# 如需在结果中画出边界轮廓（红色），可使用：
cv2.polylines(map_visible, [map_poly], isClosed=True, color=(0, 0, 255), thickness=2)

# 保存结果
cv2.imwrite("camera_fov_on_map.jpg", map_visible)
print("结果已保存为 'camera_fov_on_map.jpg'.")
