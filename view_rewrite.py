import cv2
import numpy as np
import utm

# ==============================
# ① 相机参数设置（请按实际情况替换数值）
# ==============================
# 相机内参（单位：像素）
f_x = 1200
f_y = 1200
c_x = 2000
c_y = 1500

# 相机距离地面的高度（单位：米）
h = 230.0

# 相机姿态（角度单位，注意当 pitch=yaw=roll=0 时，相机正垂直俯视地面）
# pitch_deg = 0.0  # 俯仰角 示例：10度
# yaw_deg   = 0.0   # 偏航角 示例：5度
# roll_deg  = 30.0   # 滚转角 示例：0度

pitch_deg = -19.799999  # 俯仰角 示例：10度
yaw_deg   = 3.300000   # 偏航角 示例：5度
roll_deg  = -0.00000   # 滚转角 示例：0度

# roll_deg    = 0.0  # 俯仰角，抬起机头为负 示例：10度
# yaw_deg     = 20.0   # 偏航角，机头逆时针旋转为正 示例：5度
# pitch_deg   = 0.0   # 滚转角，左侧机翼上抬为正 示例：0度

# pitch_deg   = 0.0  # 俯仰角，抬起机头为负 示例：10度
# yaw_deg     = 0.0   # 偏航角，机头逆时针旋转为正 示例：5度
# roll_deg    = 20.0   # 滚转角，左侧机翼上抬为正 示例：0度

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

# 将角度转换为弧度
# pitch = np.deg2rad(pitch_deg)
# yaw   = np.deg2rad(yaw_deg)
# roll  = np.deg2rad(roll_deg)

pitch = np.deg2rad(-roll_deg)
yaw   = np.deg2rad(-yaw_deg)
roll  = np.deg2rad(-pitch_deg)

# 相机垂足的UTM坐标（单位：米）
e, n, _, _ = utm.from_latlon(latitude=lat, longitude=lon)

# ==============================
# ② 地图信息与地理配准
# ==============================
# 遥感地图文件路径（请替换为实际路径）
map_path = r"E:\GeoVINS\Datasets\0521test\L20\0521test.tif"
# 切割后地图路径
save_path = r"E:\GeoVINS\Datasets\0521test\L20\0521test@cuttest.png"
warp_path = r"E:\GeoVINS\Datasets\0521test\L20\0521test@warptest.png"
place_path = r"E:\GeoVINS\Datasets\0521test\L20\0521test@placetest.png"

# 使用 cv2.imread 读取地图（注意：默认 BGR 顺序）
map_img = cv2.imread(map_path)
if map_img is None:
    raise ValueError("读取地图图像失败，请检查路径！")

# 地图图像的尺寸（单位：像素）
H_map, W_map = map_img.shape[:2]

# 地图配准信息：
# 左上顶点UTM坐标 (e₁, n₁) 和右下顶点UTM 坐标 (e₂, n₂)
# e1 = 499000.0    # 地图左上顶点 easting
# n1 = 4601000.0   # 地图左上顶点 northing
# e2 = 501000.0    # 地图右下顶点 easting
# n2 = 4600000.0   # 地图右下顶点 northing
e1, n1,_ ,_ = utm.from_latlon(latitude=lat_1, longitude=lon_1)
e2, n2, _, _ = utm.from_latlon(latitude=lat_2, longitude=lon_2)

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

# 计算四边形的最小外接矩形，返回的 (x, y) 为长方形左上角坐标，w, h 分别为宽和高
x, y, w, h = cv2.boundingRect(map_poly)

# 裁剪出该最小外接矩形区域
cropped = map_visible[y:y+h, x:x+w]

# 如需在结果中画出边界轮廓（红色），可使用：
# cv2.polylines(map_visible, [map_poly], isClosed=True, color=(0, 0, 255), thickness=2)

src_pts = np.array(map_poly, dtype=np.float32)
dst_pts = cam_corners.astype(np.float32)

# 计算透视变换矩阵
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

simulated_view = cv2.warpPerspective(map_img, M, (width_cam, height_cam))

# 保存结果
cv2.imwrite(save_path, cropped)
print("可视区域结果已保存！")

cv2.imwrite(warp_path, simulated_view)
print("warp结果已保存！")


# ==============================

def order_points(pts):
    """
    对 4 个点进行排序，返回的顺序为：
    左上、右上、右下、左下。
    计算方法：
      - 左上角对应 x+y 值最小，右下角对应 x+y 值最大；
      - 右上角对应 x-y 值最小，左下角对应 x-y 值最大。
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # 按照 x+y 求和
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上角：和最小
    rect[2] = pts[np.argmax(s)]  # 右下角：和最大

    # 按照 x-y 差值排序
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上角：差值最小
    rect[3] = pts[np.argmax(diff)]  # 左下角：差值最大
    
    return rect

# # 示例：假设已知的凸四边形 4 个顶点（顺序：左上、右上、右下、左下）
# # 请替换下面的坐标为实际数值
# p1 = (100, 50)    # 左上
# p2 = (400, 50)    # 右上
# p3 = (400, 300)   # 右下
# p4 = (100, 300)   # 左下

# # 将 4 个点组成 NumPy 数组 (浮点型)
# pts = np.array([p1, p2, p3, p4], dtype="float32")

# 计算凸四边形的最小外接旋转矩形（斜长方形）
# cv2.minAreaRect 返回一个元组：((center_x, center_y), (width, height), angle)
min_rect = cv2.minAreaRect(src_pts)

# 通过 cv2.boxPoints 得到矩形的 4 个顶点
box = cv2.boxPoints(min_rect)
box = np.array(box, dtype="float32")

box = order_points(box)

# 对得到的 4 个点进行重新排序，使得顺序为：左上、右上、右下、左下
# ordered_box = order_points(box)

# print("最小外接旋转矩形（斜长方形）的顶点坐标:")
# print("左上角：", ordered_box[0])
# print("右上角：", ordered_box[1])
# print("右下角：", ordered_box[2])
# print("左下角：", ordered_box[3])

# ----------------------------
# 根据旋转矩形的4个顶点进行透视变换
# ----------------------------
# 计算目标矩形的宽度和高度
# widthA = np.linalg.norm(best_box[2] - best_box[3])  # 底边宽度
# widthB = np.linalg.norm(best_box[1] - best_box[0])  # 顶边宽度
# maxWidth = int(max(widthA, widthB))

# heightA = np.linalg.norm(best_box[1] - best_box[2])  # 右侧高度
# heightB = np.linalg.norm(best_box[0] - best_box[3])  # 左侧高度
# maxHeight = int(max(heightA, heightB))

widthA = np.linalg.norm(box[2] - box[3])  # 底边宽度
widthB = np.linalg.norm(box[1] - box[0])  # 顶边宽度
maxWidth = int(max(widthA, widthB))

heightA = np.linalg.norm(box[1] - box[2])  # 右侧高度
heightB = np.linalg.norm(box[0] - box[3])  # 左侧高度
maxHeight = int(max(heightA, heightB))

cam_corners_new = np.array([
    [0, 0],                             # 左上角
    [maxWidth - 1, 0],                 # 右上角
    [maxWidth - 1, maxHeight - 1],      # 右下角
    [0, maxHeight - 1]                 # 左下角
], dtype=np.float32)

dst_pts_new = cam_corners_new.astype(np.float32)


M = cv2.getPerspectiveTransform(box, dst_pts_new)
warped = cv2.warpPerspective(map_img, M, (maxWidth, maxHeight))




def remove_black_borders(img, thresh=0):
    """
    移除图像四周全为黑色的边框。
    参数：
      img：输入图像（可以是彩色或灰度）
      thresh：对灰度图而言，大于 thresh 的值判断为有效像素（这里 thresh 默认为0，即全为0才认为是黑）。
    返回：
      裁剪后的图像。
    """
    # 如果是彩色图像则转换为灰度图便于处理
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 找出每一行和每一列是否存在非黑像素
    # 方法1：利用 np.max 沿指定轴求最大值
    rows = np.where(np.max(gray, axis=1) > thresh)[0]
    cols = np.where(np.max(gray, axis=0) > thresh)[0]

    # 如果图像中不存在非黑区域，则直接返回原图
    if rows.size == 0 or cols.size == 0:
        return img

    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]
    
    # 裁剪时注意 bottom 和 right 为包含索引，所以要加 1
    cropped = img[top:bottom+1, left:right+1]
    return cropped

# 去除图像边缘的黑边（不同边缘可能宽度不同）
warped = remove_black_borders(warped)

cv2.imwrite(place_path, warped)
print("斜长方形内的图像已保存！")


# cv2.polylines(map_img, [pts1_clamped], isClosed=True, color=(0, 0, 255), thickness=2)  # 红色
# cv2.polylines(map_img, [pts2_clamped], isClosed=True, color=(0, 255, 0), thickness=2)  # 绿色

# # ---------------------------
# # ⑤ 保存结果图像
# # 请自行设置输出路径与文件名
# output_path = "output.jpg"
# cv2.imwrite(output_path, image)
# print(f"带有边框的图像已保存为 {output_path}")