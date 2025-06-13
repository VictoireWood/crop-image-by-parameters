import cv2
import numpy as np
import utm
import pandas as pd
import os

# ==============================
# ① 相机参数设置（请按实际情况替换数值）
# ==============================
# 相机内参（单位：像素）
f_x = 600
f_y = 600
# f_x = 833.3
# f_y = 833.3
c_x = 640
c_y = 512

# ==============================
# ② 地图信息与地理配准
# ==============================
# 遥感地图文件路径（请替换为实际路径）
map_path = r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\lon1_103.122075@lat1_38.542367@lon2_103.294111@lat2_38.415714.jpg"
drone_dir = r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\drone"
# CSV 文件路径（请根据实际情况修改）
drone_csv_path = r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\drone.csv"

# 定义 CSV 表头（共11列）
columns = [
    "drone_name", "zone_number", "zone_letter",
    "left_top_e", "left_top_n",
    "right_top_e", "right_top_n",
    "right_bottom_e", "right_bottom_n",
    "left_bottom_e", "left_bottom_n"
]
area_csv_path = os.path.join(os.path.dirname(drone_dir), "area_coordinates.csv")
df_header = pd.DataFrame(columns=columns)
df_header.to_csv(area_csv_path, index=False, mode='w')

# 读入 CSV 文件，pandas 会自动解析列类型
df = pd.read_csv(drone_csv_path)

# 将每一行转换为元组，注意：使用 index=False 避免也包含 DataFrame 的索引，
# name=None 则返回普通的 tuple 类型
data_list = list(df.itertuples(index=False, name=None))
# 元组数据：drone_name,latitude,longitude,altitude,yaw,pitch,roll

def trim_and_warp(drone_dir, map_path, tuple_data):

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

    def best_cyclic_order(rot_box_pts, orig_pts):
        """
        在 rot_box_pts（4×2 数组，假定为按照一定顺序排列，如由 order_points 得到）中，
        尝试 4 种循环排列，选择使得每个点与 orig_pts（原凸四边形的 4 个顶点，
        顺序为 [top-left, top-right, bottom-right, bottom-left]）的欧氏距离和最小的排列。
        """
        best_order = None
        best_dist = float('inf')
        for shift in range(4):
            candidate = np.roll(rot_box_pts, shift, axis=0)
            # 计算每个对应点之间欧氏距离的总和
            dist = np.sum(np.linalg.norm(candidate - orig_pts, axis=1))
            if dist < best_dist:
                best_dist = dist
                best_order = candidate.copy()
        return best_order
    
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

    map_img = cv2.imread(map_path)
    if map_img is None:
        raise ValueError("读取地图图像失败，请检查路径！")

    # 地图图像的尺寸（单位：像素）
    H_map, W_map = map_img.shape[:2]

    # 元组数据：drone_name,latitude,longitude,altitude,yaw,pitch,roll
    drone_name = tuple_data[0]
    # order_name = drone_name.replace('.jpg', '')
    lat = tuple_data[1]
    lon = tuple_data[2]
    h = tuple_data[3]
    yaw_deg = tuple_data[4]
    pitch_deg = tuple_data[5]
    roll_deg = tuple_data[6]

    # 已知遥感地图的四角UTM坐标以及遥感地图大小
    # 合并后地图左上角经纬度： (103.122075, 38.542367)
    # 合并后地图右下角经纬度： (103.294111, 38.415714)
    # 地图四个角点的经纬度坐标，1为左上，2为右下
    lat_1 = 38.542367
    lon_1 = 103.122075
    lat_2 = 38.415714
    lon_2 = 103.294111

    # 切割后地图路径
    base_dir = os.path.dirname(drone_dir)
    save_dir = os.path.join(base_dir, "Crop")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(base_dir, "Crop", f"{drone_name}")
    warp_dir = os.path.join(base_dir, "Warp")
    os.makedirs(warp_dir, exist_ok=True)
    warp_path = os.path.join(base_dir, "Warp", f"{drone_name}")
    area_dir = os.path.join(base_dir, "Area")
    os.makedirs(area_dir, exist_ok=True)
    area_path = os.path.join(base_dir, "Area", f"{drone_name}")

    # 将角度转换为弧度
    pitch = np.deg2rad(roll_deg)
    yaw   = np.deg2rad(yaw_deg)
    roll  = np.deg2rad(pitch_deg)

    e, n, zone_number, zone_letter = utm.from_latlon(latitude=lat, longitude=lon)
    # 地图配准信息：
    # 左上顶点UTM坐标 (e₁, n₁) 和右下顶点UTM 坐标 (e₂, n₂)
    e1, n1, _ ,_ = utm.from_latlon(latitude=lat_1, longitude=lon_1, force_zone_number=zone_number, force_zone_letter=zone_letter)
    e2, n2, _, _ = utm.from_latlon(latitude=lat_2, longitude=lon_2, force_zone_number=zone_number, force_zone_letter=zone_letter)

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
    map_poly = []
    for pt in ground_pts_utm:
        E, N = pt
        u_map = ((E - e1) / (e2 - e1)) * W_map
        v_map = ((n1 - N) / (n1 - n2)) * H_map
        map_poly.append([u_map, v_map])
    map_poly = np.array(map_poly, dtype=np.int32)
    src_pts = np.array(map_poly, dtype=np.float32)

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

    # 保存结果
    cv2.imwrite(save_path, cropped)
    # print("可视区域结果已保存！")

    # src_pts = np.array(map_poly, dtype=np.float32)
    dst_pts = cam_corners.astype(np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    simulated_view = cv2.warpPerspective(map_img, M, (width_cam, height_cam))

    cv2.imwrite(warp_path, simulated_view)
    # print("warp结果已保存！")

    # 计算凸四边形的最小外接旋转矩形（斜长方形）
    # cv2.minAreaRect 返回一个元组：((center_x, center_y), (width, height), angle)
    min_rect = cv2.minAreaRect(src_pts)

    # 通过 cv2.boxPoints 得到矩形的 4 个顶点
    box = cv2.boxPoints(min_rect)
    box = np.array(box, dtype="float32")

    box = order_points(box)
    box = best_cyclic_order(box, src_pts)

    coordinates = []
    # 左上角对应的UTM坐标
    for i in range(4):
        e_p = box[i][0] * (e2 - e1) / W_map + e1
        n_p = n1 - box[i][1] * (n1 - n2) / H_map
        coordinates.append([e_p, n_p])
    coordinates = np.array(coordinates, dtype=np.float32)
    
    row = (
        drone_name,
        zone_number,
        zone_letter,
        coordinates[0, 0], coordinates[0, 1],
        coordinates[1, 0], coordinates[1, 1],
        coordinates[2, 0], coordinates[2, 1],
        coordinates[3, 0], coordinates[3, 1]
    )

    # 将这一行放入 DataFrame，然后追加写入 CSV（不写入表头）
    df_row = pd.DataFrame([row], columns=columns)
    df_row.to_csv(area_csv_path, mode='a', index=False, header=False)

    # 计算宽高
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

    M = cv2.getPerspectiveTransform(box, cam_corners_new)
    warped = cv2.warpPerspective(map_img, M, (maxWidth, maxHeight))

    # 去除图像边缘的黑边（不同边缘可能宽度不同）
    warped = remove_black_borders(warped)

    cv2.imwrite(area_path, warped)
    # print("斜长方形内的图像已保存！")

    print(f"处理完成：{drone_name}")

for data in data_list:
    trim_and_warp(drone_dir, map_path, data)