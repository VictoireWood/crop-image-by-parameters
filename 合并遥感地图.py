import cv2
import numpy as np
import csv
import pandas as pd

# ======== ① 参数设置 ========
# 假设7张地图的文件路径（请修改为实际文件路径）
map_paths = [
    r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\000000.jpg",
    r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\000001.jpg",
    r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\000002.jpg",
    r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\000003.jpg",
    r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\000004.jpg",
    r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\000005.jpg",
    r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\000006.jpg",
]

# CSV 文件路径，注意使用原始字符串避免转义问题
csv_path = r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\map.csv"

# 每幅图像的像素尺寸均为8000×8000
img_width = 8000
img_height = 8000

# 遥感地图的地理配准信息（经纬度），注意：经度通常增大表示向东，
# 纬度增大表示向北。
# 这里给出的6幅图每张对应的左下角经纬度（lon, lat）

# 读取 CSV 文件
df = pd.read_csv(csv_path)

# 打印前几行，检查数据（可选）
print(df.head())

# 构造左下角经纬度列表（ll_coords）
ll_coords = list(zip(df['leftdown_lon'], df['leftdown_lat']))

# 构造右上角经纬度列表（rt_coords）
rt_coords = list(zip(df['righttop_lon'], df['righttop_lat']))

# 输出结果
print("左下角经纬度列表:")
for coord in ll_coords:
    print(coord)

print("\n右上角经纬度列表:")
for coord in rt_coords:
    print(coord)

# —— 示例（仅供参考，请替换为你的实际数据）——
# ll_coords = [(100.0, 20.0), (101.0,20.5), (100.5,20.2), (100.8,19.8), (100.2,20.1), (101.2,20.3)]
# rt_coords = [(100.5, 20.5), (101.5,21.0), (101.0,20.7), (101.3,20.3), (100.7,20.6), (101.7,20.8)]

# ======== ② 计算全局经纬度范围 ========
# 对于每幅图，图像覆盖的经度区间为 [lon_left, lon_right]，纬度区间为 [lat_bottom, lat_top]。
global_left   = min(lon for (lon, lat) in ll_coords)
global_bottom = min(lat for (lon, lat) in ll_coords)
global_right  = max(lon for (lon, lat) in rt_coords)
global_top    = max(lat for (lon, lat) in rt_coords)

# 合并后地图的左上角经纬度和右下角经纬度
merged_top_left  = (global_left, global_top)
merged_bot_right = (global_right, global_bottom)
print("合并后地图左上角经纬度：", merged_top_left)
print("合并后地图右下角经纬度：", merged_bot_right)

filename = f"lon1_{global_left}@lat1_{global_top}@lon2_{global_right}@lat2_{global_bottom}"
save_path = rf"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\{filename}.jpg"

# ======== ③ 确定目标分辨率（度/像素） ========
# 对于每幅图，分辨率（经度方向、纬度方向）：
res_x_list = []
res_y_list = []
for i in range(len(map_paths)):
    lon_left, lat_bottom = ll_coords[i]
    lon_right, lat_top   = rt_coords[i]
    res_x = (lon_right - lon_left) / img_width    # 经度每像素度数
    res_y = (lat_top - lat_bottom) / img_height     # 纬度每像素度数
    res_x_list.append(res_x)
    res_y_list.append(res_y)
avg_res_x = np.mean(res_x_list)
avg_res_y = np.mean(res_y_list)
#print("目标分辨率：", avg_res_x, avg_res_y)

# ======== ④ 计算拼接后大地图的尺寸（像素） ========
mosaic_width  = int(round((global_right - global_left) / avg_res_x))
mosaic_height = int(round((global_top - global_bottom) / avg_res_y))

# mosaic_width  = int((global_right - global_left) / avg_res_x)
# mosaic_height = int((global_top - global_bottom) / avg_res_y)
print("合并后地图像素尺寸：", mosaic_width, "×", mosaic_height)

# 创建一个全黑的背景图（3通道，类型 uint8）
mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)

# 坐标转换公式：
# 对于任一经纬度 (lon, lat) 属于全图，
# 径向坐标 u = (lon - global_left) / avg_res_x
# 垂直方向坐标 v = (global_top - lat) / avg_res_y
# 注意：v 的计算中 global_top 对应图像第0行

# ======== ⑤ 依次将各个地图图像放入大地图 ========
for i, path in enumerate(map_paths):
    # 读取图像
    img = cv2.imread(path)
    if img is None:
        print("无法读取图像：", path)
        continue

    # 取当前图的地理范围
    lon_left, lat_bottom = ll_coords[i]
    lon_right, lat_top   = rt_coords[i]

    # 计算该图在大地图中的目标尺寸（像素）
    target_width  = int(round((lon_right - lon_left) / avg_res_x))
    target_height = int(round((lat_top - lat_bottom) / avg_res_y))

    # target_width  = int((lon_right - lon_left) / avg_res_x)
    # target_height = int((lat_top - lat_bottom) / avg_res_y)

    # 根据目标尺寸调整当前图大小（注意：cv2.resize 的尺寸是 (width, height)）
    resized_img = cv2.resize(img, (target_width, target_height))
    
    # 计算该图在大地图中的放置偏移（目标大图中左上角对应的像素坐标）
    x_offset = int(round((lon_left - global_left) / avg_res_x))
    y_offset = int(round((global_top - lat_top) / avg_res_y))

    # x_offset = int((lon_left - global_left) / avg_res_x)
    # y_offset = int((global_top - lat_top) / avg_res_y)

    # 将当前图贴入大地图。若区域内已有数据，
    # 这里简单使用覆盖操作，如果需要融合可进一步处理。
    mosaic[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = resized_img

# 保存合并后的大地图
cv2.imwrite(save_path, mosaic)
print("合并后的大地图已保存为 mosaic.jpg")
