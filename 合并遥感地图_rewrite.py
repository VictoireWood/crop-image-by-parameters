import cv2
import numpy as np
import pandas as pd

# ======== ① 参数设置 ========
map_paths = [
    r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\000000.jpg",
    r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\000001.jpg",
    r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\000002.jpg",
    r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\000003.jpg",
    r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\000004.jpg",
    r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\000005.jpg",
    r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\000006.jpg",
]

csv_path = r"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\map.csv"
img_width = 8000
img_height = 8000

# 读取 CSV 文件，构造左下角与右上角经纬度列表
df = pd.read_csv(csv_path)
ll_coords = list(zip(df['leftdown_lon'], df['leftdown_lat']))
rt_coords = list(zip(df['righttop_lon'], df['righttop_lat']))

# ======== ② 计算全局经纬度范围 ========
global_left   = min(lon for (lon, lat) in ll_coords)
global_bottom = min(lat for (lon, lat) in ll_coords)
global_right  = max(lon for (lon, lat) in rt_coords)
global_top    = max(lat for (lon, lat) in rt_coords)

merged_top_left  = (global_left, global_top)
merged_bot_right = (global_right, global_bottom)
print("合并后地图左上角经纬度：", merged_top_left)
print("合并后地图右下角经纬度：", merged_bot_right)

filename = f"lon1_{global_left}@lat1_{global_top}@lon2_{global_right}@lat2_{global_bottom}"
save_path = rf"D:\Dataset\qingxie\Match-Dataset-train\gs202533-ir\satellite\{filename}.jpg"

# ======== ③ 计算目标分辨率（度/像素） ========
res_x_list, res_y_list = [], []
for i in range(len(map_paths)):
    lon_left, lat_bottom = ll_coords[i]
    lon_right, lat_top   = rt_coords[i]
    res_x = (lon_right - lon_left) / img_width
    res_y = (lat_top - lat_bottom) / img_height
    res_x_list.append(res_x)
    res_y_list.append(res_y)
avg_res_x = np.mean(res_x_list)
avg_res_y = np.mean(res_y_list)

# ======== ④ 计算拼接后大地图的尺寸（像素） ========
mosaic_width  = int(round((global_right - global_left) / avg_res_x))
mosaic_height = int(round((global_top - global_bottom) / avg_res_y))
print("合并后地图像素尺寸：", mosaic_width, "×", mosaic_height)
mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)

# ======== ⑥ 依次将各个地图图像放入大地图 ========
for i, path in enumerate(map_paths):
    img = cv2.imread(path)
    if img is None:
        print("无法读取图像：", path)
        continue

    # 当前图的地理范围
    lon_left, lat_bottom = ll_coords[i]
    lon_right, lat_top   = rt_coords[i]

    # 计算目标尺寸，使用 floor 和 ceil 保证没有缝隙
    x_offset = int(np.floor((lon_left - global_left) / avg_res_x))
    y_offset = int(np.floor((global_top - lat_top) / avg_res_y))
    target_width  = int(np.ceil((lon_right - lon_left) / avg_res_x))
    target_height = int(np.ceil((lat_top - lat_bottom) / avg_res_y))

    resized_img = cv2.resize(img, (target_width, target_height))
    
    # 粘贴时检查大地图区域是否已有数据（重叠区域）
    mosaic_roi = mosaic[y_offset:y_offset+target_height, x_offset:x_offset+target_width]

    mosaic[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = resized_img

# 保存合并后的大地图
cv2.imwrite(save_path, mosaic)
print("合并后的大地图已保存为", save_path)
